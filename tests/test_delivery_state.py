"""Delivery-state machine tests — MongoDB is the source of truth.

Covers the ``deletion_log_delivery`` state machine in
``SWAGGYMUSIC.utils.deletion_audit``:

* persistent records with a unique event_key, created BEFORE sending,
  tracked through pending / sending / sent (+ honest uncertain);
* ATOMIC claims — concurrent workers can never send the same event;
* lease semantics — a live claim is protected, an expired lease is
  taken over, and a late SUCCESS still records its evidence;
* startup recovery — reads ONLY MongoDB (Telegram history is never
  scanned), recovers pending + uncertain + abandoned-sending records,
  and never touches records already marked "sent";
* crash recovery — including the crash-after-send-before-DB-update
  window, which is handled at-least-once (sendMessage has no
  idempotency key — the honest limitation is asserted, not hidden);
* legacy migration — old-format audit records are seeded into delivery
  records without loss, and an upgrade never duplicates a delivered log.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pyrogram.errors import BadRequest

from SWAGGYMUSIC.utils import deletion_audit as audit
from SWAGGYMUSIC.utils import logger as logger_mod

pytestmark = pytest.mark.asyncio


def _forbid_history(monkeypatch):
    """Any Telegram history scan during recovery is a design violation."""
    from SWAGGYMUSIC import app

    def _boom(*args, **kwargs):
        raise AssertionError("Telegram history was scanned — forbidden")

    monkeypatch.setattr(app, "get_chat_history", _boom)


def _logger_env(monkeypatch, logger_id=-1005559001):
    import config
    from SWAGGYMUSIC import app

    monkeypatch.setattr(config, "LOGGER_ID", logger_id, raising=False)
    sent = AsyncMock(return_value=SimpleNamespace(id=555900))
    monkeypatch.setattr(app, "send_message", sent)
    logger_mod._DELIVERED_EVENTS.clear()
    return sent


async def _audit_record(mock_db, chat_id, message_id, **extra):
    doc = {
        "event_key": f"{chat_id}:{message_id}",
        "chat_id": chat_id,
        "message_id": message_id,
        "reason": "BANNED_WORD",
        "source": "BANNEDWORDS",
        "channel_title": "Ch",
        "content_text": "hello",
        "media_type": "text",
    }
    doc.update(extra)
    await mock_db.deletion_audit_logs.insert_one(doc)
    return doc["event_key"]


async def _delivery(mock_db, chat_id, message_id, kind="deleted"):
    return await mock_db.deletion_log_delivery.find_one(
        {"event_key": f"{chat_id}:{message_id}", "kind": kind},
    )


async def _close_other_deliveries(mock_db, keep_key):
    """Mark every OTHER test's leftover delivery record as already sent.

    The test database is shared (mongomock has no per-test cleanup — see
    conftest.py), and the startup sweep correctly targets EVERY
    pending/uncertain/sending record.  Sweep tests only want to reason
    about their own event, so leftovers from earlier tests are closed
    out first — this mirrors production, where a sweep would have
    already delivered them.
    """
    await mock_db.deletion_log_delivery.update_many(
        {"event_key": {"$ne": keep_key}},
        {"$set": {"status": "sent", "log_message_id": 1}},
    )


async def _new_delivery(mock_db, chat_id, message_id, *, kind="deleted",
                        status="pending", **extra):
    """Insert a delivery record in an arbitrary state."""
    now = datetime.now(timezone.utc)
    doc = {
        "event_key": f"{chat_id}:{message_id}",
        "kind": kind,
        "chat_id": chat_id,
        "message_id": message_id,
        "status": status,
        "log_event_tag": logger_mod.deletion_log_event_tag(
            f"{chat_id}:{message_id}", kind),
        "log_message_id": None,
        "logs_chat_id": None,
        "attempts": 0,
        "detail": "",
        "created_at": now,
        "updated_at": now,
    }
    doc.update(extra)
    await mock_db.deletion_log_delivery.insert_one(doc)
    return doc["event_key"]


# ---------------------------------------------------------------------------
# Records + claims (atomicity, leases, evidence)
# ---------------------------------------------------------------------------

class TestDeliveryRecordsAndClaims:
    async def test_ensure_creates_record_pending_before_send(self, mock_db):
        chat_id, msg_id = -1007771001, 71001
        rec = await audit.ensure_delivery_record(
            f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
        )
        assert rec["status"] == audit.DELIVERY_PENDING
        assert rec["event_key"] == f"{chat_id}:{msg_id}"
        assert rec["kind"] == "deleted"
        assert rec["log_message_id"] is None
        assert rec["attempts"] == 0
        # idempotent — a second ensure returns the SAME record
        again = await audit.ensure_delivery_record(
            f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
        )
        assert again["created_at"] == rec["created_at"]
        count = await mock_db.deletion_log_delivery.count_documents(
            {"event_key": f"{chat_id}:{msg_id}", "kind": "deleted"},
        )
        assert count == 1

    async def test_claims_are_atomic_only_one_worker_wins(self, mock_db):
        chat_id, msg_id = -1007771002, 71002
        await audit.ensure_delivery_record(
            f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
        )
        results = await asyncio.gather(
            audit.claim_delivery(f"{chat_id}:{msg_id}", "deleted", "worker-1"),
            audit.claim_delivery(f"{chat_id}:{msg_id}", "deleted", "worker-2"),
        )
        winners = [r for r in results if r]
        assert len(winners) == 1
        assert winners[0]["status"] == audit.DELIVERY_SENDING
        assert winners[0]["claimed_by"] in ("worker-1", "worker-2")
        assert winners[0]["attempts"] == 1
        assert winners[0]["lease_expires_at"] is not None

    async def test_fresh_lease_blocks_takeover_expired_lease_allows_it(
        self, mock_db,
    ):
        chat_id, msg_id = -1007771003, 71003
        await audit.ensure_delivery_record(
            f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
        )
        first = await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-1",
        )
        assert first is not None
        # a live lease belongs to a running worker — nobody else may send
        second = await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-2",
        )
        assert second is None

        # the owner dies before finishing → the lease expires → takeover
        await mock_db.deletion_log_delivery.update_one(
            {"event_key": f"{chat_id}:{msg_id}", "kind": "deleted"},
            {"$set": {
                "lease_expires_at": datetime.now(timezone.utc)
                - timedelta(seconds=1),
            }},
        )
        takeover = await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-3",
        )
        assert takeover is not None
        assert takeover["claimed_by"] == "worker-3"
        assert takeover["attempts"] == 2

    async def test_uncertain_not_claimable_at_runtime_until_grace(
        self, mock_db,
    ):
        chat_id, msg_id = -1007771004, 71004
        await _new_delivery(
            mock_db, chat_id, msg_id, status=audit.DELIVERY_UNCERTAIN,
            uncertain_since=datetime.now(timezone.utc),
        )
        got = await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-1",
        )
        assert got is None  # inside the grace period — no blind resend

        await mock_db.deletion_log_delivery.update_one(
            {"event_key": f"{chat_id}:{msg_id}", "kind": "deleted"},
            {"$set": {
                "uncertain_since": datetime.now(timezone.utc)
                - timedelta(seconds=audit.UNCERTAIN_RUNTIME_GRACE_SECONDS + 1),
            }},
        )
        got = await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-2",
        )
        assert got is not None  # grace elapsed → safe to attempt again
        assert got["status"] == audit.DELIVERY_SENDING
        assert got["uncertain_since"] is None

    async def test_uncertain_always_claimable_at_startup(self, mock_db):
        chat_id, msg_id = -1007771005, 71005
        await _new_delivery(
            mock_db, chat_id, msg_id, status=audit.DELIVERY_UNCERTAIN,
            uncertain_since=datetime.now(timezone.utc),  # grace NOT elapsed
        )
        got = await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "startup-sweep", startup=True,
        )
        assert got is not None

    async def test_mark_sent_requires_sending_without_message_id(
        self, mock_db,
    ):
        chat_id, msg_id = -1007771006, 71006
        await audit.ensure_delivery_record(
            f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
        )
        # cannot mark "sent" while still "pending" — only a claimed
        # (sending) record accepts the verdict
        ok = await audit.mark_delivery_sent(
            f"{chat_id}:{msg_id}", "deleted", 70001,
        )
        assert ok is False
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == audit.DELIVERY_PENDING

        claimed = await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-1",
        )
        assert claimed is not None
        ok = await audit.mark_delivery_sent(
            f"{chat_id}:{msg_id}", "deleted", 70001, logs_chat_id=-1005559001,
        )
        assert ok is True
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == audit.DELIVERY_SENT
        assert rec["log_message_id"] == 70001
        assert rec["logs_chat_id"] == -1005559001
        assert rec["sent_at"] is not None
        # a second verdict cannot overwrite the recorded evidence
        ok2 = await audit.mark_delivery_sent(
            f"{chat_id}:{msg_id}", "deleted", 70002,
        )
        assert ok2 is False
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["log_message_id"] == 70001

    async def test_late_success_from_presumed_dead_worker_still_records(
        self, mock_db,
    ):
        """Worker A claimed, sent (message 71007), then stalled past its
        lease.  Worker B takes over.  A's late success is still the best
        anti-duplicate evidence and MUST be recorded — B stands down."""
        chat_id, msg_id = -1007771007, 71007
        await audit.ensure_delivery_record(
            f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
        )
        await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-a",
        )
        # A's lease expires while it is stalled; B claims
        await mock_db.deletion_log_delivery.update_one(
            {"event_key": f"{chat_id}:{msg_id}", "kind": "deleted"},
            {"$set": {
                "lease_expires_at": datetime.now(timezone.utc)
                - timedelta(seconds=1),
            }},
        )
        b_claim = await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-b",
        )
        assert b_claim is not None

        # A finally returns with Telegram's confirmation (no owner check —
        # evidence wins while the record is "sending" without a message id)
        ok = await audit.mark_delivery_sent(
            f"{chat_id}:{msg_id}", "deleted", 71007,
        )
        assert ok is True
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == audit.DELIVERY_SENT
        assert rec["log_message_id"] == 71007

    async def test_definite_failure_returns_to_pending_with_error(
        self, mock_db,
    ):
        chat_id, msg_id = -1007771008, 71008
        await audit.ensure_delivery_record(
            f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
        )
        await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-1",
        )
        await audit.mark_delivery_outcome(
            f"{chat_id}:{msg_id}", "deleted", "failed", error="BadRequest",
        )
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == audit.DELIVERY_PENDING  # retryable
        assert rec["last_error"] == "BadRequest"
        assert rec["lease_expires_at"] is None  # claim released
        assert rec["claimed_by"] is None

        await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-2",
        )
        await audit.mark_delivery_outcome(
            f"{chat_id}:{msg_id}", "deleted", "uncertain", error="lost",
        )
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == audit.DELIVERY_UNCERTAIN
        assert rec["uncertain_since"] is not None


# ---------------------------------------------------------------------------
# Startup recovery — MongoDB only, never Telegram history
# ---------------------------------------------------------------------------

class TestStartupRecoveryMongoDbOnly:
    async def test_sweep_delivers_pending_without_touching_telegram_history(
        self, monkeypatch, mock_db,
    ):
        _logger_env(monkeypatch, logger_id=-1005559101)
        _forbid_history(monkeypatch)
        chat_id, msg_id = -1007771101, 71101
        await _audit_record(mock_db, chat_id, msg_id)
        await _new_delivery(mock_db, chat_id, msg_id, status="pending")
        await _close_other_deliveries(
            mock_db, f"{chat_id}:{msg_id}",
        )

        count = await audit.recover_interrupted_deletion_logs()
        assert count == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 555900

        # idempotent: a second sweep finds nothing left to do
        count2 = await audit.recover_interrupted_deletion_logs()
        assert count2 == 0

    async def test_sweep_skips_sent_records_entirely(self, monkeypatch, mock_db):
        sent = _logger_env(monkeypatch, logger_id=-1005559102)
        _forbid_history(monkeypatch)
        chat_id, msg_id = -1007771102, 71102
        await _new_delivery(
            mock_db, chat_id, msg_id, status="sent", log_message_id=71102,
        )
        await _close_other_deliveries(mock_db, f"{chat_id}:{msg_id}")
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 0
        sent.assert_not_awaited()
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 71102  # untouched

    async def test_sweep_recovers_abandoned_sending_record_regardless_of_lease(
        self, monkeypatch, mock_db,
    ):
        """At startup ANY "sending" record is abandoned — its owner process
        is gone, so the lease from the previous process is meaningless
        (a fast restart must not strand the log until the lease expires)."""
        _logger_env(monkeypatch, logger_id=-1005559103)
        _forbid_history(monkeypatch)
        chat_id, msg_id = -1007771103, 71103
        await _new_delivery(
            mock_db, chat_id, msg_id, status="sending",
            claimed_by="dead-worker",
            claimed_at=datetime.now(timezone.utc),
            lease_expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=audit.DELIVERY_LEASE_SECONDS),  # FRESH lease
            attempts=1,
        )
        await _close_other_deliveries(mock_db, f"{chat_id}:{msg_id}")
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 555900

    async def test_sweep_recovers_uncertain_records(self, monkeypatch, mock_db):
        _logger_env(monkeypatch, logger_id=-1005559104)
        _forbid_history(monkeypatch)
        chat_id, msg_id = -1007771104, 71104
        await _new_delivery(
            mock_db, chat_id, msg_id, status="uncertain",
            uncertain_since=datetime.now(timezone.utc),
        )
        await _close_other_deliveries(mock_db, f"{chat_id}:{msg_id}")
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"

    async def test_sweep_re_records_honest_outcome_on_failure(
        self, monkeypatch, mock_db,
    ):
        """A sweep attempt that Telegram refuses stays retryable — the
        record is not lost and the next sweep tries again."""
        import config
        from SWAGGYMUSIC import app

        _logger_env(monkeypatch, logger_id=-1005559105)
        _forbid_history(monkeypatch)
        chat_id, msg_id = -1007771105, 71105
        await _new_delivery(mock_db, chat_id, msg_id, status="pending")
        await _close_other_deliveries(mock_db, f"{chat_id}:{msg_id}")

        failing = AsyncMock(side_effect=BadRequest("TEST"))
        monkeypatch.setattr(app, "send_message", failing)
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "pending"  # honest: still not delivered

        working = AsyncMock(return_value=SimpleNamespace(id=71105))
        monkeypatch.setattr(app, "send_message", working)
        count2 = await audit.recover_interrupted_deletion_logs()
        assert count2 == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 71105

    async def test_sweep_recovers_undelivered_alerts_too(
        self, monkeypatch, mock_db,
    ):
        _logger_env(monkeypatch, logger_id=-1005559106)
        _forbid_history(monkeypatch)
        chat_id, msg_id = -1007771106, 71106
        await _audit_record(mock_db, chat_id, msg_id)
        await _new_delivery(
            mock_db, chat_id, msg_id, kind="archive_alert", status="pending",
        )
        await _close_other_deliveries(mock_db, f"{chat_id}:{msg_id}")
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 1
        rec = await _delivery(mock_db, chat_id, msg_id, kind="archive_alert")
        assert rec["status"] == "sent"
        mirror = await mock_db.deletion_audit_logs.find_one(
            {"event_key": f"{chat_id}:{msg_id}"},
        )
        assert mirror["failure_alert_status"] == "sent"


# ---------------------------------------------------------------------------
# Crash recovery — including the honest at-least-once window
# ---------------------------------------------------------------------------

class TestCrashRecovery:
    async def test_crash_after_send_before_db_update_is_re_delivered(
        self, monkeypatch, mock_db,
    ):
        """THE honest window: Telegram accepted the send, the process died
        before the success was persisted.  The record is stuck in
        "sending"; the startup sweep re-delivers it.  The log cannot be
        lost; the possible duplicate is the documented at-least-once
        trade-off (sendMessage has no idempotency key)."""
        import config
        from SWAGGYMUSIC import app

        _logger_env(monkeypatch, logger_id=-1005559201)
        _forbid_history(monkeypatch)
        monkeypatch.setattr(config, "LOGGER_ID", -1005559201, raising=False)
        chat_id, msg_id = -1007771201, 71201
        await _audit_record(mock_db, chat_id, msg_id)
        await _new_delivery(
            mock_db, chat_id, msg_id, status="sending",
            claimed_by="crashed-worker",
            lease_expires_at=datetime.now(timezone.utc)
            + timedelta(seconds=audit.DELIVERY_LEASE_SECONDS),
            attempts=1,
        )
        await _close_other_deliveries(mock_db, f"{chat_id}:{msg_id}")

        working = AsyncMock(return_value=SimpleNamespace(id=71201))
        monkeypatch.setattr(app, "send_message", working)
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 1
        assert working.await_count == 1  # re-delivered — at-least-once
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 71201

    async def test_crash_before_send_leaves_no_duplicate(
        self, monkeypatch, mock_db,
    ):
        """A worker that died BEFORE Telegram accepted (record stuck in
        "sending", nothing delivered) is recovered exactly once."""
        _logger_env(monkeypatch, logger_id=-1005559202)
        _forbid_history(monkeypatch)
        chat_id, msg_id = -1007771202, 71202
        await _new_delivery(
            mock_db, chat_id, msg_id, status="sending",
            claimed_by="crashed-worker", attempts=1,
        )
        await _close_other_deliveries(mock_db, f"{chat_id}:{msg_id}")
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["attempts"] == 2

        # no double-delivery on the next sweep
        count2 = await audit.recover_interrupted_deletion_logs()
        assert count2 == 0

    async def test_runtime_worker_stands_down_when_evidence_appears(
        self, monkeypatch, mock_db,
    ):
        """Worker B claimed an abandoned event; the presumed-dead worker A
        records its late success BEFORE B's send.  B's last-moment guard
        sees the evidence in MongoDB and stands down — no second send."""
        import config
        from SWAGGYMUSIC import app

        _logger_env(monkeypatch, logger_id=-1005559203)
        chat_id, msg_id = -1007771203, 71203
        await _audit_record(mock_db, chat_id, msg_id)
        await audit.ensure_delivery_record(
            f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
        )
        await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-a",
        )
        # A stalls past its lease; B takes over
        await mock_db.deletion_log_delivery.update_one(
            {"event_key": f"{chat_id}:{msg_id}", "kind": "deleted"},
            {"$set": {
                "lease_expires_at": datetime.now(timezone.utc)
                - timedelta(seconds=1),
            }},
        )
        b_claim = await audit.claim_delivery(
            f"{chat_id}:{msg_id}", "deleted", "worker-b",
        )
        assert b_claim is not None

        # A's send HAD succeeded — its evidence lands now
        await audit.mark_delivery_sent(
            f"{chat_id}:{msg_id}", "deleted", 71203,
        )
        sent = AsyncMock(return_value=SimpleNamespace(id=999999))
        monkeypatch.setattr(app, "send_message", sent)

        # B re-checks before sending via the delivery flow and must not
        # produce a second Telegram message for a delivered log.
        result = await audit.deliver_deletion_log(
            f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
        )
        # B's claim attempt loses (record is already "sent") or its
        # last-moment guard fires — either way: no second send.
        sent.assert_not_awaited()
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 71203

    async def test_concurrent_full_flows_produce_exactly_one_send(
        self, monkeypatch, mock_db,
    ):
        """Two full delivery flows racing on the same event (e.g. the same
        deletion processed by two handlers) send exactly one message."""
        from SWAGGYMUSIC import app

        _logger_env(monkeypatch, logger_id=-1005559204)
        chat_id, msg_id = -1007771204, 71204
        await _audit_record(mock_db, chat_id, msg_id)

        async def _slow_send(*args, **kwargs):
            await asyncio.sleep(0.05)
            return SimpleNamespace(id=71204)

        sent = AsyncMock(side_effect=_slow_send)
        monkeypatch.setattr(app, "send_message", sent)

        results = await asyncio.gather(
            audit.deliver_deletion_log(
                f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
            ),
            audit.deliver_deletion_log(
                f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
            ),
        )
        assert sent.await_count == 1
        outcomes = list(results)
        assert outcomes.count(audit.DELIVERY_SENT) == 1
        assert "skipped" in outcomes
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 71204


# ---------------------------------------------------------------------------
# Legacy migration — an upgrade loses nothing and duplicates nothing
# ---------------------------------------------------------------------------

class TestLegacyMigration:
    async def test_legacy_failed_log_is_seeded_and_delivered(
        self, monkeypatch, mock_db,
    ):
        """An old-format record left retryable ("failed") by a crashed
        legacy process is migrated and delivered by the sweep."""
        _logger_env(monkeypatch, logger_id=-1005559301)
        _forbid_history(monkeypatch)
        chat_id, msg_id = -1007771301, 71301
        event_key = await _audit_record(
            mock_db, chat_id, msg_id,
            log_status="failed", log_kind="deleted",
            deletion_log_deleted_sent=False,
        )
        assert event_key
        await _close_other_deliveries(mock_db, event_key)
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec is not None
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 555900

    async def test_legacy_uncertain_attempt_is_seeded_as_uncertain(
        self, monkeypatch, mock_db,
    ):
        """A legacy attempt interrupted mid-send (claim held, no verdict)
        cannot be proven delivered without a history scan — it becomes an
        "uncertain" delivery record and the sweep recovers it."""
        _logger_env(monkeypatch, logger_id=-1005559302)
        _forbid_history(monkeypatch)
        chat_id, msg_id = -1007771302, 71302
        await _audit_record(
            mock_db, chat_id, msg_id,
            log_status="attempting", log_kind="deleted",
            deletion_log_deleted_sent=True,
        )
        await _close_other_deliveries(
            mock_db, f"{chat_id}:{msg_id}",
        )
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"  # recovered (at-least-once)

    async def test_upgrade_never_duplicates_a_delivered_legacy_log(
        self, monkeypatch, mock_db,
    ):
        """A legacy record whose log WAS delivered (log_status "sent" with
        Telegram's message id) must never be re-sent — the delivery
        record is seeded as "sent" when the event is touched again."""
        _logger_env(monkeypatch, logger_id=-1005559303)
        _forbid_history(monkeypatch)
        chat_id, msg_id = -1007771303, 71303
        await _audit_record(
            mock_db, chat_id, msg_id,
            log_status="sent", log_kind="deleted", log_message_id=800003,
            deletion_log_deleted_sent=True,
        )
        await _close_other_deliveries(
            mock_db, f"{chat_id}:{msg_id}",
        )

        # The event is re-processed after the upgrade (worker retry)…
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="re-processed",
        )
        sent = AsyncMock(return_value=SimpleNamespace(id=999913))
        from SWAGGYMUSIC import app

        monkeypatch.setattr(app, "send_message", sent)
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="again",
        )
        sent.assert_not_awaited()
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 800003  # the ORIGINAL Telegram id

        # …and the sweep has nothing to recover either.
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 0

    async def test_legacy_alert_state_is_migrated(
        self, monkeypatch, mock_db,
    ):
        """A legacy "pending" alert becomes a retryable alert delivery
        record; the sweep delivers it once."""
        _logger_env(monkeypatch, logger_id=-1005559304)
        _forbid_history(monkeypatch)
        chat_id, msg_id = -1007771304, 71304
        await _audit_record(
            mock_db, chat_id, msg_id, failure_alert_status="pending",
        )
        await _close_other_deliveries(
            mock_db, f"{chat_id}:{msg_id}",
        )
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 1
        rec = await _delivery(mock_db, chat_id, msg_id, kind="archive_alert")
        assert rec["status"] == "sent"

        count2 = await audit.recover_interrupted_deletion_logs()
        assert count2 == 0


# ---------------------------------------------------------------------------
# Transport outcomes (logger layer contract used by the machine)
# ---------------------------------------------------------------------------

class TestTransportOutcomes:
    async def test_send_post_deletion_log_outcomes(self, monkeypatch):
        import config
        from SWAGGYMUSIC import app
        from pyrogram.errors import BadRequest

        _logger_env(monkeypatch, logger_id=-1005559401)
        record = {
            "event_key": "-1007771401:71401",
            "chat_id": -1007771401,
            "message_id": 71401,
            "channel_title": "Ch",
            "content_text": "hello",
            "media_type": "text",
        }

        outcome, mid = await logger_mod.send_post_deletion_log(
            record, kind="deleted", event_key=record["event_key"],
        )
        assert outcome == "sent"
        assert mid == 555900

        monkeypatch.setattr(
            app, "send_message", AsyncMock(side_effect=BadRequest("TEST")),
        )
        outcome, mid = await logger_mod.send_post_deletion_log(
            record, kind="deleted", event_key=record["event_key"],
        )
        assert outcome == "failed"
        assert mid is None

        monkeypatch.setattr(
            app, "send_message",
            AsyncMock(side_effect=RuntimeError("connection lost mid-send")),
        )
        outcome, mid = await logger_mod.send_post_deletion_log(
            record, kind="deleted", event_key=record["event_key"],
        )
        assert outcome == "uncertain"
        assert mid is None

    async def test_long_floodwait_is_failed_short_is_slept_out(
        self, monkeypatch,
    ):
        import config
        from SWAGGYMUSIC import app
        from pyrogram.errors import FloodWait

        _logger_env(monkeypatch, logger_id=-1005559402)
        record = {
            "event_key": "-1007771402:71402",
            "chat_id": -1007771402,
            "message_id": 71402,
            "channel_title": "Ch",
            "content_text": "hello",
            "media_type": "text",
        }

        monkeypatch.setattr(
            app, "send_message", AsyncMock(side_effect=FloodWait(value=600)),
        )
        outcome, _mid = await logger_mod.send_post_deletion_log(
            record, kind="deleted", event_key=record["event_key"],
        )
        assert outcome == "failed"  # definite refusal, retryable

        responses = [FloodWait(value=2), SimpleNamespace(id=71402)]

        async def _send(*args, **kwargs):
            result = responses.pop(0)
            if isinstance(result, Exception):
                raise result
            return result

        monkeypatch.setattr(app, "send_message", _send)
        monkeypatch.setattr(logger_mod.asyncio, "sleep", AsyncMock())
        outcome, mid = await logger_mod.send_post_deletion_log(
            record, kind="deleted", event_key=record["event_key"],
        )
        assert outcome == "sent"
        assert mid == 71402
