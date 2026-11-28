"""Batched recovery + periodic retry worker tests.

Covers the final improvements to the MongoDB-based deletion-log
recovery (``SWAGGYMUSIC.utils.deletion_audit``):

* BATCHING — pending records are processed in batches of 200 and the
  sweep keeps fetching further batches until every eligible record is
  handled; it never stops after the first 200, and a failure inside a
  batch never stops the remaining records or the following batches.
* PERIODIC RETRY — a background worker re-sweeps failed / uncertain /
  stale-claimed records on a fixed cadence (recovery no longer depends
  only on bot startup), with an EXPONENTIAL per-record backoff
  (``next_attempt_at``: 30s, 60s, 120s … capped at 1h) so nothing is
  ever hammered in a tight loop.
* RELIABILITY PRESERVED — MongoDB stays the only source of truth
  (Telegram history is NEVER scanned — any attempt to scan raises),
  event keys stay unique, claims stay atomic: concurrent workers
  (startup sweep, periodic sweep, runtime handlers) can never send the
  same event twice; a stale "sending" record is taken over safely
  (startup: regardless of its lease; runtime: only after the lease
  expired — a live worker is never disturbed).
"""
import asyncio
import contextlib
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock

import pytest
from pyrogram.errors import BadRequest

from SWAGGYMUSIC.utils import deletion_audit as audit
from SWAGGYMUSIC.utils import logger as logger_mod

pytestmark = pytest.mark.asyncio

# Every test in this file uses chat ids BELOW this base; the isolation
# helper closes out leftovers from other test files (the in-memory
# mongomock database is shared across the suite) by marking everything
# ABOVE the base as already sent / inert.
_MY_CHAT_BASE = -1008880000


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
    sent = AsyncMock(return_value=__import__("types").SimpleNamespace(id=555900))
    monkeypatch.setattr(app, "send_message", sent)
    logger_mod._DELIVERED_EVENTS.clear()
    return sent


async def _isolate(mock_db):
    """Neuter ALL leftover delivery/audit records from other tests.

    The in-memory mongomock database is shared across the whole suite
    (there is no per-test cleanup — see conftest.py), and both sweeps
    correctly target EVERY eligible record.  Every test in this file
    calls this helper FIRST and inserts its own records afterwards, so
    exact-count assertions only ever see records this test created.
    """
    await mock_db.deletion_log_delivery.update_many(
        {},
        {"$set": {"status": "sent", "log_message_id": 1}},
    )
    # A legacy-format audit record (log_status failed/pending/…) would
    # be seeded into a fresh pending delivery record by the startup
    # sweep — make old leftovers look delivered so seeding skips them.
    await mock_db.deletion_audit_logs.update_many(
        {},
        {"$set": {"log_status": "sent", "failure_alert_status": "sent"}},
    )


async def _bulk_pending(mock_db, count, *, base_chat=-1008881000,
                        base_msg=88000):
    """Insert ``count`` pending delivery records (+ matching audit rows)."""
    now = datetime.now(timezone.utc)
    chat_ids = [base_chat - i for i in range(1, count + 1)]
    deliveries = [{
        "event_key": f"{cid}:{base_msg + i}",
        "kind": "deleted",
        "chat_id": cid,
        "message_id": base_msg + i,
        "status": "pending",
        "log_event_tag": logger_mod.deletion_log_event_tag(
            f"{cid}:{base_msg + i}", "deleted"),
        "log_message_id": None,
        "logs_chat_id": None,
        "attempts": 0,
        "detail": "",
        "created_at": now,
        "updated_at": now,
    } for i, cid in enumerate(chat_ids, start=1)]
    await mock_db.deletion_log_delivery.insert_many(deliveries)
    await mock_db.deletion_audit_logs.insert_many([{
        "event_key": d["event_key"],
        "chat_id": d["chat_id"],
        "message_id": d["message_id"],
        "reason": "BANNED_WORD",
        "source": "BANNEDWORDS",
        "channel_title": "Ch",
        "content_text": "hello",
        "media_type": "text",
    } for d in deliveries])
    return [d["event_key"] for d in deliveries]


async def _delivery(mock_db, chat_id, message_id, kind="deleted"):
    return await mock_db.deletion_log_delivery.find_one(
        {"event_key": f"{chat_id}:{message_id}", "kind": kind},
    )


# ---------------------------------------------------------------------------
# Batching — the sweep never stops after the first 200 records
# ---------------------------------------------------------------------------

class TestBatching:
    async def test_batch_size_constant_is_200(self):
        assert audit.DELIVERY_BATCH_SIZE == 200

    async def test_backoff_schedule_is_exponential_and_capped(self):
        b = audit._backoff_seconds
        assert b(None) == 30.0 and b(0) == 30.0 and b(1) == 30.0
        assert [b(n) for n in range(1, 8)] == [
            30.0, 60.0, 120.0, 240.0, 480.0, 960.0, 1920.0,
        ]
        assert b(8) == audit.DELIVERY_BACKOFF_MAX_SECONDS == 3600.0
        assert b(10_000) == 3600.0  # huge attempt counter never overflows

    async def test_450_pending_records_fully_processed_across_batches(
        self, monkeypatch, mock_db,
    ):
        """More pending records than ONE batch: the sweep must keep
        fetching further 200-record batches until ALL are delivered."""
        _logger_env(monkeypatch, logger_id=-1005559501)
        _forbid_history(monkeypatch)
        await _isolate(mock_db)
        keys = await _bulk_pending(mock_db, 450)
        assert len(keys) == 450 > audit.DELIVERY_BATCH_SIZE

        from SWAGGYMUSIC import app

        count = await audit.recover_interrupted_deletion_logs(
            batch_size=audit.DELIVERY_BATCH_SIZE,
        )
        assert count == 450
        assert app.send_message.await_count == 450  # exactly one send each

        undelivered = await mock_db.deletion_log_delivery.count_documents(
            {"chat_id": {"$lte": _MY_CHAT_BASE}, "status": {"$ne": "sent"}},
        )
        assert undelivered == 0

        # idempotent: a second sweep finds nothing left to do
        count2 = await audit.recover_interrupted_deletion_logs()
        assert count2 == 0

    async def test_periodic_sweep_also_fully_drains_its_backlog(
        self, monkeypatch, mock_db,
    ):
        """The periodic retry sweep shares the batched engine: a backlog
        larger than one batch is fully processed in one pass."""
        _logger_env(monkeypatch, logger_id=-1005559502)
        _forbid_history(monkeypatch)
        await _isolate(mock_db)
        keys = await _bulk_pending(
            mock_db, 220, base_chat=-1008882000, base_msg=88200,
        )
        assert len(keys) > audit.DELIVERY_BATCH_SIZE

        from SWAGGYMUSIC import app

        count = await audit.retry_undelivered_deletion_logs(
            batch_size=audit.DELIVERY_BATCH_SIZE,
        )
        assert count == 220
        assert app.send_message.await_count == 220

    async def test_sweep_continues_after_a_batch_failure(
        self, monkeypatch, mock_db,
    ):
        """A raising delivery inside one batch must not stop the other
        records of that batch nor the following batches; the failed
        record stays retryable for the next sweep."""
        _logger_env(monkeypatch, logger_id=-1005559503)
        _forbid_history(monkeypatch)
        await _isolate(mock_db)
        keys = await _bulk_pending(
            mock_db, 250, base_chat=-1008883000, base_msg=88300,
        )
        poisoned = keys[5]  # inside the FIRST batch (insertion order)
        assert poisoned

        from SWAGGYMUSIC import app

        original = audit.deliver_deletion_log
        flag = {"raise": True}

        async def _wrapper(event_key, kind, chat_id, message_id, **kwargs):
            if flag["raise"] and event_key == poisoned:
                raise RuntimeError("simulated batch-level failure")
            return await original(
                event_key, kind, chat_id, message_id, **kwargs,
            )

        monkeypatch.setattr(audit, "deliver_deletion_log", _wrapper)

        count = await audit.recover_interrupted_deletion_logs(
            batch_size=200,
        )
        assert count == 249  # every record except the poisoned one
        assert app.send_message.await_count == 249

        rec = await _delivery(mock_db, *poisoned.split(":"))
        assert rec["status"] == "pending"  # honest: still not delivered

        # next sweep (poison removed) retries exactly that record
        flag["raise"] = False
        count2 = await audit.recover_interrupted_deletion_logs()
        assert count2 == 1
        rec = await _delivery(mock_db, *poisoned.split(":"))
        assert rec["status"] == "sent"

        undelivered = await mock_db.deletion_log_delivery.count_documents(
            {"chat_id": {"$lte": _MY_CHAT_BASE}, "status": {"$ne": "sent"}},
        )
        assert undelivered == 0


# ---------------------------------------------------------------------------
# Periodic retry — backoff, runtime-safe eligibility
# ---------------------------------------------------------------------------

class TestPeriodicRetry:
    async def test_failed_record_backs_off_then_is_retried(
        self, monkeypatch, mock_db,
    ):
        """A definitely-failed record is retried by the periodic sweep —
        but only AFTER its exponential backoff stamp elapsed."""
        import config
        from SWAGGYMUSIC import app

        _logger_env(monkeypatch, logger_id=-1005559511)
        _forbid_history(monkeypatch)
        await _isolate(mock_db)
        chat_id, msg_id = -1008884100, 88410
        now = datetime.now(timezone.utc)
        await mock_db.deletion_log_delivery.insert_one({
            "event_key": f"{chat_id}:{msg_id}", "kind": "deleted",
            "chat_id": chat_id, "message_id": msg_id,
            "status": "pending",
            "log_event_tag": logger_mod.deletion_log_event_tag(
                f"{chat_id}:{msg_id}", "deleted"),
            "log_message_id": None, "logs_chat_id": None,
            "attempts": 0, "detail": "",
            "created_at": now, "updated_at": now,
        })

        failing = AsyncMock(side_effect=BadRequest("TEST"))
        monkeypatch.setattr(app, "send_message", failing)

        count = await audit.retry_undelivered_deletion_logs()
        assert count == 1
        assert failing.await_count == 1

        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "pending"  # honest outcome
        assert rec["attempts"] == 1
        window = (
            rec["next_attempt_at"] - rec["updated_at"]
        ).total_seconds()
        assert 25 <= window <= 35  # first failure → 30s backoff

        # An IMMEDIATE retry sweep must NOT re-send (backoff active)
        working = AsyncMock(return_value=__import__("types").SimpleNamespace(id=88410))
        monkeypatch.setattr(app, "send_message", working)
        count2 = await audit.retry_undelivered_deletion_logs()
        assert count2 == 0
        assert working.await_count == 0  # backoff respected — no tight loop
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "pending"

        # Backoff elapsed → the periodic sweep retries and delivers
        await mock_db.deletion_log_delivery.update_one(
            {"event_key": f"{chat_id}:{msg_id}", "kind": "deleted"},
            {"$set": {"next_attempt_at": datetime.now(timezone.utc)
                      - timedelta(seconds=1)}},
        )
        count3 = await audit.retry_undelivered_deletion_logs()
        assert count3 == 1
        assert working.await_count == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 88410

    async def test_backoff_grows_after_repeated_failures(
        self, monkeypatch, mock_db,
    ):
        """Second consecutive failure → the cooldown DOUBLES (30s → 60s)
        so a permanently failing record is never hammered."""
        import config
        from SWAGGYMUSIC import app

        _logger_env(monkeypatch, logger_id=-1005559512)
        _forbid_history(monkeypatch)
        await _isolate(mock_db)
        chat_id, msg_id = -1008884200, 88420
        now = datetime.now(timezone.utc)
        await mock_db.deletion_log_delivery.insert_one({
            "event_key": f"{chat_id}:{msg_id}", "kind": "deleted",
            "chat_id": chat_id, "message_id": msg_id,
            "status": "pending",
            "log_event_tag": logger_mod.deletion_log_event_tag(
                f"{chat_id}:{msg_id}", "deleted"),
            "log_message_id": None, "logs_chat_id": None,
            "attempts": 0, "detail": "",
            "created_at": now, "updated_at": now,
        })

        failing = AsyncMock(side_effect=BadRequest("TEST"))
        monkeypatch.setattr(app, "send_message", failing)

        await audit.retry_undelivered_deletion_logs()
        rec = await _delivery(mock_db, chat_id, msg_id)
        window1 = (
            rec["next_attempt_at"] - rec["updated_at"]
        ).total_seconds()
        assert window1 == pytest.approx(30.0, abs=3)

        # make the second attempt eligible now
        await mock_db.deletion_log_delivery.update_one(
            {"event_key": f"{chat_id}:{msg_id}", "kind": "deleted"},
            {"$set": {"next_attempt_at": datetime.now(timezone.utc)
                      - timedelta(seconds=1)}},
        )
        await audit.retry_undelivered_deletion_logs()
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["attempts"] == 2
        window2 = (
            rec["next_attempt_at"] - rec["updated_at"]
        ).total_seconds()
        assert window2 == pytest.approx(60.0, abs=3)
        assert window2 > window1 + 25  # exponential growth, not a flat loop

    async def test_startup_sweep_ignores_backoff_stamp(
        self, monkeypatch, mock_db,
    ):
        """A boot must attempt EVERY leftover record, even one whose
        backoff stamp is still in the future; the periodic sweep must
        not."""
        _logger_env(monkeypatch, logger_id=-1005559513)
        _forbid_history(monkeypatch)
        await _isolate(mock_db)
        chat_id, msg_id = -1008884300, 88430
        now = datetime.now(timezone.utc)
        await mock_db.deletion_log_delivery.insert_one({
            "event_key": f"{chat_id}:{msg_id}", "kind": "deleted",
            "chat_id": chat_id, "message_id": msg_id,
            "status": "pending",
            "log_event_tag": logger_mod.deletion_log_event_tag(
                f"{chat_id}:{msg_id}", "deleted"),
            "log_message_id": None, "logs_chat_id": None,
            "attempts": 3, "detail": "",
            "next_attempt_at": now + timedelta(hours=1),  # far future
            "created_at": now, "updated_at": now,
        })

        from SWAGGYMUSIC import app

        # periodic: stamp in the future → NOT eligible
        count = await audit.retry_undelivered_deletion_logs()
        assert count == 0
        assert app.send_message.await_count == 0

        # startup: backoff ignored — the record is attempted
        count2 = await audit.recover_interrupted_deletion_logs()
        assert count2 == 1
        assert app.send_message.await_count == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"

    async def test_uncertain_not_retried_by_periodic_until_grace(
        self, monkeypatch, mock_db,
    ):
        """An uncertain record (Telegram MAY have delivered) is left
        alone by the periodic sweep until the runtime grace elapsed —
        but the startup sweep recovers it regardless."""
        _logger_env(monkeypatch, logger_id=-1005559514)
        _forbid_history(monkeypatch)
        await _isolate(mock_db)
        chat_id, msg_id = -1008884400, 88440
        now = datetime.now(timezone.utc)
        await mock_db.deletion_log_delivery.insert_one({
            "event_key": f"{chat_id}:{msg_id}", "kind": "deleted",
            "chat_id": chat_id, "message_id": msg_id,
            "status": "uncertain",
            "log_event_tag": logger_mod.deletion_log_event_tag(
                f"{chat_id}:{msg_id}", "deleted"),
            "log_message_id": None, "logs_chat_id": None,
            "attempts": 1, "detail": "",
            "uncertain_since": now,  # grace NOT elapsed
            "created_at": now, "updated_at": now,
        })

        from SWAGGYMUSIC import app

        count = await audit.retry_undelivered_deletion_logs()
        assert count == 0
        assert app.send_message.await_count == 0  # no blind resend

        # grace elapsed (simulate time passing) → periodic retry
        await mock_db.deletion_log_delivery.update_one(
            {"event_key": f"{chat_id}:{msg_id}", "kind": "deleted"},
            {"$set": {"uncertain_since": datetime.now(timezone.utc)
                      - timedelta(
                          seconds=audit.UNCERTAIN_RUNTIME_GRACE_SECONDS + 1)}},
        )
        count2 = await audit.retry_undelivered_deletion_logs()
        assert count2 == 1
        assert app.send_message.await_count == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"

    async def test_stale_sending_recovered_by_periodic_worker(
        self, monkeypatch, mock_db,
    ):
        """A "sending" record whose lease expired belongs to a dead
        worker — the periodic sweep takes it over and delivers it."""
        _logger_env(monkeypatch, logger_id=-1005559515)
        _forbid_history(monkeypatch)
        await _isolate(mock_db)
        chat_id, msg_id = -1008884500, 88450
        now = datetime.now(timezone.utc)
        await mock_db.deletion_log_delivery.insert_one({
            "event_key": f"{chat_id}:{msg_id}", "kind": "deleted",
            "chat_id": chat_id, "message_id": msg_id,
            "status": "sending",
            "log_event_tag": logger_mod.deletion_log_event_tag(
                f"{chat_id}:{msg_id}", "deleted"),
            "log_message_id": None, "logs_chat_id": None,
            "attempts": 1, "detail": "",
            "claimed_by": "dead-worker:abc",
            "claimed_at": now - timedelta(seconds=600),
            "lease_expires_at": now - timedelta(seconds=10),  # EXPIRED
            "created_at": now, "updated_at": now,
        })

        from SWAGGYMUSIC import app

        count = await audit.retry_undelivered_deletion_logs()
        assert count == 1
        assert app.send_message.await_count == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 555900

    async def test_periodic_worker_never_disturbs_a_live_claim(
        self, monkeypatch, mock_db,
    ):
        """A "sending" record with a FRESH lease belongs to a running
        worker: the periodic sweep must not touch it (no send, no
        takeover).  The startup sweep, by contrast, recovers it —
        after a restart there is no live owner to protect."""
        _logger_env(monkeypatch, logger_id=-1005559516)
        _forbid_history(monkeypatch)
        await _isolate(mock_db)
        chat_id, msg_id = -1008884600, 88460
        now = datetime.now(timezone.utc)
        await mock_db.deletion_log_delivery.insert_one({
            "event_key": f"{chat_id}:{msg_id}", "kind": "deleted",
            "chat_id": chat_id, "message_id": msg_id,
            "status": "sending",
            "log_event_tag": logger_mod.deletion_log_event_tag(
                f"{chat_id}:{msg_id}", "deleted"),
            "log_message_id": None, "logs_chat_id": None,
            "attempts": 1, "detail": "",
            "claimed_by": "live-worker:xyz",
            "claimed_at": now,
            "lease_expires_at": now + timedelta(
                seconds=audit.DELIVERY_LEASE_SECONDS),  # FRESH
            "created_at": now, "updated_at": now,
        })

        from SWAGGYMUSIC import app

        count = await audit.retry_undelivered_deletion_logs()
        assert count == 0
        assert app.send_message.await_count == 0  # live claim untouched
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sending"
        assert rec["claimed_by"] == "live-worker:xyz"

        # startup: abandoned claim of a previous process → recovered
        count2 = await audit.recover_interrupted_deletion_logs()
        assert count2 == 1
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"


# ---------------------------------------------------------------------------
# Concurrency — workers can still never send the same event twice
# ---------------------------------------------------------------------------

class TestConcurrencyStillGuaranteed:
    async def test_sweeps_and_runtime_worker_race_send_exactly_once(
        self, monkeypatch, mock_db,
    ):
        """Startup sweep, periodic sweep and a runtime delivery flow
        race on the same pending event: the atomic claim guarantees
        exactly ONE Telegram send."""
        _logger_env(monkeypatch, logger_id=-1005559521)
        _forbid_history(monkeypatch)
        await _isolate(mock_db)
        chat_id, msg_id = -1008884700, 88470
        now = datetime.now(timezone.utc)
        await mock_db.deletion_log_delivery.insert_one({
            "event_key": f"{chat_id}:{msg_id}", "kind": "deleted",
            "chat_id": chat_id, "message_id": msg_id,
            "status": "pending",
            "log_event_tag": logger_mod.deletion_log_event_tag(
                f"{chat_id}:{msg_id}", "deleted"),
            "log_message_id": None, "logs_chat_id": None,
            "attempts": 0, "detail": "",
            "created_at": now, "updated_at": now,
        })

        from SWAGGYMUSIC import app

        async def _slow_send(*args, **kwargs):
            await asyncio.sleep(0.02)
            return __import__("types").SimpleNamespace(id=88470)

        sent = AsyncMock(side_effect=_slow_send)
        monkeypatch.setattr(app, "send_message", sent)

        results = await asyncio.gather(
            audit.recover_interrupted_deletion_logs(),
            audit.retry_undelivered_deletion_logs(),
            audit.deliver_deletion_log(
                f"{chat_id}:{msg_id}", "deleted", chat_id, msg_id,
            ),
        )
        assert sent.await_count == 1
        assert sum(results[:-1]) == 1  # exactly one sweep delivered it
        assert results[-1] in ("sent", "skipped")
        rec = await _delivery(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 88470


# ---------------------------------------------------------------------------
# The periodic retry worker loop
# ---------------------------------------------------------------------------

class TestRetryWorkerLoop:
    async def test_worker_keeps_ticking_and_survives_failures(
        self, monkeypatch,
    ):
        """The loop runs a sweep every interval, survives a raising
        sweep, and exits quietly on cancellation."""
        import types

        calls = []

        async def _fake_retry(**kwargs):
            calls.append(len(calls))
            if len(calls) == 1:
                raise RuntimeError("simulated sweep failure")
            return 0

        monkeypatch.setattr(
            audit, "retry_undelivered_deletion_logs", _fake_retry,
        )
        task = asyncio.create_task(
            audit.delivery_retry_worker(interval=0.05),
        )
        await asyncio.sleep(0.3)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert task.cancelled()
        # several ticks happened despite the first sweep raising
        assert len(calls) >= 2

    async def test_worker_skips_ticks_while_previous_sweep_running(
        self, monkeypatch,
    ):
        """A sweep slower than the interval must not stack up: ticks
        that arrive while a sweep is running are skipped."""
        import types

        release = asyncio.Event()
        started = asyncio.Event()
        calls = []

        async def _blocked_retry(**kwargs):
            calls.append(len(calls))
            started.set()
            await release.wait()
            return 0

        monkeypatch.setattr(
            audit, "retry_undelivered_deletion_logs", _blocked_retry,
        )
        task = asyncio.create_task(
            audit.delivery_retry_worker(interval=0.05),
        )
        await asyncio.wait_for(started.wait(), timeout=2)
        # several ticks pass while the first sweep is still blocked
        await asyncio.sleep(0.25)
        assert len(calls) == 1  # overlapping ticks were skipped
        release.set()
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def test_start_delivery_retry_worker_is_idempotent(self):
        """A second start while a worker runs returns the SAME task —
        no second loop is ever spawned."""
        first = audit.start_delivery_retry_worker(interval=60)
        assert first is not None and not first.done()
        second = audit.start_delivery_retry_worker(interval=60)
        assert second is first
        # cleanup — do not leak the worker into other tests
        first.cancel()
        audit._retry_worker_task = None
        with contextlib.suppress(asyncio.CancelledError):
            await first
