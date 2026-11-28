"""Regression tests for the logger fixes from the latest code review.

1. NO TIME-WINDOW DEDUPLICATION — every genuine event is logged: rapid
   repeated setting changes, several Create Post publications by the
   same user, schedule pause/resume cycles.  A send is suppressed ONLY
   when the same reliable event-specific identifier is delivered twice
   (one actual event processed more than once).  A failed send never
   keeps its slot (the retry is legitimate, not a duplicate).
2. MONGODB-BACKED DELETION LOG DELIVERY — the ``deletion_log_delivery``
   collection is the source of truth for whether a deletion log /
   archive-failure alert still needs to be sent.  A delivery record
   with the event's unique ``event_key`` is created BEFORE any send,
   tracked through ``pending`` → ``sending`` → ``sent`` (plus the
   honest ``uncertain`` state), claimed atomically so concurrent
   workers can never double-send, and marked "sent" — with Telegram's
   returned message id — only after Telegram accepted the message.
   TELEGRAM HISTORY IS NEVER SCANNED: startup recovery reads MongoDB
   alone.  The crash-after-send-before-DB-update window is handled
   at-least-once (documented; ``sendMessage`` has no idempotency key).
3. DELETION FAILURE LOGGING — Telegram deletion errors, album-sibling
   deletion failures and permission-blocked deletions from Approval
   Mode and Banned Words are reported to the central logger group.
4. PERSISTENT EVENT KEYS — every post-deletion log and archive-failure
   alert carries a unique, persistent ``EK:<tag>`` line derived
   deterministically from the record's event key + kind.  The tag is
   exact-token matchable (``find_event_tag_in_text`` — never substring
   matching, so similar event keys can never be confused) and stays
   identifiable in the logger group after any restart.

The dedicated delivery-machine tests (atomic claims, lease takeover,
late-success evidence, crash recovery, legacy migration) live in
``tests/test_delivery_state.py``.
"""
import asyncio
import re
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pyrogram.enums import ChatType
from pyrogram.errors import BadRequest, FloodWait

import SWAGGYMUSIC.plugins.misc.approvalmode as am
import SWAGGYMUSIC.plugins.misc.bannedwords as bw
from SWAGGYMUSIC.utils import approval_policy, bannedwords_policy
from SWAGGYMUSIC.utils import deletion_audit as audit
from SWAGGYMUSIC.utils import logger as logger_mod

pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _logger_env(monkeypatch, logger_id=-1005559001):
    """LOGGER_ID set + mocked app.send_message + fresh event-dedup state."""
    import config
    from SWAGGYMUSIC import app

    monkeypatch.setattr(config, "LOGGER_ID", logger_id, raising=False)
    sent = AsyncMock(return_value=SimpleNamespace(id=555900))
    monkeypatch.setattr(app, "send_message", sent)
    logger_mod._DELIVERED_EVENTS.clear()
    return sent


def _channel_message(msg_id, chat_id, *, text="", media_group_id=None):
    chat = SimpleNamespace(id=chat_id, type=ChatType.CHANNEL, title="Ch")
    msg = MagicMock()
    msg.chat = chat
    msg.id = msg_id
    msg.text = text
    msg.caption = None
    msg.entities = []
    msg.caption_entities = []
    msg.media_group_id = media_group_id
    msg.edit_date = None
    msg.outgoing = False
    msg.service = None
    msg.media = None
    msg.empty = False
    msg.date = None
    msg.delete = AsyncMock()
    return msg


def _texts(sent):
    out = []
    for call in sent.await_args_list:
        out.append(call.kwargs.get("text") or call.args[-1])
    return out


async def _audit_record(mock_db, chat_id, message_id, **extra):
    """Insert the persistent audit record (the log's content source)."""
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


async def _delivery_status(mock_db, chat_id, message_id, kind="deleted"):
    """The current status of a delivery record (None when missing)."""
    rec = await mock_db.deletion_log_delivery.find_one(
        {"event_key": f"{chat_id}:{message_id}", "kind": kind},
    )
    return rec


def _forbid_history(monkeypatch):
    """Make ANY Telegram history scan fail loudly — recovery must not
    touch Telegram history in the MongoDB-backed design."""
    from SWAGGYMUSIC import app

    def _boom(*args, **kwargs):
        raise AssertionError("Telegram history was scanned — forbidden")

    monkeypatch.setattr(app, "get_chat_history", _boom)


# ---------------------------------------------------------------------------
# FIX 1 — no time-window deduplication
# ---------------------------------------------------------------------------

class TestFix1NoTimeWindowDeduplication:
    async def test_rapid_repeated_setting_changes_are_all_logged(self, monkeypatch):
        """Three identical ENABLED changes in the same instant: 3 logs."""
        sent = _logger_env(monkeypatch)
        chat = SimpleNamespace(id=-1001234567801, title="Ch", username=None)
        for _ in range(3):
            await logger_mod.log_setlink_change(chat, None, "ENABLED")
        assert sent.await_count == 3

    async def test_rapid_on_off_on_toggle_is_all_logged(self, monkeypatch):
        """on → off → on with no delay between them: every change logged."""
        sent = _logger_env(monkeypatch)
        chat = SimpleNamespace(id=-1001234567802, title="Ch", username=None)
        await logger_mod.log_setlink_change(chat, None, "ENABLED")
        await logger_mod.log_setlink_change(chat, None, "DISABLED")
        await logger_mod.log_setlink_change(chat, None, "ENABLED")
        texts = _texts(sent)
        assert sent.await_count == 3
        assert sum("ENABLED" in t for t in texts) == 2
        assert sum("DISABLED" in t for t in texts) == 1

    async def test_multiple_publications_same_user_rapid(self, monkeypatch):
        """Several publications by the same user, same channel: each logged."""
        sent = _logger_env(monkeypatch)
        user = SimpleNamespace(id=42, first_name="Publisher", username="pub")
        for _ in range(3):
            await logger_mod.log_createpost_published(user, -1001234567803, "Normal Post")
        assert sent.await_count == 3

    async def test_schedule_pause_resume_cycle_rapid(self, monkeypatch):
        """paused → resumed → paused → resumed with no pauses in between."""
        sent = _logger_env(monkeypatch)
        for _ in range(2):
            await logger_mod.log_schedule_change(
                -1001234567804, "Ch", None, "PAUSED", schedule_id=7,
            )
            await logger_mod.log_schedule_change(
                -1001234567804, "Ch", None, "RESUMED", schedule_id=7,
            )
        assert sent.await_count == 4

    async def test_send_failure_then_retry_succeeds(self, monkeypatch):
        """A failed send returns False and can be retried immediately."""
        import config
        from SWAGGYMUSIC import app

        _logger_env(monkeypatch)
        failing = AsyncMock(side_effect=BadRequest("TEST"))
        monkeypatch.setattr(app, "send_message", failing)
        assert await logger_mod.send_log("boom") is False

        working = AsyncMock(return_value=SimpleNamespace(id=1))
        monkeypatch.setattr(app, "send_message", working)
        assert await logger_mod.send_log("boom") is True
        assert working.await_count == 1

    async def test_same_event_identifier_suppressed_only_for_same_event(self, monkeypatch):
        """event_id dedups ONE actual event; other events always log."""
        sent = _logger_env(monkeypatch)
        event = ("post", -1001234567805, 300)
        assert await logger_mod.send_log("e", event_id=event) is True
        assert await logger_mod.send_log("e", event_id=event) is False  # same event again
        assert await logger_mod.send_log("e", event_id=("post", -1001234567805, 301)) is True
        assert await logger_mod.send_log("e") is True  # no identifier → always log
        assert sent.await_count == 3

    async def test_failed_delivery_releases_event_identifier(self, monkeypatch):
        """A failed send must not swallow the retry of the same event."""
        import config
        from SWAGGYMUSIC import app

        _logger_env(monkeypatch)
        event = ("post", -1001234567806, 400)
        failing = AsyncMock(side_effect=BadRequest("TEST"))
        monkeypatch.setattr(app, "send_message", failing)
        assert await logger_mod.send_log("boom", event_id=event) is False

        working = AsyncMock(return_value=SimpleNamespace(id=2))
        monkeypatch.setattr(app, "send_message", working)
        assert await logger_mod.send_log("boom", event_id=event) is True
        assert working.await_count == 1

    async def test_helpers_without_identifier_never_deduplicate(self, monkeypatch):
        sent = _logger_env(monkeypatch)
        user = SimpleNamespace(id=1, first_name="U", username=None)
        for _ in range(3):
            await logger_mod.log_user_start(user)
        assert sent.await_count == 3


# ---------------------------------------------------------------------------
# FIX 2 — MongoDB-backed deletion-log delivery (source of truth)
# ---------------------------------------------------------------------------

class TestFix2MongoDbDeliveryState:
    async def test_definite_failure_stays_retryable_and_retry_delivers(
        self, monkeypatch, mock_db,
    ):
        import config
        from SWAGGYMUSIC import app

        audit.audit_db = mock_db.deletion_audit_logs
        audit.delivery_db = mock_db.deletion_log_delivery
        monkeypatch.setattr(config, "LOGGER_ID", -1007779001, raising=False)
        chat_id, msg_id = -1006669001, 91001
        event_key = await _audit_record(mock_db, chat_id, msg_id)

        failing = AsyncMock(side_effect=BadRequest("TEST"))
        monkeypatch.setattr(app, "send_message", failing)
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="first try",
        )
        assert failing.await_count == 1
        rec = await _delivery_status(mock_db, chat_id, msg_id)
        # Telegram refused → back to "pending" (retryable), never "sent"
        assert rec["status"] == "pending"
        assert rec["log_message_id"] is None
        assert rec["attempts"] == 1

        # The retry (next worker pass / new attempt) delivers the log.
        working = AsyncMock(return_value=SimpleNamespace(id=93001))
        monkeypatch.setattr(app, "send_message", working)
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="retry",
        )
        assert working.await_count == 1
        rec = await _delivery_status(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 93001
        # the outcome is mirrored onto the audit record for operators
        mirror = await mock_db.deletion_audit_logs.find_one(
            {"event_key": event_key},
        )
        assert mirror["log_status"] == "sent"
        assert mirror["log_message_id"] == 93001

        # …and no duplicate successful log afterwards.
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="again",
        )
        assert working.await_count == 1

    async def test_record_is_created_pending_before_sending(
        self, monkeypatch, mock_db,
    ):
        """The persistent delivery record exists BEFORE the Telegram send:
        a send that explodes mid-call still leaves a recoverable record."""
        import config
        from SWAGGYMUSIC import app

        audit.audit_db = mock_db.deletion_audit_logs
        audit.delivery_db = mock_db.deletion_log_delivery
        monkeypatch.setattr(config, "LOGGER_ID", -1007779019, raising=False)
        chat_id, msg_id = -1006669019, 91191
        await _audit_record(mock_db, chat_id, msg_id)

        observed = {}

        async def _spy_send(*args, **kwargs):
            # The delivery record MUST already exist while the send runs.
            rec = await mock_db.deletion_log_delivery.find_one(
                {"event_key": f"{chat_id}:{msg_id}", "kind": "deleted"},
            )
            observed["during_send"] = rec
            raise RuntimeError("connection lost mid-send")

        monkeypatch.setattr(app, "send_message", _spy_send)
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="boom",
        )
        rec = observed.get("during_send")
        assert rec is not None
        assert rec["status"] == "sending"
        assert rec["event_key"] == f"{chat_id}:{msg_id}"
        assert rec["kind"] == "deleted"
        assert rec["log_event_tag"] == logger_mod.deletion_log_event_tag(
            f"{chat_id}:{msg_id}", "deleted",
        )
        # the unknown outcome is recorded honestly
        rec = await _delivery_status(mock_db, chat_id, msg_id)
        assert rec["status"] == "uncertain"

    async def test_uncertain_outcome_not_immediately_retried(
        self, monkeypatch, mock_db,
    ):
        """An unknown send error may have delivered — a retry right after
        does NOT re-send (startup recovery decides later)."""
        import config
        from SWAGGYMUSIC import app

        audit.audit_db = mock_db.deletion_audit_logs
        audit.delivery_db = mock_db.deletion_log_delivery
        monkeypatch.setattr(config, "LOGGER_ID", -1007779002, raising=False)
        chat_id, msg_id = -1006669002, 91002
        await _audit_record(mock_db, chat_id, msg_id)

        ambiguous = AsyncMock(side_effect=RuntimeError("connection lost mid-send"))
        monkeypatch.setattr(app, "send_message", ambiguous)
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="try",
        )
        assert ambiguous.await_count == 1
        rec = await _delivery_status(mock_db, chat_id, msg_id)
        assert rec["status"] == "uncertain"
        assert rec["uncertain_since"] is not None

        # A retry right after does NOT re-send the log (no duplicates).
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="retry attempt",
        )
        assert ambiguous.await_count == 1

    async def test_uncertain_recovered_by_startup_sweep(
        self, monkeypatch, mock_db,
    ):
        """The startup sweep recovers uncertain records from MongoDB —
        never by scanning Telegram history."""
        import config
        from SWAGGYMUSIC import app

        audit.audit_db = mock_db.deletion_audit_logs
        audit.delivery_db = mock_db.deletion_log_delivery
        monkeypatch.setattr(config, "LOGGER_ID", -1007779020, raising=False)
        chat_id, msg_id = -1006669020, 91201
        await _audit_record(mock_db, chat_id, msg_id)

        ambiguous = AsyncMock(side_effect=RuntimeError("connection lost"))
        monkeypatch.setattr(app, "send_message", ambiguous)
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="try",
        )
        assert ambiguous.await_count == 1

        _forbid_history(monkeypatch)
        # close out leftovers from other tests (shared in-memory DB) so
        # the sweep's exact-count assertion only covers this event
        await mock_db.deletion_log_delivery.update_many(
            {"event_key": {"$ne": f"{chat_id}:{msg_id}"}},
            {"$set": {"status": "sent", "log_message_id": 1}},
        )
        working = AsyncMock(return_value=SimpleNamespace(id=93201))
        monkeypatch.setattr(app, "send_message", working)
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 1
        assert working.await_count == 1
        rec = await _delivery_status(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 93201

    async def test_claim_not_consumed_while_logger_group_unset(
        self, monkeypatch, mock_db,
    ):
        import config
        from SWAGGYMUSIC import app

        audit.audit_db = mock_db.deletion_audit_logs
        audit.delivery_db = mock_db.deletion_log_delivery
        chat_id, msg_id = -1006669003, 91003
        await _audit_record(mock_db, chat_id, msg_id)
        monkeypatch.setattr(config, "LOGGER_ID", 0, raising=False)
        sent = AsyncMock(return_value=SimpleNamespace(id=93003))
        monkeypatch.setattr(app, "send_message", sent)

        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="no logger yet",
        )
        sent.assert_not_awaited()
        # nothing can be delivered → no delivery record is created
        assert await _delivery_status(mock_db, chat_id, msg_id) is None

        # The logger group is configured later → the log is deliverable.
        monkeypatch.setattr(config, "LOGGER_ID", -1007779003, raising=False)
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="logger configured",
        )
        assert sent.await_count == 1
        rec = await _delivery_status(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 93003

    async def test_archive_failure_alert_is_retryable_until_delivered(
        self, monkeypatch, mock_db,
    ):
        import config
        from SWAGGYMUSIC import app

        audit.audit_db = mock_db.deletion_audit_logs
        audit.delivery_db = mock_db.deletion_log_delivery
        monkeypatch.setattr(config, "LOGGER_ID", -1007779004, raising=False)
        chat_id, msg_id = -1006669004, 91004
        await _audit_record(mock_db, chat_id, msg_id)
        record = {"event_key": f"{chat_id}:{msg_id}", "chat_id": chat_id,
                  "message_id": msg_id}

        failing = AsyncMock(side_effect=BadRequest("TEST"))
        monkeypatch.setattr(app, "send_message", failing)
        await audit._send_failure_alert(app, record, "Forward failed: BadRequest")
        rec = await _delivery_status(mock_db, chat_id, msg_id, kind="archive_alert")
        assert rec["status"] == "pending"  # retryable, NOT sent
        mirror = await mock_db.deletion_audit_logs.find_one(
            {"event_key": f"{chat_id}:{msg_id}"},
        )
        assert mirror["failure_alert_status"] == "pending"

        working = AsyncMock(return_value=SimpleNamespace(id=93004))
        monkeypatch.setattr(app, "send_message", working)
        await audit._send_failure_alert(app, record, "Forward failed: BadRequest")
        assert working.await_count == 1
        rec = await _delivery_status(mock_db, chat_id, msg_id, kind="archive_alert")
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 93004

        # At most one delivered alert per event.
        await audit._send_failure_alert(app, record, "again")
        assert working.await_count == 1

    async def test_alert_sends_only_once_despite_concurrent_failures(
        self, monkeypatch, mock_db,
    ):
        """Two failure paths alerting the same event concurrently → the
        atomic claim lets exactly ONE alert through."""
        import config
        from SWAGGYMUSIC import app

        audit.audit_db = mock_db.deletion_audit_logs
        audit.delivery_db = mock_db.deletion_log_delivery
        monkeypatch.setattr(config, "LOGGER_ID", -1007779021, raising=False)
        chat_id, msg_id = -1006669021, 91211
        await _audit_record(mock_db, chat_id, msg_id)
        record = {"event_key": f"{chat_id}:{msg_id}", "chat_id": chat_id,
                  "message_id": msg_id}

        async def _slow_send(*args, **kwargs):
            await asyncio.sleep(0.05)
            return SimpleNamespace(id=93211)

        sent = AsyncMock(side_effect=_slow_send)
        monkeypatch.setattr(app, "send_message", sent)
        logger_mod._DELIVERED_EVENTS.clear()

        await asyncio.gather(
            audit._send_failure_alert(app, record, "forward failed A"),
            audit._send_failure_alert(app, record, "forward failed B"),
        )
        assert sent.await_count == 1
        rec = await _delivery_status(mock_db, chat_id, msg_id, kind="archive_alert")
        assert rec["status"] == "sent"

    async def test_floodwait_failed_deletion_log_is_retryable(
        self, monkeypatch, mock_db,
    ):
        """A long FloodWait does not mark the log sent; retry delivers it."""
        import config
        from SWAGGYMUSIC import app

        audit.audit_db = mock_db.deletion_audit_logs
        audit.delivery_db = mock_db.deletion_log_delivery
        monkeypatch.setattr(config, "LOGGER_ID", -1007779005, raising=False)
        chat_id, msg_id = -1006669005, 91005
        await _audit_record(mock_db, chat_id, msg_id)

        flood = AsyncMock(side_effect=FloodWait(value=600))
        monkeypatch.setattr(app, "send_message", flood)
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="flooded",
        )
        rec = await _delivery_status(mock_db, chat_id, msg_id)
        assert rec["status"] == "pending"
        assert rec["log_message_id"] is None

        working = AsyncMock(return_value=SimpleNamespace(id=93005))
        monkeypatch.setattr(app, "send_message", working)
        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="retry",
        )
        assert working.await_count == 1
        rec = await _delivery_status(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 93005

    async def test_duplicate_event_processing_sends_once(self, monkeypatch, mock_db):
        """The same deletion event processed twice concurrently (double-fired
        update) produces exactly ONE log."""
        import config
        from SWAGGYMUSIC import app

        audit.audit_db = mock_db.deletion_audit_logs
        audit.delivery_db = mock_db.deletion_log_delivery
        monkeypatch.setattr(config, "LOGGER_ID", -1007779031, raising=False)
        chat_id, msg_id = -1006669031, 91131
        await _audit_record(mock_db, chat_id, msg_id)

        async def _slow_send(*args, **kwargs):
            await asyncio.sleep(0.05)
            return SimpleNamespace(id=555931)

        sent = AsyncMock(side_effect=_slow_send)
        monkeypatch.setattr(app, "send_message", sent)

        t1 = asyncio.create_task(audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="first",
        ))
        await asyncio.sleep(0.01)  # t1 claimed the event and is sending
        t2 = asyncio.create_task(audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="second",
        ))
        await asyncio.gather(t1, t2)

        assert sent.await_count == 1
        rec = await _delivery_status(mock_db, chat_id, msg_id)
        assert rec["status"] == "sent"
        assert rec["log_message_id"] == 555931

    async def test_event_key_kinds_are_independent_delivery_records(
        self, monkeypatch, mock_db,
    ):
        """One post can legitimately produce a "pending_retry" log and a
        later "deleted" log — two distinct delivery records, both sent."""
        import config
        from SWAGGYMUSIC import app

        audit.audit_db = mock_db.deletion_audit_logs
        audit.delivery_db = mock_db.deletion_log_delivery
        monkeypatch.setattr(config, "LOGGER_ID", -1007779022, raising=False)
        chat_id, msg_id = -1006669022, 91221
        await _audit_record(mock_db, chat_id, msg_id)

        sent = AsyncMock(return_value=SimpleNamespace(id=93222))
        monkeypatch.setattr(app, "send_message", sent)

        await audit.mark_deletion_result(
            chat_id, msg_id, status="pending_retry", detail="flood queued",
        )
        rec_retry = await _delivery_status(
            mock_db, chat_id, msg_id, kind="pending_retry",
        )
        assert rec_retry["status"] == "sent"
        assert sent.await_count == 1

        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="retry worker deleted it",
        )
        rec_deleted = await _delivery_status(
            mock_db, chat_id, msg_id, kind="deleted",
        )
        assert rec_deleted["status"] == "sent"
        assert sent.await_count == 2
        # each log carries its own exact event tag
        texts = _texts(sent)
        assert (
            f"EK:{logger_mod.deletion_log_event_tag(f'{chat_id}:{msg_id}', 'pending_retry')}"
            in texts[0]
        )
        assert (
            f"EK:{logger_mod.deletion_log_event_tag(f'{chat_id}:{msg_id}', 'deleted')}"
            in texts[1]
        )


# ---------------------------------------------------------------------------
# FIX 3 — deletion failure logging (Approval Mode + Banned Words)
# ---------------------------------------------------------------------------

class TestFix3DeletionFailureLogging:
    async def test_approval_delete_rpc_error_reaches_logger_group(
        self, monkeypatch, mock_db,
    ):
        sent = _logger_env(monkeypatch, logger_id=-1007779011)
        msg = _channel_message(55001, -1004459011, text="no approval here")
        msg.delete.side_effect = BadRequest("CHAT_ADMIN_REQUIRED")

        assert await am._delete_post(msg, reason="APPROVAL_NO_ALLOWED_USERNAME") is False
        texts = _texts(sent)
        assert any("POST DELETION FAILED" in t for t in texts)
        assert any("BadRequest" in t for t in texts)
        assert any(str(-1004459011) in t for t in texts)
        rec = await mock_db.deletion_audit_logs.find_one(
            {"event_key": "-1004459011:55001"},
        )
        assert rec["deletion_status"] == "FAILED"

    async def test_approval_delete_unknown_error_reaches_logger_group(
        self, monkeypatch, mock_db,
    ):
        sent = _logger_env(monkeypatch, logger_id=-1007779012)
        msg = _channel_message(55002, -1004459012, text="no approval here")
        msg.delete.side_effect = RuntimeError("connection reset")

        assert await am._delete_post(msg, reason="APPROVAL_NO_ALLOWED_USERNAME") is False
        texts = _texts(sent)
        assert any("POST DELETION FAILED" in t for t in texts)
        assert any("RuntimeError" in t for t in texts)
        rec = await mock_db.deletion_audit_logs.find_one(
            {"event_key": "-1004459012:55002"},
        )
        assert rec["deletion_status"] == "FAILED"
        assert "outcome unknown" in rec["deletion_result_detail"]

    async def test_approval_album_sibling_failure_reaches_logger_group(
        self, monkeypatch, mock_db,
    ):
        sent = _logger_env(monkeypatch, logger_id=-1007779013)
        chat_id, group = -1004459013, 777001
        approval_policy.remember_album_item(chat_id, group, 55011)
        approval_policy.remember_album_item(chat_id, group, 55012)
        from SWAGGYMUSIC import app

        app.delete_messages = AsyncMock(side_effect=BadRequest("TEST"))

        await am._delete_album_siblings(chat_id, group, keep=55013)

        texts = _texts(sent)
        assert len(texts) == 2  # one failure log per sibling
        assert all("POST DELETION FAILED" in t for t in texts)
        assert any("55011" in t for t in texts)
        assert any("55012" in t for t in texts)
        for mid in (55011, 55012):
            rec = await mock_db.deletion_audit_logs.find_one(
                {"event_key": f"{chat_id}:{mid}"},
            )
            assert rec["deletion_status"] == "FAILED"

    async def test_approval_no_delete_permission_reaches_logger_group(
        self, monkeypatch, mock_db,
    ):
        import config
        from SWAGGYMUSIC.utils import database as db

        monkeypatch.setattr(config, "LOGGER_ID", -1007779014, raising=False)
        sent = AsyncMock(return_value=SimpleNamespace(id=555901))
        from SWAGGYMUSIC import app

        monkeypatch.setattr(app, "send_message", sent)
        logger_mod._DELIVERED_EVENTS.clear()
        chat_id = -1004459014
        approval_policy._REJECTED.clear()
        await db.settings_db.update_one(
            {"chat_id": chat_id}, {"$set": {"chat_id": chat_id}}, upsert=True,
        )
        await db.set_approval_mode_enabled(chat_id, True)
        monkeypatch.setattr(am, "_bot_can_delete", AsyncMock(return_value=False))

        msg = _channel_message(55014, chat_id, text="plain post, no mentions")
        await am.approvalmode_new_channel_post(app, msg)

        msg.delete.assert_not_awaited()
        assert approval_policy.was_rejected(chat_id, 55014)
        texts = _texts(sent)
        assert any(
            "APPROVAL MODE" in t and "NO DELETE PERMISSION" in t for t in texts
        )

    async def test_bannedwords_delete_rpc_error_reaches_logger_group(
        self, monkeypatch, mock_db,
    ):
        sent = _logger_env(monkeypatch, logger_id=-1007779015)
        msg = _channel_message(55021, -1004459015, text="buy spam now")
        msg.delete.side_effect = BadRequest("CHAT_ADMIN_REQUIRED")

        assert await bw._delete_post(
            msg, reason="BANNED_WORD", details={"matched": "spam"},
        ) is False
        texts = _texts(sent)
        assert any("POST DELETION FAILED" in t for t in texts)
        assert any("BadRequest" in t for t in texts)
        rec = await mock_db.deletion_audit_logs.find_one(
            {"event_key": "-1004459015:55021"},
        )
        assert rec["deletion_status"] == "FAILED"

    async def test_bannedwords_album_sibling_failure_reaches_logger_group(
        self, monkeypatch, mock_db,
    ):
        sent = _logger_env(monkeypatch, logger_id=-1007779016)
        chat_id, group = -1004459016, 777002
        bannedwords_policy.remember_album_item(chat_id, group, 55031)
        bannedwords_policy.remember_album_item(chat_id, group, 55032)
        from SWAGGYMUSIC import app

        app.delete_messages = AsyncMock(side_effect=BadRequest("TEST"))

        await bw._delete_album_siblings(
            chat_id, group, keep=55033, matched="spam",
        )

        texts = _texts(sent)
        assert len(texts) == 2
        assert all("POST DELETION FAILED" in t for t in texts)
        assert any("55031" in t for t in texts)
        assert any("55032" in t for t in texts)
        for mid in (55031, 55032):
            rec = await mock_db.deletion_audit_logs.find_one(
                {"event_key": f"{chat_id}:{mid}"},
            )
            assert rec["deletion_status"] == "FAILED"

    async def test_bannedwords_no_delete_permission_reaches_logger_group(
        self, monkeypatch, mock_db,
    ):
        import config
        from SWAGGYMUSIC.utils import database as db

        monkeypatch.setattr(config, "LOGGER_ID", -1007779017, raising=False)
        sent = AsyncMock(return_value=SimpleNamespace(id=555902))
        from SWAGGYMUSIC import app

        monkeypatch.setattr(app, "send_message", sent)
        logger_mod._DELIVERED_EVENTS.clear()
        chat_id = -1004459017
        bannedwords_policy._REJECTED.clear()
        await db.settings_db.update_one(
            {"chat_id": chat_id}, {"$set": {"chat_id": chat_id}}, upsert=True,
        )
        await db.add_banned_words(chat_id, ["spam"])
        await db.set_banned_words_enabled(chat_id, True)
        monkeypatch.setattr(bw, "_bot_can_delete", AsyncMock(return_value=False))

        msg = _channel_message(55041, chat_id, text="fresh spam deals")
        await bw.bannedwords_new_channel_post(app, msg)

        msg.delete.assert_not_awaited()
        assert bannedwords_policy.was_rejected(chat_id, 55041)
        texts = _texts(sent)
        assert any(
            "BANNED WORDS" in t and "NO DELETE PERMISSION" in t for t in texts
        )

    async def test_floodwait_during_bannedwords_delete_still_logged(
        self, monkeypatch, mock_db,
    ):
        """FloodWait → pending-retry log (not DELETED) keeps working."""
        sent = _logger_env(monkeypatch, logger_id=-1007779018)
        msg = _channel_message(55051, -1004459018, text="spam again")
        msg.delete.side_effect = FloodWait(value=30)

        assert await bw._delete_post(
            msg, reason="BANNED_WORD", details={"matched": "spam"},
        ) is False
        texts = _texts(sent)
        assert any("PENDING RETRY" in t for t in texts)
        assert not any("DELETED IN CHANNEL" in t for t in texts)


# ---------------------------------------------------------------------------
# FIX 4 — persistent event keys: deterministic, unique, EXACT matching
# ---------------------------------------------------------------------------

def _delivered_log_text(chat_id, message_id, kind="deleted"):
    """Plain text as a REAL delivered deletion log appears (its EK tag)."""
    marker = {
        "deleted": "POST DELETED IN CHANNEL",
        "pending_retry": "POST DELETION PENDING",
        "failed": "POST DELETION FAILED",
    }[kind]
    tag = logger_mod.deletion_log_event_tag(f"{chat_id}:{message_id}", kind)
    return (
        f"{marker}\n\n"
        f"Channel: Ch\n"
        f"Channel ID: {chat_id}\n"
        f"Post ID: {message_id}\n"
        f"\n🧾 <code>EK:{tag}</code>"
    )


class TestEventKeyTagsUnit:
    """The persistent event_key tag: determinism, uniqueness, EXACT matching."""

    async def test_tag_is_deterministic_and_persistent(self):
        """Same event → same tag, recomputable 'after a restart'."""
        t1 = logger_mod.deletion_log_event_tag("-1006669100:91200", "deleted")
        t2 = logger_mod.deletion_log_event_tag("-1006669100:91200", "deleted")
        assert t1 == t2
        assert len(t1) == 16
        assert re.fullmatch(r"[0-9a-f]{16}", t1)

    async def test_tag_unique_per_event_and_per_kind(self):
        base = "-1006669100:91200"
        tags = {
            logger_mod.deletion_log_event_tag(base, "deleted"),
            logger_mod.deletion_log_event_tag(base, "pending_retry"),
            logger_mod.deletion_log_event_tag(base, "failed"),
            logger_mod.deletion_log_event_tag("-1006669100:91201", "deleted"),
            logger_mod.alert_event_tag(base),
        }
        assert len(tags) == 5

    async def test_exact_event_key_matches(self):
        tag = logger_mod.deletion_log_event_tag("-1006669100:91200", "deleted")
        text = _delivered_log_text(-1006669100, 91200)
        assert logger_mod.find_event_tag_in_text(text, tag)

    async def test_similar_event_keys_must_not_match(self):
        """The old substring matcher confused overlapping ids; exact tags
        must never match a similar (different) event's log."""
        chat_a, msg_a = -1001234567801, 91201
        chat_b, msg_b = -1001234567801, 9120  # "9120" ⊂ "91201"
        chat_c, msg_c = -10012345678, 91201   # "-10012345678" ⊂ "-1001234567801"

        text_a = _delivered_log_text(chat_a, msg_a)
        assert "9120" in str(msg_a)  # sanity: the old matcher WOULD match B
        for other in ((chat_b, msg_b), (chat_c, msg_c)):
            tag_other = logger_mod.deletion_log_event_tag(
                f"{other[0]}:{other[1]}", "deleted",
            )
            assert not logger_mod.find_event_tag_in_text(text_a, tag_other)

    async def test_tag_inside_longer_run_is_rejected(self):
        """Exact token equality: no substring-in-a-longer-run matches."""
        tag = "ab12cd34ef56ab12"
        assert not logger_mod.find_event_tag_in_text(f"EK:{tag}f0", tag)
        assert not logger_mod.find_event_tag_in_text(f"XEK:{tag}", tag)
        assert not logger_mod.find_event_tag_in_text(f"EK:{tag[:-1]}", tag)
        assert not logger_mod.find_event_tag_in_text(f"EK:{tag[1:]}", tag)
        assert not logger_mod.find_event_tag_in_text("no tag here", tag)
        assert not logger_mod.find_event_tag_in_text("", tag)

    async def test_kind_mixing_is_rejected(self):
        """A delivered log of one kind never satisfies another kind's
        identification for the same post."""
        event_key = "-1006669100:91200"
        deleted_tag = logger_mod.deletion_log_event_tag(event_key, "deleted")
        failed_tag = logger_mod.deletion_log_event_tag(event_key, "failed")
        text = (
            f"POST DELETION FAILED\n"
            f"Channel ID: -1006669100\n"
            f"Post ID: 91200\n"
            f"\n🧾 <code>EK:{failed_tag}</code>"
        )
        assert logger_mod.find_event_tag_in_text(text, failed_tag)
        assert not logger_mod.find_event_tag_in_text(text, deleted_tag)

    async def test_tag_line_fits_inside_max_log_text(self):
        """A huge preview must never push the tag out of the message."""
        tag = "0123456789abcdef"
        text = "x" * (logger_mod.MAX_LOG_TEXT + 500)
        out = logger_mod.append_event_tag(text, tag)
        assert len(out) <= logger_mod.MAX_LOG_TEXT
        assert out.endswith(f"EK:{tag}</code>")

    async def test_sent_log_carries_its_exact_event_key(
        self, monkeypatch, mock_db,
    ):
        """Every delivered deletion log contains EK:<tag>; the delivery
        record and the mirrored audit outcome store the SAME tag."""
        import config
        from SWAGGYMUSIC import app

        audit.audit_db = mock_db.deletion_audit_logs
        audit.delivery_db = mock_db.deletion_log_delivery
        monkeypatch.setattr(config, "LOGGER_ID", -1007779061, raising=False)
        chat_id, msg_id = -1006669061, 91601
        event_key = f"{chat_id}:{msg_id}"
        await _audit_record(mock_db, chat_id, msg_id)
        sent = AsyncMock(return_value=SimpleNamespace(id=555961))
        monkeypatch.setattr(app, "send_message", sent)

        await audit.mark_deletion_result(
            chat_id, msg_id, status="deleted", detail="carries key",
        )
        assert sent.await_count == 1
        text = _texts(sent)[0]
        expected_tag = logger_mod.deletion_log_event_tag(event_key, "deleted")
        assert f"EK:{expected_tag}" in text
        delivery = await _delivery_status(mock_db, chat_id, msg_id)
        assert delivery["log_event_tag"] == expected_tag
        assert delivery["status"] == "sent"
        mirror = await mock_db.deletion_audit_logs.find_one(
            {"event_key": event_key},
        )
        assert mirror["log_event_tag"] == expected_tag

    async def test_history_scan_helpers_are_gone(self):
        """Recovery no longer scans Telegram history — the scan machinery
        was removed, MongoDB is the single source of truth."""
        for name in (
            "scan_logger_group_tags",
            "find_delivered_event_tag",
            "find_delivered_deletion_log",
            "find_delivered_log_message",
            "SCAN_FOUND",
            "SCAN_NOT_FOUND",
            "SCAN_FAILED",
        ):
            assert not hasattr(logger_mod, name), name
        # exact-token tag matching survives as a pure text utility
        assert callable(logger_mod.find_event_tag_in_text)
