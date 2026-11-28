"""Regression tests for deletion-audit reason immutability and send recovery."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytestmark = pytest.mark.asyncio


async def test_retry_keeps_original_reason_and_does_not_duplicate_forward_or_log(
    monkeypatch, mock_db,
):
    import config
    from SWAGGYMUSIC import app
    from SWAGGYMUSIC.utils import deletion_audit as audit

    monkeypatch.setattr(config, "AUDIT_CHANNEL_ID", -1008880001, raising=False)
    monkeypatch.setattr(config, "LOGGER_ID", -1007770001, raising=False)
    audit.audit_db = mock_db.deletion_audit_logs

    message = SimpleNamespace(
        id=91001,
        empty=False,
        chat=SimpleNamespace(id=-1006660001, username="main_channel", title="Main Channel"),
        text="audit test",
        caption=None,
        media=None,
        date=None,
        from_user=None,
        sender_chat=SimpleNamespace(id=-1006660001, title="Main Channel"),
        author_signature="Editor",
    )
    app.get_messages = AsyncMock(return_value=message)
    app.get_chat = AsyncMock(return_value=SimpleNamespace(username="audit_channel"))
    app.forward_messages = AsyncMock(return_value=SimpleNamespace(id=92001))
    app.send_message = AsyncMock(return_value=SimpleNamespace(id=93001))

    first = await audit.archive_deletion(
        -1006660001, 91001, reason="SETDELAY_EXPIRED",
        source="AUTODELETE_WORKER", message=message,
    )
    # The archive forward happened; the LOGGER_ID "POST DELETED" log is
    # emitted by mark_deletion_result (centralized logger), not here.
    assert first["status"] == "forwarded"
    app.send_message.assert_not_awaited()

    # Telegram confirms the deletion → exactly ONE deletion log.
    await audit.mark_deletion_result(
        -1006660001, 91001, status="deleted", detail="test confirmed"
    )
    app.edit_message_text.assert_not_awaited()  # old edit-on-result flow is gone
    assert app.send_message.await_count == 1
    logged_text = app.send_message.await_args.kwargs.get("text") or app.send_message.await_args.args[-1]
    assert "POST DELETED" in logged_text
    assert "Main Channel" in logged_text
    assert "SETDELAY" in logged_text or "auto-delete" in logged_text.lower()

    second = await audit.archive_deletion(
        -1006660001, 91001, reason="SETDELAY_RETRY",
        source="AUTODELETE_WORKER", message=message,
    )
    assert second["status"] == "complete"
    assert app.forward_messages.await_count == 1
    assert app.send_message.await_count == 1

    record = await mock_db.deletion_audit_logs.find_one(
        {"event_key": "-1006660001:91001"}
    )
    assert record["reason"] == "SETDELAY_EXPIRED"
    assert record["source"] == "AUTODELETE_WORKER"
    assert record["last_attempt_reason"] == "SETDELAY_RETRY"
    assert record["audit_message_id"] == 92001
    assert record["log_message_id"] == 93001  # mirrored delivery outcome
    assert record["deletion_status"] == "DELETED"
    assert record.get("deleted_at") is not None
    # The MongoDB delivery record is the source of truth: sent, with the
    # Telegram message id stored on it.
    delivery = await mock_db.deletion_log_delivery.find_one(
        {"event_key": "-1006660001:91001", "kind": "deleted"},
    )
    assert delivery["status"] == "sent"
    assert delivery["log_message_id"] == 93001


async def test_mark_deletion_result_never_duplicates_the_deleted_log(
    monkeypatch, mock_db,
):
    """Two mark_deletion_result calls for one post → ONE deletion log."""
    import config
    from SWAGGYMUSIC import app
    from SWAGGYMUSIC.utils import deletion_audit as audit

    monkeypatch.setattr(config, "AUDIT_CHANNEL_ID", 0, raising=False)
    monkeypatch.setattr(config, "LOGGER_ID", -1007770002, raising=False)
    audit.audit_db = mock_db.deletion_audit_logs

    message = SimpleNamespace(
        id=91003, empty=False,
        chat=SimpleNamespace(id=-1006660003, username="main_channel", title="Ch Three"),
        text="dup log test", caption=None, media=None, date=None,
        from_user=None, sender_chat=None, author_signature=None,
    )
    app.get_messages = AsyncMock(return_value=message)
    app.send_message = AsyncMock(return_value=SimpleNamespace(id=93003))

    await audit.archive_deletion(
        -1006660003, 91003, reason="GAP_REJECT", source="SETGAP", message=message,
    )
    await audit.mark_deletion_result(
        -1006660003, 91003, status="deleted", detail="first",
    )
    await audit.mark_deletion_result(
        -1006660003, 91003, status="deleted", detail="second",
    )
    assert app.send_message.await_count == 1


async def test_failed_deletion_is_reported_separately(monkeypatch, mock_db):
    """A pending-retry outcome logs PENDING RETRY — not DELETED."""
    import config
    from SWAGGYMUSIC import app
    from SWAGGYMUSIC.utils import deletion_audit as audit

    monkeypatch.setattr(config, "AUDIT_CHANNEL_ID", 0, raising=False)
    monkeypatch.setattr(config, "LOGGER_ID", -1007770003, raising=False)
    audit.audit_db = mock_db.deletion_audit_logs

    message = SimpleNamespace(
        id=91004, empty=False,
        chat=SimpleNamespace(id=-1006660004, username="main_channel", title="Ch Four"),
        text="retry test", caption=None, media=None, date=None,
        from_user=None, sender_chat=None, author_signature=None,
    )
    app.get_messages = AsyncMock(return_value=message)
    app.send_message = AsyncMock(return_value=SimpleNamespace(id=93004))

    await audit.archive_deletion(
        -1006660004, 91004, reason="LINK_BLOCKED", source="SETLINK", message=message,
    )
    await audit.mark_deletion_result(
        -1006660004, 91004, status="pending_retry", detail="FloodWait 42s",
    )
    assert app.send_message.await_count == 1
    logged_text = app.send_message.await_args.kwargs.get("text") or app.send_message.await_args.args[-1]
    assert "RETRY" in logged_text
    assert "DELETED IN CHANNEL" not in logged_text
    record = await mock_db.deletion_audit_logs.find_one({"event_key": "-1006660004:91004"})
    assert record["deletion_status"] == "PENDING_RETRY"
    delivery = await mock_db.deletion_log_delivery.find_one(
        {"event_key": "-1006660004:91004", "kind": "pending_retry"},
    )
    assert delivery["status"] == "sent"
    assert delivery["log_message_id"] == 93004


async def test_ambiguous_forward_attempt_is_not_forwarded_twice(monkeypatch, mock_db):
    import config
    from SWAGGYMUSIC import app
    from SWAGGYMUSIC.utils import deletion_audit as audit

    monkeypatch.setattr(config, "AUDIT_CHANNEL_ID", -1008880002, raising=False)
    monkeypatch.setattr(config, "LOGGER_ID", 0, raising=False)
    audit.audit_db = mock_db.deletion_audit_logs

    event_key = "-1006660002:91002"
    await mock_db.deletion_audit_logs.insert_one({
        "event_key": event_key,
        "chat_id": -1006660002,
        "message_id": 91002,
        "created_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc),
        "reason": "LINK_BLOCKED",
        "source": "SETLINK",
        "forward_status": "attempting",
        "log_status": "pending",
    })
    message = SimpleNamespace(
        id=91002, empty=False,
        chat=SimpleNamespace(id=-1006660002, username="main_channel"),
        text="possible prior forward", caption=None, media=None, date=None,
        from_user=None, sender_chat=None, author_signature=None,
    )
    app.get_messages = AsyncMock(return_value=message)
    app.forward_messages = AsyncMock()

    result = await audit.archive_deletion(
        -1006660002, 91002, reason="LINK_REJECT_RETRY",
        source="AUTODELETE_WORKER", message=message,
    )
    assert result["status"] == "forward_uncertain"
    app.forward_messages.assert_not_awaited()
    record = await mock_db.deletion_audit_logs.find_one({"event_key": event_key})
    assert record["forward_status"] == "uncertain"
    assert record["reason"] == "LINK_BLOCKED"
    assert record["last_attempt_reason"] == "LINK_REJECT_RETRY"


def test_sender_chat_id_is_not_mislabelled_as_verified_admin():
    from SWAGGYMUSIC.utils.deletion_audit import _actor_details

    details = _actor_details(SimpleNamespace(
        from_user=None,
        sender_chat=SimpleNamespace(id=-100123, title="Channel"),
        author_signature="Channel Admin",
    ))
    assert details["sender_user_id"] is None
    assert details["sender_chat_id"] == -100123
    assert details["verified_admin_id"] is None


async def test_link_album_archives_each_member_before_bulk_delete(monkeypatch):
    from SWAGGYMUSIC import app
    from SWAGGYMUSIC.plugins.misc import setlink

    archive = AsyncMock()
    mark = AsyncMock()
    monkeypatch.setattr(setlink, "archive_deletion", archive)
    monkeypatch.setattr(setlink, "mark_deletion_result", mark)
    monkeypatch.setattr(setlink.link_policy, "take_album_items", lambda *_: [94001, 94002])
    app.delete_messages = AsyncMock(return_value=2)

    await setlink._delete_album_siblings(-1006660003, 777, keep=94003)

    assert [call.args[1] for call in archive.await_args_list] == [94001, 94002]
    assert archive.await_count == 2
    assert mark.await_count == 2
    app.delete_messages.assert_awaited_once_with(-1006660003, [94001, 94002])
