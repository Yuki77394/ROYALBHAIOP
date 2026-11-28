"""Centralized logger (SWAGGYMUSIC.utils.logger) tests.

Covers: the single-point-of-entry sender, LOGGER_ID gating, duplicate
suppression, FloodWait/RPC resilience, HTML escaping, credential safety,
the post-deletion log format (screenshot-style) and the settings-change
event helpers used by every feature.
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = pytest.mark.asyncio


@pytest.fixture
def logger_env(monkeypatch):
    """LOGGER_ID configured + mocked app.send_message, fresh dedup state."""
    import config
    from SWAGGYMUSIC import app
    from SWAGGYMUSIC.utils import logger as logger_mod

    monkeypatch.setattr(config, "LOGGER_ID", -1005550001, raising=False)
    sent = AsyncMock(return_value=SimpleNamespace(id=555001))
    monkeypatch.setattr(app, "send_message", sent)
    logger_mod._DELIVERED_EVENTS.clear()
    return logger_mod, sent


async def test_send_log_noops_without_logger_id(monkeypatch):
    import config
    from SWAGGYMUSIC import app
    from SWAGGYMUSIC.utils import logger as logger_mod

    monkeypatch.setattr(config, "LOGGER_ID", 0, raising=False)
    sent = AsyncMock()
    monkeypatch.setattr(app, "send_message", sent)
    assert await logger_mod.send_log("hello") is False
    sent.assert_not_awaited()


async def test_same_event_id_is_suppressed_but_others_are_logged(logger_env):
    """Only the SAME reliable event identifier is deduplicated."""
    logger_mod, sent = logger_env
    event = ("post", -1001234567001, 100)
    assert await logger_mod.send_log("event", event_id=event) is True
    # the same event processed again → suppressed
    assert await logger_mod.send_log("event", event_id=event) is False
    # a different event (another post) → logged
    assert await logger_mod.send_log("event", event_id=("post", -1001234567001, 101)) is True
    # without an identifier → always logged
    assert await logger_mod.send_log("event") is True
    assert sent.await_count == 3


async def test_failed_event_send_releases_its_slot(logger_env, monkeypatch):
    """A failed delivery must not suppress the retry of the same event."""
    from pyrogram.errors import BadRequest

    import config
    from SWAGGYMUSIC import app

    logger_mod, sent = logger_env
    sent.side_effect = BadRequest("TEST")
    event = ("post", -1001234567002, 200)
    assert await logger_mod.send_log("boom", event_id=event) is False

    sent.side_effect = None
    sent.reset_mock()
    assert await logger_mod.send_log("boom", event_id=event) is True
    assert sent.await_count == 1


async def test_send_log_survives_rpc_errors(logger_env, monkeypatch):
    from pyrogram.errors import BadRequest

    logger_mod, sent = logger_env
    sent.side_effect = BadRequest("TEST")
    assert await logger_mod.send_log("boom") is False  # never raises


async def test_send_log_skips_long_floodwait(logger_env):
    from pyrogram.errors import FloodWait

    logger_mod, sent = logger_env
    sent.side_effect = FloodWait(value=600)
    assert await logger_mod.send_log("busy") is False  # no 600s sleep, no crash


async def test_event_helpers_escape_and_carry_channel_info(logger_env):
    logger_mod, sent = logger_env

    chat = SimpleNamespace(id=-1001234567890, title='<b>Evil & Co</b>', username="chan")
    user = SimpleNamespace(id=42, first_name="<script>", username="admin")

    await logger_mod.log_setdelay_change(chat, user, "ENABLED", delay_value="10 minutes")
    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "&lt;b&gt;Evil &amp; Co&lt;/b&gt;" in text  # title escaped
    assert "-1001234567890" in text
    assert "ENABLED" in text

    await logger_mod.log_setgap_change(chat, user, "UPDATED", gap_value="5m", old_value="1m")
    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "SETGAP UPDATED" in text and "5m" in text

    await logger_mod.log_setlink_change(chat, user, "DISABLED", source="CHANNEL")
    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "LINK RESTRICTION DISABLED" in text


async def test_rapid_setting_changes_are_each_logged(logger_env):
    """Repeated setting changes are genuine events — every one is logged,
    even several within the same instant (no time-window dedup)."""
    logger_mod, sent = logger_env
    chat = SimpleNamespace(id=-1001234567891, title="Ch", username=None)
    for _ in range(3):
        await logger_mod.log_setdelay_change(chat, None, "ENABLED", delay_value="5m")
    assert sent.await_count == 3


async def test_approval_mode_and_banned_words_events(logger_env):
    logger_mod, sent = logger_env
    chat = SimpleNamespace(id=-1001234567892, title="Ch", username=None)

    await logger_mod.log_approval_mode_change(chat, None, "ENABLED", usernames=2)
    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "APPROVAL MODE ENABLED" in text and "Approved usernames" in text

    await logger_mod.log_approval_usernames_changed(
        chat, None, "ADDED", ["john"], total=3,
    )
    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "@john" in text and "ADDED" in text

    await logger_mod.log_bannedwords_change(chat, None, "ENABLED", total=1)
    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "BANNED WORDS ENABLED" in text


async def test_mixed_username_warning_names_both_sides(logger_env):
    logger_mod, sent = logger_env
    chat = SimpleNamespace(id=-1001234567893, title="Ch", username=None)
    message = SimpleNamespace(id=77)

    await logger_mod.log_approval_mixed_warning(
        chat, message, ["john"], ["stranger", "__nomention__"],
    )
    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "MIXED USERNAMES" in text
    assert "@john" in text and "@stranger" in text
    assert "KEPT" in text


async def test_post_deletion_log_screenshot_format(logger_env):
    """The deletion log carries channel, ID, reason, preview and status."""
    logger_mod, sent = logger_env
    from SWAGGYMUSIC.utils.database import audit_db
    from SWAGGYMUSIC.utils import logger as logger_mod2  # same module

    # the persistent record is created by archive_deletion in production
    from SWAGGYMUSIC.utils.database import audit_db as _audit_db

    await _audit_db.update_one(
        {"event_key": "-1006660009:91009"},
        {"$set": {"event_key": "-1006660009:91009"}},
        upsert=True,
    )

    record = {
        "event_key": "-1006660009:91009",
        "chat_id": -1006660009,
        "channel_title": "My Channel",
        "message_id": 91009,
        "reason": "BANNED_WORD",
        "source": "BANNEDWORDS",
        "details": {"matched": "cross"},
        "content_text": "Buy cross trainers now!!!",
        "media_type": "text",
    }
    # Transport only: returns the honest outcome + Telegram's message id;
    # the persistent delivery state is owned by deletion_audit (MongoDB).
    outcome, message_id = await logger_mod2.send_post_deletion_log(
        record, kind="deleted", event_key=record["event_key"], detail="confirmed",
    )
    assert outcome == "sent"
    assert message_id == 555001

    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "POST DELETED IN CHANNEL" in text
    assert "My Channel" in text
    assert "-1006660009" in text
    assert 'Banned word detected: "cross"' in text
    assert "Buy cross trainers now!!!" in text
    assert "DELETED" in text
    # the transport layer writes NO delivery state — and every message
    # carries its persistent event-key tag for exact identification
    doc = await audit_db.find_one({"event_key": record["event_key"]})
    assert doc.get("log_status") is None
    assert "EK:" in text


async def test_media_only_preview_falls_back_to_media_label(logger_env):
    logger_mod, sent = logger_env
    from SWAGGYMUSIC.utils import logger as logger_mod2

    record = {
        "event_key": "-1006660009:91010",
        "chat_id": -1006660009,
        "channel_title": "Ch",
        "message_id": 91010,
        "reason": "FORWARD_BLOCKED",
        "source": "SETFORWARD",
        "content_text": "",
        "media_type": "Photo",
    }
    await logger_mod2.send_post_deletion_log(
        record, kind="deleted", event_key=record["event_key"],
    )
    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "[Photo]" in text
    assert "Forwarded post not allowed" in text


async def test_preview_is_truncated_and_escaped(logger_env):
    logger_mod, sent = logger_env
    from SWAGGYMUSIC.utils import logger as logger_mod2

    record = {
        "event_key": "-1006660009:91011",
        "chat_id": -1006660009,
        "message_id": 91011,
        "reason": "GAP_REJECT",
        "source": "SETGAP",
        "content_text": "<img onerror=alert(1)>" + "x" * 5000,
        "media_type": "text",
    }
    await logger_mod2.send_post_deletion_log(
        record, kind="deleted", event_key=record["event_key"],
    )
    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "<img" not in text           # escaped
    assert "&lt;img" in text
    assert len(text) < 4096             # Telegram limit respected
    assert "…" in text                  # truncated


async def test_no_credentials_in_any_log_text(logger_env):
    """Log builders must never embed tokens/keys/URIs."""
    logger_mod, sent = logger_env
    import config as real_config

    chat = SimpleNamespace(id=-1001234567894, title="Ch", username=None)
    user = SimpleNamespace(id=1, first_name="Owner", username="owner")

    await logger_mod.log_user_start(user)
    await logger_mod.log_bot_started(SimpleNamespace(id=9, name="B", username="b"))
    await logger_mod.log_channel_first_seen(chat)
    await logger_mod.log_setforward_change(chat, user, "ENABLED")
    await logger_mod.log_bannedwords_change(chat, user, "ENABLED", total=0)
    await logger_mod.log_operation_failed("X", chat_id=1, message_id=2, detail="y")

    for call in sent.await_args_list:
        text = call.kwargs.get("text") or call.args[-1]
        assert str(real_config.BOT_TOKEN) not in text
        assert "mongodb" not in text.lower()
        assert str(real_config.MONGO_DB_URI) not in text


async def test_bot_added_to_channel_and_first_seen(logger_env):
    logger_mod, sent = logger_env
    chat = SimpleNamespace(
        id=-1001234567895, title="New Chan", username="newchan",
    )
    actor = SimpleNamespace(id=7, first_name="Admin", username=None)

    await logger_mod.log_bot_added_to_channel(chat, actor=actor)
    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "BOT ADDED TO CHANNEL" in text and "@newchan" in text and "Admin" in text

    await logger_mod.log_channel_first_seen(chat)
    text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
    assert "CHANNEL REGISTERED" in text
    # once per channel per process
    sent.reset_mock()
    await logger_mod.log_channel_first_seen(chat)
    sent.assert_not_awaited()


async def test_old_logger_systems_are_gone():
    """The rebuilt logger replaced — not layered on top of — the old ones."""
    import importlib.util

    assert importlib.util.find_spec("SWAGGYMUSIC.utils.activity_logger") is None
    assert importlib.util.find_spec("SWAGGYMUSIC.plugins.sudo.logger") is None
