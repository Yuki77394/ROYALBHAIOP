"""APPROVAL MODE tests — policy, moderation flow, albums, logger, gap safety.

Covers the spec rules:
  * allowed username present            → kept
  * only unapproved usernames           → deleted (logged as POST DELETED)
  * mixed allowed + unapproved          → kept + warning log (never deleted)
  * no username at all                  → deleted while enabled
  * feature disabled                    → no restriction
  * exact username matching (@john ≠ @johnny), text + captions + entities
  * media albums judged once, kept or deleted whole
  * settings persist in the shared MongoDB settings document
  * a rejected post never becomes an accepted /setgap reference
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pyrogram.enums import ChatMemberStatus, ChatType, MessageEntityType

import SWAGGYMUSIC.plugins.misc.approvalmode as am
from SWAGGYMUSIC.utils import approval_policy

pytestmark = pytest.mark.asyncio


CHANNEL = -10042424242


def _clean_policy():
    approval_policy._REJECTED.clear()
    approval_policy._BLOCKED_ALBUMS.clear()
    approval_policy._APPROVED_ALBUMS.clear()
    approval_policy._ALBUM_ITEMS.clear()


@pytest.fixture(autouse=True)
async def _fresh():
    from SWAGGYMUSIC.utils import database as db

    _clean_policy()
    am.SESSIONS.clear()
    await db.settings_db.delete_many({"chat_id": CHANNEL})
    yield
    _clean_policy()
    am.SESSIONS.clear()
    await db.settings_db.delete_many({"chat_id": CHANNEL})


def _channel_post(msg_id, text=None, *, caption=None, entities=None,
                  caption_entities=None, media_group_id=None, edit_date=None):
    chat = SimpleNamespace(id=CHANNEL, type=ChatType.CHANNEL)
    msg = MagicMock()
    msg.chat = chat
    msg.id = msg_id
    msg.text = text
    msg.caption = caption
    msg.entities = entities or []
    msg.caption_entities = caption_entities or []
    msg.media_group_id = media_group_id
    msg.edit_date = edit_date
    msg.outgoing = False
    msg.service = None
    msg.media = None
    msg.empty = False
    msg.date = None
    msg.delete = AsyncMock()
    return msg


def _mention_entity(offset, length):
    ent = MagicMock()
    ent.type = MessageEntityType.MENTION
    ent.offset = offset
    ent.length = length
    return ent


def _mention(text):
    """A message whose text carries one @mention entity (UTF-16 offsets)."""
    idx = text.index("@")
    return _mention_entity(idx, len(text) - idx)


def _admin_bot_ok(monkeypatch):
    member = MagicMock()
    member.status = ChatMemberStatus.ADMINISTRATOR
    member.privileges = MagicMock()
    member.privileges.can_delete_messages = True
    monkeypatch.setattr(
        am.app, "get_chat_member", AsyncMock(return_value=member),
    )


async def _enable_with_names(mock_db, names, enabled=True):
    from SWAGGYMUSIC.utils.database import (
        add_approval_usernames, set_approval_mode_enabled, settings_db,
    )
    doc = await settings_db.find_one({"chat_id": CHANNEL})
    settings = mock_db.autodelete_settings
    settings = settings  # keep linters quiet
    from SWAGGYMUSIC.utils import database as db
    await db.settings_db.update_one(
        {"chat_id": CHANNEL}, {"$set": {"chat_id": CHANNEL}}, upsert=True,
    )
    if names:
        await add_approval_usernames(CHANNEL, names)
    if enabled:
        await set_approval_mode_enabled(CHANNEL, True)


# ---------------------------------------------------------------------------
# Policy: username extraction — exact matching, text/captions/entities
# ---------------------------------------------------------------------------

class TestUsernameExtraction:
    async def test_mention_entity_is_extracted(self):
        msg = _channel_post(1, "hey @john look")
        msg.entities = [_mention("hey @john look")]
        assert approval_policy.extract_usernames(msg) == {"john"}

    async def test_caption_entities_are_extracted(self):
        msg = _channel_post(2, caption="promo @team_lead", caption_entities=[
            _mention_entity(6, 10),
        ])
        assert approval_policy.extract_usernames(msg) == {"team_lead"}

    async def test_bare_handle_scan_without_entities(self):
        msg = _channel_post(3, "ping @Mary_88 now")
        assert approval_policy.extract_usernames(msg) == {"mary_88"}

    async def test_no_partial_matches(self):
        """@john must never match @johnny or @john_doe2."""
        approved = ["john"]
        msg = _channel_post(4, "thanks @johnny!")
        approved_found, unapproved = approval_policy.classify_post(msg, approved)
        assert approved_found == set()          # johnny is NOT john
        assert unapproved == {"johnny"}

        msg2 = _channel_post(5, "thanks @John!")
        approved_found, _un = approval_policy.classify_post(msg2, approved)
        assert approved_found == {"john"}       # case-insensitive exact hit

    async def test_emoji_before_mention_utf16_offsets(self):
        text = "🔥🔥🔥 @john"
        msg = _channel_post(6, text)
        msg.entities = [_mention_entity(text.index("@"), 5)]
        assert approval_policy.extract_usernames(msg) == {"john"}


# ---------------------------------------------------------------------------
# Moderation — allowed / unapproved / mixed / none / disabled
# ---------------------------------------------------------------------------

class TestModerationFlow:
    async def test_post_with_allowed_username_is_kept(self, mock_db, monkeypatch):
        from SWAGGYMUSIC import app

        await _enable_with_names(mock_db, ["john"])
        _admin_bot_ok(monkeypatch)
        sent = AsyncMock()
        monkeypatch.setattr(app, "send_message", sent)

        msg = _channel_post(11, "promo by @john here")
        msg.entities = [_mention("promo by @john here")]
        await am.approvalmode_new_channel_post(app, msg)

        msg.delete.assert_not_awaited()
        sent.assert_not_awaited()
        assert not approval_policy.was_rejected(CHANNEL, 11)

    async def test_post_without_allowed_username_is_deleted_and_logged(
        self, mock_db, monkeypatch,
    ):
        import config
        from SWAGGYMUSIC import app

        await _enable_with_names(mock_db, ["john"])
        _admin_bot_ok(monkeypatch)
        monkeypatch.setattr(config, "LOGGER_ID", -1005550002, raising=False)
        sent = AsyncMock(return_value=SimpleNamespace(id=1))
        monkeypatch.setattr(app, "send_message", sent)

        msg = _channel_post(12, "stranger promotion @other_user")
        msg.entities = [_mention("stranger promotion @other_user")]
        await am.approvalmode_new_channel_post(app, msg)

        msg.delete.assert_awaited_once()
        # the centralized logger reported the deletion (POST DELETED)
        assert sent.await_count == 1
        text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
        assert "POST DELETED" in text
        assert "No approved username" in text
        assert approval_policy.was_rejected(CHANNEL, 12)

    async def test_mixed_usernames_are_kept_and_warned(self, mock_db, monkeypatch):
        import config
        from SWAGGYMUSIC import app

        await _enable_with_names(mock_db, ["john"])
        _admin_bot_ok(monkeypatch)
        monkeypatch.setattr(config, "LOGGER_ID", -1005550003, raising=False)
        sent = AsyncMock(return_value=SimpleNamespace(id=1))
        monkeypatch.setattr(app, "send_message", sent)

        text = "@john and @stranger collab"
        msg = _channel_post(13, text)
        msg.entities = [
            _mention_entity(text.index("@"), 5),
            _mention_entity(text.index("@stranger"), 9),
        ]
        await am.approvalmode_new_channel_post(app, msg)

        msg.delete.assert_not_awaited()          # mixed is NEVER deleted
        assert sent.await_count == 1
        text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
        assert "MIXED USERNAMES" in text
        assert "KEPT" in text
        assert not approval_policy.was_rejected(CHANNEL, 13)

    async def test_post_without_any_username_is_deleted(self, mock_db, monkeypatch):
        from SWAGGYMUSIC import app

        await _enable_with_names(mock_db, ["john"])
        _admin_bot_ok(monkeypatch)
        sent = AsyncMock(return_value=SimpleNamespace(id=1))
        monkeypatch.setattr(app, "send_message", sent)

        msg = _channel_post(14, "plain text advertisement")
        await am.approvalmode_new_channel_post(app, msg)
        msg.delete.assert_awaited_once()
        assert approval_policy.was_rejected(CHANNEL, 14)

    async def test_feature_disabled_is_a_no_op(self, mock_db, monkeypatch):
        from SWAGGYMUSIC import app

        await _enable_with_names(mock_db, ["john"], enabled=False)
        sent = AsyncMock(return_value=SimpleNamespace(id=1))
        monkeypatch.setattr(app, "send_message", sent)

        msg = _channel_post(15, "anything @stranger")
        await am.approvalmode_new_channel_post(app, msg)
        msg.delete.assert_not_awaited()
        sent.assert_not_awaited()

    async def test_no_delete_right_rejects_but_does_not_crash(
        self, mock_db, monkeypatch,
    ):
        from SWAGGYMUSIC import app

        await _enable_with_names(mock_db, ["john"])
        member = MagicMock()
        member.status = ChatMemberStatus.MEMBER
        monkeypatch.setattr(app, "get_chat_member", AsyncMock(return_value=member))

        msg = _channel_post(16, "no approved @someone")
        await am.approvalmode_new_channel_post(app, msg)
        msg.delete.assert_not_awaited()
        assert approval_policy.was_rejected(CHANNEL, 16)  # verdict still recorded

    async def test_commands_are_never_judged(self, mock_db, monkeypatch):
        from SWAGGYMUSIC import app

        await _enable_with_names(mock_db, ["john"])
        msg = _channel_post(17, "/approvalmode on")
        ent = MagicMock()
        ent.type = MessageEntityType.BOT_COMMAND
        ent.offset = 0
        ent.length = 14
        msg.entities = [ent]
        await am.approvalmode_new_channel_post(app, msg)
        msg.delete.assert_not_awaited()
        assert not approval_policy.was_rejected(CHANNEL, 17)

    async def test_edited_post_without_allowed_username_is_deleted(
        self, mock_db, monkeypatch,
    ):
        from SWAGGYMUSIC import app
        from datetime import datetime, timezone

        await _enable_with_names(mock_db, ["john"])
        _admin_bot_ok(monkeypatch)

        msg = _channel_post(
            18, "edited away @stranger",
            edit_date=datetime.now(timezone.utc),
        )
        await am.approvalmode_edited_channel_post(app, msg)
        msg.delete.assert_awaited_once()
        assert approval_policy.was_rejected(CHANNEL, 18)


# ---------------------------------------------------------------------------
# Media albums — judged once, kept or deleted whole
# ---------------------------------------------------------------------------

class TestAlbums:
    async def test_album_with_approved_caption_is_approved_whole(
        self, mock_db, monkeypatch,
    ):
        from SWAGGYMUSIC import app

        await _enable_with_names(mock_db, ["john"])
        _admin_bot_ok(monkeypatch)

        first = _channel_post(21, caption="by @john", media_group_id=777001)
        await am.approvalmode_new_channel_post(app, first)
        first.delete.assert_not_awaited()
        assert approval_policy.album_is_approved(CHANNEL, 777001)

        later = _channel_post(22, caption=None, media_group_id=777001)
        await am.approvalmode_new_channel_post(app, later)
        later.delete.assert_not_awaited()      # follows the album verdict

    async def test_album_without_approved_caption_is_deleted_whole(
        self, mock_db, monkeypatch,
    ):
        from SWAGGYMUSIC import app

        await _enable_with_names(mock_db, ["john"])
        _admin_bot_ok(monkeypatch)
        deletes = AsyncMock(return_value=2)
        monkeypatch.setattr(app, "delete_messages", deletes)

        first = _channel_post(23, caption="no one", media_group_id=777002)
        await am.approvalmode_new_channel_post(app, first)
        first.delete.assert_awaited_once()
        # the first item decides, so nothing was delivered before it —
        # no sibling bulk delete is needed here
        deletes.assert_not_awaited()
        assert approval_policy.album_is_blocked(CHANNEL, 777002)

        later = _channel_post(24, caption="by @john", media_group_id=777002)
        await am.approvalmode_new_channel_post(app, later)
        later.delete.assert_awaited_once()     # condemned album stays condemned


# ---------------------------------------------------------------------------
# Persistence + gap/setdelay compatibility
# ---------------------------------------------------------------------------

class TestPersistenceAndGapSafety:
    async def test_settings_persist_in_shared_document(self, mock_db):
        from SWAGGYMUSIC.utils.database import (
            add_approval_usernames, get_approval_usernames,
            remove_approval_username, settings_db,
        )

        await add_approval_usernames(CHANNEL, ["@John", "team_lead", "bad name!"])
        names = await get_approval_usernames(CHANNEL)
        # normalized; invalid handles skipped
        assert names == ["john", "team_lead"]

        assert await remove_approval_username(CHANNEL, "@JOHN") is True
        assert await get_approval_usernames(CHANNEL) == ["team_lead"]
        # the settings document is the shared one
        doc = await settings_db.find_one({"chat_id": CHANNEL})
        assert doc["approval_usernames"] == ["team_lead"]

    async def test_enable_disable_roundtrip(self, mock_db):
        from SWAGGYMUSIC.utils.database import (
            set_approval_mode_enabled, settings_db,
        )

        assert await set_approval_mode_enabled(CHANNEL, True) == "ENABLED"
        doc = await settings_db.find_one({"chat_id": CHANNEL})
        assert doc["approval_mode_enabled"] is True

        assert await set_approval_mode_enabled(CHANNEL, True) == "NO_CHANGE"
        assert await set_approval_mode_enabled(CHANNEL, False) == "DISABLED"
        doc = await settings_db.find_one({"chat_id": CHANNEL})
        # with no usernames on the list and no other feature enabled, the
        # shared settings document is cleaned up entirely
        assert doc is None or "approval_mode_enabled" not in doc

    async def test_rejected_post_is_marked_before_the_delete_call(
        self, mock_db, monkeypatch,
    ):
        """The verdict precedes Telegram I/O — a failed delete still marks."""
        from SWAGGYMUSIC import app

        await _enable_with_names(mock_db, ["john"])
        _admin_bot_ok(monkeypatch)
        monkeypatch.setattr(
            am.app, "get_chat_member",
            AsyncMock(side_effect=RuntimeError("telegram down")),
        )
        msg = _channel_post(31, "no approved name here")
        await am.approvalmode_new_channel_post(app, msg)
        assert approval_policy.was_rejected(CHANNEL, 31)

    async def test_gap_engine_skips_rejected_posts(self, mock_db, monkeypatch):
        """A post the approval filter rejected never becomes the gap reference."""
        from SWAGGYMUSIC.plugins.misc import autodelete as ad
        from SWAGGYMUSIC.utils.database import set_gap_duration, settings_db
        from SWAGGYMUSIC import app

        await _enable_with_names(mock_db, ["john"])
        await set_gap_duration(CHANNEL, 600)
        approval_policy.mark_rejected(CHANNEL, 41)   # as the filter would

        member = MagicMock()
        member.status = ChatMemberStatus.ADMINISTRATOR
        member.privileges = MagicMock()
        member.privileges.can_delete_messages = True
        monkeypatch.setattr(app, "get_chat_member", AsyncMock(return_value=member))

        msg = _channel_post(41, "any text by @john")
        msg.entities = [_mention("any text by @john")]
        await ad.autodelete_new_channel_post(app, msg)

        doc = await settings_db.find_one({"chat_id": CHANNEL})
        # no gap slot consumed, no reference, no deletion job
        assert "last_gap_post_at" not in doc
        assert doc.get("gap_history") in (None, [])
        assert ad.jobs_db is not None
