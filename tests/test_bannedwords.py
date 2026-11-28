"""BANNED WORDS tests — matching, moderation flow, albums, edits, persistence.

Covers the spec rules:
  * add / remove / view banned words and phrases (MongoDB persistence)
  * enable / disable per channel
  * case-insensitive, whole-word detection in text AND captions
  * posts with a banned word are deleted and logged (matched word, channel,
    preview, status)
  * media albums are deleted whole; edited posts are checked
  * bot commands are never judged as content
  * a rejected post never becomes an accepted /setgap reference
"""
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pyrogram.enums import ChatMemberStatus, ChatType, MessageEntityType

import SWAGGYMUSIC.plugins.misc.bannedwords as bw
from SWAGGYMUSIC.utils import bannedwords_policy

pytestmark = pytest.mark.asyncio


CHANNEL = -10043434343


def _clean_policy():
    bannedwords_policy._REJECTED.clear()
    bannedwords_policy._BLOCKED_ALBUMS.clear()
    bannedwords_policy._ALBUM_ITEMS.clear()


@pytest.fixture(autouse=True)
async def _fresh():
    from SWAGGYMUSIC.utils import database as db

    _clean_policy()
    bw.SESSIONS.clear()
    await db.settings_db.delete_many({"chat_id": CHANNEL})
    yield
    _clean_policy()
    bw.SESSIONS.clear()
    await db.settings_db.delete_many({"chat_id": CHANNEL})


def _channel_post(msg_id, text=None, *, caption=None, entities=None,
                  media_group_id=None, edit_date=None):
    chat = SimpleNamespace(id=CHANNEL, type=ChatType.CHANNEL)
    msg = MagicMock()
    msg.chat = chat
    msg.id = msg_id
    msg.text = text
    msg.caption = caption
    msg.entities = entities or []
    msg.caption_entities = []
    msg.media_group_id = media_group_id
    msg.edit_date = edit_date
    msg.outgoing = False
    msg.service = None
    msg.media = None
    msg.empty = False
    msg.date = None
    msg.delete = AsyncMock()
    return msg


def _admin_bot_ok(monkeypatch):
    member = MagicMock()
    member.status = ChatMemberStatus.ADMINISTRATOR
    member.privileges = MagicMock()
    member.privileges.can_delete_messages = True
    monkeypatch.setattr(
        bw.app, "get_chat_member", AsyncMock(return_value=member),
    )


async def _enable_with_words(mock_db, words, enabled=True):
    from SWAGGYMUSIC.utils import database as db

    await db.settings_db.update_one(
        {"chat_id": CHANNEL}, {"$set": {"chat_id": CHANNEL}}, upsert=True,
    )
    if words:
        await db.add_banned_words(CHANNEL, words)
    if enabled:
        await db.set_banned_words_enabled(CHANNEL, True)


# ---------------------------------------------------------------------------
# Policy — matching
# ---------------------------------------------------------------------------

class TestMatching:
    async def test_case_insensitive_single_word(self):
        msg = _channel_post(1, "Buy a CROSS trainer")
        assert bannedwords_policy.find_banned_words(msg, ["cross"]) == ["cross"]

    async def test_caption_is_checked(self):
        msg = _channel_post(2, caption="cheap meds NOW")
        assert bannedwords_policy.find_banned_words(msg, ["meds"]) == ["meds"]

    async def test_phrase_with_punctuation(self):
        msg = _channel_post(3, "Get rich quick!!! Click here")
        assert "get rich quick" in bannedwords_policy.find_banned_words(
            msg, ["get rich quick"],
        )

    async def test_whole_word_no_partial_match(self):
        msg = _channel_post(4, "the classrooms are full")
        assert bannedwords_policy.find_banned_words(msg, ["ass"]) == []
        msg2 = _channel_post(5, "he got crossed out")
        assert bannedwords_policy.find_banned_words(msg2, ["cross"]) == []

    async def test_configured_order_of_hits(self):
        msg = _channel_post(6, "spam and scam here")
        hits = bannedwords_policy.find_banned_words(msg, ["scam", "spam"])
        assert hits == ["scam", "spam"]  # configuration order, not text order

    async def test_commands_are_not_content(self):
        msg = _channel_post(7, "/bannedwords add spam")
        ent = MagicMock()
        ent.type = MessageEntityType.BOT_COMMAND
        ent.offset = 0
        ent.length = 12
        msg.entities = [ent]
        assert bannedwords_policy.is_bot_command(msg) is True


# ---------------------------------------------------------------------------
# Moderation flow
# ---------------------------------------------------------------------------

class TestModerationFlow:
    async def test_post_with_banned_word_is_deleted_and_logged(
        self, mock_db, monkeypatch,
    ):
        import config
        from SWAGGYMUSIC import app

        await _enable_with_words(mock_db, ["cross"])
        _admin_bot_ok(monkeypatch)
        monkeypatch.setattr(config, "LOGGER_ID", -1005550004, raising=False)
        sent = AsyncMock(return_value=SimpleNamespace(id=1))
        monkeypatch.setattr(app, "send_message", sent)

        msg = _channel_post(11, "selling CROSS trainers cheap")
        await bw.bannedwords_new_channel_post(app, msg)

        msg.delete.assert_awaited_once()
        assert sent.await_count == 1
        text = sent.await_args.kwargs.get("text") or sent.await_args.args[-1]
        assert "POST DELETED" in text
        assert 'Banned word detected: "cross"' in text
        assert "selling CROSS trainers cheap" in text   # preview
        assert str(CHANNEL) in text                      # channel id
        assert bannedwords_policy.was_rejected(CHANNEL, 11)

    async def test_clean_post_is_untouched(self, mock_db, monkeypatch):
        from SWAGGYMUSIC import app

        await _enable_with_words(mock_db, ["cross"])
        _admin_bot_ok(monkeypatch)
        sent = AsyncMock()
        monkeypatch.setattr(app, "send_message", sent)

        msg = _channel_post(12, "nice wholesome content")
        await bw.bannedwords_new_channel_post(app, msg)
        msg.delete.assert_not_awaited()
        sent.assert_not_awaited()

    async def test_feature_off_is_a_no_op(self, mock_db, monkeypatch):
        from SWAGGYMUSIC import app

        await _enable_with_words(mock_db, ["cross"], enabled=False)
        sent = AsyncMock()
        monkeypatch.setattr(app, "send_message", sent)

        msg = _channel_post(13, "cross everything")
        await bw.bannedwords_new_channel_post(app, msg)
        msg.delete.assert_not_awaited()
        sent.assert_not_awaited()
        assert not bannedwords_policy.was_rejected(CHANNEL, 13)

    async def test_edited_post_is_checked(self, mock_db, monkeypatch):
        from SWAGGYMUSIC import app
        from datetime import datetime, timezone

        await _enable_with_words(mock_db, ["cross"])
        _admin_bot_ok(monkeypatch)

        msg = _channel_post(
            14, "edited to add cross links",
            edit_date=datetime.now(timezone.utc),
        )
        await bw.bannedwords_edited_channel_post(app, msg)
        msg.delete.assert_awaited_once()
        assert bannedwords_policy.was_rejected(CHANNEL, 14)

    async def test_commands_are_never_judged(self, mock_db, monkeypatch):
        from SWAGGYMUSIC import app

        await _enable_with_words(mock_db, ["spam"])
        msg = _channel_post(15, "/bannedwords add spam")
        ent = MagicMock()
        ent.type = MessageEntityType.BOT_COMMAND
        ent.offset = 0
        ent.length = 12
        msg.entities = [ent]
        await bw.bannedwords_new_channel_post(app, msg)
        msg.delete.assert_not_awaited()
        assert not bannedwords_policy.was_rejected(CHANNEL, 15)

    async def test_failed_delete_still_marks_rejected(self, mock_db, monkeypatch):
        from SWAGGYMUSIC import app

        await _enable_with_words(mock_db, ["cross"])
        member = MagicMock()
        member.status = ChatMemberStatus.MEMBER
        monkeypatch.setattr(app, "get_chat_member", AsyncMock(return_value=member))

        msg = _channel_post(16, "cross post")
        await bw.bannedwords_new_channel_post(app, msg)
        msg.delete.assert_not_awaited()
        assert bannedwords_policy.was_rejected(CHANNEL, 16)


# ---------------------------------------------------------------------------
# Albums
# ---------------------------------------------------------------------------

class TestAlbums:
    async def test_album_with_banned_word_is_deleted_whole(
        self, mock_db, monkeypatch,
    ):
        from SWAGGYMUSIC import app

        await _enable_with_words(mock_db, ["cross"])
        _admin_bot_ok(monkeypatch)
        deletes = AsyncMock(return_value=2)
        monkeypatch.setattr(app, "delete_messages", deletes)

        # item 1 arrives clean
        clean = _channel_post(21, caption=None, media_group_id=888001)
        await bw.bannedwords_new_channel_post(app, clean)
        clean.delete.assert_not_awaited()

        # item 2 carries the word → the whole album goes down
        bad = _channel_post(22, caption="cross promo", media_group_id=888001)
        await bw.bannedwords_new_channel_post(app, bad)
        bad.delete.assert_awaited_once()
        deletes.assert_awaited_once()          # sibling cleanup (item 21)
        assert bannedwords_policy.album_is_blocked(CHANNEL, 888001)

        # item 3 (delivered after the verdict) is deleted on arrival
        later = _channel_post(23, caption=None, media_group_id=888001)
        await bw.bannedwords_new_channel_post(app, later)
        later.delete.assert_awaited_once()

    async def test_persisted_blocked_album_survives_restart(
        self, mock_db, monkeypatch,
    ):
        from SWAGGYMUSIC import app
        from SWAGGYMUSIC.utils.database import remember_banned_blocked_album

        await _enable_with_words(mock_db, ["cross"])
        _admin_bot_ok(monkeypatch)
        await remember_banned_blocked_album(CHANNEL, 888002)

        # simulate a restart: in-process state is empty again
        _clean_policy()
        setting = await mock_db.autodelete_settings.find_one({"chat_id": CHANNEL})
        assert bannedwords_policy.persisted_blocked_album(setting, 888002)

        item = _channel_post(24, caption=None, media_group_id=888002)
        await bw.bannedwords_new_channel_post(app, item)
        item.delete.assert_awaited_once()
        assert bannedwords_policy.album_is_blocked(CHANNEL, 888002)


# ---------------------------------------------------------------------------
# Persistence — list management
# ---------------------------------------------------------------------------

class TestPersistence:
    async def test_add_remove_view_roundtrip(self, mock_db):
        from SWAGGYMUSIC.utils.database import (
            add_banned_words, clear_banned_words, get_banned_words,
            remove_banned_word,
        )

        added, skipped = await add_banned_words(
            CHANNEL, ["Cross", "  FREE   MONEY ", "x" * 100],
        )
        assert added == ["cross", "free money"]  # normalized; too-long skipped
        assert skipped == ["x" * 100]
        assert await get_banned_words(CHANNEL) == ["cross", "free money"]

        # duplicates are skipped, never written twice
        added2, skipped2 = await add_banned_words(CHANNEL, ["cross"])
        assert added2 == [] and skipped2 == ["cross"]

        assert await remove_banned_word(CHANNEL, "CROSS") is True
        assert await get_banned_words(CHANNEL) == ["free money"]
        assert await remove_banned_word(CHANNEL, "nope") is False

        assert await clear_banned_words(CHANNEL) is True
        assert await get_banned_words(CHANNEL) == []

    async def test_enable_disable_keeps_word_list(self, mock_db):
        from SWAGGYMUSIC.utils.database import (
            add_banned_words, get_banned_words, set_banned_words_enabled,
            settings_db,
        )

        await add_banned_words(CHANNEL, ["cross"])
        assert await set_banned_words_enabled(CHANNEL, True) == "ENABLED"
        assert await set_banned_words_enabled(CHANNEL, True) == "NO_CHANGE"
        assert await set_banned_words_enabled(CHANNEL, False) == "DISABLED"
        doc = await settings_db.find_one({"chat_id": CHANNEL})
        assert "banned_words_enabled" not in doc
        # the word list survives a disable (re-enabling restores it)
        assert await get_banned_words(CHANNEL) == ["cross"]

    async def test_words_live_in_the_shared_settings_document(self, mock_db):
        from SWAGGYMUSIC.utils.database import add_banned_words, settings_db

        await add_banned_words(CHANNEL, ["cross"])
        doc = await settings_db.find_one({"chat_id": CHANNEL})
        assert doc["banned_words"] == ["cross"]


# ---------------------------------------------------------------------------
# Gap/setdelay compatibility
# ---------------------------------------------------------------------------

class TestGapSafety:
    async def test_gap_engine_skips_rejected_posts(self, mock_db, monkeypatch):
        """A post the banned-words filter rejected never becomes the gap ref."""
        from SWAGGYMUSIC.plugins.misc import autodelete as ad
        from SWAGGYMUSIC.utils.database import set_gap_duration, settings_db
        from SWAGGYMUSIC import app

        await _enable_with_words(mock_db, ["cross"])
        await set_gap_duration(CHANNEL, 600)
        bannedwords_policy.mark_rejected(CHANNEL, 41)   # as the filter would

        member = MagicMock()
        member.status = ChatMemberStatus.ADMINISTRATOR
        member.privileges = MagicMock()
        member.privileges.can_delete_messages = True
        monkeypatch.setattr(app, "get_chat_member", AsyncMock(return_value=member))

        msg = _channel_post(41, "any text")
        await ad.autodelete_new_channel_post(app, msg)

        doc = await settings_db.find_one({"chat_id": CHANNEL})
        assert "last_gap_post_at" not in doc
        assert doc.get("gap_history") in (None, [])
