"""Shared channel-input resolution tests.

Covers:

* the parser — numeric ID, @username, bare username, public t.me link,
  private invite link, legacy joinchat link, approval invite link, and
  every invalid shape;
* the resolver — numeric ids need NO Telegram call, usernames/public
  links resolve via get_chat, private/approval invites resolve ONLY
  when the bot is already a member (Telegram limitation: bots cannot
  join private channels and Telegram does not reveal the channel behind
  an invite link to non-members), expired/invalid invite hashes and
  inaccessible channels get clear, separate error messages;
* bot permission checks — an inaccessible channel or a failed admin
  verification keeps each feature's EXISTING behaviour and messages;
* EVERY channel-specific feature uses the shared resolver — /setgap,
  /setdelay, /setlink, /setforward, /setanonymous, /bannedwords,
  /approvalmode, /createpost and /Schedule — with its original feature
  workflow unchanged after resolution.
"""
import re
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pyrogram.errors import (
    ChannelPrivate,
    InviteHashExpired,
    InviteHashInvalid,
    PeerIdInvalid,
    UsernameNotOccupied,
)

import SWAGGYMUSIC.utils.channel_resolution as cr
from SWAGGYMUSIC.utils.channel_resolution import (
    CHANNEL_INPUT_PROMPT,
    ChannelResolutionError,
    parse_channel_input,
    resolve_channel_input,
)

pytestmark = pytest.mark.asyncio

CHANNEL_ID = -1001234567890


def make_chat(chat_id=CHANNEL_ID, title="My Channel", username="mychannel"):
    chat = SimpleNamespace()
    chat.id = chat_id
    chat.title = title
    chat.username = username
    return chat


def make_private_message(text, user_id=42, conftest=None):
    from conftest import make_message

    return make_message(
        chat_id=user_id, message_id=1, text=text,
        chat_type="private", from_user_id=user_id,
    )


# ---------------------------------------------------------------------------
# The exact UI text
# ---------------------------------------------------------------------------

class TestPromptText:
    def test_prompt_is_the_exact_required_message(self):
        assert CHANNEL_INPUT_PROMPT == (
            "➕ ᴀᴅᴅ ᴄʜᴀɴɴᴇʟ\n\n"
            "sᴇɴᴅ ᴛʜᴇ ᴄʜᴀɴɴᴇʟ — ɪᴛs @username, ᴀ ᴛ.ᴍᴇ ʟɪɴᴋ, ᴏʀ ɪᴛs ɪᴅ "
            "(ʟɪᴋᴇ -1001234567890).\n\n"
            "ᴘᴜʙʟɪᴄ, ᴘʀɪᴠᴀᴛᴇ ᴀɴᴅ ᴀᴘᴘʀᴏᴠᴀʟ ɪɴᴠɪᴛᴇ ʟɪɴᴋꜱ ᴀʀᴇ ꜱᴜᴘᴘᴏʀᴛᴇᴅ.\n\n"
            "ᴍᴀᴋᴇ ꜱᴜʀᴇ ᴛʜᴇ ʙᴏᴛ ɪꜱ ᴀɴ ᴀᴅᴍɪɴ ɪɴ ᴛʜᴇ ᴄʜᴀɴɴᴇʟ.\n\n"
            "sᴇɴᴅ /cancel ᴛᴏ ᴄᴀɴᴄᴇʟ."
        )


# ---------------------------------------------------------------------------
# Parsing (pure — no Telegram I/O)
# ---------------------------------------------------------------------------

class TestParse:
    def test_numeric_channel_id(self):
        assert parse_channel_input("-1001234567890") == ("id", -1001234567890)
        assert parse_channel_input(" -1001234567890 \n") == ("id", -1001234567890)

    def test_numeric_id_rejects_int_spelled_out_forms(self):
        # int() would accept these as ints; the parser must never
        # return a NUMERIC id for any of them (they end up as
        # usernames/invite hashes for Telegram to judge, or are
        # rejected outright).
        for bad in ("1_000", "+5", "٣٣٣", "12 34"):
            try:
                kind, _value = parse_channel_input(bad)
            except ChannelResolutionError:
                continue  # rejected outright — fine
            assert kind != "id", f"{bad!r} must not parse as a numeric id"

    def test_at_username(self):
        assert parse_channel_input("@channelname") == ("username", "@channelname")
        assert parse_channel_input("  @My_Channel  ") == ("username", "@My_Channel")

    def test_bare_username_becomes_at_username(self):
        assert parse_channel_input("channelname") == ("username", "@channelname")

    def test_public_tme_link(self):
        for link in (
            "https://t.me/channelname",
            "t.me/channelname",
            "https://telegram.me/channelname",
            "https://www.t.me/channelname/",
            "t.me/s/channelname",  # web-preview form
        ):
            assert parse_channel_input(link) == ("username", "@channelname"), link

    def test_private_invite_link(self):
        assert parse_channel_input("https://t.me/+AbCdEf-123") == (
            "invite", "https://t.me/+AbCdEf-123",
        )
        assert parse_channel_input("t.me/+AbCdEf") == (
            "invite", "https://t.me/+AbCdEf",
        )
        # legacy form
        assert parse_channel_input("https://t.me/joinchat/AbCdEf") == (
            "invite", "https://t.me/joinchat/AbCdEf",
        )
        # bare share-sheet form
        assert parse_channel_input("+AbCdEf") == (
            "invite", "https://t.me/+AbCdEf",
        )

    def test_approval_invite_link_parses_like_a_private_invite(self):
        # Approval invites share the "+hash" form; resolvability is
        # decided by Telegram (member or not), not by the parser.
        assert parse_channel_input("https://t.me/+ApprovalInv1te")[0] == "invite"

    def test_invalid_inputs_raise_with_clear_message(self):
        for bad in (
            "", "   ", "not a channel!!", "https://spam.com/x",
            "t.me/", "@", "@a",  # too short
            "t.me/+inv it", "two words",
        ):
            with pytest.raises(ChannelResolutionError) as exc:
                parse_channel_input(bad)
            # either the generic "unrecognised" error or the specific
            # invalid-username error — always clear, never silent
            assert (
                "couldn't recognise" in exc.value.user_message
                or "doesn't look valid" in exc.value.user_message
            )
            assert exc.value.code == "invalid_input"


# ---------------------------------------------------------------------------
# Resolution (Telegram I/O mocked)
# ---------------------------------------------------------------------------

class TestResolve:
    async def test_numeric_id_needs_no_api_call(self, monkeypatch):
        from SWAGGYMUSIC import app

        get_chat = AsyncMock()
        monkeypatch.setattr(app, "get_chat", get_chat)
        chat_id, chat = await resolve_channel_input("-1001234567890")
        assert chat_id == CHANNEL_ID
        assert chat is None
        get_chat.assert_not_awaited()  # verification is the feature's job

    async def test_username_resolves_via_get_chat(self, monkeypatch):
        from SWAGGYMUSIC import app

        get_chat = AsyncMock(return_value=make_chat())
        monkeypatch.setattr(app, "get_chat", get_chat)
        chat_id, chat = await resolve_channel_input("@mychannel")
        assert chat_id == CHANNEL_ID
        assert chat.title == "My Channel"
        get_chat.assert_awaited_once_with("@mychannel")

    async def test_public_link_resolves_to_channel_id(self, monkeypatch):
        from SWAGGYMUSIC import app

        get_chat = AsyncMock(return_value=make_chat())
        monkeypatch.setattr(app, "get_chat", get_chat)
        chat_id, _ = await resolve_channel_input("https://t.me/mychannel")
        assert chat_id == CHANNEL_ID
        get_chat.assert_awaited_once_with("@mychannel")

    async def test_private_invite_resolves_when_bot_is_member(
        self, monkeypatch,
    ):
        """Telegram's checkChatInvite answers ChatInviteAlready for an
        invite the bot already used — kurigram returns the real chat."""
        from SWAGGYMUSIC import app

        get_chat = AsyncMock(return_value=make_chat())
        monkeypatch.setattr(app, "get_chat", get_chat)
        chat_id, chat = await resolve_channel_input(
            "https://t.me/+PrivateInv1te",
        )
        assert chat_id == CHANNEL_ID
        get_chat.assert_awaited_once_with("https://t.me/+PrivateInv1te")

    async def test_private_invite_not_a_member_is_honestly_unresolvable(
        self, monkeypatch,
    ):
        """Bot NOT a member: Telegram returns invite metadata only (an
        id-less preview chat).  The resolver must NOT guess an id — it
        states the Telegram API limitation instead."""
        from SWAGGYMUSIC import app

        preview = SimpleNamespace(id=None, is_preview=True,
                                  title="Private Channel")
        get_chat = AsyncMock(return_value=preview)
        monkeypatch.setattr(app, "get_chat", get_chat)
        with pytest.raises(ChannelResolutionError) as exc:
            await resolve_channel_input("https://t.me/+NotAMember1")
        assert exc.value.code == "invite_unresolvable"
        assert "can't resolve this private invite" in exc.value.user_message
        assert "add me to the channel" in exc.value.user_message.lower()

    async def test_approval_invite_same_honest_limitation(self, monkeypatch):
        from SWAGGYMUSIC import app

        preview = SimpleNamespace(id=None, is_preview=True,
                                  title="Approval Channel",
                                  join_by_request=True)
        get_chat = AsyncMock(return_value=preview)
        monkeypatch.setattr(app, "get_chat", get_chat)
        with pytest.raises(ChannelResolutionError) as exc:
            await resolve_channel_input("https://t.me/+ApprovalInv1te")
        assert exc.value.code == "invite_unresolvable"

    async def test_expired_invite_link(self, monkeypatch):
        from SWAGGYMUSIC import app

        get_chat = AsyncMock(side_effect=InviteHashExpired())
        monkeypatch.setattr(app, "get_chat", get_chat)
        with pytest.raises(ChannelResolutionError) as exc:
            await resolve_channel_input("https://t.me/+ExpiredLink1")
        assert exc.value.code == "invite_expired"
        assert "expired or was revoked" in exc.value.user_message

    async def test_invalid_invite_link(self, monkeypatch):
        from SWAGGYMUSIC import app

        get_chat = AsyncMock(side_effect=InviteHashInvalid())
        monkeypatch.setattr(app, "get_chat", get_chat)
        with pytest.raises(ChannelResolutionError) as exc:
            await resolve_channel_input("https://t.me/+BrokenLink1")
        assert exc.value.code == "invite_invalid"
        assert "invalid" in exc.value.user_message.lower()

    async def test_inaccessible_username_gives_clear_error(self, monkeypatch):
        from SWAGGYMUSIC import app

        get_chat = AsyncMock(side_effect=ChannelPrivate())
        monkeypatch.setattr(app, "get_chat", get_chat)
        with pytest.raises(ChannelResolutionError) as exc:
            await resolve_channel_input("@gonechannel")
        assert exc.value.code == "unreachable"
        assert "can't reach that channel" in exc.value.user_message
        assert "ChannelPrivate" in exc.value.user_message

    async def test_inaccessible_numeric_username_error(self, monkeypatch):
        from SWAGGYMUSIC import app

        get_chat = AsyncMock(side_effect=UsernameNotOccupied())
        monkeypatch.setattr(app, "get_chat", get_chat)
        with pytest.raises(ChannelResolutionError) as exc:
            await resolve_channel_input("@nobodyhere")
        assert exc.value.code == "unreachable"

    async def test_peer_id_invalid_is_unreachable(self, monkeypatch):
        from SWAGGYMUSIC import app

        get_chat = AsyncMock(side_effect=PeerIdInvalid())
        monkeypatch.setattr(app, "get_chat", get_chat)
        with pytest.raises(ChannelResolutionError) as exc:
            await resolve_channel_input("@hiddenchannel")
        assert exc.value.code == "unreachable"

    async def test_invalid_input_never_touches_the_api(self, monkeypatch):
        from SWAGGYMUSIC import app

        get_chat = AsyncMock()
        monkeypatch.setattr(app, "get_chat", get_chat)
        with pytest.raises(ChannelResolutionError):
            await resolve_channel_input("https://spam.com/x")
        get_chat.assert_not_awaited()


# ---------------------------------------------------------------------------
# Every channel-specific feature uses the shared resolver
# ---------------------------------------------------------------------------

import importlib

def _setup_channel_mocks(monkeypatch, chat):
    """Bot access + admin membership mocks for the feature workflows."""
    from SWAGGYMUSIC import app

    monkeypatch.setattr(app, "get_chat", AsyncMock(return_value=chat))
    member = SimpleNamespace()
    from pyrogram.enums import ChatMemberStatus

    member.status = ChatMemberStatus.ADMINISTRATOR
    privileges = SimpleNamespace()
    privileges.can_delete_messages = True
    privileges.can_post_messages = True
    member.privileges = privileges
    monkeypatch.setattr(app, "get_chat_member", AsyncMock(return_value=member))
    return app


_PLUGINS = {
    "schedule": "SWAGGYMUSIC.plugins.misc.schedule",
    "setconfig": "SWAGGYMUSIC.plugins.bot.setconfig",
    "setlink": "SWAGGYMUSIC.plugins.misc.setlink",
    "setforward": "SWAGGYMUSIC.plugins.misc.setforward",
    "anonymous": "SWAGGYMUSIC.plugins.misc.anonymous",
    "bannedwords": "SWAGGYMUSIC.plugins.misc.bannedwords",
    "approvalmode": "SWAGGYMUSIC.plugins.misc.approvalmode",
    "createpost": "SWAGGYMUSIC.plugins.misc.createpost",
}

# Stable per-feature user ids (tests share one mongomock database).
_UIDS = {
    "setconfig": 51001, "setlink": 51002, "setforward": 51003,
    "anonymous": 51004, "bannedwords": 51005, "approvalmode": 51006,
    "createpost": 51007, "schedule": 51008,
}

# Text-input handler of each wizard (module name != handler prefix for
# /setanonymous: the module is anonymous.py, the handler setanonymous_*).
_HANDLER_NAMES = {
    "setlink": "setlink_channel_input",
    "setforward": "setforward_channel_input",
    "anonymous": "setanonymous_channel_input",
    "bannedwords": "bannedwords_channel_input",
    "approvalmode": "approvalmode_channel_input",
}

# Per-feature wizard state after a SUCCESSFUL channel resolution.
_SUCCESS_STEP = {
    "schedule": "list",       # manage purpose → the schedules list
    "createpost": "format",   # next wizard step
}
_SUCCESS_STEP_DEFAULT = "action"


def _expected_session(mod_name):
    """(success step, channel-key, session-popped-on-permission-failure)."""
    step = _SUCCESS_STEP.get(mod_name, _SUCCESS_STEP_DEFAULT)
    key = "chat_id" if mod_name == "createpost" else "channel_id"
    # createpost's own _bot_admin_status failure keeps the wizard alive
    # (existing behaviour); every other feature pops the session.
    keeps_session = mod_name in {"createpost", "schedule"}
    return step, key, keeps_session


def _seed_session(plugin, mod_name, feature):
    if mod_name == "setconfig":
        return {"feature": feature, "step": "channel_id"}
    if mod_name == "createpost":
        return {"step": "chat_id", "post_type": "normal"}
    return {"step": "channel_id"}


async def _store_session(plugin, mod_name, user_id, feature):
    if hasattr(plugin, "LOCK"):
        async with plugin.LOCK:
            plugin.SESSIONS[user_id] = _seed_session(
                plugin, mod_name, feature)
    else:
        plugin.SESSIONS[user_id] = _seed_session(plugin, mod_name, feature)


def _spy_resolver(monkeypatch, plugin):
    """Intercept the plugin's imported resolve_channel_input — proves
    the feature actually routes its channel input through the SHARED
    resolver — while delegating to the real implementation."""
    import SWAGGYMUSIC.utils.channel_resolution as _cr

    calls = []
    real = _cr.resolve_channel_input

    async def _spy(raw, **kwargs):
        calls.append(str(raw))
        return await real(raw, **kwargs)

    monkeypatch.setattr(plugin, "resolve_channel_input", _spy)
    return calls


async def drive_feature(monkeypatch, mod_name, text, *,
                        verify=(True, None), feature="setgap"):
    """Drive a feature's channel-input handler with ``text``.

    Returns (plugin, calls, replies, session, verify_mock, message).
    """
    plugin = importlib.import_module(_PLUGINS[mod_name])
    user_id = _UIDS[mod_name]

    if mod_name == "schedule":
        _setup_channel_mocks(monkeypatch, make_chat())
        calls = _spy_resolver(monkeypatch, plugin)

        async def _check(channel_id, uid,
                         *, _chat=make_chat(), verify=verify):
            if verify[0]:
                return True, _chat, None
            return False, None, verify[1]

        monkeypatch.setattr(plugin, "_check_channel", _check)
        session = _seed_session(plugin, mod_name, feature)
        message = make_private_message(text, user_id=user_id)
        await plugin._on_channel_input(message, session)
        replies = [str(c.args[0]) for c in message.reply_text.await_args_list]
        return plugin, calls, replies, session, None, message

    verifier_name = (
        "_bot_admin_status" if mod_name == "createpost"
        else "verify_channel_admin"
    )
    monkeypatch.setattr(
        plugin, verifier_name, AsyncMock(return_value=verify),
    )
    _setup_channel_mocks(monkeypatch, make_chat())
    calls = _spy_resolver(monkeypatch, plugin)
    await _store_session(plugin, mod_name, user_id, feature)

    message = make_private_message(text, user_id=user_id)
    if mod_name == "setconfig":
        await plugin.setconfig_input(None, message)
    elif mod_name == "createpost":
        await plugin.createpost_input(None, message)
    else:
        handler = getattr(
            plugin, _HANDLER_NAMES[mod_name])
        await handler(None, message)

    replies = [str(c.args[0]) for c in message.reply_text.await_args_list]
    session = plugin.SESSIONS.get(user_id)
    return plugin, calls, replies, session, None, message


def _cleanup_sessions():
    import SWAGGYMUSIC.plugins.misc.approvalmode as _am
    import SWAGGYMUSIC.plugins.misc.bannedwords as _bw
    import SWAGGYMUSIC.plugins.misc.createpost as _cp
    import SWAGGYMUSIC.plugins.misc.setforward as _sf
    import SWAGGYMUSIC.plugins.misc.setlink as _sl
    import SWAGGYMUSIC.plugins.misc.anonymous as _an
    import SWAGGYMUSIC.plugins.bot.setconfig as _sc

    for mod in (_sc, _sl, _sf, _an, _bw, _am, _cp):
        mod.SESSIONS.pop(_UIDS[mod.__name__.rsplit(".", 1)[-1]], None)


class TestEveryFeatureUsesTheSharedResolver:
    """@username must behave EXACTLY like a numeric ID in every feature:
    same session transition, same menu reply, same permission check."""

    @pytest.fixture(autouse=True)
    def _wipe(self):
        yield
        _cleanup_sessions()

    @pytest.mark.parametrize("mod_name", list(_PLUGINS))
    async def test_numeric_id_still_works_unchanged(self, monkeypatch,
                                                    mod_name):
        _plugin, calls, replies, session, _v, _m = await drive_feature(
            monkeypatch, mod_name, "-1001234567890",
        )
        step, key, _keeps = _expected_session(mod_name)
        assert calls == ["-1001234567890"]  # routed through the resolver
        assert session.get(key) == CHANNEL_ID
        assert session.get("step") == step
        # createpost's original flow replies with its format menu (no
        # "Channel ID" line) — exactly as before; the others display it.
        if mod_name not in {"schedule", "createpost"}:
            assert any("Channel ID:" in r for r in replies)

    @pytest.mark.parametrize("mod_name", list(_PLUGINS))
    async def test_at_username_works_end_to_end(self, monkeypatch,
                                                mod_name):
        _plugin, calls, replies, session, _v, _m = await drive_feature(
            monkeypatch, mod_name, "@mychannel",
        )
        step, key, _keeps = _expected_session(mod_name)
        assert calls == ["@mychannel"]  # the shared resolver was used
        assert session.get(key) == CHANNEL_ID
        assert session.get("step") == step

    @pytest.mark.parametrize("mod_name", list(_PLUGINS))
    async def test_public_tme_link_works_end_to_end(self, monkeypatch,
                                                    mod_name):
        _plugin, calls, _replies, session, _v, _m = await drive_feature(
            monkeypatch, mod_name, "https://t.me/mychannel",
        )
        step, key, _keeps = _expected_session(mod_name)
        assert calls == ["https://t.me/mychannel"]
        assert session.get(key) == CHANNEL_ID
        assert session.get("step") == step

    async def test_private_invite_link_works_in_setlink(
        self, monkeypatch,
    ):
        """Invite link + bot already a member → resolves to the real
        channel and continues the original workflow."""
        _plugin, calls, _replies, session, _v, _m = await drive_feature(
            monkeypatch, "setlink", "https://t.me/+PrivateInv1te",
        )
        assert calls == ["https://t.me/+PrivateInv1te"]
        assert session.get("channel_id") == CHANNEL_ID
        assert session.get("step") == "action"

    @pytest.mark.parametrize("mod_name", list(_PLUGINS))
    async def test_invalid_input_is_rejected_before_permissions(
        self, monkeypatch, mod_name,
    ):
        """Unrecognisable input: clear error, session kept at the
        channel step, and the permission check never ran."""
        plugin, calls, replies, session, verify_mock, _m = \
            await drive_feature(
                monkeypatch, mod_name, "https://spam.com/x",
            )
        step, _key, _keeps = _expected_session(mod_name)
        start_step = ("chat_id" if mod_name == "createpost"
                      else "channel_id")
        assert calls == ["https://spam.com/x"]  # still via the resolver
        assert session is not None  # the wizard is NOT killed
        assert session.get("step") == start_step
        assert any("couldn't recognise" in r for r in replies)
        if verify_mock is not None:
            verify_mock.assert_not_awaited()

    @pytest.mark.parametrize("mod_name", list(_PLUGINS))
    async def test_permission_failure_path_is_unchanged(
        self, monkeypatch, mod_name,
    ):
        """The user is not an admin: each feature keeps its EXISTING
        behaviour — reply with the verifier's reason and end the
        session (resolution itself succeeded)."""
        _plugin, calls, replies, session, _v, _m = await drive_feature(
            monkeypatch, mod_name, "-1001234567890",
            verify=(False, "<b>REASON-HTML</b>"),
        )
        _step, key, keeps_session = _expected_session(mod_name)
        assert calls == ["-1001234567890"]
        assert any("REASON-HTML" in r for r in replies)
        if keeps_session:
            assert session is not None  # existing behaviour preserved
        else:
            assert session is None  # session popped, exactly as before


class TestScheduleWizardInput:
    async def test_username_input_opens_the_manager(self, monkeypatch):
        mod_name = "schedule"
        plugin, calls, _replies, session, _v, _m = await drive_feature(
            monkeypatch, mod_name, "@mychannel",
        )
        assert calls == ["@mychannel"]
        assert session["channel_id"] == CHANNEL_ID
        assert session["step"] == "list"

    async def test_invalid_input_keeps_retry_markup_flow(self, monkeypatch):
        mod_name = "schedule"
        plugin, calls, replies, session, _v, _m = await drive_feature(
            monkeypatch, mod_name, "!!",
        )
        assert calls == ["!!"]
        assert session.get("channel_id") is None


# ---------------------------------------------------------------------------
# The prompt is used wherever a channel is requested
# ---------------------------------------------------------------------------

def _started_reply(mod, handler, message):
    import asyncio

    handler_fn = getattr(mod, handler)
    coro = handler_fn(None, message)
    return coro


class TestPromptShownWhereverChannelRequested:
    @pytest.fixture(autouse=True)
    def _wipe(self):
        yield
        _cleanup_sessions()

    async def _start_reply_text(self, monkeypatch, mod_name, handler,
                                feature=None):
        plugin = importlib.import_module(_PLUGINS[mod_name])
        user_id = _UIDS[mod_name]
        message = make_private_message("", user_id=user_id)
        fn = getattr(plugin, handler)
        if feature is None:
            await fn(None, message)
        else:
            await fn(None, message)
        replies = [str(c.args[0]) for c in message.reply_text.await_args_list]
        return replies

    @pytest.mark.parametrize(
        "mod_name,handler",
        [
            ("setconfig", "setgap_private_start"),
            ("setconfig", "setdelay_private_start"),
            ("setlink", "setlink_private_start"),
            ("setforward", "setforward_private_start"),
            ("anonymous", "setanonymous_start"),
            ("bannedwords", "bannedwords_private_start"),
            ("approvalmode", "approvalmode_private_start"),
        ],
    )
    async def test_start_messages_show_the_add_channel_prompt(
        self, monkeypatch, mod_name, handler,
    ):
        replies = await self._start_reply_text(
            monkeypatch, mod_name, handler,
        )
        assert replies, "the wizard must ask for the channel"
        assert any("ᴀᴅᴅ ᴄʜᴀɴɴᴇʟ" in r for r in replies)
        assert any("/cancel" in r for r in replies)

    async def test_createpost_shows_the_prompt_at_the_chat_id_step(
        self, monkeypatch,
    ):
        from conftest import make_callback_query

        plugin = importlib.import_module(_PLUGINS["createpost"])
        user_id = _UIDS["createpost"]
        await _store_session(plugin, "createpost", user_id, None)
        plugin.SESSIONS[user_id] = {"step": "post_type"}
        query = make_callback_query(
            user_id=user_id, data="createpost:type:normal",
        )
        await plugin.createpost_callbacks(None, query)
        edits = [str(c.args[0]) for c in query.message.edit_text.await_args_list]
        assert any("ᴀᴅᴅ ᴄʜᴀɴɴᴇʟ" in t for t in edits)

    async def test_schedule_panels_show_the_prompt(self):
        plugin = importlib.import_module(_PLUGINS["schedule"])
        manage_text, _markup = plugin.render_ask_channel(
            {"purpose": "manage"},
        )
        create_text, _markup = plugin.render_ask_channel(
            {"purpose": "create", "kind": plugin.store.TYPE_ONCE},
        )
        for text in (manage_text, create_text):
            assert "ᴀᴅᴅ ᴄʜᴀɴɴᴇʟ" in text
            assert "/cancel" in text
