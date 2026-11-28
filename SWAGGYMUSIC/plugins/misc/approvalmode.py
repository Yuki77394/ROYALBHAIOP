"""Channel APPROVAL MODE filter.

/approvalmode on | /approvalmode off  -> CHANNEL ONLY: enable / disable
Approval Mode for the channel the command is posted in (same shape as
/setlink and /setforward; the command message is deleted immediately and
the bot's reply auto-deletes after 120 seconds).

/approvalmode in the bot's PRIVATE chat -> the management wizard:

    /approvalmode
        → enter the Channel ID
        → status menu:
            🟢 ENABLE / 🔴 DISABLE
            👥 ADD USERNAMES / ➖ REMOVE USERNAME / 📋 VIEW LIST
            ❌ CANCEL

The wizard verifies that the user is an admin of the target channel
(utils/permissions.verify_channel_admin — re-verified on every button
press) and that the bot can delete there. The owner bypasses the
channel-admin check.

POLICY (utils/approval_policy.py — when the feature is ENABLED):
    KEEP   a post mentioning at least one approved username;
    KEEP + WARN (logger group) a post mentioning approved AND
           unapproved usernames — mixed posts are NEVER deleted;
    DELETE a post mentioning NO approved username (posts without any
           username included);
    OFF    no restriction at all.

Username detection works in message text, captions and Telegram entities
(MENTION + TEXT_MENTION, plus a conservative @handle text scan), and is
exact: the whole handle is extracted and compared case-insensitively, so
``john`` never matches ``johnny``.

MODERATION (handler group -23, directly after Banned Words):
    -40  link filter
    -30  anonymous filter
    -25  forward filter
    -24  banned-words filter
    -23  approvalmode_new_channel_post   — new channel posts
    -23  approvalmode_edited_channel_post — edited posts
    -15  /approvalmode channel command
      0  /setgap + /setdelay engine

A post an earlier filter already rejected for this update is left to it.
A post this filter rejects is marked in-process
(approval_policy.mark_rejected) BEFORE any Telegram call, so it can
never become a /setgap reference or a /setdelay job. Command messages
are never judged.

DELETION + LOGGER: deletions go through the shared flow
(``archive_deletion`` → delete → ``mark_deletion_result``); the
centralized logger turns every confirmed deletion into one "🗑️ POST
DELETED" event, and FloodWait retries / failures are reported
separately — the log never claims a deletion Telegram did not confirm.

ALBUMS: a media album is ONE post, judged once by its first delivered
item (synchronous check-and-set, atomic under asyncio): an item whose
caption carries an approved username approves the whole album; an item
without one condemns the album whole (current + already-delivered
siblings + later items). The condemned album id is persisted in
``approval_blocked_albums`` so a restart mid-delivery cannot leave a
partial album.

SETTINGS: one document per channel in mongodb.autodelete_settings (the
shared settings document — /setgap, /setdelay, /setanonymous, /setlink,
/setforward and Banned Words live there too):
    { "chat_id": <channel_id>,
      "approval_mode_enabled": True,
      "approval_usernames": ["john", "team_lead", ...],  # newest 100
      "approval_blocked_albums": [<media_group_id>, ...] }  # newest 50
Settings persist across restarts; the username list also survives a
DISABLE.
"""

import asyncio
import re
import time
from contextlib import suppress

from pyrogram import filters
from pyrogram.enums import (ButtonStyle, ChatMemberStatus, ChatType,
                            MessageServiceType)
from pyrogram.errors import FloodWait, MessageIdInvalid, RPCError
from pyrogram.types import (CallbackQuery, InlineKeyboardButton,
                            InlineKeyboardMarkup, Message)

from SWAGGYMUSIC import app
from SWAGGYMUSIC.logging import LOGGER
from SWAGGYMUSIC.utils.deletion_audit import archive_deletion, mark_deletion_result
from SWAGGYMUSIC.utils import (anonymous_policy, approval_policy,
                               bannedwords_policy, forward_policy, link_policy)
from SWAGGYMUSIC.utils.logger import (log_approval_mixed_warning,
                                      log_approval_mode_change,
                                      log_approval_usernames_changed,
                                      log_operation_failed)
from SWAGGYMUSIC.utils.database import (add_approval_usernames,
                                        clear_approval_usernames,
                                        get_approval_usernames,
                                        queue_deletion_job,
                                        remember_approval_blocked_album,
                                        remove_approval_username,
                                        set_approval_mode_enabled,
                                        settings_db)
from SWAGGYMUSIC.utils.permissions import verify_channel_admin
from SWAGGYMUSIC.utils.channel_resolution import (
    CHANNEL_INPUT_PROMPT,
    ChannelResolutionError,
    resolve_channel_input,
)

log = LOGGER(__name__)
PREFIX = "approvalmode"
SESSIONS = {}
LOCK = asyncio.Lock()

# The bot's channel-side reply auto-deletes after this many seconds —
# the same TTL the other feature replies use.
_RESPONSE_TTL = 120


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

async def _in_active_session(_, __, message) -> bool:
    return bool(
        message.from_user
        and message.from_user.id in SESSIONS
        and message.chat.type == ChatType.PRIVATE
    )


_ACTIVE_SESSION = filters.create(_in_active_session, name="ApprovalModeSessionFilter")


# ---------------------------------------------------------------------------
# Helpers (same contracts as setlink.py / bannedwords.py)
# ---------------------------------------------------------------------------

def _post_time(message: Message) -> int:
    with suppress(AttributeError, OSError, ValueError):
        return int(message.date.timestamp())
    return int(time.time())


async def _bot_can_delete(chat_id: int) -> bool:
    """True if the bot may delete posts in the channel (never raises)."""
    try:
        member = await app.get_chat_member(chat_id, app.id)
        if member.status == ChatMemberStatus.OWNER:
            return True
        privileges = getattr(member, "privileges", None)
        return bool(
            member.status == ChatMemberStatus.ADMINISTRATOR
            and privileges and privileges.can_delete_messages
        )
    except Exception as exc:
        log.warning("approvalmode bot-permission check failed for %s: %s: %s",
                    chat_id, type(exc).__name__, exc)
        return False


async def _delete_command_message(message: Message, *, reason: str) -> None:
    """Delete a command message from the channel; never raises."""
    with suppress(Exception):
        await message.delete()
    log.info(
        "deletion reason=%s chat=%s msg=%s",
        reason,
        getattr(getattr(message, "chat", None), "id", "?"),
        getattr(message, "id", "?"),
    )


def _delete_after(message: Message, delay: int) -> None:
    """Non-blocking delayed deletion of the bot's own reply."""

    async def _inner():
        try:
            await asyncio.sleep(delay)
            with suppress(Exception):
                await message.delete()
            log.info(
                "deletion reason=BOT_REPLY_CLEANUP chat=%s msg=%s",
                getattr(getattr(message, "chat", None), "id", "?"),
                getattr(message, "id", "?"),
            )
        except Exception:
            pass

    asyncio.create_task(_inner())


async def _get_chat(channel_id: int):
    try:
        return await app.get_chat(channel_id)
    except Exception as exc:
        log.warning("approvalmode could not read channel %s: %s: %s",
                    channel_id, type(exc).__name__, exc)
        return None


# ---------------------------------------------------------------------------
# Private-chat wizard
# ---------------------------------------------------------------------------

def _cancel_menu():
    return InlineKeyboardMarkup([[
        InlineKeyboardButton("❌ CANCEL", callback_data=f"{PREFIX}:cancel",
                             style=ButtonStyle.DANGER),
    ]])


def _menu(channel_id: int):
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton("🟢 ENABLE", callback_data=f"{PREFIX}:enable:{channel_id}",
                                 style=ButtonStyle.SUCCESS),
            InlineKeyboardButton("🔴 DISABLE", callback_data=f"{PREFIX}:disable:{channel_id}",
                                 style=ButtonStyle.DANGER),
        ],
        [
            InlineKeyboardButton("👥 ADD USERNAMES", callback_data=f"{PREFIX}:add:{channel_id}",
                                 style=ButtonStyle.PRIMARY),
            InlineKeyboardButton("➖ REMOVE USERNAME", callback_data=f"{PREFIX}:remove:{channel_id}",
                                 style=ButtonStyle.PRIMARY),
        ],
        [
            InlineKeyboardButton("📋 VIEW LIST", callback_data=f"{PREFIX}:view:{channel_id}",
                                 style=ButtonStyle.PRIMARY),
            InlineKeyboardButton("🗑 CLEAR LIST", callback_data=f"{PREFIX}:clear:{channel_id}",
                                 style=ButtonStyle.DANGER),
        ],
        [
            InlineKeyboardButton("❌ CANCEL", callback_data=f"{PREFIX}:cancel",
                                 style=ButtonStyle.DANGER),
        ],
    ])


_FEATURE_TEXT = (
    "✅ <b>Approval Mode:</b> every channel post must mention at least one "
    "<b>approved username</b>.\n\n"
    "• A post with an approved username is <b>kept</b>.\n"
    "• A post with approved AND unapproved usernames is <b>kept</b> and a "
    "warning is logged — mixed posts are never deleted.\n"
    "• A post with NO approved username (including posts without any "
    "username) is <b>deleted</b>.\n\n"
    "<i>Detection covers text, captions and Telegram entities; matches are "
    "exact — @john never matches @johnny.</i>"
)


def _status_line(enabled: bool, names: list) -> str:
    state = "ON 🟢" if enabled else "OFF 🔴"
    shown = ", ".join(f"@{n}" for n in names[:15])
    if len(names) > 15:
        shown += f" (+{len(names) - 15} more)"
    return f"✅ <b>Current status:</b> {state}\n👥 <b>Approved usernames ({len(names)}):</b> {shown or '—'}"


@app.on_message(filters.command(["approvalmode"]) & filters.private)
async def approvalmode_private_start(_, message: Message):
    if not message.from_user:
        return
    async with LOCK:
        SESSIONS[message.from_user.id] = {"step": "channel_id"}
    await message.reply_text(
        "✅ <b>APPROVAL MODE</b>\n\n"
        + CHANNEL_INPUT_PROMPT,
        reply_markup=_cancel_menu(),
    )


@app.on_callback_query(filters.regex(rf"^{PREFIX}:"))
async def approvalmode_callback(_, query: CallbackQuery):
    if not query.from_user:
        return await query.answer("Not available.", show_alert=True)
    user_id = query.from_user.id
    data = query.data or ""
    parts = data.split(":")

    if data == f"{PREFIX}:cancel":
        SESSIONS.pop(user_id, None)
        await query.answer("Cancelled")
        with suppress(Exception):
            await query.message.edit_text("❌ Configuration cancelled.")
        return

    if len(parts) != 3 or parts[1] not in {"enable", "disable", "add", "remove", "view", "clear"}:
        return await query.answer("Invalid option.", show_alert=True)
    try:
        channel_id = int(parts[2])
    except ValueError:
        return await query.answer("Invalid Channel ID.", show_alert=True)

    session = SESSIONS.get(user_id)
    if not session or session.get("channel_id") != channel_id:
        return await query.answer(
            "Setup expired. Use /approvalmode again.", show_alert=True,
        )

    # Never trust the UI: the admin right is re-verified on every press.
    ok, reason = await verify_channel_admin(channel_id, user_id)
    if not ok:
        SESSIONS.pop(user_id, None)
        await query.answer("You cannot manage this channel.", show_alert=True)
        with suppress(Exception):
            await query.message.edit_text(reason)
        return

    action = parts[1]
    chat = await _get_chat(channel_id)
    title = getattr(chat, "title", None) or "Channel"

    # ─── ENABLE / DISABLE ─────────────────────────────────────────
    if action in {"enable", "disable"}:
        enabled = action == "enable"
        try:
            status = await set_approval_mode_enabled(channel_id, enabled)
            names = await get_approval_usernames(channel_id)
            await query.answer("Saved")
            with suppress(Exception):
                if enabled:
                    await query.message.edit_text(
                        f"✅ <b>APPROVAL MODE ENABLED 🟢</b>\n\n"
                        f"📌 <b>Channel:</b> {title}\n"
                        f"🆔 <b>Channel ID:</b> <code>{channel_id}</code>\n\n"
                        + _FEATURE_TEXT + "\n\n"
                        + _status_line(True, names) + "\n\n"
                        + ("<i>⚠️ No approved usernames yet — add some, or "
                           "EVERY post will be deleted.</i>" if not names else "")
                    )
                else:
                    await query.message.edit_text(
                        f"✅ <b>APPROVAL MODE DISABLED 🔴</b>\n\n"
                        f"📌 <b>Channel:</b> {title}\n"
                        f"🆔 <b>Channel ID:</b> <code>{channel_id}</code>\n\n"
                        "Approval Mode is now turned off — posts are no longer "
                        "checked for approved usernames.\n\n"
                        + _status_line(False, names)
                    )
            if status != "NO_CHANGE":
                with suppress(Exception):
                    await log_approval_mode_change(
                        chat, query.from_user,
                        "ENABLED" if enabled else "DISABLED",
                        source="PRIVATE CHAT", usernames=len(names),
                    )
            SESSIONS.pop(user_id, None)
        except Exception as exc:
            log.warning("approvalmode toggle failed for %s: %s", channel_id, exc)
            await query.answer("Could not update settings.", show_alert=True)
        return

    # ─── ADD USERNAMES ────────────────────────────────────────────
    if action == "add":
        session["step"] = "add_names"
        await query.answer("Add usernames")
        with suppress(Exception):
            await query.message.edit_text(
                "👥 <b>ADD APPROVED USERNAMES</b>\n\n"
                "Send one or more usernames.\n"
                "Separate entries with <b>commas</b> or <b>new lines</b>.\n\n"
                "<i>Example:</i> <code>@john, team_lead, @Mary_88</code>\n\n"
                "Valid handles: 3-64 characters, letters / digits / underscore.",
                reply_markup=_cancel_menu(),
            )
        return

    # ─── REMOVE USERNAME ──────────────────────────────────────────
    if action == "remove":
        session["step"] = "remove_name"
        await query.answer("Remove a username")
        with suppress(Exception):
            names = await get_approval_usernames(channel_id)
            await query.message.edit_text(
                "➖ <b>REMOVE APPROVED USERNAME</b>\n\n"
                "Send the exact username to remove.\n\n"
                + _status_line(True, names),
                reply_markup=_cancel_menu(),
            )
        return

    # ─── VIEW LIST ────────────────────────────────────────────────
    if action == "view":
        await query.answer()
        names = await get_approval_usernames(channel_id)
        current = await settings_db.find_one(
            {"chat_id": channel_id}, {"approval_mode_enabled": 1},
        )
        enabled = bool(current and current.get("approval_mode_enabled"))
        with suppress(Exception):
            await query.message.edit_text(
                "📋 <b>APPROVED USERNAMES LIST</b>\n\n"
                f"📌 <b>Channel:</b> {title}\n"
                f"🆔 <b>Channel ID:</b> <code>{channel_id}</code>\n\n"
                + _status_line(enabled, names),
                reply_markup=_menu(channel_id),
            )
        return

    # ─── CLEAR LIST ───────────────────────────────────────────────
    if action == "clear":
        await query.answer("Cleared")
        with suppress(Exception):
            removed = await clear_approval_usernames(channel_id)
            names = await get_approval_usernames(channel_id)
            current = await settings_db.find_one(
                {"chat_id": channel_id}, {"approval_mode_enabled": 1},
            )
            enabled = bool(current and current.get("approval_mode_enabled"))
            await query.message.edit_text(
                "🗑 <b>APPROVED USERNAMES LIST CLEARED</b>\n\n"
                f"📌 <b>Channel:</b> {title}\n"
                f"🆔 <b>Channel ID:</b> <code>{channel_id}</code>\n\n"
                + _status_line(enabled, names),
                reply_markup=_menu(channel_id),
            )
            if removed:
                with suppress(Exception):
                    await log_approval_usernames_changed(
                        chat, query.from_user, "CLEARED", [], total=0,
                    )
        return


@app.on_message(
    _ACTIVE_SESSION
    & ~filters.service
    & ~filters.command(["approvalmode", "cancel"]),
)
async def approvalmode_channel_input(_, message: Message):
    if not message.from_user:
        return
    raw = (message.text or message.caption or "").strip()
    if raw.startswith("/"):
        # Another command (e.g. /setlink): the user left this wizard. End
        # the session and let the command reach its own handler — a
        # username list never starts with "/".
        SESSIONS.pop(message.from_user.id, None)
        from pyrogram import ContinuePropagation

        raise ContinuePropagation
    session = SESSIONS.get(message.from_user.id)
    if not session:
        return
    step = session.get("step")
    channel_id = session.get("channel_id")

    # ─── Channel input — shared resolver (ID, @username, t.me link,
    # private/approval invite link) ────────────────────────────────
    if step == "channel_id":
        try:
            channel_id, _resolved_chat = await resolve_channel_input(raw)
        except ChannelResolutionError as exc:
            return await message.reply_text(exc.user_message)
        ok, reason = await verify_channel_admin(channel_id, message.from_user.id)
        if not ok:
            SESSIONS.pop(message.from_user.id, None)
            return await message.reply_text(reason)

        session.update({"channel_id": channel_id, "step": "action"})
        current = await settings_db.find_one(
            {"chat_id": channel_id},
            {"approval_mode_enabled": 1, "approval_usernames": 1},
        )
        enabled = bool(current and current.get("approval_mode_enabled"))
        names = [str(n) for n in (current or {}).get("approval_usernames") or []]
        chat = await _get_chat(channel_id)
        title = getattr(chat, "title", None) or "Channel"
        await message.reply_text(
            f"📌 <b>Channel:</b> {title}\n"
            f"🆔 <b>Channel ID:</b> <code>{channel_id}</code>\n"
            + _status_line(enabled, names)
            + ("\n\n⚠️ I need the <b>Delete Messages</b> admin right in that "
               "channel before this filter can work."
               if not await _bot_can_delete(channel_id) else "")
            + "\n\nChoose an action:",
            reply_markup=_menu(channel_id),
        )
        return

    # ─── Add usernames input ──────────────────────────────────────
    if step == "add_names":
        if channel_id is None:
            SESSIONS.pop(message.from_user.id, None)
            return await message.reply_text("❌ Setup expired. Use /approvalmode again.")
        entries = [part.strip() for part in re.split(r"[,\n]", raw) if part.strip()]
        if not entries:
            return await message.reply_text(
                "❌ Send one or more usernames, separated by commas or new lines.",
                reply_markup=_cancel_menu(),
            )
        added, skipped = await add_approval_usernames(channel_id, entries)
        names = await get_approval_usernames(channel_id)
        chat = await _get_chat(channel_id)
        added_line = ", ".join(f"@{n}" for n in added[:15]) or "—"
        skipped_line = ", ".join(f'"{s}"' for s in skipped[:15]) or "—"
        await message.reply_text(
            "✅ <b>USERNAMES ADDED</b>\n\n"
            f"➕ <b>Added ({len(added)}):</b> {added_line}\n"
            f"⏭ <b>Skipped — duplicate or invalid ({len(skipped)}):</b> {skipped_line}\n\n"
            + _status_line(True, names),
            reply_markup=_menu(channel_id),
        )
        if added:
            with suppress(Exception):
                await log_approval_usernames_changed(
                    chat, message.from_user, "ADDED", added, total=len(names),
                )
        session["step"] = "action"
        return

    # ─── Remove username input ────────────────────────────────────
    if step == "remove_name":
        if channel_id is None:
            SESSIONS.pop(message.from_user.id, None)
            return await message.reply_text("❌ Setup expired. Use /approvalmode again.")
        removed = await remove_approval_username(channel_id, raw)
        names = await get_approval_usernames(channel_id)
        chat = await _get_chat(channel_id)
        if removed:
            await message.reply_text(
                f"✅ <b>REMOVED:</b> @{raw.strip().lstrip('@')[:64]}\n\n"
                + _status_line(True, names),
                reply_markup=_menu(channel_id),
            )
            with suppress(Exception):
                await log_approval_usernames_changed(
                    chat, message.from_user, "REMOVED",
                    [raw.strip().lstrip("@").lower()], total=len(names),
                )
        else:
            await message.reply_text(
                "❌ That username is not on the approved list.\n\n"
                + _status_line(True, names),
                reply_markup=_menu(channel_id),
            )
        session["step"] = "action"
        return


# ---------------------------------------------------------------------------
# /approvalmode — CHANNEL ONLY (on | off)
# ---------------------------------------------------------------------------

INVALID_TEXT_APPROVALMODE = (
    "Invalid option selected!\n"
    "Use on or off!\n\n"
    "Use like this:\n"
    "/approvalmode on (To turn on.)\n"
    "/approvalmode off (To turn off.)\n\n"
    "Current status for this channel: {status}."
)


@app.on_message(
    filters.channel & filters.regex(r"(?i)^/approvalmode(?:@\w+)?(?:\s+(.+?))?\s*$"),
    group=-15,
)
async def approvalmode_channel_handler(_, message: Message):
    if message.chat.type != ChatType.CHANNEL:
        return
    if not await _bot_can_delete(message.chat.id):
        with suppress(Exception):
            await message.reply_text(
                "I need <b>Delete Messages</b> admin permission to work here."
            )
        return

    text = message.text or message.caption or ""
    parts = text.split(maxsplit=1)
    value = parts[1].strip().lower() if len(parts) == 2 else ""

    if value not in {"on", "off"}:
        current = await settings_db.find_one(
            {"chat_id": message.chat.id}, {"approval_mode_enabled": 1},
        )
        status = "ON" if current and current.get("approval_mode_enabled") else "OFF"
        await _delete_command_message(message, reason="COMMAND_CLEANUP_APPROVALMODE_INVALID")
        response = None
        with suppress(Exception):
            response = await message.reply_text(
                INVALID_TEXT_APPROVALMODE.format(status=status)
            )
        if response:
            _delete_after(response, _RESPONSE_TTL)
        return

    enabled = value == "on"
    prior = await settings_db.find_one(
        {"chat_id": message.chat.id}, {"approval_mode_enabled": 1},
    )
    was_enabled = bool(prior and prior.get("approval_mode_enabled"))
    await set_approval_mode_enabled(message.chat.id, enabled)

    await _delete_command_message(
        message,
        reason="COMMAND_CLEANUP_APPROVALMODE_OFF" if not enabled
        else "COMMAND_CLEANUP_APPROVALMODE_ON",
    )

    response = None
    with suppress(Exception):
        response = await message.reply_text(
            "Successfully enabled Approval Mode!" if enabled
            else "Successfully disabled Approval Mode!"
        )
    if response:
        _delete_after(response, _RESPONSE_TTL)

    # Centralized logger — only if the state actually changed (source=CHANNEL)
    if enabled != was_enabled:
        names = await get_approval_usernames(message.chat.id)
        with suppress(Exception):
            await log_approval_mode_change(
                message.chat, message.from_user,
                "ENABLED" if enabled else "DISABLED",
                source="CHANNEL", usernames=len(names),
            )


# ---------------------------------------------------------------------------
# Moderation — new and edited channel posts (handler group -23)
# ---------------------------------------------------------------------------

async def _delete_post(message: Message, *, reason: str) -> bool:
    """Delete one post; queue a persistent retry when Telegram rate-limits.

    Mirrors the other filters. Returns True when the delete call was sent
    without a FloodWait. The centralized logger receives the confirmed
    deletion / pending retry / failure via ``mark_deletion_result``.
    """
    try:
        await archive_deletion(
            message.chat.id, message.id, reason=reason,
            source="APPROVALMODE", message=message,
        )
        await message.delete()
        await mark_deletion_result(
            message.chat.id, message.id, status="deleted",
            detail="APPROVALMODE deletion confirmed",
        )
    except FloodWait as exc:
        wait = min(max(int(getattr(exc, "value", 0) or 0), 1), 3600)
        with suppress(Exception):
            await queue_deletion_job(
                message.chat.id, message.id,
                delete_at=int(time.time()) + wait,
                original_post_time=_post_time(message),
                reason="APPROVAL_RETRY",
            )
        with suppress(Exception):
            await mark_deletion_result(
                message.chat.id, message.id, status="pending_retry",
                detail=f"FloodWait {wait}s; retry queued as APPROVAL_RETRY",
            )
        log.warning(
            "approval-mode post delete FloodWait chat=%s message=%s wait=%ss; retry queued",
            message.chat.id, message.id, wait,
        )
        return False
    except MessageIdInvalid:
        with suppress(Exception):
            await mark_deletion_result(
                message.chat.id, message.id,
                status="not_deleted_or_already_absent",
                detail="Telegram: MESSAGE_ID_INVALID — post already absent",
            )
        return False
    except RPCError as exc:
        # Telegram refused the deletion (permissions, limits, ...) — report
        # the failure through the centralized logger. Never raises.
        with suppress(Exception):
            await mark_deletion_result(
                message.chat.id, message.id, status="failed",
                detail=f"Telegram refused the deletion — {type(exc).__name__}: "
                       f"{str(exc)[:200]}",
            )
        log.warning(
            "approval-mode post delete failed chat=%s message=%s: %s: %s",
            message.chat.id, message.id, type(exc).__name__, exc,
        )
        return False
    except Exception as exc:
        # Network/unknown error during the delete call — the outcome is not
        # confirmed, so the post is reported as NOT deleted (never as
        # deleted) and the failure reaches the logger group.
        with suppress(Exception):
            await mark_deletion_result(
                message.chat.id, message.id, status="failed",
                detail=f"Deletion attempt failed — {type(exc).__name__}: "
                       f"{str(exc)[:200]} (outcome unknown)",
            )
        log.warning(
            "approval-mode post delete error chat=%s message=%s: %s: %s",
            message.chat.id, message.id, type(exc).__name__, exc,
        )
        return False
    log.info(
        "deletion reason=%s chat=%s msg=%s",
        reason, message.chat.id, message.id,
    )
    return True


async def _delete_album_siblings(chat_id: int, media_group_id: int,
                                 keep: int) -> None:
    """Best-effort deletion of the album items delivered before the verdict.

    Telegram delivered the items earlier and separately, so the whole
    album goes down in one channels.deleteMessages call. Failures (an
    item already gone, an RPC error) are logged and never raised: every
    later item is still deleted on arrival by the filter itself.
    """
    ids = [mid for mid in approval_policy.take_album_items(chat_id, media_group_id)
           if mid != keep]
    if not ids:
        return
    try:
        for mid in ids:
            await archive_deletion(
                chat_id, mid, reason="APPROVAL_ALBUM_ITEM",
                source="APPROVALMODE",
            )
        await app.delete_messages(chat_id, ids)
        for mid in ids:
            await mark_deletion_result(
                chat_id, mid, status="deleted",
                detail="APPROVALMODE album item deletion confirmed",
            )
        log.info(
            "deletion reason=APPROVAL_ALBUM_ITEM chat=%s msg=%s album=%s",
            chat_id, ids, media_group_id,
        )
    except FloodWait as exc:
        wait = min(max(int(getattr(exc, "value", 0) or 0), 1), 3600)
        for mid in ids:
            with suppress(Exception):
                await queue_deletion_job(
                    chat_id, mid,
                    delete_at=int(time.time()) + wait,
                    original_post_time=int(time.time()),
                    reason="APPROVAL_RETRY",
                )
        for mid in ids:
            with suppress(Exception):
                await mark_deletion_result(
                    chat_id, mid, status="pending_retry",
                    detail=f"FloodWait {wait}s; retry queued as APPROVAL_RETRY",
                )
        log.warning(
            "approval-mode album delete FloodWait chat=%s album=%s wait=%ss; retries queued",
            chat_id, media_group_id, wait,
        )
    except Exception as exc:
        # The batch delete failed — report every sibling as NOT deleted so
        # the failure reaches the centralized logger group, not just the
        # local log. Never raised.
        for mid in ids:
            with suppress(Exception):
                await mark_deletion_result(
                    chat_id, mid, status="failed",
                    detail=f"Album sibling deletion failed — "
                           f"{type(exc).__name__}: {str(exc)[:200]}",
                )
        log.warning(
            "approval-mode album sibling delete failed chat=%s album=%s ids=%s: %s: %s",
            chat_id, media_group_id, ids, type(exc).__name__, exc,
        )


def _names_of(handles) -> list:
    return sorted(str(h) for h in handles if h != "__nomention__")


async def _moderate_channel_post(message: Message) -> None:
    """The shared verdict path for new posts and edits."""
    chat = message.chat
    chat_id = chat.id
    group_id = approval_policy.media_group_id(message)

    # Earlier filters own this update already — leave the post to them.
    if link_policy.was_rejected(chat_id, message.id):
        return
    if anonymous_policy.was_rejected(chat_id, message.id):
        return
    if forward_policy.was_rejected(chat_id, message.id):
        return
    if bannedwords_policy.was_rejected(chat_id, message.id):
        return

    # Follow-up item of an album this filter already condemned: delete on
    # arrival, no further I/O.
    if group_id is not None and approval_policy.album_is_blocked(chat_id, group_id):
        approval_policy.mark_rejected(chat_id, message.id)
        await _delete_post(message, reason="APPROVAL_ALBUM_ITEM")
        return
    # Follow-up item of an album this filter already APPROVED: keep it.
    if group_id is not None and approval_policy.album_is_approved(chat_id, group_id):
        approval_policy.remember_album_item(chat_id, group_id, message.id)
        return

    setting = await settings_db.find_one(
        {"chat_id": chat_id},
        {"approval_mode_enabled": 1, "approval_usernames": 1,
         "approval_blocked_albums": 1},
    )
    if not setting or not setting.get("approval_mode_enabled"):
        return  # feature off — nothing is ever deleted

    approved = [str(n) for n in setting.get("approval_usernames") or []]

    # Restart recovery: this album was condemned before the restart.
    if group_id is not None and approval_policy.persisted_blocked_album(setting, group_id):
        approval_policy.block_album(chat_id, group_id)
        approval_policy.mark_rejected(chat_id, message.id)
        await _delete_post(message, reason="APPROVAL_ALBUM_ITEM")
        return

    approved_found, unapproved_found = approval_policy.classify_post(message, approved)

    if group_id is not None and not approval_policy.album_has_verdict(chat_id, group_id):
        # First item of the album decides for the whole album (check-and-set).
        if approved_found:
            if approval_policy.approve_album(chat_id, group_id):
                approval_policy.remember_album_item(chat_id, group_id, message.id)
                if unapproved_found:
                    # Mixed usernames in the deciding caption — warn, keep.
                    with suppress(Exception):
                        await log_approval_mixed_warning(
                            chat, message,
                            _names_of(approved_found),
                            _names_of(unapproved_found),
                        )
                return
            # Another item decided meanwhile: follow the stored verdict.
            if approval_policy.album_is_blocked(chat_id, group_id):
                approval_policy.mark_rejected(chat_id, message.id)
                await _delete_post(message, reason="APPROVAL_ALBUM_ITEM")
                return
            approval_policy.remember_album_item(chat_id, group_id, message.id)
            return
        # No approved username in this item: condemn the album whole.
        approval_policy.block_album(chat_id, group_id)
        try:
            await remember_approval_blocked_album(chat_id, group_id)
        except Exception as exc:
            log.warning(
                "approval album verdict not persisted chat=%s album=%s: %s: %s",
                chat_id, group_id, type(exc).__name__, exc,
            )
        approval_policy.mark_rejected(chat_id, message.id)
        if not await _bot_can_delete(chat_id):
            log.warning(
                "approval-mode post rejected but not deleted (no Delete Messages "
                "right) chat=%s message=%s", chat_id, message.id,
            )
            with suppress(Exception):
                await log_operation_failed(
                    "APPROVAL MODE — POST NOT DELETED (NO DELETE PERMISSION)",
                    chat_id=chat_id, message_id=message.id,
                    detail="The post was rejected by Approval Mode but the bot "
                           "cannot delete messages in this channel.",
                    event_id=("approval_noperm", chat_id, message.id),
                )
            return
        await _delete_post(message, reason="APPROVAL_NO_ALLOWED_USERNAME")
        await _delete_album_siblings(chat_id, group_id, keep=message.id)
        log.info(
            "approval-mode album deleted chat=%s message=%s album=%s approved_count=%s",
            chat_id, message.id, group_id, len(approved),
        )
        return

    # ─── Single post (or an album item after the verdict was taken) ──
    if group_id is not None:
        # The album verdict exists and is neither blocked nor approved in
        # this process (e.g. approved before a restart): keep the item.
        approval_policy.remember_album_item(chat_id, group_id, message.id)
        return

    if approved_found:
        # At least one approved username → keep. Mixed → warn + keep.
        if unapproved_found:
            with suppress(Exception):
                await log_approval_mixed_warning(
                    chat, message,
                    _names_of(approved_found),
                    _names_of(unapproved_found),
                )
        return

    # No approved username → delete. Verdict FIRST (so the later filters
    # and the gap engine always see it), Telegram calls after.
    approval_policy.mark_rejected(chat_id, message.id)
    if not await _bot_can_delete(chat_id):
        log.warning(
            "approval-mode post rejected but not deleted (no Delete Messages "
            "right) chat=%s message=%s", chat_id, message.id,
        )
        with suppress(Exception):
            await log_operation_failed(
                "APPROVAL MODE — POST NOT DELETED (NO DELETE PERMISSION)",
                chat_id=chat_id, message_id=message.id,
                detail="The post was rejected by Approval Mode but the bot "
                       "cannot delete messages in this channel.",
                event_id=("approval_noperm", chat_id, message.id),
            )
        return
    await _delete_post(message, reason="APPROVAL_NO_ALLOWED_USERNAME")
    log.info(
        "approval-mode post deleted chat=%s message=%s approved_count=%s",
        chat_id, message.id, len(approved),
    )


@app.on_message(filters.channel & filters.incoming, group=-23)
async def approvalmode_new_channel_post(_, message: Message):
    """Delete channel posts without an approved username while enabled."""
    try:
        if message.chat.type != ChatType.CHANNEL:
            return
        if getattr(message, "outgoing", False):
            return
        if isinstance(getattr(message, "service", None), MessageServiceType):
            return
        if approval_policy.is_bot_command(message):
            return  # commands are never content
        await _moderate_channel_post(message)
    except RPCError as exc:
        log.warning("approval-mode filter Telegram error chat=%s message=%s: %s",
                    getattr(getattr(message, "chat", None), "id", None),
                    getattr(message, "id", None), exc)
    except Exception as exc:
        log.exception(
            "approval-mode filter failed chat=%s message=%s: %s",
            getattr(getattr(message, "chat", None), "id", None),
            getattr(message, "id", None), exc,
        )


@app.on_edited_message(filters.channel & filters.incoming, group=-23)
async def approvalmode_edited_channel_post(_, message: Message):
    """Edited channel posts run through the SAME verdict path.

    An edit can never smuggle the post past Approval Mode: a post edited
    to no longer carry an approved username is deleted.
    """
    try:
        if message.chat.type != ChatType.CHANNEL:
            return
        if getattr(message, "outgoing", False):
            return
        if isinstance(getattr(message, "service", None), MessageServiceType):
            return
        if approval_policy.is_bot_command(message):
            return
        await _moderate_channel_post(message)
    except RPCError as exc:
        log.warning("approval-mode filter Telegram error (edit) chat=%s message=%s: %s",
                    getattr(getattr(message, "chat", None), "id", None),
                    getattr(message, "id", None), exc)
    except Exception as exc:
        log.exception(
            "approval-mode filter failed (edit) chat=%s message=%s: %s",
            getattr(getattr(message, "chat", None), "id", None),
            getattr(message, "id", None), exc,
        )
