"""Channel BANNED WORDS filter.

/bannedwords on | /bannedwords off  -> CHANNEL ONLY: enable / disable the
banned-words filter for the channel the command is posted in (same shape
as /setlink and /setforward; the command message is deleted immediately
and the bot's reply auto-deletes after 120 seconds).

/bannedwords in the bot's PRIVATE chat -> the management wizard:

    /bannedwords
        → enter the Channel ID
        → status menu:
            🟢 ENABLE / 🔴 DISABLE
            ➕ ADD WORDS / ➖ REMOVE WORD / 📋 VIEW LIST
            ❌ CANCEL

The wizard verifies that the user is an admin of the target channel
(utils/permissions.verify_channel_admin — re-verified on every button
press) and that the bot can delete there. The owner bypasses the
channel-admin check.

MODERATION (handler group -24, see utils/bannedwords_policy.py):
    -40  link filter            (link_blocker_new_channel_post)
    -30  anonymous filter       (anonymous_post_filter)
    -25  forward filter         (forward_post_remover)
    -24  bannedwords_new_channel_post   — new channel posts
    -24  bannedwords_edited_channel_post — edited posts (edits cannot
          smuggle a banned word past the filter)
    -16  /bannedwords channel command
      0  /setgap + /setdelay engine

A post the earlier filters already rejected for this update is left to
them (no second delete call, no duplicate deletion log). A post this
filter rejects is marked in-process (bannedwords_policy.mark_rejected)
BEFORE any Telegram call, so it can never become a /setgap reference or
a /setdelay job. Command messages are never judged (a Telegram
BOT_COMMAND entity marks them): ``/bannedwords add spam`` must reach the
wizard even while ``spam`` is banned.

DELETION + LOGGER: deletions go through the shared flow
(``archive_deletion`` → delete → ``mark_deletion_result``); the
centralized logger (SWAGGYMUSIC.utils.logger) turns every confirmed
deletion into one "🗑️ POST DELETED" event with the matched word in the
reason line — "Do not claim a post was deleted unless Telegram confirms
the deletion": FloodWait retries and failures are reported separately.

ALBUMS: a media album with a banned word in any item's caption is
deleted whole — the current item, every item already delivered
(in-process tracking) and every later item of the same media_group_id;
the verdict is persisted in ``banned_blocked_albums`` so a restart
mid-delivery cannot leave a partial album.

SETTINGS: one document per channel in mongodb.autodelete_settings (the
shared settings document — /setgap, /setdelay, /setanonymous, /setlink,
/setforward and Approval Mode live there too):
    { "chat_id": <channel_id>,
      "banned_words_enabled": True,
      "banned_words": ["cross", "buy now", ...],      # newest 100
      "banned_blocked_albums": [<media_group_id>, ...] }  # newest 50
Settings persist across restarts; the word list also survives a DISABLE.
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
from SWAGGYMUSIC.utils import anonymous_policy, bannedwords_policy, forward_policy, link_policy
from SWAGGYMUSIC.utils.logger import (log_bannedwords_change,
                                      log_bannedwords_list_changed,
                                      log_operation_failed)
from SWAGGYMUSIC.utils.database import (add_banned_words, clear_banned_words,
                                        get_banned_words,
                                        queue_deletion_job,
                                        remember_banned_blocked_album,
                                        remove_banned_word,
                                        set_banned_words_enabled,
                                        settings_db)
from SWAGGYMUSIC.utils.permissions import verify_channel_admin
from SWAGGYMUSIC.utils.channel_resolution import (
    CHANNEL_INPUT_PROMPT,
    ChannelResolutionError,
    resolve_channel_input,
)

log = LOGGER(__name__)
PREFIX = "bannedwords"
SESSIONS = {}
LOCK = asyncio.Lock()

# The bot's channel-side reply auto-deletes after this many seconds —
# the same TTL the /setdelay, /setgap, /setlink and /setforward replies use.
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


_ACTIVE_SESSION = filters.create(_in_active_session, name="BannedWordsSessionFilter")


# ---------------------------------------------------------------------------
# Helpers (same contracts as setlink.py)
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
        log.warning("bannedwords bot-permission check failed for %s: %s: %s",
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
        log.warning("bannedwords could not read channel %s: %s: %s",
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
            InlineKeyboardButton("➕ ADD WORDS", callback_data=f"{PREFIX}:add:{channel_id}",
                                 style=ButtonStyle.PRIMARY),
            InlineKeyboardButton("➖ REMOVE WORD", callback_data=f"{PREFIX}:remove:{channel_id}",
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
    "🤬 <b>Banned:</b> posts, captions and edited posts containing a "
    "configured banned word or phrase (case-insensitive, whole-word "
    "matching — banned <code>cross</code> never matches <code>crossed</code>).\n\n"
    "✅ <b>Kept:</b> posts without any banned word.\n\n"
    "<i>A media album containing a banned word is deleted whole; service "
    "messages are never touched.</i>"
)


def _status_line(enabled: bool, words: list) -> str:
    state = "ON 🟢" if enabled else "OFF 🔴"
    shown = ", ".join(f'"{w}"' for w in words[:15])
    if len(words) > 15:
        shown += f" (+{len(words) - 15} more)"
    return f"🤬 <b>Current status:</b> {state}\n🗂 <b>Banned words ({len(words)}):</b> {shown or '—'}"


@app.on_message(filters.command(["bannedwords"]) & filters.private)
async def bannedwords_private_start(_, message: Message):
    if not message.from_user:
        return
    async with LOCK:
        SESSIONS[message.from_user.id] = {"step": "channel_id"}
    await message.reply_text(
        "🤬 <b>BANNED WORDS</b>\n\n"
        + CHANNEL_INPUT_PROMPT,
        reply_markup=_cancel_menu(),
    )


@app.on_callback_query(filters.regex(rf"^{PREFIX}:"))
async def bannedwords_callback(_, query: CallbackQuery):
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
            "Setup expired. Use /bannedwords again.", show_alert=True,
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
            status = await set_banned_words_enabled(channel_id, enabled)
            words = await get_banned_words(channel_id)
            await query.answer("Saved")
            with suppress(Exception):
                if enabled:
                    await query.message.edit_text(
                        f"🤬 <b>BANNED WORDS ENABLED 🟢</b>\n\n"
                        f"📌 <b>Channel:</b> {title}\n"
                        f"🆔 <b>Channel ID:</b> <code>{channel_id}</code>\n\n"
                        + _FEATURE_TEXT + "\n\n"
                        + _status_line(True, words) + "\n\n"
                        + "<i>Prohibited posts are deleted automatically.</i>"
                    )
                else:
                    await query.message.edit_text(
                        f"🤬 <b>BANNED WORDS DISABLED 🔴</b>\n\n"
                        f"📌 <b>Channel:</b> {title}\n"
                        f"🆔 <b>Channel ID:</b> <code>{channel_id}</code>\n\n"
                        "The filter is now turned off — posts containing "
                        "banned words are no longer deleted.\n\n"
                        + _status_line(False, words)
                    )
            if status != "NO_CHANGE":
                with suppress(Exception):
                    await log_bannedwords_change(
                        chat, query.from_user,
                        "ENABLED" if enabled else "DISABLED",
                        source="PRIVATE CHAT", total=len(words),
                    )
            SESSIONS.pop(user_id, None)
        except Exception as exc:
            log.warning("bannedwords toggle failed for %s: %s", channel_id, exc)
            await query.answer("Could not update settings.", show_alert=True)
        return

    # ─── ADD WORDS ────────────────────────────────────────────────
    if action == "add":
        session["step"] = "add_words"
        await query.answer("Add words")
        with suppress(Exception):
            await query.message.edit_text(
                "➕ <b>ADD BANNED WORDS</b>\n\n"
                "Send one or more words or phrases.\n"
                "Separate entries with <b>commas</b> or <b>new lines</b>.\n\n"
                "<i>Example:</i> <code>cross, free money, buy now</code>",
                reply_markup=_cancel_menu(),
            )
        return

    # ─── REMOVE WORD ──────────────────────────────────────────────
    if action == "remove":
        session["step"] = "remove_word"
        await query.answer("Remove a word")
        with suppress(Exception):
            words = await get_banned_words(channel_id)
            await query.message.edit_text(
                "➖ <b>REMOVE BANNED WORD</b>\n\n"
                "Send the exact word or phrase to remove.\n\n"
                + _status_line(True, words),
                reply_markup=_cancel_menu(),
            )
        return

    # ─── VIEW LIST ────────────────────────────────────────────────
    if action == "view":
        await query.answer()
        words = await get_banned_words(channel_id)
        current = await settings_db.find_one(
            {"chat_id": channel_id}, {"banned_words_enabled": 1},
        )
        enabled = bool(current and current.get("banned_words_enabled"))
        with suppress(Exception):
            await query.message.edit_text(
                "📋 <b>BANNED WORDS LIST</b>\n\n"
                f"📌 <b>Channel:</b> {title}\n"
                f"🆔 <b>Channel ID:</b> <code>{channel_id}</code>\n\n"
                + _status_line(enabled, words),
                reply_markup=_menu(channel_id),
            )
        return

    # ─── CLEAR LIST ───────────────────────────────────────────────
    if action == "clear":
        await query.answer("Cleared")
        with suppress(Exception):
            removed = await clear_banned_words(channel_id)
            words = await get_banned_words(channel_id)
            current = await settings_db.find_one(
                {"chat_id": channel_id}, {"banned_words_enabled": 1},
            )
            enabled = bool(current and current.get("banned_words_enabled"))
            await query.message.edit_text(
                "🗑 <b>BANNED WORDS LIST CLEARED</b>\n\n"
                f"📌 <b>Channel:</b> {title}\n"
                f"🆔 <b>Channel ID:</b> <code>{channel_id}</code>\n\n"
                + _status_line(enabled, words),
                reply_markup=_menu(channel_id),
            )
            if removed:
                with suppress(Exception):
                    await log_bannedwords_list_changed(
                        chat, query.from_user, "CLEARED", [], total=0,
                    )
        return


@app.on_message(
    _ACTIVE_SESSION
    & ~filters.service
    & ~filters.command(["bannedwords", "cancel"]),
)
async def bannedwords_channel_input(_, message: Message):
    if not message.from_user:
        return
    raw = (message.text or message.caption or "").strip()
    if raw.startswith("/"):
        # Another command (e.g. /setlink): the user left this wizard. End
        # the session and let the command reach its own handler — a word
        # list never starts with "/".
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
            {"banned_words_enabled": 1, "banned_words": 1},
        )
        enabled = bool(current and current.get("banned_words_enabled"))
        words = [str(w) for w in (current or {}).get("banned_words") or []]
        chat = await _get_chat(channel_id)
        title = getattr(chat, "title", None) or "Channel"
        await message.reply_text(
            f"📌 <b>Channel:</b> {title}\n"
            f"🆔 <b>Channel ID:</b> <code>{channel_id}</code>\n"
            + _status_line(enabled, words)
            + ("\n\n⚠️ I need the <b>Delete Messages</b> admin right in that "
               "channel before this filter can work."
               if not await _bot_can_delete(channel_id) else "")
            + "\n\nChoose an action:",
            reply_markup=_menu(channel_id),
        )
        return

    # ─── Add words input ──────────────────────────────────────────
    if step == "add_words":
        if channel_id is None:
            SESSIONS.pop(message.from_user.id, None)
            return await message.reply_text("❌ Setup expired. Use /bannedwords again.")
        entries = [part.strip() for part in re.split(r"[,\n]", raw) if part.strip()]
        if not entries:
            return await message.reply_text(
                "❌ Send one or more words or phrases, separated by commas "
                "or new lines.",
                reply_markup=_cancel_menu(),
            )
        added, skipped = await add_banned_words(channel_id, entries)
        words = await get_banned_words(channel_id)
        chat = await _get_chat(channel_id)
        added_line = ", ".join(f'"{w}"' for w in added[:15]) or "—"
        skipped_line = ", ".join(f'"{s}"' for s in skipped[:15]) or "—"
        await message.reply_text(
            "✅ <b>WORDS ADDED</b>\n\n"
            f"➕ <b>Added ({len(added)}):</b> {added_line}\n"
            f"⏭ <b>Skipped — duplicate or invalid ({len(skipped)}):</b> {skipped_line}\n\n"
            + _status_line(True, words),
            reply_markup=_menu(channel_id),
        )
        if added:
            with suppress(Exception):
                await log_bannedwords_list_changed(
                    chat, message.from_user, "ADDED", added, total=len(words),
                )
        session["step"] = "action"
        return

    # ─── Remove word input ────────────────────────────────────────
    if step == "remove_word":
        if channel_id is None:
            SESSIONS.pop(message.from_user.id, None)
            return await message.reply_text("❌ Setup expired. Use /bannedwords again.")
        removed = await remove_banned_word(channel_id, raw)
        words = await get_banned_words(channel_id)
        chat = await _get_chat(channel_id)
        if removed:
            await message.reply_text(
                f"✅ <b>REMOVED:</b> \"{raw.strip()[:64]}\"\n\n"
                + _status_line(True, words),
                reply_markup=_menu(channel_id),
            )
            with suppress(Exception):
                await log_bannedwords_list_changed(
                    chat, message.from_user, "REMOVED",
                    [" ".join(raw.split()).lower()], total=len(words),
                )
        else:
            await message.reply_text(
                "❌ That word is not on the banned list.\n\n"
                + _status_line(True, words),
                reply_markup=_menu(channel_id),
            )
        session["step"] = "action"
        return


# ---------------------------------------------------------------------------
# /bannedwords — CHANNEL ONLY (on | off)
# ---------------------------------------------------------------------------

INVALID_TEXT_BANNEDWORDS = (
    "Invalid option selected!\n"
    "Use on or off!\n\n"
    "Use like this:\n"
    "/bannedwords on (To turn on.)\n"
    "/bannedwords off (To turn off.)\n\n"
    "Current status for this channel: {status}."
)


@app.on_message(
    filters.channel & filters.regex(r"(?i)^/bannedwords(?:@\w+)?(?:\s+(.+?))?\s*$"),
    group=-16,
)
async def bannedwords_channel_handler(_, message: Message):
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
            {"chat_id": message.chat.id}, {"banned_words_enabled": 1},
        )
        status = "ON" if current and current.get("banned_words_enabled") else "OFF"
        await _delete_command_message(message, reason="COMMAND_CLEANUP_BANNEDWORDS_INVALID")
        response = None
        with suppress(Exception):
            response = await message.reply_text(
                INVALID_TEXT_BANNEDWORDS.format(status=status)
            )
        if response:
            _delete_after(response, _RESPONSE_TTL)
        return

    enabled = value == "on"
    prior = await settings_db.find_one(
        {"chat_id": message.chat.id}, {"banned_words_enabled": 1},
    )
    was_enabled = bool(prior and prior.get("banned_words_enabled"))
    await set_banned_words_enabled(message.chat.id, enabled)

    await _delete_command_message(
        message,
        reason="COMMAND_CLEANUP_BANNEDWORDS_OFF" if not enabled
        else "COMMAND_CLEANUP_BANNEDWORDS_ON",
    )

    response = None
    with suppress(Exception):
        response = await message.reply_text(
            "Successfully enabled banned words filter!" if enabled
            else "Successfully disabled banned words filter!"
        )
    if response:
        _delete_after(response, _RESPONSE_TTL)

    # Centralized logger — only if the state actually changed (source=CHANNEL)
    if enabled != was_enabled:
        words = await get_banned_words(message.chat.id)
        with suppress(Exception):
            await log_bannedwords_change(
                message.chat, message.from_user,
                "ENABLED" if enabled else "DISABLED",
                source="CHANNEL", total=len(words),
            )


# ---------------------------------------------------------------------------
# Moderation — new and edited channel posts (handler group -24)
# ---------------------------------------------------------------------------

async def _delete_post(message: Message, *, reason: str, details=None) -> bool:
    """Delete one post; queue a persistent retry when Telegram rate-limits.

    Mirrors the link and anonymous filters. Returns True when the delete
    call was sent without a FloodWait.  The centralized logger receives
    the confirmed deletion / pending retry / failure via
    ``mark_deletion_result``.
    """
    try:
        await archive_deletion(
            message.chat.id, message.id, reason=reason,
            source="BANNEDWORDS", message=message, details=details,
        )
        await message.delete()
        await mark_deletion_result(
            message.chat.id, message.id, status="deleted",
            detail="BANNEDWORDS deletion confirmed",
        )
    except FloodWait as exc:
        wait = min(max(int(getattr(exc, "value", 0) or 0), 1), 3600)
        with suppress(Exception):
            await queue_deletion_job(
                message.chat.id, message.id,
                delete_at=int(time.time()) + wait,
                original_post_time=_post_time(message),
                reason="BANNED_WORD_RETRY",
            )
        with suppress(Exception):
            await mark_deletion_result(
                message.chat.id, message.id, status="pending_retry",
                detail=f"FloodWait {wait}s; retry queued as BANNED_WORD_RETRY",
            )
        log.warning(
            "banned-words post delete FloodWait chat=%s message=%s wait=%ss; retry queued",
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
            "banned-words post delete failed chat=%s message=%s: %s: %s",
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
            "banned-words post delete error chat=%s message=%s: %s: %s",
            message.chat.id, message.id, type(exc).__name__, exc,
        )
        return False
    log.info(
        "deletion reason=%s chat=%s msg=%s",
        reason, message.chat.id, message.id,
    )
    return True


async def _delete_album_siblings(chat_id: int, media_group_id: int,
                                 keep: int, *, matched: str) -> None:
    """Best-effort deletion of the album items delivered before the verdict.

    Telegram delivered the clean items earlier and separately, so the
    whole album goes down in one channels.deleteMessages call. Failures
    (an item already gone, an RPC error) are logged and never raised:
    every later item is still deleted on arrival by the filter itself.
    """
    ids = [mid for mid in bannedwords_policy.take_album_items(chat_id, media_group_id)
           if mid != keep]
    if not ids:
        return
    try:
        for mid in ids:
            await archive_deletion(
                chat_id, mid, reason="BANNED_WORD_ALBUM",
                source="BANNEDWORDS", details={"matched": matched},
            )
        await app.delete_messages(chat_id, ids)
        for mid in ids:
            await mark_deletion_result(
                chat_id, mid, status="deleted",
                detail="BANNEDWORDS album item deletion confirmed",
            )
        log.info(
            "deletion reason=BANNED_WORD_ALBUM chat=%s msg=%s album=%s",
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
                    reason="BANNED_WORD_RETRY",
                )
        for mid in ids:
            with suppress(Exception):
                await mark_deletion_result(
                    chat_id, mid, status="pending_retry",
                    detail=f"FloodWait {wait}s; retry queued as BANNED_WORD_RETRY",
                )
        log.warning(
            "banned-words album delete FloodWait chat=%s album=%s wait=%ss; retries queued",
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
            "banned-words album sibling delete failed chat=%s album=%s ids=%s: %s: %s",
            chat_id, media_group_id, ids, type(exc).__name__, exc,
        )


async def _moderate_channel_post(message: Message) -> None:
    """The shared verdict path for new posts and edits."""
    chat = message.chat
    chat_id = chat.id
    group_id = bannedwords_policy.media_group_id(message)

    # Earlier filters own this update already — leave the post to them.
    if link_policy.was_rejected(chat_id, message.id):
        return
    if anonymous_policy.was_rejected(chat_id, message.id):
        return
    if forward_policy.was_rejected(chat_id, message.id):
        return

    # Follow-up item of an album this filter already condemned: delete on
    # arrival, no further I/O (an album is never split — the same rule the
    # link filter and the /setgap engine apply).
    if group_id is not None and bannedwords_policy.album_is_blocked(chat_id, group_id):
        bannedwords_policy.mark_rejected(chat_id, message.id)
        await _delete_post(message, reason="BANNED_WORD_ALBUM")
        return

    setting = await settings_db.find_one(
        {"chat_id": chat_id},
        {"banned_words_enabled": 1, "banned_words": 1, "banned_blocked_albums": 1},
    )
    if not setting or not setting.get("banned_words_enabled"):
        return  # feature off — nothing is ever deleted

    words = [str(w) for w in setting.get("banned_words") or []]
    hits = bannedwords_policy.find_banned_words(message, words)

    # Restart recovery: this album was condemned before the restart, so
    # the delivered item must not survive the gap in in-process state.
    if group_id is not None and bannedwords_policy.persisted_blocked_album(setting, group_id):
        bannedwords_policy.block_album(chat_id, group_id)
        bannedwords_policy.mark_rejected(chat_id, message.id)
        await _delete_post(message, reason="BANNED_WORD_ALBUM",
                           details={"matched": hits[0] if hits else ""})
        return

    if not hits:
        if group_id is not None:
            # Clean album item: remember it so a later item's banned word
            # takes the whole album down.
            bannedwords_policy.remember_album_item(chat_id, group_id, message.id)
        return

    # Banned word while the feature is enabled. Verdict FIRST (so
    # concurrent album items and the later filters always see it),
    # Telegram calls after — the same order the other filters use.
    matched = hits[0]
    if group_id is not None:
        bannedwords_policy.block_album(chat_id, group_id)
        try:
            await remember_banned_blocked_album(chat_id, group_id)
        except Exception as exc:
            log.warning(
                "banned-words album verdict not persisted chat=%s album=%s: %s: %s",
                chat_id, group_id, type(exc).__name__, exc,
            )
    bannedwords_policy.mark_rejected(chat_id, message.id)

    if not await _bot_can_delete(chat_id):
        log.warning(
            "banned-words post rejected but not deleted (no Delete Messages right) "
            "chat=%s message=%s", chat_id, message.id,
        )
        with suppress(Exception):
            await log_operation_failed(
                "BANNED WORDS — POST NOT DELETED (NO DELETE PERMISSION)",
                chat_id=chat_id, message_id=message.id,
                detail=f'The post containing the banned word "{matched}" was '
                       "rejected but the bot cannot delete messages in this "
                       "channel.",
                event_id=("bannedwords_noperm", chat_id, message.id),
            )
        return

    await _delete_post(message, reason="BANNED_WORD", details={"matched": matched})
    if group_id is not None:
        await _delete_album_siblings(
            chat_id, group_id, keep=message.id, matched=matched,
        )
    log.info(
        "banned-words post deleted chat=%s message=%s matched=%r album=%s edited=%s",
        chat_id, message.id, matched, group_id or "-",
        getattr(message, "edit_date", None) is not None,
    )


@app.on_message(filters.channel & filters.incoming, group=-24)
async def bannedwords_new_channel_post(_, message: Message):
    """Delete channel posts carrying banned words while the feature is on."""
    try:
        if message.chat.type != ChatType.CHANNEL:
            return
        if getattr(message, "outgoing", False):
            return
        if isinstance(getattr(message, "service", None), MessageServiceType):
            return
        if bannedwords_policy.is_bot_command(message):
            return  # commands are never content (e.g. "/bannedwords add spam")
        await _moderate_channel_post(message)
    except RPCError as exc:
        log.warning("banned-words filter Telegram error chat=%s message=%s: %s",
                    getattr(getattr(message, "chat", None), "id", None),
                    getattr(message, "id", None), exc)
    except Exception as exc:
        log.exception(
            "banned-words filter failed chat=%s message=%s: %s",
            getattr(getattr(message, "chat", None), "id", None),
            getattr(message, "id", None), exc,
        )


@app.on_edited_message(filters.channel & filters.incoming, group=-24)
async def bannedwords_edited_channel_post(_, message: Message):
    """Edited channel posts run through the SAME verdict path.

    An edit can never smuggle a banned word past the filter: a clean
    post edited to contain a banned word is deleted.
    """
    try:
        if message.chat.type != ChatType.CHANNEL:
            return
        if getattr(message, "outgoing", False):
            return
        if isinstance(getattr(message, "service", None), MessageServiceType):
            return
        if bannedwords_policy.is_bot_command(message):
            return
        await _moderate_channel_post(message)
    except RPCError as exc:
        log.warning("banned-words filter Telegram error (edit) chat=%s message=%s: %s",
                    getattr(getattr(message, "chat", None), "id", None),
                    getattr(message, "id", None), exc)
    except Exception as exc:
        log.exception(
            "banned-words filter failed (edit) chat=%s message=%s: %s",
            getattr(getattr(message, "chat", None), "id", None),
            getattr(message, "id", None), exc,
        )
