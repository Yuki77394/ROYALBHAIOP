#
# Copyright (C) 2021-2022 by Yuki77394@Github, < https://github.com/yuki77394/SWAGGYHELPER >.
# This file is part of < https://github.com/Yuki77394/SWAGGYHELPER > project,
# and is released under the "MIT License".
# Please see < https://github.com/Yuki77394/SWAGGYHELPER/blob/main/LICENSE >
# All rights reserved.
#
"""SWAGGYHELPER — THE centralized logger system.
=====================================================================

This module is the ONLY place in the whole bot that is allowed to send
messages to the configured logger group (``config.LOGGER_ID``).  Every
feature event — bot lifecycle, channel registration, settings changes,
moderation deletions, Create Post activity, failed operations — is
reported through the helpers in this file, in one consistent format.

What replaced what
---------------------------------------------------------------------
The previous generations of logging were removed and folded into here:

* ``utils/activity_logger.py`` (the old "activity logger" with the
  ``/logger`` master toggle ``is_on_off(2)``) — REMOVED.  Its event
  helpers were rebuilt here with the SAME names and signatures, so call
  sites and tests keep working.
* ``plugins/sudo/logger.py`` (the ``/logger enable|disable`` command) —
  REMOVED.  Logging is always on when LOGGER_ID is configured; there is
  deliberately no switch that could silently disable the audit trail.
* The detail message that ``utils/deletion_audit.py`` used to send to
  LOGGER_ID (and its edit-on-result flow) — REPLACED by the unified
  "POST DELETED" log in this module (see ``send_post_deletion_log``).
  The deletion ARCHIVE (forwarding originals to AUDIT_CHANNEL_ID) is an
  independent feature and stays untouched in deletion_audit.py.
* The history-scan recovery helpers (``scan_logger_group_tags``,
  ``find_delivered_event_tag``, ``find_delivered_deletion_log``,
  ``find_delivered_log_message`` and the ``SCAN_*`` result codes) —
  REMOVED.  Recovery no longer inspects Telegram history: MongoDB is
  the source of truth for deletion-log delivery state (see
  ``utils/deletion_audit.py``).  Every log message still carries its
  persistent ``EK:<tag>`` line (``append_event_tag``) so an operator —
  or any future verifier — can identify the exact event a delivered log
  belongs to by EXACT token match (``find_event_tag_in_text``), never
  substring guessing.

Delivery responsibilities
---------------------------------------------------------------------
This module is deliberately only the TRANSPORT + formatting layer: it
sends text to the logger group and reports an honest per-send OUTCOME
(``_send_with_outcome`` → "sent" / "failed" / "uncertain" plus the
delivered Telegram message id).  All persistent delivery state — the
``pending`` / ``sending`` / ``sent`` / ``uncertain`` records, atomic
claims and startup recovery — lives in MongoDB and is owned by
``utils/deletion_audit.py``.

Reliability contract (never crash the bot)
---------------------------------------------------------------------
* If ``config.LOGGER_ID`` is 0 / missing, every helper silently no-ops.
* Telegram ``FloodWait``: waits of up to ``MAX_FLOODWAIT_SLEEP`` seconds
  are slept out once and the send is retried once; longer waits are
  abandoned (logged locally) instead of stalling a handler.
* Any ``RPCError`` / network error / unexpected exception is caught and
  reported to the application logger (``log.txt`` / console).  Callers
  are NEVER disturbed: every public helper returns ``False`` on failure
  and never raises.
* Duplicate protection — event identifiers only, NO time windows.
  Every genuine event is logged: a setting toggled twice, two Create
  Post publications by the same user, a schedule paused again — all of
  them produce their own log, even when several happen within the same
  second.  A send is suppressed ONLY when the caller supplies a
  RELIABLE event-specific identifier (``event_id``) that was already
  delivered in this process — i.e. the same actual event being
  processed more than once (double-fired update, retried handler).
  Callers without such an identifier pass nothing and are always
  logged.  The post-deletion log additionally claims its event in
  MongoDB (the ``deletion_log_delivery`` record — the persistent
  delivery state owned by ``deletion_audit.py``) so a worker retry
  after a restart cannot send a second "deleted" message for the same
  post; the claim is released again when delivery definitely failed,
  and an attempt interrupted by a crash is recovered from the MongoDB
  delivery state by the startup sweep and the periodic retry worker —
  Telegram history is never consulted — so a log is never silently
  lost.
* Event-key identification — every post-deletion log carries a unique,
  PERSISTENT ``event_key`` tag (``EK:<16 hex chars>``, derived
  deterministically from the audit record's event key + log kind) in a
  compact code-span line at the end of the message.  The tag survives
  restarts (it is recomputable from MongoDB) and stays searchable in
  the logger group forever.  It is identification metadata: an
  operator — or any verifier — ties a delivered log to its EXACT
  MongoDB delivery record by EXACT token match
  (``find_event_tag_in_text``), never substring guessing.  Delivery
  state itself lives ONLY in MongoDB; nothing ever scans Telegram
  history to decide whether a log still needs to be sent.
* HTML safety: every user-controlled string (names, previews, reasons)
  is HTML-escaped before being embedded; long previews are truncated to
  ``MAX_PREVIEW`` characters on a UTF-16-safe boundary.
* Credential safety: no API keys, bot tokens, session strings or Mongo
  URIs are ever passed to or emitted by this module.
"""

import asyncio
import hashlib
import html
import re
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import config
from pyrogram.errors import FloodWait, RPCError

from SWAGGYMUSIC.logging import LOGGER

log = LOGGER("SWAGGYMUSIC.logger")


# ---------------------------------------------------------------------------
# Tuning
# ---------------------------------------------------------------------------

# In-process event-identifier memory (same scale as the policy verdict LRUs).
_DELIVERED_MAX = 10_000
_DELIVERED_EVENTS: "OrderedDict[Tuple, None]" = OrderedDict()

# Telegram message limit is 4096 characters; keep logs comfortably inside it.
MAX_LOG_TEXT = 3800
# Preview length for deleted-post logs.
MAX_PREVIEW = 600
# A FloodWait up to this many seconds is slept out once (then retried once).
MAX_FLOODWAIT_SLEEP = 10

# Honest per-send outcomes returned by the transport layer
# (``_send_with_outcome``):
#   "sent"     — Telegram ACCEPTED the message (message id returned).
#   "failed"   — Telegram definitely did NOT deliver (it refused the
#                send: RPCError, or a FloodWait too long to sleep out).
#   "uncertain"— the outcome is UNKNOWN (connection lost mid-send and
#                similar); the message MAY have been delivered.
OUTCOME_SENT = "sent"
OUTCOME_FAILED = "failed"
OUTCOME_UNCERTAIN = "uncertain"

# Friendly names for deletion sources (the feature responsible).
_SOURCE_LABELS = {
    "SETDELAY": "Set Delay (auto-delete)",
    "AUTODELETE_WORKER": "Set Delay (auto-delete worker)",
    "SETGAP": "Post Gap Control",
    "SETANONYMOUS": "Anonymous Post Protection",
    "SETLINK": "Link Restriction",
    "SETLINK_ALBUM": "Link Restriction (album)",
    "SETFORWARD": "Forward Post Remover",
    "BANNEDWORDS": "Banned Words",
    "APPROVALMODE": "Approval Mode",
    "LOGGER": "Logger",
}

# Human-readable deletion reasons (record.reason → text). Reasons carrying
# dynamic data (the matched banned word) build their line from record details.
_REASON_LABELS = {
    "GAP_REJECT": "Post gap violation (/setgap)",
    "SETDELAY_EXPIRED": "Auto-delete delay expired (/setdelay)",
    "GAP_REJECT_RETRY": "Post gap violation (/setgap) — delayed retry",
    "ANON_REJECT_RETRY": "Anonymous post rejected — delayed retry",
    "LINK_REJECT_RETRY": "Prohibited link rejected — delayed retry",
    "FORWARD_REJECT_RETRY": "Forwarded post rejected — delayed retry",
    "ANON_REJECT": "Anonymous post not allowed (/setanonymous)",
    "LINK_BLOCKED": "Prohibited link detected (/setlink)",
    "LINK_ALBUM_ITEM": "Album contained a prohibited link (/setlink)",
    "FORWARD_BLOCKED": "Forwarded post not allowed (/setforward)",
    "BANNED_WORD": "Banned word detected (/bannedwords)",
    "BANNED_WORD_ALBUM": "Album contained a banned word (/bannedwords)",
    "APPROVAL_NO_ALLOWED_USERNAME": "No approved username found (Approval Mode)",
    "APPROVAL_ALBUM_ITEM": "Album rejected by Approval Mode",
}

# Status lines per deletion-log kind.
_KIND_HEADERS = {
    "deleted": "🗑️ <b>POST DELETED IN CHANNEL</b>",
    "pending_retry": "⏳ <b>POST DELETION PENDING — RETRY QUEUED</b>",
    "failed": "⚠️ <b>POST DELETION FAILED</b>",
}
_KIND_STATUS = {
    "deleted": "DELETED",
    "pending_retry": "PENDING RETRY",
    "failed": "FAILED",
}


# ---------------------------------------------------------------------------
# Same-event dedup — event identifiers only, no time windows
# ---------------------------------------------------------------------------

def _claim_event(event_id: Tuple) -> bool:
    """Claim the delivery slot for a RELIABLE event-specific identifier.

    False only when the SAME identifier was already DELIVERED earlier in
    this process — the signature of one actual event being processed
    more than once (double-fired update, retried handler).  There is
    deliberately no time window: identifiers are only claimed for
    events that a single concrete object identifies (a post's message
    id, a MongoDB event key), so the same identifier can never belong to
    a later, different event.  Never raises.
    """
    try:
        if event_id in _DELIVERED_EVENTS:
            return False
        _DELIVERED_EVENTS[event_id] = None
        _DELIVERED_EVENTS.move_to_end(event_id)
        while len(_DELIVERED_EVENTS) > _DELIVERED_MAX:
            _DELIVERED_EVENTS.popitem(last=False)
        return True
    except Exception:
        return True


def _release_event(event_id: Tuple) -> None:
    """Give a claimed event identifier back after a FAILED delivery.

    A log that never reached the logger group must not stay registered
    as delivered — otherwise the legitimate retry of the same event
    would be swallowed as a duplicate.  Never raises.
    """
    try:
        _DELIVERED_EVENTS.pop(event_id, None)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Persistent event-key tags — the exact identity of one log message
# ---------------------------------------------------------------------------

# The exact event-key tag embedded in every post-deletion log message:
# ``EK:<16 hex chars>`` inside a compact code-span line.  The tag is
# deterministic (recomputable from the MongoDB delivery record at any
# time) and unique per (event, kind).  The log TEXT is transport only —
# delivery state lives in MongoDB — but the embedded tag still lets an
# operator (or any verifier) match a delivered message to its exact
# event by EXACT token equality (``find_event_tag_in_text``), never by
# substring guessing.
EVENT_TAG_PREFIX = "EK:"
_TAG_TOKEN_LENGTH = 16

def deletion_log_event_tag(event_key: str, kind: str) -> str:
    """The unique, persistent tag of ONE post-deletion log.

    Deterministic SHA-256 over ``event_key`` (the audit record's
    ``chat_id:message_id``) + the log kind — recomputable at any time
    after a restart, so a delivered log can always be tied back to its
    MongoDB delivery record.  One tag per (event, kind): a "deleted"
    log and a "failed" log of the same post are distinct events with
    distinct tags.

    16 hex chars → 64 bits — collision-proof for any realistic audit
    volume, and compact enough to embed in every log message.
    """
    raw = f"v1|log|{event_key}|{kind}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_TAG_TOKEN_LENGTH]


def alert_event_tag(event_key: str) -> str:
    """The persistent tag of ONE archive-failure alert (same scheme)."""
    raw = f"v1|alert|{event_key}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:_TAG_TOKEN_LENGTH]


def append_event_tag(text: str, tag: str, *, max_len: int = MAX_LOG_TEXT) -> str:
    """Append the compact event-key line to a log text.

    The line ``🧾 <code>EK:<tag></code>`` is metadata-safe: plain ASCII
    inside a code span, HTML-clean, visually subordinate, and readable
    in Telegram clients after any restart.  The base text is truncated
    FIRST so the tag line always survives inside ``max_len`` (a
    truncated-off tag would defeat the log's identification).
    """
    line = f"\n\n🧾 <code>{EVENT_TAG_PREFIX}{tag}</code>"
    room = max_len - len(line)
    if room < 0:
        room = 0
    return str(text)[:room] + line


def _event_tag_pattern(tag: str) -> "re.Pattern":
    """Regex that matches the EXACT tag token — and nothing similar.

    Lookarounds forbid surrounding alphanumerics, so a tag never
    matches inside a longer run (another tag, a bigger id, a hex blob):
    this is exact-token equality, never substring matching.
    """
    needle = re.escape(f"{EVENT_TAG_PREFIX}{tag}")
    return re.compile(rf"(?<![0-9a-zA-Z]){needle}(?![0-9a-zA-Z])")


def find_event_tag_in_text(text: str, tag: str) -> bool:
    """True only when ``text`` contains the EXACT event-key token ``tag``.

    ``EK:<tag>`` embedded in a longer alphanumeric run (e.g. a similar
    tag or a bigger number) does NOT match.  Never raises.
    """
    try:
        if not text or not tag:
            return False
        return bool(_event_tag_pattern(tag).search(str(text)))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Formatting helpers (moved here from the old activity_logger)
# ---------------------------------------------------------------------------

def _build_mention(user) -> str:
    """HTML ``<a href="tg://user?id=...">`` mention; escaped, never raises."""

    if user is None:
        return "Unknown User"

    try:
        name = (user.first_name or "").strip()
        if not name:
            name = (getattr(user, "last_name", None) or "").strip()
        if not name:
            name = "User"

        safe_name = html.escape(name, quote=False)
        user_id = int(user.id)
        return f'<a href="tg://user?id={user_id}">{safe_name}</a>'

    except Exception:
        return "Unknown User"


def _format_username(user) -> str:
    """``@username`` or ``Not Available``."""
    if user is None:
        return "Not Available"
    username = getattr(user, "username", None)
    return f"@{username}" if username else "Not Available"


def _format_timestamp() -> str:
    """Human-readable UTC timestamp for log entries."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _format_channel_title(chat) -> str:
    """Escaped channel title, or ``Not Available``. Never raises."""
    if chat is None:
        return "Not Available"
    try:
        title = getattr(chat, "title", None) or getattr(chat, "first_name", None)
        if title:
            return html.escape(str(title), quote=False)
    except Exception:
        pass
    return "Not Available"


def _format_channel_username(chat) -> str:
    """``@username`` for the channel, or ``Not Available``."""
    if chat is None:
        return "Not Available"
    try:
        username = getattr(chat, "username", None)
        if username:
            return f"@{username}"
    except Exception:
        pass
    return "Not Available"


def _format_channel_id(chat) -> int:
    """Numeric channel ID, or 0 if unavailable."""
    if chat is None:
        return 0
    try:
        return int(getattr(chat, "id", 0))
    except Exception:
        return 0


def _format_user_id(user) -> str:
    """Numeric user ID as a string, or ``Not Available``."""
    if user is None:
        return "Not Available"
    try:
        return str(int(getattr(user, "id", 0)))
    except Exception:
        return "Not Available"


def _format_duration(seconds: int) -> str:
    """Seconds → human-readable duration ("3 seconds", "22 hours", ...)."""

    try:
        seconds = int(seconds)
    except Exception:
        return str(seconds)

    if seconds <= 0:
        return "0 seconds"
    if seconds < 60:
        return f"{seconds} second{'s' if seconds != 1 else ''}"
    if seconds < 3600:
        minutes = seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''}"
    hours = seconds // 3600
    return f"{hours} hour{'s' if hours != 1 else ''}"


def _escape(value) -> str:
    """HTML-escape any value for safe embedding in a log message."""
    try:
        return html.escape(str(value), quote=False)
    except Exception:
        return ""


def _short_preview(text: Optional[str], limit: int = MAX_PREVIEW) -> str:
    """Truncate a preview on a UTF-16-safe boundary (Telegram entity rule)."""
    if not text:
        return ""
    try:
        data = text.encode("utf-16-le")
        if len(data) <= limit * 2:
            return text
        cut = data[: limit * 2]
        # do not split a surrogate pair in half
        if len(cut) % 2:
            cut = cut[:-1]
        out = cut.decode("utf-16-le", errors="replace")
        return out.rstrip() + "…"
    except Exception:
        return str(text)[:limit]


def _source_label(source: Optional[str]) -> str:
    if not source:
        return "Unknown"
    return _SOURCE_LABELS.get(str(source), _escape(source))


def reason_line(record: dict) -> str:
    """The human-readable ``Reason:`` line for a deletion event."""
    reason = str((record or {}).get("reason") or "UNKNOWN")
    details = (record or {}).get("details") or {}
    if reason == "BANNED_WORD":
        matched = details.get("matched")
        if matched:
            return f'Banned word detected: "{_escape(matched)}"'
    if reason == "BANNED_WORD_ALBUM":
        matched = details.get("matched")
        if matched:
            return f'Album contained the banned word: "{_escape(matched)}"'
    return _REASON_LABELS.get(reason, _escape(reason.replace("_", " ").title()))


# ---------------------------------------------------------------------------
# Core sender — every helper below routes through this single function.
# ---------------------------------------------------------------------------

async def _send_with_outcome(text: str) -> Tuple[str, Optional[int]]:
    """THE single Telegram send to the logger group + honest outcome.

    Returns ``(OUTCOME_SENT, message_id)`` when Telegram ACCEPTED the
    message, ``(OUTCOME_FAILED, None)`` when Telegram definitely did
    NOT deliver (it refused: RPCError, or a FloodWait too long to sleep
    out), and ``(OUTCOME_UNCERTAIN, None)`` when the outcome is unknown
    (connection lost mid-send and similar — the message MAY have been
    delivered).  Never raises.
    """
    try:
        logger_id = int(getattr(config, "LOGGER_ID", 0) or 0)
    except Exception:
        logger_id = 0
    if not logger_id:
        # Nothing can ever be delivered without a logger group — a
        # definite non-delivery, not an ambiguity.
        return OUTCOME_FAILED, None

    async def _send():
        from SWAGGYMUSIC import app  # local import avoids import cycles

        return await app.send_message(
            chat_id=logger_id,
            text=text[:MAX_LOG_TEXT],
            disable_web_page_preview=True,
        )

    try:
        sent = await _send()
        return OUTCOME_SENT, getattr(sent, "id", None)

    except FloodWait as exc:
        wait = int(getattr(exc, "value", 0) or 0)
        if 0 < wait <= MAX_FLOODWAIT_SLEEP:
            await asyncio.sleep(wait + 1)
            try:
                sent = await _send()
                return OUTCOME_SENT, getattr(sent, "id", None)
            except RPCError:
                return OUTCOME_FAILED, None
            except Exception:
                # Telegram MAY have accepted the retry — unknown.
                return OUTCOME_UNCERTAIN, None
        log.warning(
            "Log send skipped (FloodWait %ss): %.80s",
            wait, text.replace("\n", " "),
        )
        return OUTCOME_FAILED, None
    except RPCError as exc:
        log.warning(
            "Log send failed (%s): %.80s",
            type(exc).__name__, text.replace("\n", " "),
        )
        return OUTCOME_FAILED, None
    except Exception as exc:
        # Never crash the caller. The outcome is unknown, not definitely
        # failed: the connection may have dropped after Telegram accepted.
        log.warning(
            "Logger send outcome unknown: %s: %s", type(exc).__name__, exc,
        )
        return OUTCOME_UNCERTAIN, None


async def send_log(text: str, *, event_id: Optional[Tuple] = None) -> bool:
    """Send ``text`` to ``config.LOGGER_ID`` — THE single point of entry.

    Every event helper in this module calls this; nothing else in the
    codebase may call ``app.send_message(LOGGER_ID, ...)`` directly.

    ``event_id`` is optional and must only be passed when a RELIABLE
    event-specific identifier exists (a post's message id, a MongoDB
    event key) — two sends with the same ``event_id`` are one actual
    event processed twice, so the second is suppressed.  WITHOUT an
    identifier every call is sent: repeated setting changes, multiple
    Create Post publications and schedule pause/resume cycles are all
    genuine events and are never deduplicated.

    Returns True when Telegram accepted the message, False when logging
    is disabled, suppressed as a same-event duplicate, or the send
    failed/ended uncertain.  A failed send releases its event slot, so
    retrying the same event later is always possible; a delivered send
    keeps it, which is what suppresses true duplicates.  Never raises.
    """
    if event_id is not None and not _claim_event(event_id):
        return False

    delivered = False
    try:
        outcome, _message_id = await _send_with_outcome(text)
        delivered = outcome == OUTCOME_SENT
        return delivered
    finally:
        # A failed delivery must not keep the event slot: the same event
        # may legitimately be retried (and then delivered). Only success
        # makes the slot stick.
        if event_id is not None and not delivered:
            _release_event(event_id)


# ---------------------------------------------------------------------------
# Event helpers — one per logical event.  Each builds its text and
# delegates to send_log().
# ---------------------------------------------------------------------------

async def log_user_start(user) -> None:
    """A user started the bot (private /start)."""

    text = (
        f"{_build_mention(user)} ᴊᴜsᴛ sᴛᴀʀᴛᴇᴅ ᴛʜᴇ ʙᴏᴛ.\n\n"
        f"<b>ᴜsᴇʀ ɪᴅ :</b> <code>{int(getattr(user, 'id', 0))}</code>\n"
        f"<b>ᴜsᴇʀɴᴀᴍᴇ :</b> {_format_username(user)}"
    )
    await send_log(text)


async def log_bot_started(app_client) -> bool:
    """The bot finished booting — identity summary to the logger group.

    Returns True when the message was delivered.  ``core/bot.py`` uses the
    return value as its startup health gate (it exits when the logger
    group is unreachable), so this deliberately returns False both when
    LOGGER_ID is unset and when the send failed.
    """
    try:
        name = getattr(app_client, "name", None) or "Bot"
        username = getattr(app_client, "username", None)
        bot_id = int(getattr(app_client, "id", 0) or 0)
    except Exception:
        name, username, bot_id = "Bot", None, 0

    text = (
        "🚀 <b>BOT STARTED</b>\n\n"
        f"🤖 <b>Bot:</b> {_escape(name)}\n"
        f"🆔 <b>Bot ID:</b> <code>{bot_id}</code>\n"
        f"🔗 <b>Username:</b> {'@' + username if username else 'Not Available'}\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    return await send_log(text)


async def log_bot_added_to_channel(chat, *, actor=None) -> None:
    """The bot was added to (or promoted in) a chat — channel info + ID."""

    text = (
        "➕ <b>BOT ADDED TO CHANNEL</b>\n\n"
        f"📌 <b>Channel:</b> {_format_channel_title(chat)}\n"
        f"🆔 <b>Channel ID:</b> <code>{_format_channel_id(chat)}</code>\n"
        f"🔗 <b>Username:</b> {_format_channel_username(chat)}\n"
        f"🧩 <b>Type:</b> {_escape(getattr(getattr(chat, 'type', None), 'name', 'Unknown'))}"
    )
    if actor is not None:
        text += (
            f"\n\n👤 <b>Added By:</b> {_build_mention(actor)}\n"
            f"🆔 <b>User ID:</b> <code>{_format_user_id(actor)}</code>"
        )
    text += f"\n\n🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    await send_log(text)


# Channels whose first activity was already reported in this process.
_CHANNEL_SEEN_MAX = 2048
_CHANNELS_SEEN: "OrderedDict[int, None]" = OrderedDict()


async def log_channel_first_seen(chat) -> None:
    """Channel information + ID — once per channel per process.

    Telegram does not reliably notify bots that they were added to a
    channel, so the first post/command the bot processes there doubles
    as the registration event. Bounded LRU keeps this at zero cost for
    channels already reported.
    """
    try:
        chat_id = int(getattr(chat, "id", 0) or 0)
    except Exception:
        return
    if not chat_id:
        return
    if chat_id in _CHANNELS_SEEN:
        return
    _CHANNELS_SEEN[chat_id] = None
    _CHANNELS_SEEN.move_to_end(chat_id)
    while len(_CHANNELS_SEEN) > _CHANNEL_SEEN_MAX:
        _CHANNELS_SEEN.popitem(last=False)

    text = (
        "📢 <b>CHANNEL REGISTERED</b>\n\n"
        f"📌 <b>Channel:</b> {_format_channel_title(chat)}\n"
        f"🆔 <b>Channel ID:</b> <code>{chat_id}</code>\n"
        f"🔗 <b>Username:</b> {_format_channel_username(chat)}\n\n"
        "The bot processed its first activity in this channel.\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


async def log_setdelay_change(
    chat, user, status: str, *,
    delay_value: Optional[str] = None,
    old_value: Optional[str] = None,
    source: str = "CHANNEL",
) -> None:
    """A /setdelay state change for a specific channel (ENABLED/UPDATED/DISABLED)."""

    ch_title = _format_channel_title(chat)
    ch_id = _format_channel_id(chat)
    ch_uname = _format_channel_username(chat)
    actor = _build_mention(user) if user is not None else "Channel Admin"
    actor_id = _format_user_id(user)

    if status == "ENABLED":
        header = "🟢 <b>SETDELAY ENABLED</b>"
        value_line = f"⏱ <b>Delay:</b> <code>{delay_value or 'N/A'}</code>\n\n"
    elif status == "DISABLED":
        header = "🔴 <b>SETDELAY DISABLED</b>"
        value_line = ""
    elif status == "UPDATED":
        header = "⚙️ <b>SETDELAY UPDATED</b>"
        value_line = (
            f"⏱ <b>Previous Delay:</b> <code>{old_value or 'N/A'}</code>\n"
            f"⏱ <b>New Delay:</b> <code>{delay_value or 'N/A'}</code>\n\n"
        )
    else:
        header = f"⏱ <b>SETDELAY {_escape(status)}</b>"
        value_line = ""

    text = (
        f"{header}\n\n"
        f"📍 <b>Source:</b> {_escape(source)}\n\n"
        f"📌 <b>Channel:</b> {ch_title}\n"
        f"🆔 <b>Channel ID:</b> <code>{ch_id}</code>\n"
        f"🔗 <b>Username:</b> {ch_uname}\n\n"
        f"👤 <b>Changed By:</b> {actor}\n"
        f"🆔 <b>User ID:</b> <code>{actor_id}</code>\n\n"
        f"{value_line}"
        f"📊 <b>Status:</b> {_escape(status)}\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


async def log_setgap_change(
    chat, user, status: str, *,
    gap_value: Optional[str] = None,
    old_value: Optional[str] = None,
    source: str = "CHANNEL",
) -> None:
    """A /setgap state change for a specific channel (ENABLED/UPDATED/DISABLED)."""

    ch_title = _format_channel_title(chat)
    ch_id = _format_channel_id(chat)
    ch_uname = _format_channel_username(chat)
    actor = _build_mention(user) if user is not None else "Channel Admin"
    actor_id = _format_user_id(user)

    if status == "ENABLED":
        header = "🟢 <b>SETGAP ENABLED</b>"
        value_line = f"📏 <b>Gap:</b> <code>{gap_value or 'N/A'}</code>\n\n"
    elif status == "DISABLED":
        header = "🔴 <b>SETGAP DISABLED</b>"
        value_line = ""
    elif status == "UPDATED":
        header = "⚙️ <b>SETGAP UPDATED</b>"
        value_line = (
            f"⏱ <b>Previous Gap:</b> <code>{old_value or 'N/A'}</code>\n"
            f"⏱ <b>New Gap:</b> <code>{gap_value or 'N/A'}</code>\n\n"
        )
    else:
        header = f"📏 <b>SETGAP {_escape(status)}</b>"
        value_line = ""

    text = (
        f"{header}\n\n"
        f"📍 <b>Source:</b> {_escape(source)}\n\n"
        f"📌 <b>Channel:</b> {ch_title}\n"
        f"🆔 <b>Channel ID:</b> <code>{ch_id}</code>\n"
        f"🔗 <b>Username:</b> {ch_uname}\n\n"
        f"👤 <b>Changed By:</b> {actor}\n"
        f"🆔 <b>User ID:</b> <code>{actor_id}</code>\n\n"
        f"{value_line}"
        f"📊 <b>Status:</b> {_escape(status)}\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


async def log_setlink_change(
    chat, user, status: str, *, source: str = "CHANNEL",
) -> None:
    """A /setlink (Link Restriction) state change for a channel."""

    header = (
        "🟢 <b>LINK RESTRICTION ENABLED</b>" if status == "ENABLED"
        else "🔴 <b>LINK RESTRICTION DISABLED</b>" if status == "DISABLED"
        else f"🔗 <b>LINK RESTRICTION {_escape(status)}</b>"
    )
    text = (
        f"{header}\n\n"
        f"📍 <b>Source:</b> {_escape(source)}\n\n"
        f"📌 <b>Channel:</b> {_format_channel_title(chat)}\n"
        f"🆔 <b>Channel ID:</b> <code>{_format_channel_id(chat)}</code>\n"
        f"🔗 <b>Username:</b> {_format_channel_username(chat)}\n\n"
        f"👤 <b>Changed By:</b> {_build_mention(user) if user is not None else 'Channel Admin'}\n"
        f"🆔 <b>User ID:</b> <code>{_format_user_id(user)}</code>\n\n"
        f"📊 <b>Status:</b> {_escape(status)}\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


async def log_setforward_change(
    chat, user, status: str, *, source: str = "CHANNEL",
) -> None:
    """A /setforward (Forward Post Remover) state change for a channel."""

    header = (
        "🟢 <b>FORWARD POST REMOVER ENABLED</b>" if status == "ENABLED"
        else "🔴 <b>FORWARD POST REMOVER DISABLED</b>" if status == "DISABLED"
        else f"↪️ <b>FORWARD POST REMOVER {_escape(status)}</b>"
    )
    text = (
        f"{header}\n\n"
        f"📍 <b>Source:</b> {_escape(source)}\n\n"
        f"📌 <b>Channel:</b> {_format_channel_title(chat)}\n"
        f"🆔 <b>Channel ID:</b> <code>{_format_channel_id(chat)}</code>\n"
        f"🔗 <b>Username:</b> {_format_channel_username(chat)}\n\n"
        f"👤 <b>Changed By:</b> {_build_mention(user) if user is not None else 'Channel Admin'}\n"
        f"🆔 <b>User ID:</b> <code>{_format_user_id(user)}</code>\n\n"
        f"📊 <b>Status:</b> {_escape(status)}\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


async def log_setanonymous_change(
    chat, user, status: str, *, source: str = "CHANNEL",
) -> None:
    """A /setanonymous (Anonymous Post protection) state change for a channel."""

    header = (
        "🟢 <b>ANONYMOUS POST PROTECTION ENABLED</b>" if status == "ENABLED"
        else "🔴 <b>ANONYMOUS POST PROTECTION DISABLED</b>" if status == "DISABLED"
        else f"🛡️ <b>ANONYMOUS POST PROTECTION {_escape(status)}</b>"
    )
    text = (
        f"{header}\n\n"
        f"📍 <b>Source:</b> {_escape(source)}\n\n"
        f"📌 <b>Channel:</b> {_format_channel_title(chat)}\n"
        f"🆔 <b>Channel ID:</b> <code>{_format_channel_id(chat)}</code>\n"
        f"🔗 <b>Username:</b> {_format_channel_username(chat)}\n\n"
        f"👤 <b>Changed By:</b> {_build_mention(user) if user is not None else 'Channel Admin'}\n"
        f"🆔 <b>User ID:</b> <code>{_format_user_id(user)}</code>\n\n"
        f"📊 <b>Status:</b> {_escape(status)}\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


async def log_approval_mode_change(
    chat, user, status: str, *, source: str = "CHANNEL",
    usernames: Optional[int] = None,
) -> None:
    """An Approval Mode enable/disable/change event for a channel."""

    header = (
        "🟢 <b>APPROVAL MODE ENABLED</b>" if status == "ENABLED"
        else "🔴 <b>APPROVAL MODE DISABLED</b>" if status == "DISABLED"
        else f"✅ <b>APPROVAL MODE {_escape(status)}</b>"
    )
    count_line = ""
    if usernames is not None:
        count_line = f"👥 <b>Approved usernames:</b> {int(usernames)}\n\n"
    text = (
        f"{header}\n\n"
        f"📍 <b>Source:</b> {_escape(source)}\n\n"
        f"📌 <b>Channel:</b> {_format_channel_title(chat)}\n"
        f"🆔 <b>Channel ID:</b> <code>{_format_channel_id(chat)}</code>\n"
        f"🔗 <b>Username:</b> {_format_channel_username(chat)}\n\n"
        f"👤 <b>Changed By:</b> {_build_mention(user) if user is not None else 'Channel Admin'}\n"
        f"🆔 <b>User ID:</b> <code>{_format_user_id(user)}</code>\n\n"
        f"{count_line}"
        f"📊 <b>Status:</b> {_escape(status)}\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


async def log_approval_usernames_changed(
    chat, user, action: str, usernames: list, *, total: int,
) -> None:
    """Allowed usernames were added to / removed from a channel's list."""

    shown = ", ".join(f"@{_escape(u)}" for u in usernames[:10])
    if len(usernames) > 10:
        shown += f" (+{len(usernames) - 10} more)"
    if action == "ADDED":
        header = "➕ <b>APPROVAL MODE — USERNAMES ADDED</b>"
    elif action == "REMOVED":
        header = "➖ <b>APPROVAL MODE — USERNAMES REMOVED</b>"
    elif action == "CLEARED":
        header = "🧹 <b>APPROVAL MODE — LIST CLEARED</b>"
    else:
        header = f"👥 <b>APPROVAL MODE LIST {_escape(action)}</b>"

    text = (
        f"{header}\n\n"
        f"📌 <b>Channel:</b> {_format_channel_title(chat)}\n"
        f"🆔 <b>Channel ID:</b> <code>{_format_channel_id(chat)}</code>\n\n"
        f"👥 <b>Usernames:</b> {shown or '—'}\n"
        f"🗂 <b>List size:</b> {int(total)}\n\n"
        f"👤 <b>Changed By:</b> {_build_mention(user) if user is not None else 'Channel Admin'}\n"
        f"🆔 <b>User ID:</b> <code>{_format_user_id(user)}</code>\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


async def log_approval_mixed_warning(
    chat, message, allowed: list, unapproved: list,
) -> None:
    """A post carried BOTH approved and unapproved usernames: kept + warned.

    Per the Approval Mode policy the post is NOT deleted in the mixed
    case — this warning goes to the logger group instead.

    The post itself (chat id + message id) is the reliable event-specific
    identifier here: one post yields at most one mixed warning per
    process, while different posts each get their own warning.
    """

    def _names(values: list) -> str:
        shown = ", ".join(f"@{_escape(v)}" for v in values[:8])
        if len(values) > 8:
            shown += f" (+{len(values) - 8} more)"
        return shown or "—"

    text = (
        "⚠️ <b>APPROVAL MODE — MIXED USERNAMES WARNING</b>\n\n"
        f"📌 <b>Channel:</b> {_format_channel_title(chat)}\n"
        f"🆔 <b>Channel ID:</b> <code>{_format_channel_id(chat)}</code>\n"
        f"💬 <b>Post ID:</b> <code>{getattr(message, 'id', '?')}</code>\n\n"
        f"✅ <b>Approved found:</b> {_names(list(allowed))}\n"
        f"🚫 <b>Unapproved found:</b> {_names(list(unapproved))}\n\n"
        "📊 <b>Status:</b> KEPT (contains at least one approved username)\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(
        text,
        event_id=(
            "approval_mixed", _format_channel_id(chat),
            getattr(message, "id", 0),
        ),
    )


async def log_bannedwords_change(
    chat, user, status: str, *, source: str = "CHANNEL",
    total: Optional[int] = None,
) -> None:
    """A Banned Words enable/disable event for a channel."""

    header = (
        "🟢 <b>BANNED WORDS ENABLED</b>" if status == "ENABLED"
        else "🔴 <b>BANNED WORDS DISABLED</b>" if status == "DISABLED"
        else f"🤬 <b>BANNED WORDS {_escape(status)}</b>"
    )
    count_line = ""
    if total is not None:
        count_line = f"🗂 <b>Banned words:</b> {int(total)}\n\n"
    text = (
        f"{header}\n\n"
        f"📍 <b>Source:</b> {_escape(source)}\n\n"
        f"📌 <b>Channel:</b> {_format_channel_title(chat)}\n"
        f"🆔 <b>Channel ID:</b> <code>{_format_channel_id(chat)}</code>\n"
        f"🔗 <b>Username:</b> {_format_channel_username(chat)}\n\n"
        f"👤 <b>Changed By:</b> {_build_mention(user) if user is not None else 'Channel Admin'}\n"
        f"🆔 <b>User ID:</b> <code>{_format_user_id(user)}</code>\n\n"
        f"{count_line}"
        f"📊 <b>Status:</b> {_escape(status)}\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


async def log_bannedwords_list_changed(
    chat, user, action: str, words: list, *, total: int,
) -> None:
    """Words were added to / removed from a channel's banned-words list."""

    shown = ", ".join(f'"{_escape(w)}"' for w in words[:10])
    if len(words) > 10:
        shown += f" (+{len(words) - 10} more)"
    if action == "ADDED":
        header = "➕ <b>BANNED WORDS — WORDS ADDED</b>"
    elif action == "REMOVED":
        header = "➖ <b>BANNED WORDS — WORD REMOVED</b>"
    elif action == "CLEARED":
        header = "🧹 <b>BANNED WORDS — LIST CLEARED</b>"
    else:
        header = f"🤬 <b>BANNED WORDS LIST {_escape(action)}</b>"

    text = (
        f"{header}\n\n"
        f"📌 <b>Channel:</b> {_format_channel_title(chat)}\n"
        f"🆔 <b>Channel ID:</b> <code>{_format_channel_id(chat)}</code>\n\n"
        f"🤬 <b>Words:</b> {shown or '—'}\n"
        f"🗂 <b>List size:</b> {int(total)}\n\n"
        f"👤 <b>Changed By:</b> {_build_mention(user) if user is not None else 'Channel Admin'}\n"
        f"🆔 <b>User ID:</b> <code>{_format_user_id(user)}</code>\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


_SCHEDULE_HEADERS = {
    "CREATED": "🟢 <b>SCHEDULED POST CREATED</b>",
    "EDITED": "⚙️ <b>SCHEDULED POST EDITED</b>",
    "RESCHEDULED": "🕒 <b>SCHEDULED POST RESCHEDULED</b>",
    "CANCELLED": "🚫 <b>SCHEDULED POST CANCELLED</b>",
    "DELETED": "🗑 <b>SCHEDULED POST DELETED</b>",
    "SENT": "✅ <b>SCHEDULED POST SENT</b>",
    "FAILED": "🔴 <b>SCHEDULED POST FAILED</b>",
    "MISSED": "⌛ <b>SCHEDULED POST MISSED</b>",
    "SKIPPED": "⌛ <b>SCHEDULED POST SKIPPED</b>",
    "PAUSED": "⏸ <b>SCHEDULED POST PAUSED</b>",
    "RESUMED": "▶️ <b>SCHEDULED POST RESUMED</b>",
    "TIME_CHANGED": "🕒 <b>SCHEDULED POST TIME CHANGED</b>",
}
_SCHEDULE_KINDS = {"once": "One-Time", "daily": "Daily"}


async def log_schedule_change(
    channel_id: int, channel_title: Optional[str], user, action: str, *,
    schedule_id: int, when: Optional[str] = None, details: Optional[str] = None,
    kind: Optional[str] = None, protected: Optional[bool] = None,
) -> None:
    """A /Schedule (scheduled post) event. (Same signature as before.)"""

    header = _SCHEDULE_HEADERS.get(
        action, f"🗓 <b>SCHEDULED POST {_escape(action)}</b>"
    )
    actor = _build_mention(user) if user is not None else "Scheduler"
    title = _escape(channel_title) if channel_title else "Not Available"
    text = (
        f"{header}\n\n"
        f"🆔 <b>Schedule:</b> <code>#{int(schedule_id)}</code>\n"
        f"📌 <b>Channel:</b> {title}\n"
        f"🆔 <b>Channel ID:</b> <code>{int(channel_id)}</code>\n"
        + (f"🗂 <b>Type:</b> {_SCHEDULE_KINDS[kind]}\n" if kind in _SCHEDULE_KINDS else "")
        + (f"📅 <b>Post Time:</b> {_escape(when)}\n" if when else "")
        + (f"🛡 <b>Protection:</b> {'ON' if protected else 'OFF'}\n" if protected is not None else "")
        + f"\n👤 <b>By:</b> {actor}\n"
        f"🆔 <b>User ID:</b> <code>{_format_user_id(user)}</code>\n\n"
        + (f"📝 <b>Details:</b> {_escape(details)}\n\n" if details else "")
        + f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    # Every schedule transition (create, pause, resume, ...) is its own
    # genuine event — a schedule paused again later must be logged again.
    await send_log(text)


async def log_createpost_attempt(user) -> None:
    """A non-authorized user reached the final CreatePost gate (denied)."""

    text = (
        "🔒 <b>CREATE POST ACCESS ATTEMPT</b>\n\n"
        f"👤 <b>User:</b> {_build_mention(user)}\n"
        f"🆔 <b>ID:</b> <code>{int(getattr(user, 'id', 0))}</code>\n"
        f"🔗 <b>Username:</b> {_format_username(user)}\n\n"
        "📌 <b>Action:</b>\n"
        "User attempted to create a post but does not have access to CreatePost.\n\n"
        "📊 <b>Status:</b> ❌ ACCESS DENIED\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


async def log_createpost_published(user, chat_id: int, kind: str) -> None:
    """A Create Post session published a post into a channel successfully.

    Every publication is its own event — several posts by the same user
    to the same channel are each logged.
    """

    text = (
        "📝 <b>CREATE POST PUBLISHED</b>\n\n"
        f"👤 <b>By:</b> {_build_mention(user)}\n"
        f"🆔 <b>User ID:</b> <code>{_format_user_id(user)}</code>\n"
        f"📌 <b>Channel ID:</b> <code>{int(chat_id)}</code>\n"
        f"🗂 <b>Post type:</b> {_escape(kind)}\n\n"
        f"📊 <b>Status:</b> ✅ SENT\n\n"
        f"🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    await send_log(text)


async def log_operation_failed(
    title: str, *, chat_id: Optional[int] = None,
    message_id: Optional[int] = None, detail: str = "",
    event_id: Optional[Tuple] = None,
    event_tag: Optional[str] = None,
) -> Tuple[str, Optional[int]]:
    """A failed operation (archive failure, verification failure, ...).

    Routed through the centralized logger so operators see failures in
    the logger group without each subsystem talking to Telegram itself.

    ``event_id`` is only for callers with a RELIABLE event-specific
    identifier (e.g. the MongoDB event key of the audit record); calls
    without one are always logged.

    ``event_tag`` embeds the caller's persistent ``EK:<tag>`` line in
    the message, so a delivered alert can always be tied back to its
    MongoDB delivery record by EXACT token match.

    Returns ``(outcome, message_id)`` from the transport layer:
    ``OUTCOME_SENT`` with the delivered Telegram message id, or
    ``OUTCOME_FAILED`` / ``OUTCOME_UNCERTAIN`` with None — callers that
    track delivery state in MongoDB use this to mark a record "sent"
    ONLY after Telegram accepted the message.  Never raises.
    """
    text = (
        f"🔴 <b>{_escape(title)}</b>\n\n"
        + (f"🆔 <b>Channel ID:</b> <code>{int(chat_id)}</code>\n" if chat_id else "")
        + (f"💬 <b>Post ID:</b> <code>{int(message_id)}</code>\n" if message_id else "")
        + (f"\n📝 <b>Detail:</b> {_escape(detail)}\n" if detail else "")
        + f"\n🕐 <b>Time:</b> <code>{_format_timestamp()}</code>"
    )
    if event_tag:
        text = append_event_tag(text, event_tag)
    if event_id is not None and not _claim_event(event_id):
        return OUTCOME_FAILED, None  # same actual event, already delivered
    delivered = False
    try:
        outcome, sent_id = await _send_with_outcome(text)
        delivered = outcome == OUTCOME_SENT
        return outcome, sent_id
    finally:
        if event_id is not None and not delivered:
            _release_event(event_id)


# ---------------------------------------------------------------------------
# Post-deletion log — the unified "🗑️ POST DELETED" event
# ---------------------------------------------------------------------------
#
# deletion_audit.mark_deletion_result() owns the deletion outcome and the
# persistent audit record; it claims the event (one log per post per
# outcome) and then calls send_post_deletion_log() below.  The record
# provides: chat_id, channel_title, message_id, reason, source, details,
# content_text (preview) and media_type — everything the log needs.

def _media_label(message) -> str:
    name = getattr(getattr(message, "media", None), "name", None)
    return name if isinstance(name, str) else "text"


def build_post_deletion_text(record: dict, *, kind: str, detail: str = "") -> str:
    """The screenshot-style deletion log body from an audit record."""

    header = _KIND_HEADERS.get(kind, _KIND_HEADERS["failed"])
    status = _KIND_STATUS.get(kind, "FAILED")

    chat_id = (record or {}).get("chat_id")
    message_id = (record or {}).get("message_id")
    title = (record or {}).get("channel_title")

    preview = str((record or {}).get("content_text") or "").strip()
    media_type = str((record or {}).get("media_type") or "text")
    if not preview:
        preview = f"[{media_type}]" if media_type and media_type != "text" else "[media post]"

    audit_link = (record or {}).get("original_link")
    audit_msg_id = (record or {}).get("audit_message_id")
    archive_chat = (record or {}).get("audit_channel_id")

    lines = [
        header,
        "",
        f"📌 <b>Channel:</b> {_escape(title) if title else 'Not Available'}",
        f"🆔 <b>Channel ID:</b> <code>{chat_id}</code>",
        f"💬 <b>Post ID:</b> <code>{message_id}</code>",
        "",
        f"<b>Reason:</b> {reason_line(record)}",
        f"🔧 <b>Feature:</b> {_source_label((record or {}).get('source'))}",
        "",
        f"<b>Preview:</b> {_escape(_short_preview(preview))}",
        "",
        f"📊 <b>Status:</b> {status}",
    ]
    if detail:
        lines.append(f"📝 <b>Detail:</b> {_escape(detail)}")
    if audit_msg_id and archive_chat:
        # link built by deletion_audit for the forwarded archive copy
        link = (record or {}).get("archive_link")
        if link:
            lines.append(f'\n📎 <a href="{html.escape(str(link), quote=True)}">View archived copy</a>')
    elif audit_link:
        lines.append(
            f'\n🔗 <a href="{html.escape(str(audit_link), quote=True)}">Original post link</a>'
        )
    lines.append(f"\n🕐 <b>Time:</b> <code>{_format_timestamp()}</code>")
    return "\n".join(lines)


async def send_post_deletion_log(
    record: dict, *, kind: str, event_key: str, detail: str = "",
) -> Tuple[str, Optional[int]]:
    """Send ONE post-deletion log for ``record`` — transport only.

    Formatting + the Telegram send with an honest, never-raising
    outcome.  All persistent delivery state (the ``pending`` /
    ``sending`` / ``sent`` / ``uncertain`` record, atomic claims and
    recovery) is owned by ``utils/deletion_audit.py`` — MongoDB is the
    source of truth, this function touches no database.

    Returns ``(outcome, message_id)``:

    * ``(OUTCOME_SENT, telegram_message_id)`` — Telegram accepted the
      message (this is the id of the delivered log in the logger group).
    * ``(OUTCOME_FAILED, None)`` — Telegram definitely did NOT deliver
      (RPCError / FloodWait it refused to absorb); the caller may retry.
    * ``(OUTCOME_UNCERTAIN, None)`` — unknown outcome; the message MAY
      have been delivered.  The caller records that honestly and lets
      the startup recovery decide.

    Every delivered message carries the event's persistent ``EK:<tag>``
    line so it can always be tied back to its delivery record.  Never
    raises.
    """
    text = build_post_deletion_text(record, kind=kind, detail=detail)
    text = append_event_tag(text, deletion_log_event_tag(event_key, kind))
    try:
        return await _send_with_outcome(text)
    except Exception as exc:  # absolute backstop — the caller must not crash
        log.warning(
            "Post-deletion log transport failure for %s: %s: %s",
            event_key, type(exc).__name__, exc,
        )
        return OUTCOME_UNCERTAIN, None


# ---------------------------------------------------------------------------
# Delivery state lives in MongoDB — see utils/deletion_audit.py
# ---------------------------------------------------------------------------
#
# This module is transport + formatting only.  Whether a deletion log or
# archive-failure alert still needs to be sent is tracked in the
# ``deletion_log_delivery`` collection (``pending`` / ``sending`` /
# ``sent`` / ``uncertain``), claimed atomically by the workers and
# recovered from MongoDB alone — by the startup sweep AND the periodic
# retry worker (batched, with per-record exponential backoff) — Telegram
# history is never scanned.  The ``EK:<tag>`` line embedded in every
# message keeps each delivered log identifiable as its exact event.

def get_logger(name: str):
    """Back-compat shim: the plain application logger (log.txt / console).

    The old placeholder exposed this; keep it so any stray import keeps
    working. Telegram-side logging must use the helpers above.
    """
    return LOGGER(name)
