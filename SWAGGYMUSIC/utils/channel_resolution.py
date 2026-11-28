"""Shared channel-input resolution — ONE mechanism for every feature.

=====================================================================
 WHY THIS MODULE EXISTS
=====================================================================

Every channel-specific feature (/setgap, /setdelay, /setlink,
/setforward, /setanonymous, /bannedwords, /approvalmode, /createpost,
/Schedule …) used to ask for a raw numeric Channel ID and parse it
with ``int(raw)``.  Users, however, naturally send the channel's
@username, a t.me link, or a private invite link — and every one of
those is a valid way to name a channel on Telegram.

This module is the SINGLE parser/resolver all of those wizards share:

  * ``parse_channel_input``      — pure, no I/O: recognise the input
    form (numeric ID / @username / bare username / public t.me link /
    private or approval invite link) and normalise it;
  * ``resolve_channel_input``    — turn that into a real channel id via
    the Telegram API (only when needed) and return ``(chat_id, chat)``;
  * ``CHANNEL_INPUT_PROMPT``     — the exact UI text every wizard shows
    when it asks for a channel;
  * ``ChannelResolutionError``   — one failure type carrying a clear,
    user-facing HTML message (``.user_message``) and a machine code
    (``.code``) for logging.

=====================================================================
 WHAT EACH INPUT FORM BECOMES
=====================================================================

  * ``-1001234567890``                → kind ``id``       (int; NO API
    call here — the feature's own ``verify_channel_admin`` /
    ``_check_channel`` already checks bot access + permissions and
    keeps its existing behaviour and messages)
  * ``@channelname`` / ``channelname``→ kind ``username`` →
    ``app.get_chat("@channelname")`` (Telegram resolves public
    usernames; a private/inexistent one raises and is mapped to a
    clear error)
  * ``https://t.me/channelname``      → kind ``username`` (public link)
  * ``https://t.me/+invitecode``      → kind ``invite``  → kurigram's
    ``get_chat`` runs ``messages.checkChatInvite`` for us:
      - the bot is ALREADY a member  → ``ChatInviteAlready`` → the real
        channel is returned and the workflow continues;
      - the bot is NOT a member      → Telegram answers with invite
        metadata only (a *preview* chat with NO id) — Telegram does
        not reveal the channel behind an invite link to non-members,
        and a bot cannot join a private channel by itself.  That is an
        API limitation, reported honestly: the user is told to add the
        bot as an admin first and then send the @username, link or ID.
  * ``https://t.me/joinchat/…``       → legacy private invite (same
    handling)
  * approval invites (``+`` links of channels with join requests) are
    the same object on the API — resolvable exactly when the bot is
    already a member; never claimed otherwise.

Invalid / expired / inaccessible inputs each get their own message
(``INVITE_HASH_INVALID`` / ``INVITE_HASH_EXPIRED`` / ``ChannelPrivate``
/ ``PeerIdInvalid`` / preview-without-id …), so the user always knows
what went wrong and what to do next.

Feature-specific logic (settings, verification, menus, logging) is
NOT touched: a feature calls ``resolve_channel_input`` in place of its
old ``int(raw)`` line and otherwise continues its original workflow.
"""

import re
from typing import Any, Optional, Tuple

from SWAGGYMUSIC.logging import LOGGER

log = LOGGER(__name__)


# ---------------------------------------------------------------------------
# The prompt shown wherever a channel is requested (exact UI text)
# ---------------------------------------------------------------------------

CHANNEL_INPUT_PROMPT = (
    "➕ ᴀᴅᴅ ᴄʜᴀɴɴᴇʟ\n\n"
    "sᴇɴᴅ ᴛʜᴇ ᴄʜᴀɴɴᴇʟ — ɪᴛs @username, ᴀ ᴛ.ᴍᴇ ʟɪɴᴋ, ᴏʀ ɪᴛs ɪᴅ "
    "(ʟɪᴋᴇ -1001234567890).\n\n"
    "ᴘᴜʙʟɪᴄ, ᴘʀɪᴠᴀᴛᴇ ᴀɴᴅ ᴀᴘᴘʀᴏᴠᴀʟ ɪɴᴠɪᴛᴇ ʟɪɴᴋꜱ ᴀʀᴇ ꜱᴜᴘᴘᴏʀᴛᴇᴅ.\n\n"
    "ᴍᴀᴋᴇ ꜱᴜʀᴇ ᴛʜᴇ ʙᴏᴛ ɪꜱ ᴀɴ ᴀᴅᴍɪɴ ɪɴ ᴛʜᴇ ᴄʜᴀɴɴᴇʟ.\n\n"
    "sᴇɴᴅ /cancel ᴛᴏ ᴄᴀɴᴄᴇʟ."
)


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ChannelResolutionError(Exception):
    """A channel input could not be parsed or resolved.

    ``user_message`` is a ready-to-send HTML string for the user;
    ``code`` is a stable machine identifier ("invalid_input",
    "unreachable", "invite_unresolvable", "invite_expired",
    "invite_invalid") for logging and tests.  Never carries secrets.
    """

    def __init__(self, user_message: str, *, code: str = "invalid_input",
                 cause: Optional[BaseException] = None) -> None:
        super().__init__(user_message)
        self.user_message = user_message
        self.code = code
        self.cause = cause


_INVALID_INPUT_TEXT = (
    "❌ <b>I couldn't recognise that as a channel.</b>\n\n"
    "Send the channel's <b>@username</b>, a <b>t.me link</b> "
    "(public or private invite), or its numeric <b>ID</b> — "
    "for example <code>-1001234567890</code>."
)

_UNREACHABLE_TEXT = (
    "❌ <b>I can't reach that channel.</b>\n\n"
    "Make sure I'm added there as an <b>admin</b>. If the channel is "
    "private, add me first, then send its @username, t.me link or ID.\n\n"
    "<code>{detail}</code>"
)

_INVITE_UNRESOLVABLE_TEXT = (
    "❌ <b>I can't resolve this private invite link.</b>\n\n"
    "Telegram only reveals which channel a private invite link belongs "
    "to <b>after I've been added to that channel</b> — bots cannot join "
    "a private channel by themselves.\n\n"
    "Add me to the channel as an <b>admin</b> first, then send its "
    "<b>@username</b>, its <b>t.me link</b> or its numeric <b>ID</b>."
)

_INVITE_EXPIRED_TEXT = (
    "❌ <b>This invite link has expired or was revoked.</b>\n\n"
    "Ask the channel owner for a fresh link, or send the channel's "
    "<b>@username</b> or numeric <b>ID</b>."
)

_INVITE_INVALID_TEXT = (
    "❌ <b>This invite link is invalid.</b>\n\n"
    "Check the link and send it again, or use the channel's "
    "<b>@username</b> or numeric <b>ID</b>."
)

_USERNAME_INVALID_TEXT = (
    "❌ <b>That username doesn't look valid.</b>\n\n"
    "Telegram usernames use 4-64 letters, digits or underscores — "
    "for example <code>@my_channel</code>."
)


# ---------------------------------------------------------------------------
# Pure parsing (no Telegram I/O)
# ---------------------------------------------------------------------------

# A Telegram channel/user id: an ASCII integer, optional minus, ≤15 digits.
# (int() alone would also accept "1_000", "+5", "٣٣٣" and other spellings
# that are NOT Telegram ids — mirroring utils/user_resolution.py's strict
# parse.)
_ID_RE = re.compile(r"^-?[0-9]{1,15}$")

# A Telegram username handle: 4-64 letters/digits/underscores.
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_]{4,64}$")

# An invite-link hash: letters, digits, "-" and "_" (Telegram's own
# INVITE_LINK_RE accepts exactly ``[\w-]+``).
_INVITE_HASH_RE = re.compile(r"^[A-Za-z0-9_-]{1,256}$")

# t.me / telegram.me (also t.org / t.dog) links.
_TME_LINK_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?(?:t(?:elegram)?\.(?:me|org|dog))/(\S+)$",
    re.IGNORECASE,
)

# Parsed input: ("id", int) | ("username", "@name") | ("invite", link).
_Parsed = Tuple[str, Any]


def _invalid(code: str = "invalid_input") -> ChannelResolutionError:
    return ChannelResolutionError(_INVALID_INPUT_TEXT, code=code)


def parse_channel_input(raw: Any) -> _Parsed:
    """Classify raw user input WITHOUT any Telegram call.  Raises
    :class:`ChannelResolutionError` (code ``invalid_input``) when the
    text cannot name a channel at all.

    Accepted forms — the ones named in ``CHANNEL_INPUT_PROMPT``:

    * numeric id:            ``-1001234567890`` (also plain ``-123…``)
    * @username:             ``@channelname``
    * bare username:         ``channelname`` (treated as @username)
    * public link:           ``https://t.me/channelname`` (scheme and
      ``www.`` optional; ``telegram.me``, ``t.me/s/name`` previews too)
    * private invite:        ``https://t.me/+hash`` or the legacy
      ``https://t.me/joinchat/hash`` (scheme optional, ``+hash`` alone
      accepted) — private AND approval-invite links share this form
    """
    text = str(raw or "").strip()
    if not text:
        raise _invalid()

    # Numeric Channel ID — no API call needed to know it.
    if _ID_RE.match(text):
        return "id", int(text)

    # @username
    if text.startswith("@"):
        handle = text[1:]
        if not _USERNAME_RE.match(handle):
            raise ChannelResolutionError(
                _USERNAME_INVALID_TEXT, code="invalid_input",
            )
        return "username", "@" + handle

    # t.me links (public + private invite + approval invite)
    link_match = _TME_LINK_RE.match(text)
    if link_match:
        # Cut any query string / fragment before inspecting the path.
        path = re.split(r"[?#]", link_match.group(1), maxsplit=1)[0]
        path = path.rstrip("/")
        lowered = path.lower()
        if lowered.startswith("+"):
            invite_hash = path[1:]
            if not _INVITE_HASH_RE.match(invite_hash):
                raise _invalid()
            return "invite", f"https://t.me/{path}"
        if lowered.startswith("joinchat/"):
            invite_hash = path[len("joinchat/"):]
            if not _INVITE_HASH_RE.match(invite_hash) or not invite_hash:
                raise _invalid()
            return "invite", f"https://t.me/joinchat/{invite_hash}"
        # t.me/s/<name> is the web preview of a public channel.
        if lowered.startswith("s/"):
            path = path[2:]
        if "/" in path or not _USERNAME_RE.match(path):
            raise _invalid()
        return "username", "@" + path

    # A bare "+hash" — the share-sheet form of an invite link.
    if text.startswith("+"):
        invite_hash = text[1:]
        if not _INVITE_HASH_RE.match(invite_hash):
            raise _invalid()
        return "invite", f"https://t.me/{text}"

    # A bare word — most likely the username typed without the @.
    if _USERNAME_RE.match(text):
        return "username", "@" + text

    raise _invalid()


# ---------------------------------------------------------------------------
# Resolution (Telegram I/O only where an id cannot be known locally)
# ---------------------------------------------------------------------------

def _unreachable(exc: Exception) -> ChannelResolutionError:
    return ChannelResolutionError(
        _UNREACHABLE_TEXT.format(detail=type(exc).__name__),
        code="unreachable",
        cause=exc,
    )


def _invite_failure(exc: Exception) -> ChannelResolutionError:
    name = type(exc).__name__.upper()
    if "EXPIRED" in name:
        return ChannelResolutionError(
            _INVITE_EXPIRED_TEXT, code="invite_expired", cause=exc,
        )
    if "INVALID" in name:
        return ChannelResolutionError(
            _INVITE_INVALID_TEXT, code="invite_invalid", cause=exc,
        )
    return _unreachable(exc)


async def resolve_channel_input(
    raw: Any,
    *,
    app_client: Any = None,
) -> Tuple[int, Any]:
    """Parse the user's input and resolve it to a real channel id.

    Returns ``(chat_id, chat)``.  ``chat`` is the Telegram Chat object
    when one was fetched during resolution (``None`` for a pure numeric
    id, where the feature's own verification fetches it as before).

    Raises :class:`ChannelResolutionError` with a clear, user-facing
    ``.user_message`` when the input cannot be understood, is
    unreachable, or is a private/approval invite Telegram refuses to
    resolve for a non-member bot.  NEVER claims that every private
    invite can be resolved — the API limitation is stated as-is.

    Access + admin-permission checks stay with each feature's existing
    ``verify_channel_admin`` (unchanged messages and behaviour).
    """
    kind, value = parse_channel_input(raw)  # may raise invalid_input

    if kind == "id":
        # Nothing to resolve: the numeric id IS the channel id.  Access
        # and permission checks belong to the feature's own verifier.
        return int(value), None

    if app_client is None:
        from SWAGGYMUSIC import app as app_client  # default client

    try:
        chat = await app_client.get_chat(value)
    except Exception as exc:
        if kind == "invite":
            raise _invite_failure(exc)
        raise _unreachable(exc)

    chat_id = getattr(chat, "id", None)
    if chat_id is None:
        # Kurigram's get_chat returns an id-less PREVIEW chat when
        # checkChatInvite answered with invite metadata only — i.e. the
        # bot is NOT a member of that private channel.  Telegram will
        # not tell a non-member which channel sits behind the link, and
        # bots cannot self-join private channels: report the limitation
        # honestly instead of guessing.
        raise ChannelResolutionError(
            _INVITE_UNRESOLVABLE_TEXT, code="invite_unresolvable",
        )
    try:
        chat_id = int(chat_id)
    except (TypeError, ValueError):
        # A "chat" without an integer id cannot be addressed — treat it
        # as unreachable rather than leaking a raw string into the
        # feature workflows (every feature stores an int chat id).
        raise ChannelResolutionError(
            _UNREACHABLE_TEXT.format(detail="chat id not resolvable"),
            code="unreachable",
        )

    return chat_id, chat


__all__ = [
    "CHANNEL_INPUT_PROMPT",
    "ChannelResolutionError",
    "parse_channel_input",
    "resolve_channel_input",
]
