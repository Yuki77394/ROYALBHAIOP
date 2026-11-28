"""Approval-mode policy shared by the /approvalmode filter, its wizard and tests.

APPROVAL MODE (per channel): every channel post must carry at least one
APPROVED username. Applied by ``plugins/misc/approvalmode.py`` when the
feature is enabled for a channel:

* KEEP a post that mentions at least one approved username (text,
  captions and Telegram entities are all inspected).
* KEEP + WARN (logger group) a post that mentions approved AND
  unapproved usernames — the mixed case is deliberately never deleted.
* DELETE a post that mentions NO approved username (including posts
  without any username at all).
* Feature disabled → no restriction at all.

Username detection
---------------------------------------------------------------------
* MENTION entities (``@user``) — the authoritative source. The entity
  span is decoded through UTF-16 (Telegram entity offsets are UTF-16
  code units) so emoji before the mention cannot corrupt the slice.
* TEXT_MENTION entities (an inline user mention without a visible
  handle) — the mentioned user's ``username`` is used when present; a
  mention without any username counts as an UNAPPROVED mention.
* A conservative text scan for ``@handle`` tokens Telegram did not
  entity-tag (forwarded captions, forks, ...).
* NO PARTIAL MATCHES: the complete handle is extracted (Telegram
  usernames are 3-64 chars of letters/digits/underscore) and compared by
  exact, case-insensitive equality against the approved list — ``john``
  never matches ``johnny``.

Media albums
---------------------------------------------------------------------
Telegram delivers every item of a media group as a separate update, in
any order, possibly concurrently. The album is ONE post and is judged
ONCE: the FIRST item processed decides the verdict for the whole album
(synchronous check-and-set on an in-process map — atomic under asyncio).
An item whose caption carries an approved username approves the album
(all items kept, later items included — the mixed rule applies at album
level, the deciding item's own caption is what is judged); an item
without an approved username condemns the album whole: the current item
and every item already delivered are deleted, and every later item of
the same ``media_group_id`` is deleted on arrival. A condemned album id
is persisted in the channel's settings document
(``approval_blocked_albums``, bounded) so a restart between item
deliveries cannot leave a partial album alive.
"""
from collections import OrderedDict
from typing import List, Optional, Set, Tuple

from pyrogram.enums import MessageEntityType, MessageServiceType

from SWAGGYMUSIC.logging import LOGGER

log = LOGGER(__name__)

# ---------------------------------------------------------------------------
# In-process verdicts and album registries (bounded; oldest dropped first)
# ---------------------------------------------------------------------------

# How many per-post verdicts are kept (same scale as the other policies).
_REJECTED_MAX = 10_000
_REJECTED: "OrderedDict[Tuple[int, int], None]" = OrderedDict()

# How many (chat, media_group_id) albums are tracked at once.
_ALBUM_TRACK_MAX = 512
# Items remembered per album (for sibling cleanup of a condemned album).
_ALBUM_ITEMS_MAX = 32
_ALBUM_ITEMS: "OrderedDict[Tuple[int, int], OrderedDict[int, None]]" = OrderedDict()
# Albums condemned by Approval Mode: (chat_id, media_group_id) -> None
_BLOCKED_ALBUMS: "OrderedDict[Tuple[int, int], None]" = OrderedDict()
# Albums APPROVED by Approval Mode (first item had an approved username):
# (chat_id, media_group_id) -> None. In-process only: approved albums need
# no persistence — items are kept by default.
_APPROVED_ALBUMS: "OrderedDict[Tuple[int, int], None]" = OrderedDict()


def _touch(book: OrderedDict, key, max_entries: int) -> None:
    book[key] = None
    book.move_to_end(key)
    while len(book) > max_entries:
        book.popitem(last=False)


def mark_rejected(chat_id: int, message_id: int) -> None:
    """Remember (in this process) that Approval Mode rejected a post."""
    _touch(_REJECTED, (int(chat_id), int(message_id)), _REJECTED_MAX)


def was_rejected(chat_id: int, message_id: int) -> bool:
    try:
        return (int(chat_id), int(message_id)) in _REJECTED
    except (TypeError, ValueError):
        return False


def remember_album_item(chat_id: int, media_group_id: int, message_id: int) -> None:
    """Track a delivered item of a media album (for sibling cleanup)."""
    key = (int(chat_id), int(media_group_id))
    items = _ALBUM_ITEMS.get(key)
    if items is None:
        items = OrderedDict()
        _touch(_ALBUM_ITEMS, key, _ALBUM_TRACK_MAX)
        _ALBUM_ITEMS[key] = items
    _touch(items, int(message_id), _ALBUM_ITEMS_MAX)


def block_album(chat_id: int, media_group_id: int) -> None:
    """Condemn a whole album (verdict recorded before any Telegram call)."""
    _touch(_BLOCKED_ALBUMS, (int(chat_id), int(media_group_id)), _ALBUM_TRACK_MAX)
    _APPROVED_ALBUMS.pop((int(chat_id), int(media_group_id)), None)


def album_is_blocked(chat_id: int, media_group_id: int) -> bool:
    try:
        return (int(chat_id), int(media_group_id)) in _BLOCKED_ALBUMS
    except (TypeError, ValueError):
        return False


def approve_album(chat_id: int, media_group_id: int) -> None:
    """Approve a whole album (first item carried an approved username).

    Synchronous check-and-set: returns True when THIS call decided the
    album (no verdict existed yet), False when a verdict already exists
    (the caller must follow the stored one).
    """
    key = (int(chat_id), int(media_group_id))
    if key in _APPROVED_ALBUMS:
        return False
    if key in _BLOCKED_ALBUMS:
        return False
    _touch(_APPROVED_ALBUMS, key, _ALBUM_TRACK_MAX)
    return True


def album_is_approved(chat_id: int, media_group_id: int) -> bool:
    try:
        return (int(chat_id), int(media_group_id)) in _APPROVED_ALBUMS
    except (TypeError, ValueError):
        return False


def album_has_verdict(chat_id: int, media_group_id: int) -> bool:
    return album_is_approved(chat_id, media_group_id) or album_is_blocked(
        chat_id, media_group_id,
    )


def take_album_items(chat_id: int, media_group_id: int) -> List[int]:
    """Return the items delivered so far for one album and forget them."""
    key = (int(chat_id), int(media_group_id))
    items = _ALBUM_ITEMS.pop(key, None)
    return list(items) if items else []


def forget_album(chat_id: int, media_group_id: int) -> None:
    """Drop all in-process state for one album."""
    key = (int(chat_id), int(media_group_id))
    _BLOCKED_ALBUMS.pop(key, None)
    _APPROVED_ALBUMS.pop(key, None)
    _ALBUM_ITEMS.pop(key, None)


def media_group_id(message) -> Optional[int]:
    """The message's media-group id (albums), as a plain int, or None."""
    value = getattr(message, "media_group_id", None)
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def persisted_blocked_album(setting: Optional[dict], media_group_id: int) -> bool:
    """True if ``setting``'s persisted ``approval_blocked_albums`` lists the id."""
    blocked = (setting or {}).get("approval_blocked_albums") or []
    for value in blocked:
        try:
            if int(value) == int(media_group_id):
                return True
        except (TypeError, ValueError):
            continue
    return False


# ---------------------------------------------------------------------------
# Username extraction — entities first (authoritative), then a text scan.
# ---------------------------------------------------------------------------

# A complete Telegram handle inside text: 3-64 of letters/digits/underscore,
# never preceded or followed by a handle character (no partial matches).
_BARE_HANDLE_RE = r"(?<![A-Za-z0-9_@])@([A-Za-z0-9_]{3,64})(?![A-Za-z0-9_])"


def _entity_span(text: str, entity) -> str:
    """The text an entity spans (UTF-16-safe slice; "" when malformed)."""
    try:
        data = text.encode("utf-16-le")
        start = int(entity.offset) * 2
        end = start + int(entity.length) * 2
        if start < 0 or end > len(data) or start > end:
            return ""
        return data[start:end].decode("utf-16-le", errors="replace")
    except Exception:
        return ""


def _handle_of(value: Optional[str]) -> Optional[str]:
    """``@User``/``user`` → lowercase ``user``; None for invalid shapes."""
    if not value:
        return None
    text = str(value).strip()
    if text.startswith("@"):
        text = text[1:]
    if not text or len(text) < 3 or len(text) > 64:
        return None
    if not all(ch.isalnum() or ch == "_" for ch in text):
        return None
    return text.lower()


def extract_usernames(message) -> Set[str]:
    """Every username the message mentions (lowercase, without ``@``).

    Sources: MENTION entities (text + caption), TEXT_MENTION entities
    (the mentioned user's handle, when Telegram exposes one), and a
    conservative ``@handle`` scan of the raw text. Pure function; never
    raises.
    """
    try:
        if isinstance(getattr(message, "service", None), MessageServiceType):
            return set()
        text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
        if not isinstance(text, str):
            text = str(text)
        found: Set[str] = set()

        entities = list(getattr(message, "entities", None) or [])
        entities += list(getattr(message, "caption_entities", None) or [])
        for entity in entities:
            etype = getattr(entity, "type", None)
            if etype is MessageEntityType.MENTION:
                handle = _handle_of(_entity_span(text, entity))
                if handle:
                    found.add(handle)
            elif etype is MessageEntityType.TEXT_MENTION:
                user = getattr(entity, "user", None)
                handle = _handle_of(getattr(user, "username", None))
                if handle:
                    found.add(handle)
                else:
                    # An inline mention Telegram did not give a handle for:
                    # count it as an unapproved mention so the mixed-case
                    # warning still fires.
                    found.add("__nomention__")

        if text:
            for match in __import__("re").finditer(_BARE_HANDLE_RE, text):
                handle = _handle_of(match.group(1))
                if handle:
                    found.add(handle)
        return found
    except Exception as exc:  # never let detection break the dispatcher
        log.warning("approval_policy.extract_usernames failed: %s: %s",
                    type(exc).__name__, exc)
        return set()


def classify_post(message, approved: List[str]) -> Tuple[Set[str], Set[str]]:
    """``(approved_found, unapproved_found)`` for one message.

    The special ``__nomention__`` marker means "a mention without an
    exposed username" (TEXT_MENTION) and is always unapproved.
    """
    found = extract_usernames(message)
    approved_set = {
        _handle_of(name) for name in (approved or []) if _handle_of(name)
    }
    approved_set.discard(None)
    approved_found = found & approved_set
    unapproved_found = set(found) - approved_set
    return approved_found, unapproved_found


def is_bot_command(message) -> bool:
    """True if the message is a bot command — commands are never content."""
    try:
        if isinstance(getattr(message, "service", None), MessageServiceType):
            return False
        entities = list(getattr(message, "entities", None) or [])
        entities += list(getattr(message, "caption_entities", None) or [])
        for entity in entities:
            if getattr(entity, "type", None) is MessageEntityType.BOT_COMMAND:
                return True
        text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
        return isinstance(text, str) and text.strip().startswith("/")
    except Exception:
        return False
