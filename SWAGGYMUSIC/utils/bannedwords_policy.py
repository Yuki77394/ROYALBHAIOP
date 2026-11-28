"""Banned-words policy shared by the /bannedwords filter, its wizard and tests.

Policy (applied by ``plugins/misc/bannedwords.py`` when the Banned Words
feature is enabled for a channel):

* DELETE posts whose text or caption contains a configured banned word or
  phrase:
    - matching is CASE-INSENSITIVE;
    - single words and multi-word phrases are both supported;
    - matching is whole-word: a banned word never matches inside a longer
      word (banned ``ass`` does not match ``class``; banned ``cross`` does
      not match ``crossed``). A phrase boundary check is applied per side,
      so punctuation around the phrase is fine;
    - the visible text is scanned, which already includes what Telegram
      entities render (a TEXT_LINK's display text and a URL entity's text
      are part of the message text).
* Media albums: Telegram delivers every item of a media group as a
  separate update (same ``media_group_id``). While the feature is
  enabled, clean items are remembered in ``_ALBUM_ITEMS`` (bounded,
  in-process). The first item found to carry a banned word condemns the
  whole album: the verdict is recorded BEFORE any Telegram call (so a
  concurrent item always sees it), the current item and every item
  already delivered are deleted, and every later item of the same
  ``media_group_id`` is deleted on arrival. The condemned album id is
  also persisted in the channel's settings document
  (``banned_blocked_albums``, bounded) so a restart between item
  deliveries cannot leave a partial album alive — the same mechanism the
  link filter and the /setgap engine use.
* Edited posts are judged like new posts (an edit cannot smuggle a
  banned word in).

Coordination with the other filters and /setgap — the banned-words
filter runs in handler group -24 (after the link -40 / anonymous -30 /
forward -25 filters, before Approval Mode -23, the channel commands
-20..-16 and the gap/delay engine 0). A post it rejects must never
become an accepted /setgap reference or a /setdelay job, whatever
happens to its deletion:

1. ``mark_rejected`` records the verdict in-process (bounded LRU) before
   any Telegram call; the later filters and the gap engine check
   ``was_rejected`` and skip the post (same pattern as link_policy,
   anonymous_policy and forward_policy).
2. Deletions that Telegram refuses with FloodWait are queued for the
   persistent deletion worker (``autodelete_jobs``, reason
   ``BANNED_WORD_RETRY``) instead of being abandoned.

Detection is a pure function (``find_banned_words``) — no I/O, never
raises.
"""
import re
from collections import OrderedDict
from typing import List, Optional, Tuple

from pyrogram.enums import MessageServiceType

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
# Items remembered per album (Telegram media groups hold 2..10 items).
_ALBUM_ITEMS_MAX = 32
# Items seen per album: (chat_id, media_group_id) -> {msg_id: None}
_ALBUM_ITEMS: "OrderedDict[Tuple[int, int], OrderedDict[int, None]]" = OrderedDict()
# Albums condemned by the banned-words filter.
_BLOCKED_ALBUMS: "OrderedDict[Tuple[int, int], None]" = OrderedDict()


def _touch(book: OrderedDict, key, max_entries: int) -> None:
    book[key] = None
    book.move_to_end(key)
    while len(book) > max_entries:
        book.popitem(last=False)


def mark_rejected(chat_id: int, message_id: int) -> None:
    """Remember (in this process) that the banned-words filter rejected a post."""
    _touch(_REJECTED, (int(chat_id), int(message_id)), _REJECTED_MAX)


def was_rejected(chat_id: int, message_id: int) -> bool:
    try:
        return (int(chat_id), int(message_id)) in _REJECTED
    except (TypeError, ValueError):
        return False


def remember_album_item(chat_id: int, media_group_id: int, message_id: int) -> None:
    """Track a delivered (clean) item of a media album."""
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


def album_is_blocked(chat_id: int, media_group_id: int) -> bool:
    try:
        return (int(chat_id), int(media_group_id)) in _BLOCKED_ALBUMS
    except (TypeError, ValueError):
        return False


def take_album_items(chat_id: int, media_group_id: int) -> List[int]:
    """Return the items delivered so far for one album and forget them."""
    key = (int(chat_id), int(media_group_id))
    items = _ALBUM_ITEMS.pop(key, None)
    return list(items) if items else []


def forget_album(chat_id: int, media_group_id: int) -> None:
    """Drop all in-process state for one album (verdict + items)."""
    _BLOCKED_ALBUMS.pop((int(chat_id), int(media_group_id)), None)
    _ALBUM_ITEMS.pop((int(chat_id), int(media_group_id)), None)


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
    """True if ``setting``'s persisted ``banned_blocked_albums`` lists the id."""
    blocked = (setting or {}).get("banned_blocked_albums") or []
    for value in blocked:
        try:
            if int(value) == int(media_group_id):
                return True
        except (TypeError, ValueError):
            continue
    return False


# ---------------------------------------------------------------------------
# Detection — whole-word, case-insensitive phrase matching.
# ---------------------------------------------------------------------------

_WORD_CHAR_CLASS = r"\w"


def _phrase_pattern(word: str) -> "re.Pattern":
    """A case-insensitive whole-phrase pattern for one banned word/phrase.

    A side that begins/ends with a word character gets a boundary
    assertion so partial-word matches are impossible; a side that
    begins/ends with punctuation (e.g. inside ``f***``) matches anywhere.
    """
    escaped = re.escape(word)
    head = (
        rf"(?<!{_WORD_CHAR_CLASS})" if word[:1].isalnum() or word[:1] == "_" else ""
    )
    tail = (
        rf"(?!{_WORD_CHAR_CLASS})" if word[-1:].isalnum() or word[-1:] == "_" else ""
    )
    return re.compile(head + escaped + tail, re.IGNORECASE)


def find_banned_words(message, words: List[str]) -> List[str]:
    """Every configured banned word present in ``message`` (deduplicated).

    Scans the message text and caption (the visible text already covers
    what entities render). Pure function: no I/O, never raises. Returns
    the MATCHED words in the order they appear in the configuration.
    """
    try:
        if isinstance(getattr(message, "service", None), MessageServiceType):
            return []
        text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
        if not isinstance(text, str):
            text = str(text)
        if not text or not words:
            return []
        lowered = text.lower()
        hits: List[str] = []
        for word in words or []:
            if not word:
                continue
            pattern = _phrase_pattern(word)
            if pattern.search(lowered):
                hits.append(str(word))
        return hits
    except Exception as exc:  # never let detection break the dispatcher
        log.warning("bannedwords_policy.find_banned_words failed: %s: %s",
                    type(exc).__name__, exc)
        return []


def message_has_banned_word(message, words: List[str]) -> bool:
    """True if the message carries any configured banned word (pure check)."""
    return bool(find_banned_words(message, words))


def is_bot_command(message) -> bool:
    """True if the message is a bot command — commands are never content.

    Telegram marks commands with a BOT_COMMAND entity; a leading "/" is
    accepted as a fallback for forks that drop the entity. Content
    filters MUST skip command messages (e.g. ``/bannedwords add spam``
    contains the word ``spam`` and must still reach the wizard).
    """
    try:
        if isinstance(getattr(message, "service", None), MessageServiceType):
            return False
        entities = list(getattr(message, "entities", None) or [])
        entities += list(getattr(message, "caption_entities", None) or [])
        for entity in entities:
            from pyrogram.enums import MessageEntityType

            if getattr(entity, "type", None) is MessageEntityType.BOT_COMMAND:
                return True
        text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
        return isinstance(text, str) and text.strip().startswith("/")
    except Exception:
        return False
