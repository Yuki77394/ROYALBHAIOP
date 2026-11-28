"""Deletion audit archiver + deletion-outcome reporting.

Two independent responsibilities live here:

1. ARCHIVE (independent feature, unchanged behaviour): a deleted channel
   post is forwarded (not copied/re-uploaded) to the dedicated audit
   channel (``config.AUDIT_CHANNEL_ID``). MongoDB retains the audit
   record without a TTL. Forward delivery is best-effort and must never
   prevent the original moderation deletion. Ambiguous Telegram-send
   outcomes are marked "uncertain" rather than blindly retried, avoiding
   duplicate forwards after a crash.

2. LOGGER OUTPUT (replaced by the centralized logger): the old detail
   message this module used to send to LOGGER_ID (``_format_log``) and
   its edit-on-result flow were direct logger-group sends. They are now
   routed through ``SWAGGYMUSIC.utils.logger`` — the only module allowed
   to talk to the logger group — via the unified "POST DELETED" event.

MongoDB is the source of truth for deletion-log delivery state
---------------------------------------------------------------------
The ``deletion_log_delivery`` collection holds ONE delivery record per
(event_key, log kind) — created BEFORE any send — and MongoDB alone
decides whether a log still needs to be delivered.  Telegram history is
NEVER scanned, neither at runtime nor during startup recovery.  The
state machine per record:

    pending ──claim──▶ sending ──Telegram accepted──▶ sent
       ▲                  │
       │   definite       │ unknown outcome / owner died
       └────failure───────┴──────────────▶ uncertain
                          (uncertain is reclaimed by the startup sweep
                           or the periodic retry worker after a grace)

* ``pending``   — created before the first send; retryable by any worker.
* ``sending``   — one worker owns it via an ATOMIC claim
  (``find_one_and_update``), with a lease stamp; concurrent workers can
  never send the same event.
* ``sent``      — set ONLY after Telegram accepted the message, together
  with the returned logger-group message id.  A sent record is final:
  no retry, sweep or re-processed event ever sends it again.
* ``uncertain`` — the honest state for "Telegram MAY have delivered"
  (unknown exception mid-send, or a worker that died mid-claim).
  Runtime workers do not instantly re-send an uncertain record (that
  would duplicate a possibly-delivered log); the STARTUP sweep and the
  PERIODIC RETRY WORKER recover pending + uncertain + abandoned-sending
  records from MongoDB.

Batching + periodic retry
-------------------------
The recovery sweep processes candidates in batches of
``DELIVERY_BATCH_SIZE`` (200) and keeps fetching further batches until
NO eligible record is left — the batch size bounds per-pass work and
memory, never the total amount recovered; recovery never stops after
the first 200 records.  Pagination is by ``_id`` (each batch continues
strictly AFTER the last ``_id`` of the previous one), so the query
never grows a ``$nin`` exclusion list and a record that keeps failing
simply falls behind the cursor instead of blocking or looping the
sweep.  Recovery also does not depend on a restart:
``start_delivery_retry_worker()`` runs a periodic retry sweep every
``DELIVERY_RETRY_INTERVAL_SECONDS`` that re-delivers failed and
uncertain records, with an EXPONENTIAL PER-RECORD BACKOFF
(``next_attempt_at``: 30s, 60s, 120s … capped at 1 hour) so a record
that keeps failing is never hammered in a tight loop and Telegram is
not flooded with retry calls.  A stale "sending" record (its worker
died) is taken over safely: at startup regardless of its lease, at
runtime only after the lease expired.

Crash-after-send-before-DB-update (the honest limitation)
---------------------------------------------------------
``sendMessage`` has no idempotency key, so the window in which a
process dies AFTER Telegram accepted the message but BEFORE the success
was persisted cannot be resolved without inspecting Telegram history —
which recovery deliberately does not do.  That window is therefore
handled at-least-once: the record stays "sending", the startup sweep
treats it as uncertain and re-delivers it, which GUARANTEES the log is
not lost but may (rarely) produce a duplicate in the logger group.
Every message carries its exact ``EK:<tag>`` line so duplicates are
trivially identifiable.  Duplicates are always preferred over a
silently lost audit log; outside that crash window delivery is
exactly-once.

The legacy per-kind claim booleans (``deletion_log_<kind>_sent``) and
the ``log_status`` / ``log_attempted_at`` fields on the audit record are
no longer written by new code; old records are migrated into delivery
records at startup (legacy "sent" outcomes are preserved so an upgrade
never duplicates already-delivered logs).  Final outcomes are still
MIRRORED onto the audit record for operators and older readers.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple
import uuid as _uuid

import config
from pyrogram.errors import FloodWait, RPCError

from SWAGGYMUSIC.logging import LOGGER

log = LOGGER("SWAGGYMUSIC.deletion_audit")


# Human-visible title of the archive-failure alert (the alert's identity
# in the logger group is its EXACT ``EK:<tag>`` — see ``_send_failure_alert``);
# this phrase is what operators see, not what recovery matches on.
_ALERT_MARKER = "DELETION ARCHIVE NEEDS ATTENTION"

# ---------------------------------------------------------------------------
# Delivery state machine — MongoDB is the source of truth
# ---------------------------------------------------------------------------

# Delivery record states (``deletion_log_delivery.status``).
DELIVERY_PENDING = "pending"      # created, never sent — retryable
DELIVERY_SENDING = "sending"      # claimed by one worker (lease held)
DELIVERY_SENT = "sent"            # Telegram accepted; message id persisted
DELIVERY_UNCERTAIN = "uncertain"  # outcome unknown — recovered by sweeps

# Deletion-log kinds tracked by the machine + the archive-failure alert.
DELETION_LOG_KINDS = ("deleted", "pending_retry", "failed")
ALERT_LOG_KIND = "archive_alert"

# How long one worker's claim on a "sending" record stays valid.  The
# send path sleeps at most ~11s (FloodWait + retry); anything far beyond
# the lease belongs to a dead worker and may be taken over.
DELIVERY_LEASE_SECONDS = 120.0

# Runtime workers leave an explicitly-"uncertain" record alone for this
# long after the ambiguous outcome (an immediate retry could duplicate a
# log Telegram actually delivered).  The STARTUP sweep ignores the grace
# and recovers uncertain records regardless — after a restart there is
# no concurrent in-process owner to protect.
UNCERTAIN_RUNTIME_GRACE_SECONDS = 600.0

# How many delivery records ONE sweep pass fetches and processes at a
# time.  The sweep keeps fetching further batches until NO eligible
# record is left — the batch size bounds memory and per-pass MongoDB
# work, never the total number of records recovered (recovery NEVER
# stops after the first 200 records).
DELIVERY_BATCH_SIZE = 200

# Cadence of the periodic delivery-retry worker (see
# ``start_delivery_retry_worker``).  Recovery must not depend only on
# bot startup: failed / uncertain / stale-claimed records are re-swept
# from MongoDB on this interval while the bot keeps running.
DELIVERY_RETRY_INTERVAL_SECONDS = 300.0

# Exponential retry backoff for records that keep failing: after the
# Nth failed attempt the record is not touched again for
# ``min(BASE * 2^(N-1), MAX)`` seconds (30s → 60s → 120s → … → capped
# at 1 hour).  This keeps the retry sweeps away from tight loops and
# keeps the Telegram API call volume sane while still guaranteeing that
# every undelivered record keeps being retried.
DELIVERY_BACKOFF_BASE_SECONDS = 30.0
DELIVERY_BACKOFF_MAX_SECONDS = 3600.0
# Hard cap on the backoff exponent so ``2 ** exponent`` can never grow
# unboundedly for a record with a huge ``attempts`` counter.
_DELIVERY_BACKOFF_MAX_EXPONENT = 16

# Legacy per-kind claim fields (previous design).  Read ONLY by the
# migration that seeds delivery records from old audit records; new code
# never writes them.
_LOG_CLAIM_FIELDS = {
    "deleted": "deletion_log_deleted_sent",
    "pending_retry": "deletion_log_pending_retry_sent",
    "failed": "deletion_log_failed_sent",
}


def _worker_token(prefix: str) -> str:
    """A per-attempt worker identity for delivery-record claims."""
    return f"{prefix}:{_uuid.uuid4().hex[:12]}"


def _backoff_seconds(attempts: Optional[int]) -> float:
    """The retry cooldown after the Nth failed delivery attempt.

    Exponential with base ``DELIVERY_BACKOFF_BASE_SECONDS`` — 30s after
    the first failure, 60s after the second, 120s after the third, … —
    capped at ``DELIVERY_BACKOFF_MAX_SECONDS`` (1 hour).  The exponent
    is clamped so a record with an absurdly large ``attempts`` counter
    can never overflow.  Never raises.
    """
    try:
        n = max(0, int(attempts or 0))
    except Exception:
        n = 0
    exponent = min(max(n - 1, 0), _DELIVERY_BACKOFF_MAX_EXPONENT)
    return min(
        DELIVERY_BACKOFF_BASE_SECONDS * (2 ** exponent),
        DELIVERY_BACKOFF_MAX_SECONDS,
    )


def _event_key(chat_id: int, message_id: int) -> str:
    return f"{int(chat_id)}:{int(message_id)}"


def _message_link(chat, chat_id: int, message_id: int) -> str:
    username = getattr(chat, "username", None) if chat is not None else None
    if username:
        return f"https://t.me/{username}/{int(message_id)}"
    raw = str(int(chat_id))
    internal = raw[4:] if raw.startswith("-100") else raw.lstrip("-")
    return f"https://t.me/c/{internal}/{int(message_id)}"


def _actor_details(message) -> dict:
    """Capture Telegram-exposed sender data without inventing admin identity.

    Values are sanitised: only plain ints/strings are stored, so an exotic
    message object (or a test mock) can never poison the MongoDB document.
    """
    def _clean(value):
        if isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            return value[:256]
        return None

    sender = getattr(message, "from_user", None)
    sender_chat = getattr(message, "sender_chat", None)
    return {
        "sender_user_id": _clean(getattr(sender, "id", None)),
        "sender_user_name": _clean(getattr(sender, "first_name", None)),
        "sender_username": _clean(getattr(sender, "username", None)),
        "sender_chat_id": _clean(getattr(sender_chat, "id", None)),
        "sender_chat_title": _clean(getattr(sender_chat, "title", None)),
        "author_signature": _clean(getattr(message, "author_signature", None)),
        "verified_admin_id": None,
        "identity_note": (
            "Telegram does not expose the responsible admin ID for this channel post."
        ),
    }


def _channel_title_of(message) -> Optional[str]:
    """Best-effort channel title from the message's chat object."""
    chat = getattr(message, "chat", None)
    title = getattr(chat, "title", None) or getattr(chat, "first_name", None)
    if not isinstance(title, str):
        return None
    return title[:256]


# ---------------------------------------------------------------------------
# Persistent record + failure alerts
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Delivery state machine — MongoDB is the source of truth
# ---------------------------------------------------------------------------
#
# Every deletion log / archive-failure alert gets ONE persistent record in
# the ``deletion_log_delivery`` collection, created BEFORE the send.  The
# record — not Telegram history — decides whether a send still happens:
#
#   pending ──claim──▶ sending ──Telegram accepted──▶ sent
#      ▲                  │
#      │  definite        │ unknown outcome / dead owner
#      └───failure────────┴─────────────▶ uncertain
#
# Claims are atomic (``find_one_and_update`` with a status filter), so
# concurrent workers can never send the same event.  The startup sweep
# and the periodic retry worker read ONLY this collection.

from pymongo import ASCENDING, ReturnDocument


async def _get_delivery_db():
    """The delivery-state collection (lazy import avoids cycles)."""
    from SWAGGYMUSIC.utils.database import delivery_db

    return delivery_db


def _delivery_event_tag(event_key: str, kind: str) -> str:
    """The persistent ``EK:`` tag of one delivery record's log message."""
    from SWAGGYMUSIC.utils import logger as central_logger

    if kind == ALERT_LOG_KIND:
        return central_logger.alert_event_tag(event_key)
    return central_logger.deletion_log_event_tag(event_key, kind)


def _legacy_deletion_status(rec: dict, kind: str) -> Tuple[str, Optional[int]]:
    """Map LEGACY audit-record log fields to a delivery state.

    Used when a delivery record does not exist yet for an old event
    (upgrade path).  A legacy "sent" outcome for THIS kind is preserved
    exactly (with its message id) so an upgrade can never re-send a log
    that was already delivered; a held claim without a verdict means an
    interrupted attempt whose outcome is unknown → "uncertain".
    """
    claim_field = _LOG_CLAIM_FIELDS.get(kind, f"deletion_log_{kind}_sent")
    status = rec.get("log_status")
    if status == "sent" and rec.get("log_kind") == kind:
        try:
            mid = rec.get("log_message_id")
            return DELIVERY_SENT, (int(mid) if mid is not None else None)
        except Exception:
            return DELIVERY_SENT, None
    if rec.get(claim_field):
        return DELIVERY_UNCERTAIN, None
    if status in ("failed", "pending") and rec.get("log_kind") == kind:
        return DELIVERY_PENDING, None
    if status in ("uncertain", "attempting") and rec.get("log_kind") == kind:
        return DELIVERY_UNCERTAIN, None
    # Nothing attributable to this kind → a fresh, deliverable event.
    return DELIVERY_PENDING, None


def _legacy_alert_status(rec: dict) -> Tuple[str, Optional[int]]:
    """Map the LEGACY ``failure_alert_status`` field to a delivery state."""
    status = rec.get("failure_alert_status")
    if status == "sent":
        return DELIVERY_SENT, None
    if status in ("attempting", "uncertain"):
        return DELIVERY_UNCERTAIN, None
    return DELIVERY_PENDING, None


async def ensure_delivery_record(
    event_key: str,
    kind: str,
    chat_id: int,
    message_id: int,
    *,
    detail: str = "",
) -> dict:
    """Create-or-get the persistent delivery record — BEFORE any send.

    The record is the MongoDB source of truth for this event's delivery.
    When it does not exist yet and the event predates this design, the
    initial state is seeded from the legacy audit-record fields so an
    upgrade never loses an undelivered log nor re-sends a delivered one.
    Never raises (returns {} on failure).
    """
    try:
        delivery_db = await _get_delivery_db()
        existing = await delivery_db.find_one(
            {"event_key": event_key, "kind": kind},
        )
        if existing:
            return existing

        insert_status, insert_message_id = DELIVERY_PENDING, None
        try:
            from SWAGGYMUSIC.utils.database import audit_db

            legacy = await audit_db.find_one({"event_key": event_key}) or {}
            if kind == ALERT_LOG_KIND:
                insert_status, insert_message_id = _legacy_alert_status(legacy)
            else:
                insert_status, insert_message_id = \
                    _legacy_deletion_status(legacy, kind)
        except Exception:
            pass

        now = datetime.now(timezone.utc)
        insert_doc = {
            "event_key": event_key,
            "kind": kind,
            "chat_id": int(chat_id),
            "message_id": int(message_id),
            "status": insert_status,
            "log_event_tag": _delivery_event_tag(event_key, kind),
            "log_message_id": insert_message_id,
            "logs_chat_id": None,
            "attempts": 0,
            "claimed_by": None,
            "claimed_at": None,
            "lease_expires_at": None,
            "uncertain_since": None,
            "next_attempt_at": None,
            "last_error": None,
            "detail": str(detail or "")[:300],
            "created_at": now,
            "updated_at": now,
            "sent_at": None,
        }
        await delivery_db.update_one(
            {"event_key": event_key, "kind": kind},
            {"$setOnInsert": insert_doc},
            upsert=True,
        )
        return await delivery_db.find_one(
            {"event_key": event_key, "kind": kind},
        ) or {}
    except Exception as exc:
        log.warning(
            "Delivery-record creation failed for %s/%s: %s: %s",
            event_key, kind, type(exc).__name__, exc,
        )
        return {}


async def claim_delivery(
    event_key: str,
    kind: str,
    worker_id: str,
    *,
    startup: bool = False,
) -> Optional[dict]:
    """Atomically claim the right to send ONE event's log.

    A single ``find_one_and_update`` transitions the record to
    "sending" ONLY when the current state allows a send:

    * ``pending`` — never sent: any worker may claim it.
    * ``sending`` — another worker owns it, but at RUNTIME only after
      its lease expired (a dead/slow owner); at STARTUP any "sending"
      record belongs to a process that no longer exists, so it is
      reclaimable regardless of its lease.
    * ``uncertain`` — an unknown outcome: at RUNTIME only after the
      grace period (an immediate resend could duplicate a log Telegram
      actually delivered); at STARTUP always (there is no concurrent
      in-process owner to protect).

    Only ONE concurrent claim can win; the loser gets None.  Never
    raises (returns None on failure).
    """
    try:
        delivery_db = await _get_delivery_db()
        now = datetime.now(timezone.utc)
        if startup:
            sending_clause = {"status": DELIVERY_SENDING}
            uncertain_clause = {"status": DELIVERY_UNCERTAIN}
        else:
            sending_clause = {
                "status": DELIVERY_SENDING,
                "lease_expires_at": {"$lte": now},
            }
            uncertain_clause = {
                "status": DELIVERY_UNCERTAIN,
                "uncertain_since": {
                    "$lte": now - timedelta(
                        seconds=UNCERTAIN_RUNTIME_GRACE_SECONDS)
                },
            }
        claimed = await delivery_db.find_one_and_update(
            {
                "event_key": event_key,
                "kind": kind,
                "$or": [
                    {"status": DELIVERY_PENDING},
                    sending_clause,
                    uncertain_clause,
                ],
            },
            {
                "$set": {
                    "status": DELIVERY_SENDING,
                    "claimed_by": worker_id,
                    "claimed_at": now,
                    "lease_expires_at": now + timedelta(
                        seconds=DELIVERY_LEASE_SECONDS),
                    "uncertain_since": None,
                    "updated_at": now,
                },
                "$inc": {"attempts": 1},
            },
            return_document=ReturnDocument.AFTER,
        )
        if claimed:
            await _mirror_outcome_to_audit(
                event_key, kind, DELIVERY_SENDING,
            )
        return claimed
    except Exception as exc:
        log.warning(
            "Delivery claim failed for %s/%s: %s: %s",
            event_key, kind, type(exc).__name__, exc,
        )
        return None


async def _mirror_outcome_to_audit(
    event_key: str,
    kind: str,
    status: str,
    *,
    message_id: Optional[int] = None,
    logs_chat_id: Optional[int] = None,
) -> None:
    """Best-effort mirror of a delivery outcome onto the audit record.

    Keeps the legacy ``log_status`` / ``failure_alert_status`` fields
    meaningful for operators and for older readers (e.g.
    ``archive_deletion``'s "complete" check via ``log_message_id``).
    Never raises.
    """
    try:
        from SWAGGYMUSIC.utils.database import audit_db

        now = datetime.now(timezone.utc)
        legacy_map = {
            DELIVERY_SENT: "sent",
            DELIVERY_SENDING: "attempting",
            DELIVERY_UNCERTAIN: "uncertain",
        }
        fields = {}
        if kind == ALERT_LOG_KIND:
            legacy = legacy_map.get(status, "pending")
            fields = {"failure_alert_status": legacy, "updated_at": now}
        else:
            legacy = legacy_map.get(status, "failed")
            fields = {"log_status": legacy, "log_kind": kind,
                      "updated_at": now}
            if status == DELIVERY_SENT:
                fields["log_event_tag"] = _delivery_event_tag(event_key, kind)
                fields["logged_at"] = now
        if status == DELIVERY_SENT and message_id is not None:
            fields["log_message_id"] = int(message_id)
            fields["logs_chat_id"] = int(logs_chat_id or 0) or None
        if not fields:
            return
        await audit_db.update_one(
            {"event_key": event_key}, {"$set": fields},
        )
    except Exception:
        pass  # bookkeeping is best-effort


async def mark_delivery_sent(
    event_key: str,
    kind: str,
    message_id: Optional[int],
    *,
    logs_chat_id: Optional[int] = None,
) -> bool:
    """Record a CONFIRMED delivery — Telegram accepted the message.

    The update is conditional: only a record still in "sending" without
    a stored message id accepts the verdict.  There is deliberately no
    owner check — a worker whose lease expired (or that was presumed
    dead) may still record its late SUCCESS, because that evidence is
    the best anti-duplicate information available: a second worker
    re-checks the record right before sending and stands down when a
    message id already exists.

    Never raises; returns True when THIS call recorded the verdict.
    """
    try:
        delivery_db = await _get_delivery_db()
        now = datetime.now(timezone.utc)
        fields = {
            "status": DELIVERY_SENT,
            "claimed_by": None,
            "lease_expires_at": None,
            "uncertain_since": None,
            "next_attempt_at": None,
            "last_error": None,
            "sent_at": now,
            "updated_at": now,
            "log_event_tag": _delivery_event_tag(event_key, kind),
        }
        if message_id is not None:
            try:
                fields["log_message_id"] = int(message_id)
            except Exception:
                pass
            if logs_chat_id:
                fields["logs_chat_id"] = int(logs_chat_id)
        result = await delivery_db.update_one(
            {
                "event_key": event_key,
                "kind": kind,
                "status": DELIVERY_SENDING,
                "log_message_id": None,
            },
            {"$set": fields},
        )
        recorded = bool(result.modified_count)
        if recorded:
            await _mirror_outcome_to_audit(
                event_key, kind, DELIVERY_SENT,
                message_id=message_id, logs_chat_id=logs_chat_id,
            )
        return recorded
    except Exception as exc:
        log.warning(
            "Delivery sent-marking failed for %s/%s: %s: %s",
            event_key, kind, type(exc).__name__, exc,
        )
        return False


async def mark_delivery_outcome(
    event_key: str,
    kind: str,
    outcome: str,
    *,
    error: Optional[str] = None,
    claimed_by: Optional[str] = None,
    attempts: Optional[int] = None,
) -> None:
    """Record a NON-success outcome of a claimed send attempt.

    ``outcome`` comes from the logger's transport layer:
    ``OUTCOME_FAILED`` (Telegram definitely refused → back to
    ``pending``, retryable by the next worker/sweep) or
    ``OUTCOME_UNCERTAIN`` (unknown outcome → ``uncertain``, recovered by
    the startup sweep / the periodic retry worker).

    ``claimed_by`` identifies the attempt's owner; when given, the
    outcome is only recorded while THAT worker still owns the claim — a
    takeover worker's live claim is never overwritten by a stale
    owner's late failure bookkeeping.

    ``attempts`` is the claim-time attempt counter; it drives the
    EXPONENTIAL RETRY BACKOFF — the record stores ``next_attempt_at``
    (now + 30s, 60s, 120s … capped at 1h) so the periodic retry worker
    never hammers a failing event in a tight loop.  Startup sweeps
    ignore the backoff stamp.  Never raises.
    """
    try:
        delivery_db = await _get_delivery_db()
        now = datetime.now(timezone.utc)
        status = (
            DELIVERY_UNCERTAIN
            if outcome == "uncertain" else DELIVERY_PENDING
        )
        fields = {
            "status": status,
            "claimed_by": None,
            "lease_expires_at": None,
            "last_error": str(error or "")[:200] or None,
            # Exponential per-record retry backoff: the retry sweeps
            # skip this record until the stamp elapses (30s after the
            # 1st failure, 60s after the 2nd, … capped at 1 hour) — no
            # tight retry loops, no Telegram API flooding.
            "next_attempt_at": now + timedelta(
                seconds=_backoff_seconds(attempts),
            ),
            "updated_at": now,
        }
        if status == DELIVERY_UNCERTAIN:
            fields["uncertain_since"] = now
        claim_filter = {
            "event_key": event_key,
            "kind": kind,
            "status": DELIVERY_SENDING,
        }
        if claimed_by:
            claim_filter["claimed_by"] = claimed_by
        await delivery_db.update_one(claim_filter, {"$set": fields})
        await _mirror_outcome_to_audit(event_key, kind, status)
    except Exception as exc:
        log.warning(
            "Delivery outcome recording failed for %s/%s: %s: %s",
            event_key, kind, type(exc).__name__, exc,
        )


async def deliver_deletion_log(
    event_key: str,
    kind: str,
    chat_id: int,
    message_id: int,
    *,
    detail: str = "",
    startup: bool = False,
) -> str:
    """The full delivery flow for ONE post-deletion log.

    ensure record (before sending) → atomic claim → last-moment
    duplicate guard → send through the centralized logger → record the
    honest outcome.  Returns the resulting record status, or
    ``"skipped"`` when another worker owns/sent the event, or
    ``"error"`` when the flow itself failed (the record stays
    recoverable).  Never raises.
    """
    try:
        record = await ensure_delivery_record(
            event_key, kind, chat_id, message_id, detail=detail,
        )
        if record.get("status") == DELIVERY_SENT:
            return DELIVERY_SENT  # already delivered — never a duplicate

        claimed = await claim_delivery(
            event_key, kind, _worker_token("dlog"), startup=startup,
        )
        if not claimed:
            return "skipped"
        # Last-moment duplicate guard: a worker presumed dead (expired
        # lease) may have recorded its confirmed delivery between our
        # claim and now.  MongoDB knows the log landed — stand down.
        try:
            delivery_db = await _get_delivery_db()
            current = await delivery_db.find_one(
                {"event_key": event_key, "kind": kind},
            ) or {}
            if current.get("log_message_id") is not None:
                await mark_delivery_sent(
                    event_key, kind, current.get("log_message_id"),
                    logs_chat_id=current.get("logs_chat_id"),
                )
                return DELIVERY_SENT
        except Exception:
            pass  # the guard is best-effort; the send below is still safe

        try:
            from SWAGGYMUSIC.utils.database import audit_db

            fresh = await audit_db.find_one({"event_key": event_key}) or {
                "event_key": event_key,
                "chat_id": int(chat_id),
                "message_id": int(message_id),
            }
        except Exception:
            fresh = {
                "event_key": event_key,
                "chat_id": int(chat_id),
                "message_id": int(message_id),
            }

        from SWAGGYMUSIC.utils import logger as central_logger

        outcome, sent_id = await central_logger.send_post_deletion_log(
            fresh, kind=kind, event_key=event_key, detail=str(detail)[:300],
        )
        if outcome == central_logger.OUTCOME_SENT:
            await mark_delivery_sent(
                event_key, kind, sent_id, logs_chat_id=config.LOGGER_ID,
            )
            return DELIVERY_SENT
        await mark_delivery_outcome(
            event_key, kind, outcome, error=outcome,
            claimed_by=claimed.get("claimed_by"),
            attempts=claimed.get("attempts"),
        )
        return (
            DELIVERY_UNCERTAIN if outcome == "uncertain"
            else DELIVERY_PENDING
        )
    except Exception as exc:
        log.warning(
            "Deletion-log delivery failed for %s/%s: %s: %s",
            event_key, kind, type(exc).__name__, exc,
        )
        return "error"


async def deliver_archive_alert(
    event_key: str,
    chat_id: int,
    message_id: int,
    *,
    detail: str = "",
    startup: bool = False,
) -> str:
    """The full delivery flow for ONE archive-failure alert.

    Same machine as ``deliver_deletion_log`` (kind ``archive_alert``):
    the alert is created before sending, claimed atomically, and only a
    Telegram-accepted send marks it "sent".  Never raises.
    """
    try:
        record = await ensure_delivery_record(
            event_key, ALERT_LOG_KIND, chat_id, message_id, detail=detail,
        )
        if record.get("status") == DELIVERY_SENT:
            return DELIVERY_SENT

        claimed = await claim_delivery(
            event_key, ALERT_LOG_KIND, _worker_token("alert"),
            startup=startup,
        )
        if not claimed:
            return "skipped"

        try:
            delivery_db = await _get_delivery_db()
            current = await delivery_db.find_one(
                {"event_key": event_key, "kind": ALERT_LOG_KIND},
            ) or {}
            if current.get("log_message_id") is not None:
                await mark_delivery_sent(
                    event_key, ALERT_LOG_KIND,
                    current.get("log_message_id"),
                    logs_chat_id=current.get("logs_chat_id"),
                )
                return DELIVERY_SENT
        except Exception:
            pass

        from SWAGGYMUSIC.utils import logger as central_logger

        outcome, sent_id = await central_logger.log_operation_failed(
            "DELETION ARCHIVE NEEDS ATTENTION",
            chat_id=chat_id,
            message_id=message_id,
            detail=f"{detail}. The original deletion flow will continue; "
                   "check this event in MongoDB.",
            event_id=("audit_failure", event_key),
            event_tag=central_logger.alert_event_tag(event_key),
        )
        if outcome == central_logger.OUTCOME_SENT:
            await mark_delivery_sent(
                event_key, ALERT_LOG_KIND, sent_id,
                logs_chat_id=config.LOGGER_ID,
            )
            return DELIVERY_SENT
        await mark_delivery_outcome(
            event_key, ALERT_LOG_KIND, outcome, error=outcome,
            claimed_by=claimed.get("claimed_by"),
            attempts=claimed.get("attempts"),
        )
        return (
            DELIVERY_UNCERTAIN if outcome == "uncertain"
            else DELIVERY_PENDING
        )
    except Exception as exc:
        log.warning(
            "Archive-alert delivery failed for %s: %s: %s",
            event_key, type(exc).__name__, exc,
        )
        return "error"


async def _send_failure_alert(app, record: dict, detail: str) -> None:
    """Send at most one archive-failure alert — THROUGH the logger module.

    The alert's delivery state lives in its ``deletion_log_delivery``
    record (kind ``archive_alert``): created before the send, claimed
    atomically so concurrent failures never double-send, marked "sent"
    ONLY after Telegram accepted the message, "pending" after a
    definite failure (retryable) and "uncertain" after an unknown
    outcome.  The STARTUP sweep recovers pending/uncertain/abandoned
    alerts from MongoDB — Telegram history is never scanned.

    The legacy ``failure_alert_status`` field on the audit record is
    mirrored (best-effort) for operators.
    """
    if not config.LOGGER_ID or app is None:
        return
    event_key = record.get("event_key")
    if not event_key:
        return
    try:
        chat_id = int(record.get("chat_id") or 0)
        message_id = int(record.get("message_id") or 0)
    except Exception:
        return
    try:
        await deliver_archive_alert(
            event_key, chat_id, message_id,
            detail=str(detail)[:300], startup=False,
        )
    except Exception as exc:
        log.warning(
            "Could not send audit failure alert for %s: %s",
            event_key, type(exc).__name__,
        )


async def _store_record(
    chat_id: int,
    message_id: int,
    *,
    reason: str,
    source: str,
    now: datetime,
) -> None:
    """Upsert the persistent deletion-event record (first reason wins).

    Always called — even when the archive channel is not configured —
    because the centralized logger reads the record (preview, channel
    name, reason) and uses its atomic claim for duplicate-free logs.
    """
    from SWAGGYMUSIC.utils.database import audit_db

    await audit_db.update_one(
        {"event_key": _event_key(chat_id, message_id)},
        {
            "$setOnInsert": {
                "event_key": _event_key(chat_id, message_id),
                "chat_id": int(chat_id),
                "message_id": int(message_id),
                "created_at": now,
                "reason": str(reason),
                "source": str(source),
                "forward_status": "pending",
                "archive_status": "pending",
                "deletion_status": "PENDING",
            },
            "$set": {
                "last_attempt_reason": str(reason),
                "last_attempt_source": str(source),
                "last_attempt_at": now,
                "updated_at": now,
            },
            "$inc": {"attempt_count": 1},
        },
        upsert=True,
    )


async def _store_content(chat_id: int, message_id: int, message) -> None:
    """Best-effort store of the content needed by the deletion log."""
    from SWAGGYMUSIC.utils.database import audit_db

    chat = getattr(message, "chat", None)
    text = getattr(message, "text", None) or getattr(message, "caption", None) or ""
    if not isinstance(text, str):
        text = ""
    media_obj = getattr(message, "media", None)
    media_name = getattr(media_obj, "name", None)
    media_type = media_name if isinstance(media_name, str) else "text"
    date = getattr(message, "date", None)
    await audit_db.update_one(
        {"event_key": _event_key(chat_id, message_id)},
        {"$set": {
            "author": _actor_details(message),
            "content_text": text[:4000],
            "media_type": media_type,
            "original_date": date if hasattr(date, "timestamp") else None,
            "channel_title": _channel_title_of(message),
            "original_link": _message_link(chat, chat_id, message_id),
            "updated_at": datetime.now(timezone.utc),
        }},
    )


# ---------------------------------------------------------------------------
# Deletion outcome → centralized logger
# ---------------------------------------------------------------------------

# status (mark_deletion_result) → log kind
_STATUS_KINDS = {
    "deleted": "deleted",
    "pending_retry": "pending_retry",
    "not_deleted_or_already_absent": "failed",
    "failed": "failed",
}


async def mark_deletion_result(
    chat_id: int,
    message_id: int,
    *,
    status: str,
    detail: str = "",
) -> None:
    """Persist the actual delete outcome and emit the deletion log.

    Called only after the Telegram delete operation returned, so the log
    is honest: an archived/forwarded post is never labelled deleted
    merely because the bot attempted to delete it.

    ``status``:
      * ``"deleted"``                      — Telegram confirmed the delete.
      * ``"pending_retry"``                — FloodWait; a persistent retry
                                             job was queued (deletion failed
                                             for now — logged separately).
      * ``"failed"``                       — the delete call failed.
      * ``"not_deleted_or_already_absent"``— Telegram reports the post was
                                             already gone; nothing deleted.

    The log itself is delivered through the MongoDB-backed delivery
    machine (``deliver_deletion_log``): a persistent delivery record
    with the event's unique ``event_key`` is created BEFORE any send,
    claimed atomically (concurrent workers can never double-send), and
    marked "sent" — with Telegram's returned message id — only after
    Telegram accepted the message.  Audit maintenance never raises.
    """
    event_key = _event_key(chat_id, message_id)
    now = datetime.now(timezone.utc)
    status_l = str(status).lower()
    try:
        from SWAGGYMUSIC.utils.database import audit_db

        # Upsert: even when archive_deletion could not create the record
        # (or a caller skipped it), the deletion outcome always has a
        # record to live on — a deletion is never left unreported just
        # because its audit row was missing.
        await audit_db.update_one(
            {"event_key": event_key},
            {
                "$setOnInsert": {
                    "event_key": event_key,
                    "chat_id": int(chat_id),
                    "message_id": int(message_id),
                    "created_at": now,
                },
                "$set": {
                    "deletion_status": str(status).upper(),
                    "deletion_result_detail": str(detail)[:300],
                    "deleted_at": now if status_l == "deleted" else None,
                    "updated_at": now,
                },
            },
            upsert=True,
        )
    except Exception as exc:
        log.warning(
            "Could not persist deletion outcome chat=%s message=%s: %s: %s",
            chat_id, message_id, type(exc).__name__, exc,
        )

    kind = _STATUS_KINDS.get(status_l, "failed")
    # Nothing can be delivered while the logger group is unconfigured;
    # do not create a delivery record for a send that cannot happen.
    if not config.LOGGER_ID:
        return
    try:
        # MongoDB-backed delivery: create-before-send, atomic claim,
        # "sent" only after Telegram accepts — concurrent workers and
        # retries included, duplicates excluded (outside the honest
        # crash window documented in the module docstring).
        await deliver_deletion_log(
            event_key, kind, int(chat_id), int(message_id),
            detail=str(detail)[:300], startup=False,
        )
    except Exception as exc:
        # Logger/audit maintenance must never interfere with the deletion flow.
        log.warning(
            "Could not report deletion outcome in log chat=%s message=%s: %s: %s",
            chat_id, message_id, type(exc).__name__, exc,
        )


async def _seed_legacy_delivery_records(*, max_records: int = 500) -> int:
    """Create delivery records for UNDELIVERED legacy-format audit records.

    Upgrade path only.  Old code tracked deletion-log delivery on the
    audit record itself (``log_status`` / per-kind claim booleans /
    ``failure_alert_status``).  Records whose legacy state proves the
    log was never confirmed delivered are seeded into the
    ``deletion_log_delivery`` collection so the startup sweep finds and
    delivers them; a legacy "sent" verdict is intentionally NOT seeded
    here (already-delivered logs need no recovery — if such an event is
    ever re-processed, ``ensure_delivery_record`` maps it to "sent"
    directly, so an upgrade never duplicates a delivered log either).

    Idempotent: records that already have a delivery record are skipped.
    Never raises; returns the number of records seeded.
    """
    seeded = 0
    try:
        from SWAGGYMUSIC.utils.database import audit_db, delivery_db

        legacy_statuses = ["failed", "pending", "uncertain", "attempting"]
        cursor = audit_db.find(
            {"log_status": {"$in": legacy_statuses},
             "log_kind": {"$in": list(DELETION_LOG_KINDS)}},
        ).limit(max_records)
        async for rec in cursor:
            event_key = rec.get("event_key")
            kind = rec.get("log_kind")
            if not event_key or kind not in DELETION_LOG_KINDS:
                continue
            try:
                chat_id = int(rec.get("chat_id") or 0)
                message_id = int(rec.get("message_id") or 0)
            except Exception:
                continue
            existing = await delivery_db.find_one(
                {"event_key": event_key, "kind": kind},
            )
            if existing:
                continue
            status, _mid = _legacy_deletion_status(rec, kind)
            now = datetime.now(timezone.utc)
            await delivery_db.update_one(
                {"event_key": event_key, "kind": kind},
                {"$setOnInsert": {
                    "event_key": event_key,
                    "kind": kind,
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "status": status,
                    "log_event_tag": _delivery_event_tag(event_key, kind),
                    "log_message_id": None,
                    "logs_chat_id": None,
                    "attempts": 0,
                    "detail": str(rec.get("deletion_result_detail") or "")[:300],
                    "uncertain_since": (
                        now if status == DELIVERY_UNCERTAIN else None
                    ),
                    "next_attempt_at": None,
                    "created_at": now,
                    "updated_at": now,
                }},
                upsert=True,
            )
            seeded += 1

        alert_cursor = audit_db.find(
            {"failure_alert_status": {"$in": legacy_statuses}},
        ).limit(max_records)
        async for rec in alert_cursor:
            event_key = rec.get("event_key")
            if not event_key:
                continue
            try:
                chat_id = int(rec.get("chat_id") or 0)
                message_id = int(rec.get("message_id") or 0)
            except Exception:
                continue
            existing = await delivery_db.find_one(
                {"event_key": event_key, "kind": ALERT_LOG_KIND},
            )
            if existing:
                continue
            status, _mid = _legacy_alert_status(rec)
            now = datetime.now(timezone.utc)
            await delivery_db.update_one(
                {"event_key": event_key, "kind": ALERT_LOG_KIND},
                {"$setOnInsert": {
                    "event_key": event_key,
                    "kind": ALERT_LOG_KIND,
                    "chat_id": chat_id,
                    "message_id": message_id,
                    "status": status,
                    "log_event_tag": _delivery_event_tag(
                        event_key, ALERT_LOG_KIND),
                    "log_message_id": None,
                    "logs_chat_id": None,
                    "attempts": 0,
                    "detail": "",
                    "uncertain_since": (
                        now if status == DELIVERY_UNCERTAIN else None
                    ),
                    "next_attempt_at": None,
                    "created_at": now,
                    "updated_at": now,
                }},
                upsert=True,
            )
            seeded += 1
    except Exception as exc:
        log.warning(
            "Legacy delivery-record seeding failed: %s: %s",
            type(exc).__name__, exc,
        )
    return seeded


_SWEEP_KINDS = tuple(DELETION_LOG_KINDS) + (ALERT_LOG_KIND,)


async def _fetch_sweep_batch(delivery_db, query, *, batch_size, after_id=None):
    """One batch of sweep candidates, paginating FORWARD by ``_id``.

    Efficient keyset pagination: every batch after the first fetches
    only records whose ``_id`` is strictly GREATER than the last ``_id``
    of the previous batch (``{"_id": {"$gt": after_id}}``), sorted by
    ``_id`` ascending.  The query therefore NEVER grows a ``$nin``
    exclusion list — its size is constant no matter how many batches
    the sweep needs.  A record that failed (and stayed non-"sent") is
    not re-fetched within this sweep — it simply falls behind the
    cursor and is retried by the NEXT sweep — so a failing record can
    neither block the other records nor loop the sweep forever.
    """
    q = dict(query)
    if after_id is not None:
        q["_id"] = {"$gt": after_id}
    cursor = delivery_db.find(q).sort("_id", ASCENDING).limit(batch_size)
    return [rec async for rec in cursor]


async def _sweep_delivery_records(
    query,
    *,
    startup: bool,
    batch_size: int,
    max_batches: Optional[int] = None,
) -> int:
    """Process eligible delivery records batch-by-batch until exhausted.

    Shared engine of the startup recovery and the periodic retry sweep.
    Fetches up to ``batch_size`` candidate records per pass, delivers
    each one independently (a failing record or a failing delivery can
    never abort the batch or the sweep), then fetches the NEXT batch —
    repeating until a fetch comes back empty, i.e. until every eligible
    record has been handled.  It never stops after the first batch.

    The cursor is the last ``_id`` of the previous batch (keyset
    pagination): each fetch continues strictly AFTER it, so the query
    stays constant-sized — no growing ``$nin`` list — and the sweep
    always advances forward.  A record whose delivery raises is behind
    the cursor by the time the next batch is fetched, so it can never
    be re-fetched in THIS sweep (it is retried by the next one) and can
    never wedge the sweep into an infinite loop.

    Every candidate goes through the normal delivery flow (atomic
    claim, last-moment duplicate guard, send, honest outcome), so
    concurrent workers can never double-send.  Returns the number of
    records whose delivery attempt produced an honest outcome.
    """
    reconciled = 0
    after_id = None  # last ``_id`` consumed — the pagination cursor
    batches_done = 0
    delivery_db = await _get_delivery_db()
    while True:
        try:
            batch = await _fetch_sweep_batch(
                delivery_db, query,
                batch_size=batch_size, after_id=after_id,
            )
        except Exception as exc:
            # Cursor-level failure: this pass cannot read more
            # candidates.  Stop cleanly — already-processed batches
            # keep their recorded outcomes and the next sweep retries.
            log.warning(
                "Delivery-sweep batch fetch failed after %d batch(es): "
                "%s: %s", batches_done, type(exc).__name__, exc,
            )
            break
        if not batch:
            break  # no eligible record left — the sweep is complete
        batches_done += 1

        # Advance the pagination cursor BEFORE delivering: batches are
        # sorted by ``_id`` ascending, so the last ``_id`` of this batch
        # is the exact position every later fetch continues after.  Even
        # a record whose delivery raises is thus excluded from this
        # sweep's next fetches — the sweep always advances and always
        # terminates.
        cursor_advanced = False
        last_id = batch[-1].get("_id")
        if last_id is not None:
            after_id = last_id
            cursor_advanced = True

        for rec in batch:
            event_key = rec.get("event_key")
            kind = rec.get("kind")
            if not event_key or kind not in _SWEEP_KINDS:
                continue
            try:
                chat_id = int(rec.get("chat_id") or 0)
                message_id = int(rec.get("message_id") or 0)
            except Exception:
                continue
            detail = str(rec.get("detail") or "")[:300]
            try:
                if kind == ALERT_LOG_KIND:
                    result = await deliver_archive_alert(
                        event_key, chat_id, message_id,
                        detail=detail, startup=startup,
                    )
                else:
                    result = await deliver_deletion_log(
                        event_key, kind, chat_id, message_id,
                        detail=detail, startup=startup,
                    )
                if result in (DELIVERY_SENT, DELIVERY_PENDING,
                              DELIVERY_UNCERTAIN):
                    reconciled += 1
                # "skipped" — a runtime worker owns/sent it concurrently;
                # "error" — the record stays in MongoDB for the next sweep.
            except Exception as exc:
                log.warning(
                    "Recovery delivery failed for %s/%s: %s: %s",
                    event_key, kind, type(exc).__name__, exc,
                )

        if len(batch) < batch_size:
            break  # last batch — nothing eligible beyond it
        if max_batches is not None and batches_done >= max_batches:
            break  # test/operator guard; production passes None (unlimited)
        if not cursor_advanced:
            # Defensive: a collection handing out records without usable
            # ``_id``s cannot be paginated — stop rather than loop forever.
            break
    return reconciled


async def recover_interrupted_deletion_logs(
    *, batch_size: Optional[int] = None, max_batches: Optional[int] = None,
) -> int:
    """Deliver deletion logs a crashed or failing process left undelivered.

    Called once at bot startup (fire-and-forget).  MONGODB IS THE ONLY
    SOURCE OF TRUTH — Telegram history is never scanned, neither here
    nor anywhere else in recovery, so recovery is O(pending records)
    MongoDB reads and independent of logger-group traffic, message
    volume and any scan window.

    1. LEGACY seeding — undelivered old-format audit records (previous
       design's ``log_status`` / claim booleans / alert statuses) are
       migrated into delivery records first, so an upgrade loses nothing.
    2. CANDIDATES — every delivery record that is not "sent":
       ``pending`` (never delivered / definitely refused),
       ``uncertain`` (outcome unknown — Telegram MAY have delivered),
       ``sending`` (a worker claimed it and died; at startup ANY
       "sending" record is abandoned, its lease from the previous
       process is meaningless).
    3. BATCHED DELIVERY — candidates are processed in batches of
       ``batch_size`` (default ``DELIVERY_BATCH_SIZE`` = 200) and the
       sweep keeps fetching further batches until NO eligible record is
       left.  It never stops after the first 200 records: a backlog of
       any size is fully processed in one sweep.  Each candidate is
       claimed ATOMICALLY (concurrent workers can never double-send),
       sent through the centralized logger, and its honest outcome
       recorded: "sent" only after Telegram accepted (with the returned
       message id), "pending" after a definite refusal, "uncertain"
       after an unknown outcome.  A failure inside one batch (a raising
       delivery, a bad record) never stops the remaining records or the
       following batches.

    The startup sweep deliberately IGNORES the retry-backoff stamp
    (``next_attempt_at``): a boot is exactly the moment when every
    leftover record must be attempted.

    Honest limitation (documented in the module docstring): a record
    abandoned in "sending" may represent a send Telegram ALREADY
    accepted (crash after send, before the success was persisted).
    Without scanning Telegram history — which recovery deliberately
    does not do — that cannot be distinguished from a dead send, so the
    sweep re-delivers it: the audit log can never be lost, at the cost
    of a rare, exactly-identifiable (``EK:`` tag) duplicate in the
    logger group.  Telegram's ``sendMessage`` has no idempotency key, so
    absolute exactly-once delivery is not achievable without it.

    Returns the number of records the sweep processed.  Never raises.
    """
    if not config.LOGGER_ID:
        return 0
    reconciled = 0
    try:
        await _seed_legacy_delivery_records()

        query = {"status": {"$in": [
            DELIVERY_PENDING, DELIVERY_UNCERTAIN, DELIVERY_SENDING,
        ]}}
        reconciled = await _sweep_delivery_records(
            query,
            startup=True,
            batch_size=int(batch_size or DELIVERY_BATCH_SIZE),
            max_batches=max_batches,
        )
    except Exception as exc:
        log.warning(
            "Interrupted-deletion-log recovery failed: %s: %s",
            type(exc).__name__, exc,
        )
    return reconciled


async def retry_undelivered_deletion_logs(
    *, batch_size: Optional[int] = None, max_batches: Optional[int] = None,
) -> int:
    """PERIODIC retry sweep — recovery that does not depend on a restart.

    Runs on the delivery-retry worker's cadence
    (``start_delivery_retry_worker``) while the bot is up.  Same engine
    and guarantees as the startup sweep (MongoDB-only, batched until
    exhausted, atomic claims, honest outcomes), with RUNTIME-SAFE
    eligibility so a live worker is never disturbed:

    * ``pending``   — retried only after its exponential backoff stamp
      (``next_attempt_at``) elapsed: 30s, 60s, 120s … capped at 1 hour.
      This is what keeps a permanently failing record away from tight
      retry loops and keeps the Telegram API call volume sane.
    * ``uncertain`` — retried only after BOTH the runtime grace (an
      immediate resend could duplicate a log Telegram actually
      delivered) and the backoff stamp elapsed.
    * ``sending``   — only when its lease EXPIRED (its worker died or
      stalled); a live claim is never taken over at runtime.

    Legacy seeding is deliberately NOT run here — it is a startup-time
    migration; records it has not yet covered are picked up on the next
    boot.  Never raises; returns the number of records processed.
    """
    if not config.LOGGER_ID:
        return 0
    now = datetime.now(timezone.utc)
    backoff_open = {"$or": [
        {"next_attempt_at": {"$exists": False}},
        {"next_attempt_at": None},
        {"next_attempt_at": {"$lte": now}},
    ]}
    query = {"$or": [
        # pending — definite non-delivery: retry after the backoff
        {"$and": [
            {"status": DELIVERY_PENDING},
            backoff_open,
        ]},
        # uncertain — unknown outcome: runtime grace AND backoff elapsed
        {"$and": [
            {"status": DELIVERY_UNCERTAIN},
            {"uncertain_since": {"$lte": now - timedelta(
                seconds=UNCERTAIN_RUNTIME_GRACE_SECONDS)}},
            backoff_open,
        ]},
        # stale claim — the owning worker died (lease expired)
        {"$and": [
            {"status": DELIVERY_SENDING},
            {"lease_expires_at": {"$lte": now}},
        ]},
    ]}
    try:
        return await _sweep_delivery_records(
            query,
            startup=False,
            batch_size=int(batch_size or DELIVERY_BATCH_SIZE),
            max_batches=max_batches,
        )
    except Exception as exc:
        log.warning(
            "Periodic delivery retry sweep failed: %s: %s",
            type(exc).__name__, exc,
        )
        return 0


# The periodic delivery-retry worker's task + its sweep lock.  The lock
# guarantees two sweeps never overlap (a slow sweep must not stack up
# with the next tick); ``_retry_worker_task`` keeps the task referenced
# so asyncio never garbage-collects it mid-flight.
_sweep_lock = asyncio.Lock()
_retry_worker_task: Optional["asyncio.Task"] = None


async def delivery_retry_worker(*, interval: Optional[float] = None) -> None:
    """The periodic delivery-retry loop — runs until cancelled.

    Every ``interval`` seconds (default ``DELIVERY_RETRY_INTERVAL_SECONDS``,
    5 minutes) it runs ``retry_undelivered_deletion_logs()``: failed and
    uncertain records are re-delivered with per-record exponential
    backoff, and stale claims of dead workers are taken over.  Ticks are
    skipped while a previous sweep is still running (a slow sweep never
    stacks up with the next one).  The loop survives sweep failures — a
    raising sweep is logged and the next tick simply tries again — and
    never lets an exception escape (a cancelled loop exits quietly).
    """
    interval_s = float(interval or DELIVERY_RETRY_INTERVAL_SECONDS)
    while True:
        await asyncio.sleep(interval_s)
        if _sweep_lock.locked():
            continue  # previous sweep still running — skip this tick
        async with _sweep_lock:
            try:
                await retry_undelivered_deletion_logs()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning(
                    "Delivery retry tick failed: %s: %s",
                    type(exc).__name__, exc,
                )


def start_delivery_retry_worker(
    *, interval: Optional[float] = None,
) -> Optional["asyncio.Task"]:
    """Schedule the periodic delivery-retry worker (fire-and-forget).

    Idempotent per process: while a worker task is already running, the
    existing task is returned and no second loop is started.  Returns
    ``None`` when there is no running event loop.  Never raises.
    """
    global _retry_worker_task
    try:
        running = _retry_worker_task
        if running is not None and not running.done():
            return running
        loop = asyncio.get_running_loop()
        task = loop.create_task(delivery_retry_worker(interval=interval))
    except Exception as exc:
        log.warning(
            "Delivery retry worker could not start: %s: %s",
            type(exc).__name__, exc,
        )
        return None
    _retry_worker_task = task
    return task


async def archive_deletion(
    chat_id: int,
    message_id: int,
    *,
    reason: str,
    source: str,
    message=None,
    details: Optional[dict] = None,
) -> dict:
    """Persist, forward, and report a deletion event.

    The first observed reason/source are immutable. Retry details are stored
    separately, so e.g. SETDELAY_EXPIRED is not overwritten by a later
    SETDELAY_RETRY. Forward send attempts are claimed atomically. If the
    process crashes after Telegram accepts a send but before MongoDB stores
    its returned message ID, the state becomes uncertain and is not blindly
    resent (which avoids duplicate archive messages).

    ``details`` (optional) carries feature-specific dynamic data for the
    deletion log (e.g. ``{"matched": "cross"}`` for Banned Words); the
    first observation is kept, like the reason.

    The LOGGER_ID detail message of the old implementation was replaced
    by the centralized logger: ``mark_deletion_result`` reports the
    outcome through ``SWAGGYMUSIC.utils.logger``.
    """
    event_key = _event_key(chat_id, message_id)
    now = datetime.now(timezone.utc)
    result = {
        "event_key": event_key,
        "forwarded": False,
        "logged": False,
        "audit_link": None,
        "status": "disabled",
    }

    app = None
    record = {
        "event_key": event_key,
        "chat_id": int(chat_id),
        "message_id": int(message_id),
        "reason": str(reason),
        "source": str(source),
    }
    try:
        from SWAGGYMUSIC import app

        # Keep the first/original reason and source immutable.
        from SWAGGYMUSIC.utils.database import audit_db

        insert_doc = {
            **record,
            "created_at": now,
            "reason": str(reason),
            "source": str(source),
            "forward_status": "pending",
            "archive_status": "pending",
            "deletion_status": "PENDING",
        }
        if details:
            insert_doc["details"] = dict(details)
        await audit_db.update_one(
            {"event_key": event_key},
            {
                "$setOnInsert": insert_doc,
                "$set": {
                    "last_attempt_reason": str(reason),
                    "last_attempt_source": str(source),
                    "last_attempt_at": now,
                    "updated_at": now,
                },
                "$inc": {"attempt_count": 1},
            },
            upsert=True,
        )
        current = await audit_db.find_one({"event_key": event_key}) or {}
        if message is None:
            # Fetch the original post ONLY when someone will consume it: the
            # archive channel and/or the centralized logger. With both unset
            # the fetch would be a pointless Telegram call.
            if config.AUDIT_CHANNEL_ID or config.LOGGER_ID:
                message = await app.get_messages(int(chat_id), int(message_id))
            else:
                result["status"] = "disabled"
                return result
        if not message or getattr(message, "empty", False):
            await audit_db.update_one(
                {"event_key": event_key},
                {"$set": {"archive_status": "source_message_unavailable",
                           "updated_at": datetime.now(timezone.utc)}},
            )
            result["status"] = "source_message_unavailable"
            await _send_failure_alert(app, current, result["status"])
            return result

        await _store_content(chat_id, message_id, message)
        current = await audit_db.find_one({"event_key": event_key}) or current

        if not config.AUDIT_CHANNEL_ID:
            # Archive channel not configured — the record still exists for
            # the centralized logger; nothing is forwarded.
            result["status"] = "disabled"
            return result

        # Forward claim: only one worker can move pending -> attempting.
        audit_message_id = current.get("audit_message_id")
        if not audit_message_id:
            claim = await audit_db.update_one(
                {
                    "event_key": event_key,
                    "$or": [
                        {"forward_status": {"$in": ["pending", "retryable_failed"]}},
                        {"forward_status": {"$exists": False}},
                    ],
                },
                {"$set": {"forward_status": "attempting",
                           "forward_attempted_at": datetime.now(timezone.utc)}},
            )
            if claim.modified_count:
                try:
                    forwarded = await app.forward_messages(
                        chat_id=config.AUDIT_CHANNEL_ID,
                        from_chat_id=int(chat_id),
                        message_ids=int(message_id),
                    )
                    if isinstance(forwarded, (list, tuple)):
                        forwarded = forwarded[0] if forwarded else None
                    audit_message_id = getattr(forwarded, "id", None)
                    if not audit_message_id:
                        raise RuntimeError("Telegram forward returned no message ID")
                    archive_link = None
                    try:
                        audit_chat = await app.get_chat(config.AUDIT_CHANNEL_ID)
                        archive_link = _message_link(
                            audit_chat, config.AUDIT_CHANNEL_ID, audit_message_id,
                        )
                    except Exception:
                        archive_link = _message_link(
                            None, config.AUDIT_CHANNEL_ID, audit_message_id,
                        )
                    await audit_db.update_one(
                        {"event_key": event_key},
                        {"$set": {
                            "audit_channel_id": int(config.AUDIT_CHANNEL_ID),
                            "audit_message_id": int(audit_message_id),
                            "archive_link": archive_link,
                            "forward_status": "sent",
                            "forwarded_at": datetime.now(timezone.utc),
                            "updated_at": datetime.now(timezone.utc),
                        }},
                    )
                except (FloodWait, RPCError) as exc:
                    await audit_db.update_one(
                        {"event_key": event_key},
                        {"$set": {"forward_status": "retryable_failed",
                                   "last_forward_error": type(exc).__name__,
                                   "updated_at": datetime.now(timezone.utc)}},
                    )
                    await _send_failure_alert(app, current, f"Forward failed: {type(exc).__name__}")
                    result["status"] = "forward_failed"
                    return result
                except Exception as exc:
                    # Network timeout / unknown outcome: Telegram may have
                    # accepted the forward. Do not auto-forward a duplicate.
                    await audit_db.update_one(
                        {"event_key": event_key},
                        {"$set": {"forward_status": "uncertain",
                                   "last_forward_error": type(exc).__name__,
                                   "updated_at": datetime.now(timezone.utc)}},
                    )
                    await _send_failure_alert(app, current, f"Forward outcome uncertain: {type(exc).__name__}")
                    result["status"] = "forward_uncertain"
                    return result
            else:
                current = await audit_db.find_one({"event_key": event_key}) or current
                audit_message_id = current.get("audit_message_id")
                if not audit_message_id:
                    state = current.get("forward_status", "pending")
                    if state == "attempting":
                        await audit_db.update_one(
                            {"event_key": event_key, "forward_status": "attempting"},
                            {"$set": {"forward_status": "uncertain",
                                       "updated_at": datetime.now(timezone.utc)}},
                        )
                        state = "uncertain"
                    result["status"] = f"forward_{state}"
                    if state in {"uncertain", "retryable_failed"}:
                        await _send_failure_alert(app, current, f"Forward state: {state}")
                    return result

        audit_chat = await app.get_chat(config.AUDIT_CHANNEL_ID)
        audit_link = _message_link(audit_chat, config.AUDIT_CHANNEL_ID, audit_message_id)
        result.update({"forwarded": True, "audit_link": audit_link, "status": "forwarded"})
        await audit_db.update_one(
            {"event_key": event_key},
            {"$set": {"archive_link": audit_link,
                       "updated_at": datetime.now(timezone.utc)}},
        )

        # The centralized logger (mark_deletion_result) owns the LOGGER_ID
        # output; "complete" means forwarded AND already reported there.
        fresh = await audit_db.find_one({"event_key": event_key}, {"log_message_id": 1}) or {}
        result["logged"] = bool(fresh.get("log_message_id"))
        result["status"] = "complete" if result["logged"] else "forwarded"
        await audit_db.update_one(
            {"event_key": event_key},
            {"$set": {"archive_status": result["status"],
                       "updated_at": datetime.now(timezone.utc)}},
        )
        return result

    except Exception as exc:
        log.warning(
            "Deletion audit failed chat=%s message=%s: %s: %s",
            chat_id, message_id, type(exc).__name__, exc,
        )
        try:
            from SWAGGYMUSIC.utils.database import audit_db

            await audit_db.update_one(
                {"event_key": event_key},
                {"$set": {
                    "archive_status": "failed",
                    "last_error": type(exc).__name__,
                    "updated_at": datetime.now(timezone.utc),
                }},
                upsert=True,
            )
        except Exception as db_exc:
            log.error("Could not persist failed deletion audit: %s", type(db_exc).__name__)
        await _send_failure_alert(app, record, f"Audit processing failed: {type(exc).__name__}")
        result["status"] = "failed"
        return result
