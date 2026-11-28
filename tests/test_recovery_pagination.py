"""Recovery pagination tests — efficient ``_id`` keyset pagination.

The delivery-recovery sweep must:

* handle MORE than ``DELIVERY_BATCH_SIZE`` (200) pending records in one
  pass, paginating by ``_id`` (each batch continues strictly AFTER the
  last ``_id`` of the previous batch);
* NEVER issue a growing ``$nin`` exclusion list — every query stays
  constant-sized no matter how many batches are processed;
* keep failed records from blocking other records or looping the sweep
  (a failed record falls behind the cursor and is retried by the NEXT
  sweep, later batches still continue);
* preserve atomic claims, unique event keys, retry/backoff and crash
  recovery (covered further in test_delivery_state.py /
  test_delivery_retry_worker.py).
"""
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from SWAGGYMUSIC.utils import deletion_audit as audit
from SWAGGYMUSIC.utils import logger as logger_mod

pytestmark = pytest.mark.asyncio

# Every test uses its own chat-id window [base - n, base]; leftovers of
# other tests (shared mongomock database — see conftest.py) are closed
# out with _close_outside() before each sweep.
_MY_TOP = -1009996000


def _forbid_history(monkeypatch):
    """Any Telegram history scan during recovery is a design violation."""
    from SWAGGYMUSIC import app

    def _boom(*args, **kwargs):
        raise AssertionError("Telegram history was scanned — forbidden")

    monkeypatch.setattr(app, "get_chat_history", _boom)


def _logger_env(monkeypatch, logger_id=-1009996900):
    import config
    from SWAGGYMUSIC import app

    monkeypatch.setattr(config, "LOGGER_ID", logger_id, raising=False)
    sent = AsyncMock(return_value=SimpleNamespace(id=999600))
    monkeypatch.setattr(app, "send_message", sent)
    logger_mod._DELIVERED_EVENTS.clear()
    return sent


async def _close_outside(mock_db, low: int, high: int):
    """Mark every delivery record OUTSIDE [low, high] as already sent,
    so a sweep only sees THIS test's backlog."""
    await mock_db.deletion_log_delivery.update_many(
        {"$or": [
            {"chat_id": {"$lt": low}},
            {"chat_id": {"$gt": high}},
        ]},
        {"$set": {"status": "sent", "log_message_id": 1}},
    )


async def _bulk_pending(mock_db, n, *, base_chat):
    """Insert ``n`` pending records with chat ids [base-n+1 … base]."""
    keys = []
    now = datetime.now(timezone.utc)
    docs = []
    for i in range(n):
        chat_id = base_chat - i
        message_id = 96000 + i
        keys.append(f"{chat_id}:{message_id}")
        docs.append({
            "event_key": f"{chat_id}:{message_id}",
            "kind": "deleted",
            "chat_id": int(chat_id),
            "message_id": int(message_id),
            "status": "pending",
            "log_event_tag": logger_mod.deletion_log_event_tag(
                f"{chat_id}:{message_id}", "deleted"),
            "log_message_id": None,
            "logs_chat_id": None,
            "attempts": 0,
            "detail": "",
            "created_at": now,
            "updated_at": now,
        })
    await mock_db.deletion_log_delivery.insert_many(docs)
    return keys


async def _remaining(mock_db, low, high):
    return await mock_db.deletion_log_delivery.count_documents({
        "chat_id": {"$gte": low, "$lte": high},
        "status": {"$ne": "sent"},
    })


class _FindSpy:
    """Record every ``find`` query the sweep issues on the delivery
    collection, then delegate to the ORIGINAL ``find``.

    The patch target MUST be the exact object ``deletion_audit`` uses
    (``SWAGGYMUSIC.utils.database.delivery_db``) — mongomock-motor
    creates a fresh collection wrapper on every attribute access, so
    patching ``mock_db.deletion_log_delivery`` would miss the sweep's
    calls.  The original ``find`` is captured BEFORE patching, so the
    spy never recurses into itself.
    """

    def __init__(self):
        import SWAGGYMUSIC.utils.database as _db_mod

        self._collection = _db_mod.delivery_db
        self._original_find = self._collection.find
        self.queries = []

    def _spy(self, query=None, *args, **kwargs):
        self.queries.append(dict(query or {}))
        return self._original_find(query, *args, **kwargs)

    def patch(self, monkeypatch):
        monkeypatch.setattr(self._collection, "find", self._spy)
        return self


class _RecordingCollection:
    """A collection-like wrapper for UNIT tests: records every ``find``
    query and delegates to the real mongomock collection.  Passed
    directly to ``_fetch_sweep_batch``, no patching involved."""

    def __init__(self, collection):
        self._collection = collection
        self.queries = []

    def find(self, query=None, *args, **kwargs):
        self.queries.append(dict(query or {}))
        return self._collection.find(query, *args, **kwargs)


def _patch_find(monkeypatch):
    """Spy on the sweep's ``find`` calls; returns the spy."""
    return _FindSpy().patch(monkeypatch)


# ---------------------------------------------------------------------------
# Unit: the batch fetch uses ``$gt`` on ``_id`` — never ``$nin``
# ---------------------------------------------------------------------------

class TestFetchBatchPagination:
    async def test_no_after_id_fetches_from_the_beginning(self, mock_db):
        base = -1009996100
        await _bulk_pending(mock_db, 3, base_chat=base)
        await _close_outside(mock_db, base - 10, base)
        fake = _RecordingCollection(mock_db.deletion_log_delivery)
        batch = await audit._fetch_sweep_batch(
            fake,
            {"status": "pending", "chat_id": {"$gte": base - 10}},
            batch_size=2,
        )
        assert len(batch) == 2
        assert fake.queries[0]["chat_id"] == {"$gte": base - 10}
        assert "_id" not in fake.queries[0]  # first fetch has no cursor

    async def test_after_id_fetches_strictly_greater_ids(self, mock_db):
        base = -1009996200
        await _bulk_pending(mock_db, 5, base_chat=base)
        await _close_outside(mock_db, base - 10, base)
        chat_q = {"status": "pending", "chat_id": {"$gte": base - 10}}
        fake = _RecordingCollection(mock_db.deletion_log_delivery)
        first = await audit._fetch_sweep_batch(fake, chat_q, batch_size=2)
        cursor = first[-1]["_id"]
        second = await audit._fetch_sweep_batch(
            fake, chat_q, batch_size=2, after_id=cursor,
        )
        assert len(second) == 2
        assert fake.queries[1]["_id"] == {"$gt": cursor}
        assert {r["_id"] for r in first}.isdisjoint({r["_id"] for r in second})

        third = await audit._fetch_sweep_batch(
            fake, chat_q, batch_size=2, after_id=second[-1]["_id"],
        )
        assert len(third) == 1  # 5 records — 2 + 2 + 1: pagination drains all
        seen = {r["_id"] for r in first + second + third}
        assert len(seen) == 5  # no record fetched twice
        # No ``$nin`` anywhere — the cursor is a single ``$gt`` clause.
        for q in fake.queries:
            assert "$nin" not in str(q)

    async def test_cursor_advances_monotonically(self, mock_db):
        base = -1009996250
        await _bulk_pending(mock_db, 6, base_chat=base)
        await _close_outside(mock_db, base - 10, base)
        chat_q = {"status": "pending", "chat_id": {"$gte": base - 10}}
        seen_ids = []
        after_id = None
        while True:
            batch = await audit._fetch_sweep_batch(
                mock_db.deletion_log_delivery, chat_q,
                batch_size=2, after_id=after_id,
            )
            if not batch:
                break
            seen_ids.extend(r["_id"] for r in batch)
            after_id = batch[-1]["_id"]
        assert len(seen_ids) == 6  # every record exactly once
        assert len(seen_ids) == len(set(seen_ids))


# ---------------------------------------------------------------------------
# Full sweep: > 200 records via _id pagination, no $nin, failures don't block
# ---------------------------------------------------------------------------

class TestSweepPagination:
    async def test_450_records_paginated_by_id_without_nin(
        self, monkeypatch, mock_db,
    ):
        """A backlog larger than one batch is fully processed; every
        fetch paginates with an ``$gt`` cursor on ``_id`` and NO query
        ever carries a ``$nin`` list."""
        _logger_env(monkeypatch, logger_id=-1009996300)
        _forbid_history(monkeypatch)
        base = -1009996400  # 450 records → [base-449, base]
        keys = await _bulk_pending(mock_db, 450, base_chat=base)
        assert len(keys) > audit.DELIVERY_BATCH_SIZE
        await _close_outside(mock_db, base - 450, base)

        spy = _patch_find(monkeypatch)

        from SWAGGYMUSIC import app

        count = await audit.recover_interrupted_deletion_logs(
            batch_size=audit.DELIVERY_BATCH_SIZE,
        )
        assert count == 450
        assert app.send_message.await_count == 450  # one send per record

        # 450 records / 200 per batch = 3 paginated fetches
        assert len(spy.queries) == 3
        # The first fetch has no cursor; every later one continues
        # strictly after the previous batch's last _id — and no query
        # ever carries a growing ``$nin`` exclusion list.
        assert "_id" not in spy.queries[0]
        for prev_q, q in zip(spy.queries, spy.queries[1:]):
            assert "$nin" not in str(q)
            assert "_id" in q and "$gt" in q["_id"]

        assert await _remaining(mock_db, base - 450, base) == 0

        # Idempotent: a second sweep finds nothing left to do.
        count2 = await audit.recover_interrupted_deletion_logs()
        assert count2 == 0

    async def test_periodic_sweep_paginates_the_same_way(
        self, monkeypatch, mock_db,
    ):
        """The periodic retry sweep shares the pagination engine: > 200
        eligible records, ``$gt`` cursors, never ``$nin``."""
        _logger_env(monkeypatch, logger_id=-1009996500)
        _forbid_history(monkeypatch)
        base = -1009996800  # 230 records → [base-229, base]
        await _bulk_pending(mock_db, 230, base_chat=base)
        await _close_outside(mock_db, base - 230, base)

        spy = _patch_find(monkeypatch)

        from SWAGGYMUSIC import app

        count = await audit.retry_undelivered_deletion_logs(
            batch_size=audit.DELIVERY_BATCH_SIZE,
        )
        assert count == 230
        assert app.send_message.await_count == 230
        assert len(spy.queries) == 2  # 200 + 30
        for q in spy.queries:
            assert "$nin" not in str(q)
        assert "_id" not in spy.queries[0]
        assert "$gt" in spy.queries[1].get("_id", {})

    async def test_failed_record_does_not_block_later_batches(
        self, monkeypatch, mock_db,
    ):
        """A raising delivery in the FIRST batch must not stop the other
        records, nor the following batches, nor loop the sweep: the
        failed record falls behind the ``_id`` cursor and later batches
        continue; the failure is retried by the NEXT sweep."""
        _logger_env(monkeypatch, logger_id=-1009996600)
        _forbid_history(monkeypatch)
        base = -1009997000  # 250 records → [base-249, base]
        keys = await _bulk_pending(mock_db, 250, base_chat=base)
        poisoned = keys[5]  # inside the first batch
        await _close_outside(mock_db, base - 250, base)

        from SWAGGYMUSIC import app

        original = audit.deliver_deletion_log
        flag = {"raise": True}

        async def _wrapper(event_key, kind, chat_id, message_id, **kwargs):
            if flag["raise"] and event_key == poisoned:
                raise RuntimeError("simulated per-record failure")
            return await original(
                event_key, kind, chat_id, message_id, **kwargs,
            )

        monkeypatch.setattr(audit, "deliver_deletion_log", _wrapper)

        spy = _patch_find(monkeypatch)

        count = await audit.recover_interrupted_deletion_logs(batch_size=200)
        # The other 249 records were still processed — later batches
        # continued after the failure.
        assert count == 249
        assert app.send_message.await_count == 249
        # Two batches were fetched (200 + 50) — the failure did not
        # stop the sweep after the first batch.
        assert len(spy.queries) == 2
        for q in spy.queries:
            assert "$nin" not in str(q)

        rec = await mock_db.deletion_log_delivery.find_one(
            {"event_key": poisoned},
        )
        assert rec["status"] == "pending"  # honest: still not delivered

        # The failed record is retried by the NEXT sweep (the cursor
        # resets every sweep), then everything is delivered.
        flag["raise"] = False
        count2 = await audit.recover_interrupted_deletion_logs()
        assert count2 == 1
        rec = await mock_db.deletion_log_delivery.find_one(
            {"event_key": poisoned},
        )
        assert rec["status"] == "sent"

        assert await _remaining(mock_db, base - 250, base) == 0

    async def test_failed_record_cannot_loop_the_sweep_forever(
        self, monkeypatch, mock_db,
    ):
        """Even when a record fails on EVERY attempt, the sweep
        terminates (each record is visited at most once per sweep) —
        no infinite loop."""
        _logger_env(monkeypatch, logger_id=-1009996700)
        _forbid_history(monkeypatch)
        base = -1009997100
        keys = await _bulk_pending(mock_db, 5, base_chat=base)
        poisoned = keys[2]
        await _close_outside(mock_db, base - 10, base)

        original = audit.deliver_deletion_log

        async def _always_fails(event_key, kind, chat_id, message_id, **kwargs):
            if event_key == poisoned:
                raise RuntimeError("permanent failure")
            return await original(
                event_key, kind, chat_id, message_id, **kwargs,
            )

        monkeypatch.setattr(audit, "deliver_deletion_log", _always_fails)

        spy = _patch_find(monkeypatch)

        count = await audit.recover_interrupted_deletion_logs(batch_size=2)
        assert count == 4  # everyone except the permanently failing one
        # Exactly 3 fetches (5 records / batch 2 → 2+2+1): the sweep
        # advanced monotonically and STOPPED — it did not loop.
        assert len(spy.queries) == 3
        cursors = [q["_id"]["$gt"] for q in spy.queries[1:]]
        assert all(
            cursors[i] < cursors[i + 1] for i in range(len(cursors) - 1)
        )

        # The failing record stays pending for the next sweep.
        rec = await mock_db.deletion_log_delivery.find_one(
            {"event_key": poisoned},
        )
        assert rec["status"] == "pending"

    async def test_recovery_still_never_scans_telegram_history(
        self, monkeypatch, mock_db,
    ):
        """Pagination changed nothing about the source of truth: the
        sweep reads MongoDB only — Telegram history is never scanned."""
        _logger_env(monkeypatch, logger_id=-1009996850)
        _forbid_history(monkeypatch)
        base = -1009997200
        await _bulk_pending(mock_db, 3, base_chat=base)
        await _close_outside(mock_db, base - 10, base)
        count = await audit.recover_interrupted_deletion_logs()
        assert count == 3
