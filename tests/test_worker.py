"""The event-driven path: outcome persistence and the worker loop.

The replay harness proves the pipeline is *correct*. This file covers what
only the worker has: per-order serialisation, scheduler ticks that
actually fire, and rows landing in the shape `analytics/matrix.sql` reads.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.outcomes import (
    COLUMNS,
    VETO_COLUMNS,
    NullOutcomeStore,
    outcome_row,
)
from core.verdicts import Action, Verdict
from harness import scenarios as sc
from services.pipeline import Pipeline
from services.worker import Worker


# ---------------------------------------------------------- outcomes --

async def test_outcome_row_matches_the_analytics_schema():
    """A row that does not match matrix.sql is a silently broken metric."""
    s = sc.SCENARIO_A
    p = Pipeline()
    d = await p.process(s.observations(), s.evaluate_at, order_id=s.order_id)

    row = outcome_row(d, scenario="A", ground_truth=s.ground_truth.value)
    assert set(row) == set(COLUMNS), "outcome_row drifted from the outcomes table"
    assert row["verdict"] == "ORDER_SETTLED"
    assert row["action"] == "NOOP"
    assert row["gate_allowed"] in (0, 1)
    assert isinstance(row["latency_ms"], int)


async def test_veto_rows_match_their_schema():
    """The veto log is a deliverable, so its shape is asserted too."""
    s = sc.SCENARIO_D
    p = Pipeline()
    await p.process(s.observations(), s.start + 3600, order_id=s.order_id)

    store = NullOutcomeStore()
    rows = [{**v, "trace_id": "t"} for v in p.vetoes]
    await store.write_vetoes(rows)

    for r in store.vetoes:
        missing = set(VETO_COLUMNS) - set(r)
        assert not missing, f"veto row missing {missing}"


async def test_null_store_keeps_rows_when_no_database_is_up():
    store = NullOutcomeStore()
    assert store.backend == "memory"
    await store.init()
    await store.write_outcome({"order_id": "o1"})
    await store.write_vetoes([{"rule": "I3"}])
    assert store.outcomes and store.vetoes


# ------------------------------------------------------------ worker --

@pytest.fixture
async def worker():
    """A worker with an in-memory outcome store.

    These are unit tests of the worker's own behaviour - locking, the
    scheduler tick, what gets written - so the outcome backend is pinned
    rather than left to whichever database happens to be running. With
    Docker up the worker would otherwise pick ClickHouse and these
    assertions would depend on the environment. The real ClickHouse and
    Postgres writes are covered in tests/test_integration.py.
    """
    w = Worker(dry_run=True)
    await w.start()
    w.outcomes = NullOutcomeStore()
    yield w
    await w.stop()


async def test_worker_starts_degraded_without_crashing(worker):
    """Chaos-adjacent: no backing services means degraded, not dead."""
    assert worker.pipeline is not None
    assert worker.outcomes is not None
    assert worker.scheduler is not None


async def test_worker_resolves_an_order_and_persists_the_outcome(worker):
    s = sc.SCENARIO_A
    for o in s.observations():
        await worker.infra.store.append(o)

    await worker.handle({"order_id": s.order_id}, now=s.evaluate_at)

    assert worker.processed == 1
    assert worker.outcomes.outcomes, "no outcome row was written"
    assert worker.outcomes.outcomes[-1]["verdict"] == Verdict.ORDER_SETTLED.value


async def test_worker_ignores_events_with_no_stored_observations(worker):
    await worker.handle({"order_id": "order_that_does_not_exist"})
    assert worker.processed == 0
    assert worker.skipped == 1


async def test_worker_serialises_concurrent_events_for_one_order(worker):
    """Two concurrent decisions on one order could each see no prior
    action and both send a link. The per-order lock is what stops that."""
    s = sc.SCENARIO_B
    for o in s.observations():
        await worker.infra.store.append(o)

    # `now` is pinned to the scenario's clock: the fixture is dated
    # January, and a real-time clock would put it outside the 72h
    # service-implicit consent window and veto on compliance instead.
    await asyncio.gather(
        *[worker.handle({"order_id": s.order_id}, now=s.evaluate_at) for _ in range(5)]
    )

    executed = [
        o for o in worker.pipeline.executor.outcomes
        if o.action == Action.SEND_RECOVERY_LINK and o.status in ("EXECUTED", "STUBBED")
    ]
    assert len(executed) == 1, f"sent {len(executed)} links for one order"
    assert worker.processed == 5, "later passes must still be evaluated, just not act"


async def test_scheduler_tick_refolds_a_due_order(worker):
    """PENDING_TAT is a deferral. Something has to come back for it."""
    s = sc.SCENARIO_D
    for o in s.observations():
        await worker.infra.store.append(o)

    # Due in the past, so the first tick picks it up immediately.
    await worker.scheduler.schedule(s.order_id, int(time.time()) - 10)

    task = asyncio.create_task(worker.tick_scheduler(interval_s=0.05))
    await asyncio.sleep(0.3)
    worker._stop.set()
    await asyncio.gather(task, return_exceptions=True)
    worker._stop.clear()

    assert worker.processed >= 1
    assert not await worker.scheduler.pending(), "due item was not consumed"
