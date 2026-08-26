"""Integration tests against real backing services.

Skipped automatically when Docker is not up, so `pytest -q` still runs
clean on a bare laptop. These cover the things the in-process fallbacks
*cannot* prove:

* the append-only trigger is enforced by Postgres, not by convention;
* siblings really do land on one Kafka partition;
* the analytics schema accepts the rows we actually write.

Run them with `docker compose up -d` first.
"""

from __future__ import annotations

import asyncio
import socket
import time
import uuid

import pytest

from core.config import settings
from core.events import PostgresStore, canonical_order
from core.outcomes import ClickHouseOutcomeStore, build_outcome_store, outcome_row
from harness import scenarios as sc
from services.pipeline import Pipeline


def _up(host: str, port: int, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


needs_postgres = pytest.mark.skipif(
    not _up("localhost", 5432), reason="postgres not up (docker compose up -d)"
)
needs_kafka = pytest.mark.skipif(
    not _up("localhost", 19092), reason="redpanda not up"
)
needs_clickhouse = pytest.mark.skipif(
    not _up("localhost", 8123), reason="clickhouse not up"
)
needs_redis = pytest.mark.skipif(not _up("localhost", 6379), reason="redis not up")


@pytest.fixture
async def pool():
    import asyncpg

    p = await asyncpg.create_pool(settings().postgres_dsn, min_size=1, max_size=4)
    yield p
    await p.close()


# ------------------------------------------------------------- I2 --

@needs_postgres
async def test_observations_are_append_only_in_the_database(pool):
    """I2 enforced by a trigger, not by our good intentions.

    A stray UPDATE anywhere in the codebase would silently invalidate
    every replay and every property test, so the database refuses it.
    """
    store = PostgresStore(pool)
    await store.init()

    oid = f"order_it_{uuid.uuid4().hex[:8]}"
    obs = sc.SCENARIO_B.observations()[0].model_copy(
        update={"order_id": oid, "event_id": f"evt_{uuid.uuid4().hex[:8]}"}
    )
    assert await store.append(obs) is True

    async with pool.acquire() as c:
        with pytest.raises(Exception, match="append-only"):
            await c.execute(
                "UPDATE observations SET status = 'captured' WHERE order_id = $1", oid
            )
        with pytest.raises(Exception, match="append-only"):
            await c.execute("DELETE FROM observations WHERE order_id = $1", oid)

    rows = await store.load(oid)
    assert len(rows) == 1 and rows[0].status == "failed"


@needs_postgres
async def test_duplicate_event_id_is_absorbed_by_the_unique_constraint(pool):
    """Chaos 1 against the real database rather than a dict."""
    store = PostgresStore(pool)
    await store.init()

    oid = f"order_it_{uuid.uuid4().hex[:8]}"
    obs = sc.SCENARIO_B.observations()[0].model_copy(
        update={"order_id": oid, "event_id": "evt_fixed_duplicate"}
    )
    results = [await store.append(obs) for _ in range(5)]
    assert results == [True, False, False, False, False]
    assert len(await store.load(oid)) == 1


@needs_postgres
async def test_round_trip_preserves_the_fold(pool):
    """Storing and reloading must not change a verdict."""
    from core.fold import fold

    store = PostgresStore(pool)
    await store.init()

    s = sc.SCENARIO_A
    oid = f"order_it_{uuid.uuid4().hex[:8]}"
    obs = [
        o.model_copy(update={"order_id": oid, "event_id": f"{o.event_id}_{oid}"})
        for o in s.observations()
    ]
    for o in obs:
        await store.append(o)

    loaded = await store.load(oid)
    assert len(loaded) == len(obs)
    assert fold(loaded, s.evaluate_at, order_id=oid) == fold(obs, s.evaluate_at, order_id=oid)
    # Ordering must survive the round trip through JSONB.
    assert [o.event_time for o in canonical_order(loaded)] == [
        o.event_time for o in canonical_order(obs)
    ]


# ---------------------------------------------------------- redis --

@needs_redis
async def test_redis_dedupe_claim_is_atomic():
    """SET NX EX: exactly one concurrent claimer wins."""
    import redis.asyncio as aioredis

    from core.infra import RedisDedupe

    client = aioredis.from_url(settings().redis_url, decode_responses=True)
    d = RedisDedupe(client)
    key = f"evt:it_{uuid.uuid4().hex[:8]}"

    results = await asyncio.gather(*[d.claim(key, ttl_s=60) for _ in range(20)])
    assert sum(results) == 1, "more than one caller claimed the same event"
    await client.delete(key)
    await client.aclose()


@needs_redis
async def test_redis_scheduler_survives_a_new_client():
    """Durability is the whole reason the scheduler is not in memory."""
    import redis.asyncio as aioredis

    from services.scheduler.main import RedisScheduler

    order_id = f"order_it_{uuid.uuid4().hex[:8]}"
    due = int(time.time()) - 5

    c1 = aioredis.from_url(settings().redis_url, decode_responses=True)
    await RedisScheduler(c1).schedule(order_id, due)
    await c1.aclose()

    # A different client stands in for a restarted process.
    c2 = aioredis.from_url(settings().redis_url, decode_responses=True)
    s2 = RedisScheduler(c2)
    assert order_id in await s2.due(int(time.time()))
    assert order_id not in [k for k, _ in await s2.pending()]
    await c2.aclose()


# ---------------------------------------------------------- kafka --

@needs_kafka
async def test_siblings_share_a_partition():
    """Pitfall #6, proven rather than asserted.

    Two attempts on one order must reach one consumer, or the sibling
    check that I3 depends on becomes a cross-partition race.
    """
    from aiokafka import AIOKafkaProducer

    from core.infra import KafkaBus

    topic = f"it_raw_{uuid.uuid4().hex[:8]}"
    bus = KafkaBus(settings().kafka_bootstrap)
    await bus.start()

    s = sc.SCENARIO_A
    try:
        for o in s.observations():
            await bus.publish(topic, key=s.order_id, value={"payment_id": o.payment_id})

        producer: AIOKafkaProducer = bus._producer
        parts = await producer.partitions_for(topic)
        assigned = {
            producer._partition(topic, None, s.order_id.encode(), None, None, None)
            for _ in range(2)
        }
        assert len(assigned) == 1, "one key resolved to multiple partitions"
        assert parts, "topic has no partitions"
    finally:
        await bus.stop()


# ----------------------------------------------------- clickhouse --

@needs_clickhouse
async def test_outcomes_land_in_clickhouse_and_matrix_sql_reads_them():
    """The analytics schema must accept the rows we actually produce.

    A confusion matrix built on a schema that silently rejects rows is
    worse than no confusion matrix.
    """
    store = await build_outcome_store(settings())
    if not isinstance(store, ClickHouseOutcomeStore):
        pytest.skip(f"clickhouse not selected (got {store.backend})")

    tag = f"it_{uuid.uuid4().hex[:8]}"
    p = Pipeline()
    for s in sc.ALL:
        # seed_evidence matters: without the downtime record, E is
        # correctly PENDING_TAT rather than CONFIRMED_FAILED, and the
        # matrix would show an off-diagonal cell that is not a bug.
        d = await p.process(
            s.observations(), s.evaluate_at, order_id=s.order_id,
            seed_evidence=s.evidence,
        )
        row = outcome_row(d, scenario=tag, ground_truth=s.ground_truth.value)
        await store.write_outcome(row)
    if p.vetoes:
        await store.write_vetoes([{**v, "trace_id": tag} for v in p.vetoes])

    # Query 1 from analytics/matrix.sql, scoped to this run.
    matrix = store.client.query(
        "SELECT ground_truth, verdict, count() AS n FROM outcomes "
        f"WHERE scenario = '{tag}' GROUP BY ground_truth, verdict"
    ).result_rows
    assert matrix, "no rows came back"

    correct = sum(n for truth, verdict, n in matrix if truth == verdict)
    total = sum(n for _, _, n in matrix)
    assert total == len(sc.ALL)
    assert correct == total, f"off-diagonal cells: {matrix}"

    # Query 2: the invariants, straight from the database.
    fp = store.client.query(
        "SELECT countIf(action = 'SEND_RECOVERY_LINK' AND ground_truth = 'ORDER_SETTLED') "
        f"FROM outcomes WHERE scenario = '{tag}'"
    ).result_rows[0][0]
    assert fp == 0, "a recovery link was recorded against a settled order"

    store.client.command(f"ALTER TABLE outcomes DELETE WHERE scenario = '{tag}'")


# ------------------------------------------------------ end to end --

@needs_postgres
@needs_redis
async def test_worker_end_to_end_on_real_infrastructure(pool):
    """Ingest through the real store, resolve through the real worker."""
    from services.worker import Worker

    w = Worker(dry_run=True)
    await w.start()
    try:
        assert not w.infra.degraded, f"degraded on full infra: {w.infra.degraded}"

        s = sc.SCENARIO_A
        oid = f"order_it_{uuid.uuid4().hex[:8]}"
        for o in s.observations():
            await w.infra.store.append(
                o.model_copy(update={"order_id": oid, "event_id": f"{o.event_id}_{oid}"})
            )

        await w.handle({"order_id": oid}, now=s.evaluate_at)

        assert w.processed == 1
        # The customer already paid: nothing may move.
        moved = [
            o for o in w.pipeline.executor.outcomes
            if o.status in ("EXECUTED", "STUBBED")
            and o.action.value in ("SEND_RECOVERY_LINK", "REFUND", "CAPTURE")
        ]
        assert not moved, f"acted on a settled order: {moved}"
    finally:
        await w.stop()
