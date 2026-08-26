"""Infrastructure with local fallbacks.

Every dependency here has an in-process twin. That is not convenience:
the slip protocol says Kafka may be dropped for an asyncio queue and
ClickHouse for Postgres, and chaos 3/4/6 all require killing a dependency
and watching the system degrade rather than crash. Making the fallback a
first-class object means "degraded" is a tested state, not an outage.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from typing import Any, AsyncIterator, Protocol

from core.config import Settings, settings


# --------------------------- dedupe ---------------------------

class Dedupe(Protocol):
    async def claim(self, key: str, ttl_s: int = 7 * 86_400) -> bool: ...


class InMemoryDedupe:
    """Process-local. Correct for one replica; the demo runs one."""

    def __init__(self) -> None:
        self._seen: dict[str, float] = {}

    async def claim(self, key: str, ttl_s: int = 7 * 86_400) -> bool:
        now = time.time()
        exp = self._seen.get(key)
        if exp is not None and exp > now:
            return False
        self._seen[key] = now + ttl_s
        return True


class RedisDedupe:
    """SET NX EX - the claim and the check are one atomic operation."""

    def __init__(self, client: Any) -> None:
        self.client = client

    async def claim(self, key: str, ttl_s: int = 7 * 86_400) -> bool:
        return bool(await self.client.set(key, 1, nx=True, ex=ttl_s))


# ----------------------------- bus -----------------------------

class Bus(Protocol):
    async def publish(self, topic: str, key: str, value: dict[str, Any]) -> None: ...
    def subscribe(self, topic: str) -> AsyncIterator[dict[str, Any]]: ...


class InProcessBus:
    """asyncio queues. The documented Kafka fallback (Day 5 cut)."""

    def __init__(self) -> None:
        self._topics: dict[str, list[asyncio.Queue]] = {}
        self.published: list[tuple[str, str, dict]] = []

    def _queues(self, topic: str) -> list[asyncio.Queue]:
        return self._topics.setdefault(topic, [])

    async def publish(self, topic: str, key: str, value: dict[str, Any]) -> None:
        self.published.append((topic, key, value))
        for q in self._queues(topic):
            q.put_nowait({"key": key, "value": value})

    async def subscribe(self, topic: str) -> AsyncIterator[dict[str, Any]]:
        q: asyncio.Queue = asyncio.Queue()
        self._queues(topic).append(q)
        try:
            while True:
                yield await q.get()
        finally:
            with contextlib.suppress(ValueError):
                self._queues(topic).remove(q)


class KafkaBus:
    """Redpanda over the Kafka API.

    The key is always `order_id` (pitfall #6). Siblings on different
    partitions cannot be compared by a single consumer, and comparing
    siblings is the entire purpose of the system (I3).
    """

    def __init__(self, bootstrap: str) -> None:
        self.bootstrap = bootstrap
        self._producer: Any = None

    async def start(self) -> None:
        from aiokafka import AIOKafkaProducer

        producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap,
            value_serializer=lambda v: json.dumps(v).encode(),
            key_serializer=lambda k: k.encode(),
            enable_idempotence=True,
        )
        try:
            await producer.start()
        except Exception:
            # Close the half-built producer before propagating, or its
            # background tasks outlive the failed connection attempt.
            with contextlib.suppress(Exception):
                await producer.stop()
            raise
        self._producer = producer

    async def stop(self) -> None:
        if self._producer:
            await self._producer.stop()

    async def publish(self, topic: str, key: str, value: dict[str, Any]) -> None:
        await self._producer.send_and_wait(topic, value=value, key=key)

    async def subscribe(self, topic: str, group: str = "nishchay") -> AsyncIterator[dict]:
        from aiokafka import AIOKafkaConsumer

        c = AIOKafkaConsumer(
            topic,
            bootstrap_servers=self.bootstrap,
            group_id=group,
            value_deserializer=lambda v: json.loads(v.decode()),
            key_deserializer=lambda k: k.decode() if k else "",
            auto_offset_reset="earliest",
        )
        await c.start()
        try:
            async for msg in c:
                yield {"key": msg.key, "value": msg.value}
        finally:
            await c.stop()


# --------------------------- wiring ---------------------------

class Infra:
    """Everything a service needs, with whatever backend is reachable.

    `degraded` names the things we could not reach. It is surfaced on the
    console rather than swallowed - an operator must be able to see that
    a verdict was reached without a dependency (I9).
    """

    def __init__(self, cfg: Settings | None = None) -> None:
        self.cfg = cfg or settings()
        self.store: Any = None
        self.dedupe: Dedupe = InMemoryDedupe()
        self.bus: Any = InProcessBus()
        self.pool: Any = None
        self.redis: Any = None
        self.degraded: list[str] = []

    async def start(self) -> None:
        from core.events import InMemoryStore, PostgresStore

        try:
            import asyncpg

            self.pool = await asyncpg.create_pool(
                self.cfg.postgres_dsn, min_size=1, max_size=8, timeout=5
            )
            self.store = PostgresStore(self.pool)
            await self.store.init()
        except Exception as e:                       # noqa: BLE001 - degrade, never crash
            self.degraded.append(f"postgres: {type(e).__name__}")
            self.store = InMemoryStore()

        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(self.cfg.redis_url, decode_responses=True)
            await client.ping()
            # Only publish the handle once the ping has actually
            # succeeded. Assigning before the probe leaves a dead client
            # on `self.redis`, and every downstream truthiness check
            # ("use Redis if we have it") then picks the broken path.
            self.redis = client
            self.dedupe = RedisDedupe(client)
        except Exception as e:                       # noqa: BLE001
            self.degraded.append(f"redis: {type(e).__name__}")
            self.redis = None
            self.dedupe = InMemoryDedupe()

        if self.cfg.enable_kafka:
            try:
                bus = KafkaBus(self.cfg.kafka_bootstrap)
                await bus.start()
                self.bus = bus
            except Exception as e:                   # noqa: BLE001
                self.degraded.append(f"kafka: {type(e).__name__}")
                self.bus = InProcessBus()

    async def stop(self) -> None:
        for closer in (
            getattr(self.bus, "stop", None),
            getattr(self.redis, "aclose", None),
            getattr(self.pool, "close", None),
        ):
            if closer:
                with contextlib.suppress(Exception):
                    await closer()


def use_psycopg_compatible_loop() -> bool:
    """Windows only: switch off the ProactorEventLoop before any loop exists.

    psycopg refuses to run in async mode on Windows' default
    ProactorEventLoop, and LangGraph's Postgres checkpointer is built on
    psycopg. Left alone, the resolver silently degrades to an in-memory
    checkpointer on a Windows dev box - which is precisely where chaos 5
    ("kill the resolver mid-flight, resume, identical verdict") gets
    demonstrated, so the degradation would hide the thing being shown.

    In the containers this is a no-op: they run Linux, where the default
    loop is already compatible.

    Must be called before the first event loop is created. Returns True
    if the policy was changed.
    """
    import sys

    if sys.platform != "win32":
        return False
    selector = getattr(asyncio, "WindowsSelectorEventLoopPolicy", None)
    if selector is None:
        return False
    if isinstance(asyncio.get_event_loop_policy(), selector):
        return False
    # The tradeoff: SelectorEventLoop cannot spawn subprocesses and caps
    # at 512 sockets. Neither matters here - nothing in this service
    # shells out, and a single-merchant demo is nowhere near the cap.
    asyncio.set_event_loop_policy(selector())
    return True
