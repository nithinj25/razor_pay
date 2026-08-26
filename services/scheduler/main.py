"""Durable recheck timers. Redis sorted set, score = due timestamp.

PENDING_TAT is not an answer, it is a deferral - so something has to come
back. This is that something. It must survive a restart, because the
whole point of waiting out the RBI window is that the wait is longer than
a process lifetime (scenario D waits from Friday evening to Tuesday).

Due times are computed in banking days by `core.banking`. A naive
`now + 86400` here would re-fold on the Saturday, find nothing changed,
and hand a link to the strategist while the bank was still going to
reverse the debit on Tuesday.
"""

from __future__ import annotations

import time
from typing import Any, Protocol

ZKEY = "nishchay:recheck"


class Scheduler(Protocol):
    async def schedule(self, order_id: str, due_ts: int) -> None: ...
    async def due(self, now: int) -> list[str]: ...
    async def pending(self) -> list[tuple[str, int]]: ...


class InMemoryScheduler:
    """Non-durable twin. Correct for a single-process demo run."""

    def __init__(self) -> None:
        self._z: dict[str, int] = {}

    async def schedule(self, order_id: str, due_ts: int) -> None:
        # Earliest wins: a tighter deadline must not be pushed out by a
        # later re-fold of the same order.
        cur = self._z.get(order_id)
        self._z[order_id] = min(cur, due_ts) if cur is not None else due_ts

    async def due(self, now: int) -> list[str]:
        ready = sorted([k for k, v in self._z.items() if v <= now], key=lambda k: self._z[k])
        for k in ready:
            del self._z[k]
        return ready

    async def pending(self) -> list[tuple[str, int]]:
        return sorted(self._z.items(), key=lambda kv: kv[1])


class RedisScheduler:
    """Durable across restarts, which is the entire requirement."""

    def __init__(self, client: Any) -> None:
        self.client = client

    async def schedule(self, order_id: str, due_ts: int) -> None:
        # GT=False + NX semantics: keep the earliest deadline.
        existing = await self.client.zscore(ZKEY, order_id)
        if existing is None or due_ts < existing:
            await self.client.zadd(ZKEY, {order_id: due_ts})

    async def due(self, now: int) -> list[str]:
        ids = await self.client.zrangebyscore(ZKEY, 0, now)
        if ids:
            await self.client.zrem(ZKEY, *ids)
        return list(ids)

    async def pending(self) -> list[tuple[str, int]]:
        rows = await self.client.zrange(ZKEY, 0, -1, withscores=True)
        return [(m, int(s)) for m, s in rows]


async def run_worker(
    scheduler: Scheduler,
    on_due,
    stop_after: int | None = None,
    tick_s: float = 1.0,
    clock=time.time,
) -> int:
    """Poll for due rechecks and re-fold. Returns how many it processed.

    `on_due(order_id, now)` is the re-resolution. It is passed in rather
    than imported so the worker can be tested without a resolver.
    """
    import asyncio

    processed = 0
    while True:
        now = int(clock())
        for order_id in await scheduler.due(now):
            await on_due(order_id, now)
            processed += 1
        if stop_after is not None and processed >= stop_after:
            return processed
        await asyncio.sleep(tick_s)
