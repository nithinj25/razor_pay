"""Fetchers: pull wide, return narrow.

Every fetcher returns an `Evidence` - a handful of fields with a source,
a confidence and a provenance string - never raw API JSON. The downtime
endpoint alone returns ~40KB; the Evidence it produces is ~180 bytes.
That is Bumblebee's "prune at the edge" practice, and it is what keeps
the resolver's prompt small enough to reason over.

The second rule is that a fetcher never raises into the graph. A timeout,
a 429, an open circuit and a malformed body all produce
`Evidence.unavailable(...)` with confidence 0.0, which lowers the
verdict's confidence, which fails the gate's floor, which yields NOOP.
Degradation biases toward inaction (I9), and it does so through the
ordinary path rather than an error handler.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from core.verdicts import Evidence

FetcherFn = Callable[["FetchContext"], Awaitable[Evidence]]


@dataclass
class FetchContext:
    """Everything a fetcher may look at. Deliberately small."""

    order_id: str
    payment_id: str | None = None
    method: str | None = None
    amount: int = 0
    now: int = 0
    rrn: str | None = None
    client: Any = None                       # httpx.AsyncClient or a double
    extra: dict[str, Any] = field(default_factory=dict)


class CircuitBreaker:
    """Trip after N consecutive failures; recover after a cooldown.

    Without this, a downed Razorpay endpoint costs every resolution its
    full timeout, and the chaos-6 429 storm turns into a thundering herd.
    """

    def __init__(self, name: str, threshold: int = 3, cooldown_s: float = 30.0):
        self.name = name
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_at: float | None = None

    def is_open(self, now: float) -> bool:
        if self.opened_at is None:
            return False
        if now - self.opened_at >= self.cooldown_s:
            self.opened_at = None
            self.failures = 0
            return False
        return True

    def record(self, ok: bool, now: float) -> None:
        if ok:
            self.failures = 0
            self.opened_at = None
        else:
            self.failures += 1
            if self.failures >= self.threshold:
                self.opened_at = now


async def guarded(
    name: str,
    fn: FetcherFn,
    ctx: FetchContext,
    breaker: CircuitBreaker,
    timeout_s: float = 3.0,
    clock=None,
) -> Evidence:
    """Run one fetcher under a breaker and a timeout. Never raises."""
    import time as _t

    now = (clock or _t.monotonic)()
    if breaker.is_open(now):
        return Evidence.unavailable(name, "circuit open")
    try:
        ev = await asyncio.wait_for(fn(ctx), timeout=timeout_s)
        breaker.record(True, now)
        return ev
    except asyncio.TimeoutError:
        breaker.record(False, now)
        return Evidence.unavailable(name, f"timeout after {timeout_s}s")
    except Exception as e:                     # noqa: BLE001
        breaker.record(False, now)
        return Evidence.unavailable(name, f"{type(e).__name__}: {e}")


async def gather_evidence(
    fetchers: dict[str, FetcherFn],
    ctx: FetchContext,
    breakers: dict[str, CircuitBreaker],
    timeout_s: float = 3.0,
) -> tuple[Evidence, ...]:
    """Run every fetcher concurrently. Partial results are the norm."""
    names = list(fetchers)
    results = await asyncio.gather(
        *[
            guarded(n, fetchers[n], ctx, breakers.setdefault(n, CircuitBreaker(n)), timeout_s)
            for n in names
        ],
        return_exceptions=True,
    )
    out: list[Evidence] = []
    for n, r in zip(names, results):
        out.append(r if isinstance(r, Evidence) else Evidence.unavailable(n, str(r)))
    return tuple(out)
