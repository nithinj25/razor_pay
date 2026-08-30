"""The resolution worker. Consumes the `raw` topic and drives the pipeline.

This is the process that runs in production. `harness/replay.py` calls
the same `Pipeline` synchronously for deterministic tests; this one is
the event-driven path: Kafka in, triage, resolve, gate, execute, persist,
and a scheduler tick for the deferrals.

Two properties are load-bearing here and not in the replay path:

* **Per-order serialisation.** Messages are keyed on `order_id`, so
  siblings share a partition — but a single consumer can still interleave
  two messages for one order across an `await`. A per-order lock makes the
  read-fold-decide sequence atomic, because two concurrent decisions on
  one order could each see no prior action and both send a link.
* **The scheduler actually runs.** PENDING_TAT is a deferral, and without
  this loop nothing ever comes back for scenario D.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import signal
import time
from collections import defaultdict
from typing import Any

from core.config import Settings, settings
from core.fold import FoldConfig
from core.infra import Infra, use_psycopg_compatible_loop
from core.llm import build_llm
from core.outcomes import build_outcome_store, decision_row, outcome_row
from services.pipeline import Pipeline
from services.scheduler.main import InMemoryScheduler, RedisScheduler
from services.triage.classify import Route, triage

TOPIC_RAW = "raw"


class Worker:
    def __init__(self, cfg: Settings | None = None, dry_run: bool = True):
        self.cfg = cfg or settings()
        self.infra = Infra(self.cfg)
        self.dry_run = dry_run
        self.pipeline: Pipeline | None = None
        self.outcomes: Any = None
        self.scheduler: Any = None
        self._locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)
        self._stop = asyncio.Event()
        self.processed = 0
        self.skipped = 0

    async def start(self) -> None:
        await self.infra.start()
        self.outcomes = await build_outcome_store(self.cfg, self.infra.pool)

        self.scheduler = (
            RedisScheduler(self.infra.redis) if self.infra.redis else InMemoryScheduler()
        )

        from services.executor.main import Executor

        self.pipeline = Pipeline(
            llm=build_llm(self.cfg),
            executor=Executor(dry_run=self.dry_run, cfg=self.cfg),
            scheduler=self.scheduler,
            fold_cfg=FoldConfig(
                tat_banking_days=self.cfg.tat_window_banking_days,
                settle_horizon_days=self.cfg.settle_horizon_days,
            ),
            resolver=await self._resolver(),
        )

        print(
            f"[worker] outcomes={self.outcomes.backend} "
            f"scheduler={'redis' if self.infra.redis else 'memory'} "
            f"bus={type(self.infra.bus).__name__} "
            f"dry_run={self.dry_run}"
            + (f" degraded={self.infra.degraded}" if self.infra.degraded else "")
        )

    async def _resolver(self):
        """Resolver with a durable checkpointer when Postgres is reachable.

        The checkpointer is what makes chaos 5 a property of the framework
        rather than something we reimplement: a graph killed mid-flight
        resumes from its last checkpoint. Without Postgres it falls back to
        in-memory, which still recovers within a process.
        """
        from services.resolver.graph import Resolver

        checkpointer = None
        if self.infra.pool is not None:
            try:
                from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

                cm = AsyncPostgresSaver.from_conn_string(self.cfg.postgres_dsn)
                checkpointer = await cm.__aenter__()
                await checkpointer.setup()
                self._checkpoint_cm = cm
            except Exception as e:                 # noqa: BLE001
                self.infra.degraded.append(f"checkpointer: {type(e).__name__}")
                checkpointer = None

        return Resolver(llm=build_llm(self.cfg), cfg=self.cfg, checkpointer=checkpointer)

    # ------------------------------ loops ------------------------------

    async def consume(self) -> None:
        """Kafka `raw` -> triage -> pipeline. One order at a time."""
        sub = self.infra.bus.subscribe(TOPIC_RAW)
        async for msg in sub:
            if self._stop.is_set():
                break
            await self.handle(msg.get("value", {}))

    async def handle(self, event: dict, now: int | None = None) -> None:
        """Resolve one order.

        `now` is a parameter for the same reason it is one in `fold`:
        this is the only place in the service that reads a clock, and a
        clock that cannot be injected cannot be tested against a dated
        fixture.
        """
        order_id = event.get("order_id") or event.get("key") or ""
        if not order_id:
            return

        # Serialise per order: two concurrent decisions on one order could
        # each observe no prior action and both act.
        async with self._locks[order_id]:
            observations = await self.infra.store.load(order_id)
            if not observations:
                self.skipped += 1
                return

            latest = max(observations, key=lambda o: o.event_time)
            route, _, why = triage(latest)
            if route is Route.IGNORE:
                self.skipped += 1
                return

            await self._decide(order_id, observations, now or int(time.time()), why)

    async def _decide(self, order_id: str, observations: list, now: int, why: str = "") -> None:
        assert self.pipeline is not None
        d = await self.pipeline.process(observations, now, order_id=order_id)
        self.processed += 1

        await self.outcomes.write_outcome(outcome_row(d))
        # The trace only exists in memory during the run. A merchant asking
        # "what did the agent do on my order?" needs it written down.
        await self.outcomes.write_decision(decision_row(d))
        if self.pipeline.vetoes:
            await self.outcomes.write_vetoes(self.pipeline.vetoes)
            self.pipeline.vetoes = []

        print(
            f"[worker] {order_id} {d.verdict.verdict.value}@{d.verdict.confidence:.2f} "
            f"-> {d.action.value} [{d.outcome.status if d.outcome else '-'}]"
            + (f"  veto: {d.gate.reason}" if d.gate and not d.gate.allowed else "")
        )

    async def tick_scheduler(self, interval_s: float = 5.0) -> None:
        """Re-fold orders whose banking-day recheck has come due.

        Without this, PENDING_TAT is a promise nobody keeps.
        """
        while not self._stop.is_set():
            now = int(time.time())
            for order_id in await self.scheduler.due(now):
                observations = await self.infra.store.load(order_id)
                if observations:
                    async with self._locks[order_id]:
                        await self._decide(order_id, observations, now, "scheduled recheck")
            with contextlib.suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stop.wait(), timeout=interval_s)

    async def run(self) -> None:
        await self.start()
        tasks = [
            asyncio.create_task(self.consume()),
            asyncio.create_task(self.tick_scheduler()),
        ]
        try:
            await self._stop.wait()
        finally:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self.stop()

    async def stop(self) -> None:
        self._stop.set()
        cm = getattr(self, "_checkpoint_cm", None)
        if cm is not None:
            with contextlib.suppress(Exception):
                await cm.__aexit__(None, None, None)
        await self.infra.stop()
        print(f"[worker] stopped. processed={self.processed} skipped={self.skipped}")


async def main_async(args) -> None:
    w = Worker(dry_run=not args.live)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        with contextlib.suppress(NotImplementedError, AttributeError):
            loop.add_signal_handler(sig, w._stop.set)

    await w.run()


def main() -> None:
    ap = argparse.ArgumentParser(description="Nishchay resolution worker.")
    ap.add_argument(
        "--live", action="store_true",
        help="actually call the Razorpay API (default is dry run)",
    )
    # Before any loop is created, or the Postgres checkpointer degrades.
    use_psycopg_compatible_loop()
    try:
        asyncio.run(main_async(ap.parse_args()))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
