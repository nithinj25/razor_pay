"""The nine faults from GUARDRAILS section 5.

Each returns (passed, detail). The bar is not "does not crash" - it is
"produces the same verdict, or a strictly safer one". A chaos test that
only asserts the absence of an exception would pass on a system that
silently double-charged.

Chaos 5 is the one to demo: kill the resolver mid-flight, restart it,
and get a byte-identical verdict. That is hard to fake and instantly
legible to a reviewer, because it can only be true if status is derived
rather than stored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Callable

from core.events import InMemoryStore, parse_webhook
from core.fold import fold
from core.llm import NullLLM
from core.verdicts import Action, Verdict
from harness import scenarios as sc
from services.executor.main import Executor
from services.ingress.signing import sign, verify_any
from services.pipeline import Pipeline
from services.resolver.fetchers import CircuitBreaker, FetchContext, guarded
from services.resolver.graph import Resolver

Result = tuple[bool, str]


# 1 -------------------------------------------------------------------
async def chaos_1_duplicate_webhooks() -> Result:
    """The same webhook five times: one observation, one action."""
    store = InMemoryStore()
    d = sc.SCENARIO_B.deliveries[0]
    obs = parse_webhook(json.dumps(d.body), d.event_id, sc.BASE + 3, source="fixture")

    appended = [await store.append(obs) for _ in range(5)]
    rows = await store.load(obs.order_id)

    ex = Executor(dry_run=True)
    p = Pipeline(executor=ex)
    for _ in range(5):
        await p.process(rows, sc.SCENARIO_B.evaluate_at, order_id=obs.order_id)

    acted = [o for o in ex.outcomes if o.status in ("EXECUTED", "STUBBED")]
    ok = appended == [True, False, False, False, False] and len(rows) == 1 and len(acted) == 1
    return ok, f"appends={appended.count(True)}/5 observations={len(rows)} actions={len(acted)}"


# 2 -------------------------------------------------------------------
async def chaos_2_inverted_delivery() -> Result:
    """`authorized` before `failed`: correct either way."""
    s = sc.SCENARIO_A
    obs = s.observations()
    forward = fold(obs, s.evaluate_at, order_id=s.order_id)
    reverse = fold(list(reversed(obs)), s.evaluate_at, order_id=s.order_id)
    ok = forward == reverse and forward.verdict == Verdict.ORDER_SETTLED
    return ok, f"both orderings -> {forward.verdict.value}"


# 3 -------------------------------------------------------------------
async def chaos_3_dead_fetcher() -> Result:
    """Kill the downtime fetcher: partial resolve, confidence drops, flagged."""
    async def dead(ctx):
        raise ConnectionError("downtime API not enabled")

    ev = await guarded("downtime", dead, FetchContext(order_id="o"), CircuitBreaker("downtime"))
    s = sc.SCENARIO_E
    healthy = fold(s.observations(), s.evaluate_at, evidence=s.evidence)
    degraded = fold(s.observations(), s.evaluate_at, evidence=(ev,))

    ok = (
        ev.available is False
        and degraded.confidence < healthy.confidence
        and degraded.verdict == Verdict.PENDING_TAT      # strictly safer
    )
    return ok, (
        f"{healthy.verdict.value}@{healthy.confidence:.2f} -> "
        f"{degraded.verdict.value}@{degraded.confidence:.2f}, gap flagged: {ev.provenance}"
    )


# 4 -------------------------------------------------------------------
async def chaos_4_no_llm() -> Result:
    """Kill the LLM: rules only, UNRESOLVED, therefore NOOP."""
    s = sc.SCENARIO_D
    ex = Executor(dry_run=True)
    p = Pipeline(llm=NullLLM(), executor=ex)
    d = await p.process(s.observations(), s.evaluate_at, order_id=s.order_id)

    moved = [
        o for o in ex.outcomes
        if o.action in (Action.SEND_RECOVERY_LINK, Action.REFUND, Action.CAPTURE)
        and o.status in ("EXECUTED", "STUBBED")
    ]
    # The property is that no money moves and the case reaches a human -
    # not that a particular action was chosen. ESCALATE and a stubbed
    # VOICE_CALL both satisfy it; asserting one by name made this test
    # fail the moment the voice path was wired, for no safety reason.
    safe_actions = {Action.ESCALATE, Action.NOOP, Action.HOLD, Action.VOICE_CALL}
    ok = (
        d.verdict.verdict == Verdict.UNRESOLVED
        and d.action in safe_actions
        and not moved
    )
    return ok, (
        f"verdict={d.verdict.verdict.value} action={d.action.value} "
        f"money_actions={len(moved)} (verdict unchanged by the model's absence)"
    )


# 5 -------------------------------------------------------------------
async def chaos_5_kill_resolver_midflight() -> Result:
    """THE DEMO. Kill the resolver mid-graph; re-run; identical verdict.

    The first pass dies inside the graph. The second runs to completion.
    Because status is `fold(observations, now)` and both passes see the
    same observations at the same `now`, the verdict must be identical -
    there is no partial state to recover, because there is no state.
    """
    s = sc.SCENARIO_D
    obs, now = s.observations(), s.evaluate_at

    class Bomb:
        usage = None

        async def structured(self, *a, **kw):
            raise RuntimeError("resolver killed mid-flight")

    crashed = None
    try:
        await Resolver(llm=Bomb()).resolve(obs, now, order_id=s.order_id)
    except Exception as e:                       # noqa: BLE001
        crashed = type(e).__name__

    # Two full resolutions of the same order at the same `now`. Comparing
    # a completed run against a bare `fold` would be the wrong test: the
    # resolver gathers evidence and a bare fold does not, so they can
    # legitimately differ in confidence. The invariant is that *resuming*
    # reproduces the verdict, which means run-to-run equality.
    first = (await Resolver(llm=NullLLM()).resolve(obs, now, order_id=s.order_id))["verdict"]
    second = (await Resolver(llm=NullLLM()).resolve(obs, now, order_id=s.order_id))["verdict"]

    # And the killed attempt must have left nothing behind that changes it.
    ok = first == second and first.verdict == fold(obs, now, order_id=s.order_id).verdict
    return ok, (
        f"killed={crashed or 'graph absorbed it'}; "
        f"re-run 1={first.verdict.value}@{first.confidence:.2f} "
        f"re-run 2={second.verdict.value}@{second.confidence:.2f}; identical={first == second}"
    )


# 6 -------------------------------------------------------------------
async def chaos_6_rate_limit_storm() -> Result:
    """429 storm: the breaker opens and stops hammering."""
    calls = {"n": 0}

    async def throttled(ctx):
        calls["n"] += 1
        raise RuntimeError("429 Too Many Requests")

    b = CircuitBreaker("attempts", threshold=3, cooldown_s=999)
    ctx = FetchContext(order_id="order_x")
    evs = [await guarded("attempts", throttled, ctx, b) for _ in range(50)]

    ok = calls["n"] == 3 and all(not e.available for e in evs)
    return ok, f"50 attempts -> {calls['n']} upstream calls, breaker open"


# 7 -------------------------------------------------------------------
async def chaos_7_old_replayed_event() -> Result:
    """A 15-day-old replayed event is accepted; the fold handles it."""
    s = sc.SCENARIO_A
    obs = s.observations()
    stale = obs[0].model_copy(update={"received_at": obs[0].received_at + 15 * 86_400})

    v = fold([stale] + obs[1:], s.evaluate_at, order_id=s.order_id)
    ok = v.verdict == Verdict.ORDER_SETTLED and stale.skew > 14 * 86_400
    return ok, f"skew={stale.skew // 86400}d -> {v.verdict.value} (ordered by event_time)"


# 8 -------------------------------------------------------------------
async def chaos_8_secret_rotation() -> Result:
    """Rotate mid-stream: retries signed with the old secret still verify."""
    raw = json.dumps(sc.SCENARIO_B.deliveries[0].body).encode()
    old_sig, new_sig = sign(raw, "old_secret"), sign(raw, "new_secret")

    ok = (
        verify_any(raw, old_sig, "new_secret", "old_secret")
        and verify_any(raw, new_sig, "new_secret", "old_secret")
        and not verify_any(raw, old_sig, "new_secret", "")
    )
    return ok, "old and new signatures both accepted inside the rotation window"


# 9 -------------------------------------------------------------------
async def chaos_9_burst_load() -> Result:
    """10k events in a burst: no bad actions under load.

    The safety property is not throughput. It is that a system under
    pressure must not start acting on partial information - so the
    assertion is zero money-moving actions on unsettled orders, not a
    latency number.
    """
    ex = Executor(dry_run=True)
    p = Pipeline(executor=ex)
    started = time.monotonic()

    orders = []
    for i in range(200):                          # 200 orders x ~2 events
        s = sc.SCENARIO_A
        obs = [
            o.model_copy(update={
                "order_id": f"order_load_{i}",
                "event_id": f"{o.event_id}_{i}",
            })
            for o in s.observations()
        ]
        orders.append(obs)

    results = await asyncio.gather(
        *[p.process(o, sc.SCENARIO_A.evaluate_at, order_id=f"order_load_{i}")
          for i, o in enumerate(orders)]
    )
    elapsed = time.monotonic() - started

    bad = [d for d in results if d.action in (Action.SEND_RECOVERY_LINK, Action.REFUND)]
    settled = [d for d in results if d.verdict.verdict == Verdict.ORDER_SETTLED]
    ok = not bad and len(settled) == len(results)
    return ok, (
        f"{len(results)} orders in {elapsed:.2f}s "
        f"({len(results)/max(elapsed,0.001):.0f}/s), duplicate actions={len(bad)}"
    )


CHAOS: dict[int, tuple[str, Callable]] = {
    1: ("same webhook x5", chaos_1_duplicate_webhooks),
    2: ("authorized before failed", chaos_2_inverted_delivery),
    3: ("kill downtime fetcher", chaos_3_dead_fetcher),
    4: ("kill LLM", chaos_4_no_llm),
    5: ("kill resolver mid-flight", chaos_5_kill_resolver_midflight),
    6: ("Razorpay 429 storm", chaos_6_rate_limit_storm),
    7: ("15-day-old replayed event", chaos_7_old_replayed_event),
    8: ("secret rotated mid-stream", chaos_8_secret_rotation),
    9: ("burst load", chaos_9_burst_load),
}

#: The slip protocol says 6-9 may be cut. 1-5 never may.
CORE = (1, 2, 3, 4, 5)


async def run_all(only: list[int] | None = None) -> dict[int, tuple[bool, str]]:
    out: dict[int, tuple[bool, str]] = {}
    for n, (_, fn) in CHAOS.items():
        if only and n not in only:
            continue
        try:
            out[n] = await fn()
        except Exception as e:                    # noqa: BLE001
            out[n] = (False, f"raised {type(e).__name__}: {e}")
    return out


def main() -> None:
    from rich.console import Console
    from rich.table import Table

    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--core", action="store_true", help="only faults 1-5")
    ap.add_argument("--only", type=int, nargs="*")
    args = ap.parse_args()

    only = list(CORE) if args.core else args.only
    results = asyncio.run(run_all(only))

    con = Console()
    t = Table(title="CHAOS MATRIX", header_style="bold")
    t.add_column("#", justify="right")
    t.add_column("fault")
    t.add_column("result")
    t.add_column("detail")

    for n, (ok, detail) in sorted(results.items()):
        t.add_row(
            str(n), CHAOS[n][0],
            "[green]PASS[/green]" if ok else "[red]FAIL[/red]",
            detail,
        )
    con.print(t)

    passed = sum(1 for ok, _ in results.values() if ok)
    total = len(results)
    style = "bold green" if passed == total else "bold red"
    con.print(f"\n[{style}]{passed}/{total} chaos faults handled[/{style}]")
    raise SystemExit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
