"""Replay a scenario on a synthetic clock, through the whole pipeline.

Two modes:

* **in-process** (default) - drives the pipeline directly. Deterministic,
  fast, and what the accuracy run and the console use.
* **--live** - HMAC-signs each fixture and POSTs it at a running ingress,
  so the same bytes exercise the real signature path. This is what proves
  the fixtures are not a private dialect.

The synthetic clock is the point. Scenario D spans a long weekend; nobody
is going to sit through it, and compressing it must not change a verdict.
`--speed` only affects wall-clock sleeps, never the `now` handed to the
fold - so a 16x replay and a real-time replay produce identical answers.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import dataclass
from typing import Any, Callable

from core.events import Observation
from core.llm import build_llm
from harness import scenarios as sc
from core.verdicts import Action
from services.pipeline import Decision, Pipeline

#: How consequential an action is, for picking the decisive outcome of a
#: multi-step replay. Moving money outranks escalating, which outranks
#: doing nothing.
ACTION_SIGNIFICANCE: dict[Action, int] = {
    Action.NOOP: 1,
    Action.NOTIFY_MERCHANT: 2,
    Action.HOLD: 2,
    Action.ESCALATE: 3,
    Action.VOICE_CALL: 4,
    Action.CAPTURE: 5,
    Action.SEND_RECOVERY_LINK: 5,
    Action.REFUND: 5,
}


@dataclass
class ReplayStep:
    at: int                     # synthetic epoch second
    observation: Observation | None
    decision: Decision | None = None


class Replay:
    def __init__(self, pipeline: Pipeline | None = None, speed: float = 1.0):
        self.pipeline = pipeline or Pipeline()
        self.speed = speed
        self.steps: list[ReplayStep] = []

    async def run(
        self,
        scenario: sc.Scenario,
        on_event: Callable[[dict], Any] | None = None,
        realtime: bool = False,
    ) -> list[ReplayStep]:
        """Feed deliveries in ARRIVAL order, re-deciding after each one.

        Re-deciding after every delivery is what makes the console show
        the verdict *changing* - which is the visible proof that status is
        derived rather than stored.
        """
        delivered: list[Observation] = []
        obs_by_delivery = scenario.observations()

        for d, obs in zip(scenario.deliveries, obs_by_delivery):
            now = scenario.start + d.at
            delivered.append(obs)

            if realtime and self.speed > 0:
                await asyncio.sleep(min(d.at / self.speed, 2.0) / max(len(scenario.deliveries), 1))

            if on_event:
                await _maybe_await(on_event({
                    "type": "observation",
                    "at": now,
                    "event_type": obs.event_type,
                    "payment_id": obs.payment_id,
                    "order_id": obs.order_id,
                    "event_time": obs.event_time,
                    "received_at": obs.received_at,
                    "inverted": obs.event_time < max(
                        (o.event_time for o in delivered[:-1]), default=obs.event_time
                    ),
                }))

            decision = await self.pipeline.process(
                list(delivered), now, order_id=scenario.order_id,
                seed_evidence=scenario.evidence,
                extra=_extra_for(scenario),
            )
            step = ReplayStep(at=now, observation=obs, decision=decision)
            self.steps.append(step)

            if on_event:
                await _maybe_await(on_event({"type": "decision", **decision.to_row()}))

        # Final evaluation at the scenario's labelled `now`. For D that is
        # after the banking window has closed, which is the whole point.
        if scenario.evaluate_at > (scenario.start + scenario.deliveries[-1].at):
            final = await self.pipeline.process(
                list(delivered), scenario.evaluate_at, order_id=scenario.order_id,
                seed_evidence=scenario.evidence,
                extra=_extra_for(scenario),
            )
            self.steps.append(ReplayStep(at=scenario.evaluate_at, observation=None, decision=final))
            if on_event:
                await _maybe_await(on_event({"type": "decision", **final.to_row()}))

        return self.steps

    @property
    def final(self) -> Decision | None:
        """The last verdict - what the system believes at the end."""
        for s in reversed(self.steps):
            if s.decision:
                return s.decision
        return None

    @property
    def effective(self) -> Decision | None:
        """The decisive action of the run.

        A replay re-decides after every delivery, which produces a
        sequence of outcomes rather than one: scenario C waits, then
        captures; scenario D waits across a long weekend, then escalates.
        Reporting the first is misleading (it looks like the system only
        ever waited) and so is reporting the last (a repeat pass is
        correctly vetoed on idempotency, which would read as a refusal).

        So: the most consequential outcome that actually happened.
        """
        best, best_rank = None, -1
        for s in self.steps:
            d = s.decision
            if not d or not d.outcome or d.outcome.status not in ("EXECUTED", "STUBBED"):
                continue
            rank = ACTION_SIGNIFICANCE.get(d.action, 0)
            if rank >= best_rank:            # ties break toward the later one
                best, best_rank = d, rank
        return best or self.final


def _extra_for(scenario) -> dict:
    """Fetch-context inputs a scenario carries but the event stream does not.

    Customer messages arrive through a support inbox, not a webhook, so
    they cannot be replayed as observations - they are fetched.
    """
    extra: dict = {}
    if getattr(scenario, "customer_messages", None):
        extra["customer_messages"] = list(scenario.customer_messages)
    return extra


async def _maybe_await(x):
    if asyncio.iscoroutine(x):
        await x


# ----------------------------- live mode -----------------------------

async def post_live(scenario: sc.Scenario, base_url: str, secret: str, speed: float = 4.0) -> int:
    """Sign each fixture and POST it at a running ingress."""
    import httpx

    from services.ingress.signing import sign

    sent = 0
    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as c:
        prev_at = 0
        for d in scenario.deliveries:
            gap = (d.at - prev_at) / max(speed, 0.001)
            if gap > 0:
                await asyncio.sleep(min(gap, 3.0))
            prev_at = d.at
            raw = json.dumps(d.body).encode()
            r = await c.post(
                "/webhook/razorpay",
                content=raw,
                headers={
                    "X-Razorpay-Signature": sign(raw, secret),
                    "X-Razorpay-Event-Id": d.event_id,
                    "Content-Type": "application/json",
                },
            )
            print(f"  {d.event_id:22} -> {r.status_code}")
            sent += 1 if r.status_code == 200 else 0
    return sent


# ------------------------------- cli -------------------------------

async def main_async(args) -> None:
    from rich.console import Console

    con = Console()

    if args.live:
        from core.config import settings

        cfg = settings()
        for key in _keys(args.scenario):
            s = sc.BY_KEY[key]
            con.print(f"[bold]scenario {key}[/bold] -> {args.url}")
            await post_live(s, args.url, cfg.rzp_webhook_secret, args.speed)
        return

    from harness.scripted_agents import pipeline_for

    llm = build_llm() if args.real_llm else None
    for key in _keys(args.scenario):
        s = sc.BY_KEY[key]
        r = Replay(pipeline_for(s, llm), speed=args.speed)
        await r.run(s, realtime=args.realtime)
        f, eff = r.final, r.effective
        ok = f.verdict.verdict == s.ground_truth
        mark = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
        total_llm = sum(st.decision.llm_calls for st in r.steps if st.decision)
        con.print(
            f"{mark} [bold]{key}[/bold] {s.title}\n"
            f"      verdict   {f.verdict.verdict.value} "
            f"(conf {f.verdict.confidence:.2f}, truth {s.ground_truth.value})\n"
            f"      action    {eff.action.value} [{eff.outcome.status if eff.outcome else '-'}]\n"
            f"      rules     {', '.join(f.verdict.rules_fired)}\n"
            f"      llm calls {total_llm}"
            + (f"\n      detail    {eff.outcome.detail}" if eff.outcome and eff.outcome.detail else "")
            + (f"\n      veto      {eff.gate.reason}" if eff.gate and not eff.gate.allowed else "")
        )


def _keys(arg: str) -> list[str]:
    return list(sc.BY_KEY) if arg in ("all", "ALL") else [arg.upper()]


def main() -> None:
    p = argparse.ArgumentParser(description="Replay labelled scenarios.")
    p.add_argument("--scenario", default="all", help="A-F, or 'all'")
    p.add_argument("--speed", type=float, default=4.0, help="clock compression for live/realtime")
    p.add_argument("--live", action="store_true", help="sign and POST at a running ingress")
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--realtime", action="store_true", help="insert wall-clock gaps")
    p.add_argument("--real-llm", action="store_true", help="use the Anthropic API")
    asyncio.run(main_async(p.parse_args()))


if __name__ == "__main__":
    main()
