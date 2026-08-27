"""End-to-end orchestration: triage -> resolve -> strategise -> gate -> execute.

This is the only place the services are wired together, and the order is
load-bearing. The gate sits between every intent and every action, and
the executor is reachable from nowhere else. Reading top to bottom should
make it obvious that no model output can reach money without passing a
re-derivation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from core.events import Observation
from core.fold import FoldConfig, fold
from core.intents import Category, RecoveryIntent
from core.llm import LLM, NullLLM
from core.trace import AgentStep, AgentSummary
from core.verdicts import Action, Evidence, Verdict, VerdictResult
from services.executor.main import Executor, Outcome
from services.gate.rules import CustomerContext, GateDecision, evaluate, evidence_version
from services.resolver.graph import Resolver
from services.scheduler.main import InMemoryScheduler, Scheduler
from services.strategist.graph import Strategist
from services.triage.classify import Route, triage


@dataclass
class Decision:
    """One complete pass over an order. The unit the console renders."""

    order_id: str
    now: int
    verdict: VerdictResult
    route: Route
    triage_reason: str
    intent: RecoveryIntent | None = None
    gate: GateDecision | None = None
    outcome: Outcome | None = None
    llm_calls: int = 0
    latency_s: float = 0.0
    trace: list[str] = field(default_factory=list)
    degraded: list[str] = field(default_factory=list)
    evidence: tuple[Evidence, ...] = ()
    narrative: Any = None
    #: Node-by-node record of what each agent did, and whether a model
    #: was actually involved. Without this, `llm_calls: 0` is ambiguous
    #: between "the rules did their job" and "the model is misconfigured".
    steps: tuple[AgentStep, ...] = ()

    @property
    def action(self) -> Action:
        return self.outcome.action if self.outcome else Action.NOOP

    def to_row(self) -> dict[str, Any]:
        return {
            "ts": self.now,
            "order_id": self.order_id,
            "verdict": self.verdict.verdict.value,
            "confidence": round(self.verdict.confidence, 4),
            "action": self.action.value,
            "status": self.outcome.status if self.outcome else "NONE",
            "gate_allowed": bool(self.gate and self.gate.allowed),
            "veto_reason": self.gate.reason if self.gate else "",
            "rules_fired": list(self.verdict.rules_fired),
            "llm_calls": self.llm_calls,
            "latency_s": self.latency_s,
            "amount_due": self.verdict.amount_due,
            "amount_paid": self.verdict.amount_paid,
            "agents": self.agents.to_row(),
        }

    @property
    def agents(self) -> AgentSummary:
        return AgentSummary.of(self.steps)


class Pipeline:
    def __init__(
        self,
        llm: LLM | None = None,
        executor: Executor | None = None,
        scheduler: Scheduler | None = None,
        fold_cfg: FoldConfig | None = None,
        resolver: Resolver | None = None,
        strategist: Strategist | None = None,
        merchant: str = "Acme Store",
    ):
        self.llm = llm or NullLLM()
        self.fold_cfg = fold_cfg or FoldConfig()
        self.resolver = resolver or Resolver(llm=self.llm, fold_cfg=self.fold_cfg)
        self.strategist = strategist or Strategist(llm=self.llm)
        self.executor = executor or Executor(dry_run=True)
        self.scheduler = scheduler or InMemoryScheduler()
        self.merchant = merchant
        self.vetoes: list[dict] = []
        self.decisions: list[Decision] = []

    async def process(
        self,
        observations: list[Observation],
        now: int,
        order_id: str | None = None,
        customer: CustomerContext | None = None,
        client: Any = None,
        extra: dict | None = None,
        seed_evidence: tuple[Evidence, ...] = (),
    ) -> Decision:
        started = time.monotonic()
        oid = order_id or (observations[0].order_id if observations else "")

        # -- Triage. Pure, no I/O, no LLM. Decides whether the resolver
        #    graph is worth running at all.
        latest = max(observations, key=lambda o: o.event_time) if observations else None
        if latest is not None:
            route, _, why = triage(latest)
        else:
            route, why = Route.IGNORE, "no observations"

        # -- Resolve. The graph's precheck short-circuits deterministic
        #    cases, so this is cheap when the rules already suffice.
        state = await self.resolver.resolve(
            observations, now, order_id=oid, client=client,
            extra=extra, thread_id=f"{oid}:{now}",
        )
        verdict: VerdictResult = state["verdict"]
        evidence = tuple(state.get("evidence", ())) + tuple(seed_evidence)
        if seed_evidence:
            # Fixture-supplied evidence (the demo path when the downtime
            # API is not enabled) must be folded in like any other.
            verdict = fold(observations, now, order_id=oid, evidence=evidence, cfg=self.fold_cfg)

        d = Decision(
            order_id=oid,
            now=now,
            verdict=verdict,
            route=route,
            triage_reason=why,
            llm_calls=state.get("llm_calls", 0),
            degraded=list(state.get("degraded", [])),
            evidence=evidence,
            narrative=state.get("narrative"),
            steps=tuple(state.get("steps", ())),
        )

        # -- A time-dependent verdict is a deferral, not an answer.
        if verdict.recheck_at:
            await self.scheduler.schedule(oid, verdict.recheck_at)
            d.trace.append(f"recheck scheduled for {verdict.recheck_at}")

        amount = verdict.amount_due or (latest.amount if latest else 0)
        payment_id = _target_payment(observations, verdict)
        ev_version = evidence_version(observations)

        # -- Intent. Only CONFIRMED_FAILED reaches the strategist; every
        #    other verdict has a deterministic proposal.
        if verdict.verdict == Verdict.CONFIRMED_FAILED:
            s = await self.strategist.run(
                verdict, now, amount=amount, evidence=evidence,
                merchant=self.merchant,
                ctx={"evidence_version": ev_version,
                     "method": latest.method if latest else None,
                     "notes": _notes_of(latest)},
                thread_id=f"{oid}:{now}",
            )
            d.intent = s.get("intent")
            d.llm_calls += s.get("llm_calls", 0)
            d.trace += s.get("trace", [])
            d.degraded += s.get("degraded", [])
            d.steps = d.steps + tuple(s.get("steps", ()))
        else:
            d.intent = self._deterministic_intent(verdict, ev_version)

        # -- Gate. Re-derives everything. This is the only authority.
        if d.intent is not None:
            d.gate = evaluate(
                d.intent, observations, now,
                customer=customer, seen_keys=self.executor.seen_keys,
                cfg=self.fold_cfg, evidence=evidence,
            )
            if not d.gate.allowed:
                for v in d.gate.vetoes:
                    self.vetoes.append(
                        {
                            "ts": now, "order_id": oid, "action": d.intent.action.value,
                            "rule": v.rule, "reason": v.reason,
                            "confidence": d.intent.confidence,
                            "evidence": ",".join(sorted({e.source for e in evidence})),
                        }
                    )

            contact, email = _customer_of(latest)
            d.outcome = await self.executor.execute(
                d.gate, d.intent, order_id=oid, payment_id=payment_id,
                amount=amount, merchant=self.merchant,
                contact=contact, email=email,
            )

        d.latency_s = round(time.monotonic() - started, 4)
        self.decisions.append(d)
        return d

    def _deterministic_intent(self, v: VerdictResult, ev_version: str) -> RecoveryIntent:
        """No model involved. NOOP, CAPTURE, ESCALATE and HOLD are all
        fully determined by the verdict - there is nothing to decide."""
        return RecoveryIntent(
            action=v.proposed_action,
            template_id=None,
            variables=[],
            channel=None,
            category=Category.SERVICE_IMPLICIT,
            confidence=v.confidence,
            reasoning=f"{v.verdict.value} via {', '.join(v.rules_fired)}",
            evidence_version=ev_version,
        )


def _target_payment(observations: list[Observation], v: VerdictResult) -> str:
    """The payment an action applies to.

    For a capture that is the authorised attempt; for everything else the
    most recent failure. Picking the wrong one would key idempotency to
    the wrong entity.
    """
    from core.fold import build_state
    from core.verdicts import PaymentStatus

    st = build_state(observations, v.order_id)
    if v.verdict == Verdict.UNCAPTURED_AUTH:
        for p in st.payments.values():
            if p.status == PaymentStatus.AUTHORIZED:
                return p.payment_id
    if st.failed:
        return max(st.failed, key=lambda p: p.failed_at or 0).payment_id
    return next(iter(st.payments), "")


def _customer_of(obs: Observation | None) -> tuple[str, str]:
    """The payer's contact details, straight off the payment entity.

    Razorpay puts `contact` and `email` on the payment; the recovery link
    needs them to notify, and the WhatsApp sender needs the contact to
    address the message at all.
    """
    if obs is None:
        return "", ""
    ent = (obs.payload.get("payload", {}).get("payment", {}) or {}).get("entity", {}) or {}
    return str(ent.get("contact") or ""), str(ent.get("email") or "")


def _notes_of(obs: Observation | None) -> dict:
    if obs is None:
        return {}
    ent = (obs.payload.get("payload", {}).get("payment", {}) or {}).get("entity", {}) or {}
    return {"notes": ent.get("notes", {}), "description": ent.get("error_description", "")}
