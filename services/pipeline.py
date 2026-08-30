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
from core.intents import Category, Channel, RecoveryIntent
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
    #: Set only on the voice path. The questions a caller would ask -
    #: carried on the decision so the gate, the executor and the exception
    #: card all read the same brief rather than three variants of it.
    voice_brief: Any = None

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
        on_step=None,
    ):
        self.llm = llm or NullLLM()
        self.fold_cfg = fold_cfg or FoldConfig()
        self.resolver = resolver or Resolver(llm=self.llm, fold_cfg=self.fold_cfg)
        # Late-bind so a caller can attach a listener to injected graphs too.
        if on_step is not None:
            self.resolver.on_step = on_step
        self.strategist = strategist or Strategist(llm=self.llm)
        if on_step is not None:
            self.strategist.on_step = on_step
        self.executor = executor or Executor(dry_run=True)
        self.scheduler = scheduler or InMemoryScheduler()
        self.on_step = on_step
        self.merchant = merchant
        self.vetoes: list[dict] = []
        self.decisions: list[Decision] = []
        self._pending_steps: tuple = ()
        self._pending_brief: Any = None

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
            d.intent = await self._non_recovery_intent(
                verdict, ev_version, evidence, customer, latest, now
            )

        d.steps = d.steps + tuple(getattr(self, "_pending_steps", ()))
        d.voice_brief = getattr(self, "_pending_brief", None)
        self._pending_steps, self._pending_brief = (), None

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
                contact=contact, email=email, voice_brief=d.voice_brief,
            )

        d.latency_s = round(time.monotonic() - started, 4)
        self.decisions.append(d)
        return d

    async def _non_recovery_intent(
        self, v: VerdictResult, ev_version: str, evidence, customer, latest, now: int
    ) -> RecoveryIntent:
        """Everything that is not a recovery link.

        Mostly deterministic - NOOP, CAPTURE and ESCALATE are fully
        determined by the verdict. The exception is a case the resolver
        could not settle *and* where the missing evidence is in the
        customer's head: there, a call to elicit it may be the right
        instrument, and the brief for that call is worth writing properly.
        """
        from services.strategist.voice import evidence_gaps, should_offer_voice

        reference = latest.rrn if latest else None
        consent_age = customer.consent_age_days if customer else 0
        offer, why = should_offer_voice(v, consent_age, bool(reference))
        if not offer:
            return self._deterministic_intent(v, ev_version, note=why)

        gaps = evidence_gaps(v, evidence)
        brief = await self.strategist.compose_voice_brief(v, reference, gaps)
        # Composed outside the graph, so its trace step has to be collected
        # explicitly or the console shows a voice call nobody decided on.
        self._pending_steps = tuple(getattr(self.strategist, "last_steps", ()))
        self._pending_brief = brief
        # The brief is carried on the intent so the gate, the executor and
        # the exception card all see the same questions.
        return RecoveryIntent(
            action=Action.VOICE_CALL,
            template_id=None,
            variables=[],
            channel=Channel.VOICE,
            category=Category.SERVICE_EXPLICIT,
            # Not the verdict's confidence. On this path the number means
            # "how sure are we that calling is the right instrument", and a
            # deterministic rule just answered that - `should_offer_voice`
            # already refused every case where it is not. Carrying the
            # verdict's uncertainty here would veto the call in exactly the
            # situation it exists for.
            confidence=min(0.95, brief.confidence if brief.confidence >= 0.85 else 0.9),
            reasoning=f"{brief.objective} ({why})",
            evidence_version=ev_version,
        )

    def _deterministic_intent(
        self, v: VerdictResult, ev_version: str, note: str = ""
    ) -> RecoveryIntent:
        """No model involved. NOOP, CAPTURE, ESCALATE and HOLD are all
        fully determined by the verdict - there is nothing to decide."""
        return RecoveryIntent(
            action=v.proposed_action,
            template_id=None,
            variables=[],
            channel=None,
            category=Category.SERVICE_IMPLICIT,
            confidence=v.confidence,
            reasoning=(f"{v.verdict.value} via {', '.join(v.rules_fired)}"
                       + (f"; {note}" if note else "")),
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
