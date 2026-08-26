"""Agent 1 - the resolver. LangGraph: precheck -> plan -> fetch -> analyze.

The shape that matters is the *precheck*. Before any planning, the
deterministic fold runs on the observations alone. If it produces a
confident, non-ambiguous verdict, the graph ends there having made zero
LLM calls - which is what keeps scenarios A, B and C free of the model
entirely, and makes "AI where it earns its place" a measured claim rather
than a slogan.

The model is reached only when the deterministic rules are genuinely out
of road: unstructured evidence to weigh (a customer email, a pasted bank
SMS), or an escalation narrative to write. Both are judgement, neither
moves money.

LangGraph specifically, rather than a hand-rolled loop: Razorpay's own
Viveka is built on it, and the checkpointer is what makes chaos 5 - kill
the resolver mid-flight, resume, get an identical verdict - a property of
the framework rather than something we have to hand-roll.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

from core.config import Settings, settings
from core.events import Observation
from core.fold import FoldConfig, fold
from core.llm import LLM, LLMUnavailable, NullLLM, Usage, render_untrusted_block
from core.verdicts import Evidence, Verdict, VerdictResult
from services.resolver.fetchers import CircuitBreaker, FetchContext, gather_evidence
from services.resolver.fetchers.razorpay import FETCHERS, NEEDS_CLIENT, PLANNABLE

#: Below this, a deterministic verdict is not trusted on its own and the
#: graph goes and gathers evidence.
PRECHECK_CONFIDENCE = 0.90

#: Hard ceiling on fetch rounds. Enforced in the router, not the prompt.
MAX_ROUNDS = 3


class Plan(BaseModel):
    """Which evidence to gather next, and why."""

    reasoning: str = Field(description="One sentence. Why this evidence resolves the ambiguity.")
    fetchers: list[Literal["payment", "attempts", "downtime", "history", "settlement", "bank_prior"]]
    confidence: float = Field(ge=0.0, le=1.0)


class Narrative(BaseModel):
    """The escalation packet's human-readable half.

    Written by the model because it is prose for a human, and prose is
    the one thing a model is unambiguously better at than a rule.
    """

    summary: str = Field(description="What happened, in two sentences, for an ops reviewer.")
    what_was_checked: list[str]
    what_is_missing: list[str]
    suggested_next_step: str


def _merge(a: tuple, b: tuple) -> tuple:
    return tuple(a) + tuple(b)


class ResolveState(TypedDict, total=False):
    order_id: str
    now: int
    observations: list[Observation]
    evidence: Annotated[tuple[Evidence, ...], _merge]
    rounds: int
    verdict: VerdictResult | None
    narrative: Narrative | None
    llm_calls: int
    plan_reasoning: str
    fetch_ctx: dict[str, Any]
    degraded: list[str]


PLAN_SYSTEM = """You are the evidence planner for a payment-recovery agent.

A payment failed and the deterministic rules could not settle whether the
customer's money actually moved. Choose the smallest set of probes that
would resolve it.

What each probe gives you:
- attempts:   every sibling payment on this order. The single highest-signal
              probe: a captured sibling means the customer already paid.
- payment:    authoritative current status for this one attempt.
- downtime:   whether this payment method is in a known outage right now.
- history:    which payment methods this customer has succeeded with before.
- settlement: whether the money reached the merchant's bank account.
- bank_prior: this bank's baseline technical-decline rate.

Choose at most three. Prefer `attempts` whenever siblings could exist.
Do not choose a probe whose answer would not change the verdict."""

ANALYZE_SYSTEM = """You are writing the escalation packet for a payment that
could not be resolved automatically.

You are NOT deciding whether money moved - that decision is already made by
deterministic rules and is not yours to revisit. Your job is to describe the
situation accurately for a human reviewer, listing what was checked, what is
missing, and what they should do next.

Be specific about identifiers. If an RRN is present, it is the single most
useful thing in the packet: it is what ties this payment to the customer's
bank statement."""


class Resolver:
    """Compiled graph plus its dependencies."""

    def __init__(
        self,
        llm: LLM | None = None,
        cfg: Settings | None = None,
        fold_cfg: FoldConfig | None = None,
        checkpointer: Any = None,
        fetchers: dict | None = None,
    ):
        self.cfg = cfg or settings()
        self.llm = llm or NullLLM()
        self.fold_cfg = fold_cfg or FoldConfig(
            tat_banking_days=self.cfg.tat_window_banking_days,
            settle_horizon_days=self.cfg.settle_horizon_days,
        )
        self.fetchers = fetchers if fetchers is not None else FETCHERS
        # NEEDS_CLIENT describes the *real* HTTP fetchers. An injected
        # substitute borrowing one of their names does not inherit their
        # dependencies, so the skip only applies to the default set.
        self._default_fetchers = fetchers is None
        self.breakers: dict[str, CircuitBreaker] = {}
        self.usage = getattr(self.llm, "usage", Usage())
        self.app = self._build(checkpointer)

    # ------------------------------ nodes ------------------------------

    async def precheck(self, state: ResolveState) -> ResolveState:
        """Deterministic rules first. This node makes zero LLM calls."""
        v = fold(
            state["observations"],
            state["now"],
            order_id=state.get("order_id"),
            evidence=state.get("evidence", ()),
            cfg=self.fold_cfg,
        )
        return {"verdict": v, "rounds": state.get("rounds", 0)}

    async def plan(self, state: ResolveState) -> ResolveState:
        """LLM chooses probes. Falls back to a fixed plan if unavailable."""
        v = state["verdict"]
        obs = state["observations"]
        latest = obs[-1] if obs else None

        # The deterministic fallback is not a degraded afterthought: it is
        # what runs during chaos 4, and it must be good enough that the
        # system still resolves most cases without a model at all.
        fallback = ["attempts", "payment"]
        if latest is not None and latest.error_step in ("payment_initiation", "payment_authentication"):
            fallback.append("downtime")

        try:
            user = (
                f"<verdict>{v.verdict.value} confidence={v.confidence:.2f} "
                f"rules={list(v.rules_fired)}</verdict>\n"
                f"<payment>method={latest.method if latest else None} "
                f"error_source={latest.error_source if latest else None} "
                f"error_step={latest.error_step if latest else None} "
                f"error_reason={latest.error_reason if latest else None}</payment>\n"
                f"<siblings>{len({o.payment_id for o in obs if o.payment_id})}</siblings>\n"
                f"<available_probes>{list(PLANNABLE)}</available_probes>"
            )
            plan: Plan = await self.llm.structured(
                Plan, PLAN_SYSTEM, user, node="plan", temperature=0.0
            )
            chosen = [f for f in plan.fetchers if f in self.fetchers][:3] or fallback
            return {
                "fetch_ctx": {"chosen": chosen},
                "plan_reasoning": plan.reasoning,
                "llm_calls": state.get("llm_calls", 0) + 1,
            }
        except LLMUnavailable as e:
            return {
                "fetch_ctx": {"chosen": fallback},
                "plan_reasoning": f"deterministic plan (LLM unavailable: {e})",
                "degraded": state.get("degraded", []) + [f"plan: {e}"],
            }

    async def fetch(self, state: ResolveState) -> ResolveState:
        """Run the chosen probes concurrently. Never raises."""
        obs = state["observations"]
        latest = obs[-1] if obs else None
        chosen = state.get("fetch_ctx", {}).get("chosen", [])
        client = state.get("fetch_ctx", {}).get("client")

        # An unconfigured probe is skipped, not failed. Reporting "no
        # HTTP client" as unavailable evidence would tax every offline
        # verdict as though Razorpay were down, and the demo would look
        # permanently degraded. A *configured* probe that fails still
        # produces unavailable evidence, which is what chaos 3 asserts.
        needs_client = NEEDS_CLIENT if self._default_fetchers else frozenset()
        skipped = [n for n in chosen if n in needs_client and client is None]
        runnable = [n for n in chosen if n not in skipped]
        subset = {n: self.fetchers[n] for n in runnable if n in self.fetchers}

        ctx = FetchContext(
            order_id=state.get("order_id", ""),
            payment_id=latest.payment_id if latest else None,
            method=latest.method if latest else None,
            amount=latest.amount if latest else 0,
            now=state["now"],
            rrn=latest.rrn if latest else None,
            client=state.get("fetch_ctx", {}).get("client"),
            extra=state.get("fetch_ctx", {}).get("extra", {}),
        )
        ev = await gather_evidence(subset, ctx, self.breakers, self.cfg.fetch_timeout_s)
        out: dict = {"evidence": ev, "rounds": state.get("rounds", 0) + 1}
        if skipped:
            out["degraded"] = state.get("degraded", []) + [
                f"skipped (no API client): {', '.join(skipped)}"
            ]
        return out

    async def analyze(self, state: ResolveState) -> ResolveState:
        """Re-fold with the new evidence. Rules decide; the model narrates."""
        v = fold(
            state["observations"],
            state["now"],
            order_id=state.get("order_id"),
            evidence=state.get("evidence", ()),
            cfg=self.fold_cfg,
        )
        return {"verdict": v}

    async def narrate(self, state: ResolveState) -> ResolveState:
        """Only for UNRESOLVED. Writes the human's evidence packet."""
        v = state["verdict"]
        obs = state["observations"]
        latest = obs[-1] if obs else None

        untrusted = {
            "notes": (latest.payload.get("payload", {}).get("payment", {}).get("entity", {}).get("notes")
                      if latest else {}),
            "description": (latest.payload.get("payload", {}).get("payment", {}).get("entity", {}).get("error_description")
                            if latest else ""),
        }
        user = (
            f"<verdict>{v.verdict.value} confidence={v.confidence:.2f}</verdict>\n"
            f"<rules_fired>{list(v.rules_fired)}</rules_fired>\n"
            f"<evidence>{[e.model_dump() for e in v.evidence]}</evidence>\n"
            f"<identifiers>rrn={latest.rrn if latest else None} "
            f"upi_txn={latest.upi_transaction_id if latest else None} "
            f"payment_id={latest.payment_id if latest else None}</identifiers>\n"
            + render_untrusted_block(untrusted)
        )
        try:
            n: Narrative = await self.llm.structured(
                Narrative, ANALYZE_SYSTEM, user, node="narrate", temperature=0.2
            )
            return {"narrative": n, "llm_calls": state.get("llm_calls", 0) + 1}
        except LLMUnavailable as e:
            # Chaos 4: no model means no prose. The verdict is unchanged,
            # because the verdict never came from the model.
            return {
                "narrative": None,
                "degraded": state.get("degraded", []) + [f"narrate: {e}"],
            }

    # ----------------------------- routing -----------------------------

    def route_precheck(self, state: ResolveState) -> str:
        """The zero-LLM fast path. A, B and C leave here."""
        v = state["verdict"]
        if v.verdict == Verdict.UNRESOLVED:
            return "plan"
        if v.confidence >= PRECHECK_CONFIDENCE:
            return "done"
        return "plan"

    def route_analyze(self, state: ResolveState) -> str:
        v = state["verdict"]
        if state.get("rounds", 0) >= MAX_ROUNDS:
            return "narrate" if v.verdict == Verdict.UNRESOLVED else "done"
        if v.verdict == Verdict.UNRESOLVED:
            # One more round only if a probe we have not tried could help.
            tried = {e.source for e in state.get("evidence", ())}
            if set(PLANNABLE) - tried and state.get("rounds", 0) < MAX_ROUNDS:
                return "plan"
            return "narrate"
        return "done"

    def _build(self, checkpointer: Any):
        from langgraph.graph import END, StateGraph

        g = StateGraph(ResolveState)
        g.add_node("precheck", self.precheck)
        g.add_node("plan", self.plan)
        g.add_node("fetch", self.fetch)
        g.add_node("analyze", self.analyze)
        g.add_node("narrate", self.narrate)

        g.set_entry_point("precheck")
        g.add_conditional_edges("precheck", self.route_precheck, {"plan": "plan", "done": END})
        g.add_edge("plan", "fetch")
        g.add_edge("fetch", "analyze")
        g.add_conditional_edges(
            "analyze", self.route_analyze,
            {"plan": "plan", "narrate": "narrate", "done": END},
        )
        g.add_edge("narrate", END)
        return g.compile(checkpointer=checkpointer)

    # ------------------------------ entry ------------------------------

    async def resolve(
        self,
        observations: list[Observation],
        now: int,
        order_id: str | None = None,
        client: Any = None,
        extra: dict | None = None,
        thread_id: str | None = None,
    ) -> ResolveState:
        oid = order_id or (observations[0].order_id if observations else "")
        state: ResolveState = {
            "order_id": oid,
            "now": now,
            "observations": observations,
            "evidence": (),
            "rounds": 0,
            "llm_calls": 0,
            "degraded": [],
            "fetch_ctx": {"client": client, "extra": extra or {}},
        }
        config = {"configurable": {"thread_id": thread_id or oid}}
        started = time.monotonic()
        out = await self.app.ainvoke(state, config=config)
        out["latency_s"] = round(time.monotonic() - started, 4)
        return out
