"""Agent 2 - the strategist. The genuinely branchy loop.

Runs only on CONFIRMED_FAILED. The question is no longer "did money
move" (settled) but "what intervention actually works here", and that is
where the branch structure stops being knowable in advance:

    turn 1  act now, or wait?
    turn 2  is the outage method-scoped or bank-wide?
    turn 3  if scoped: which alternative rail?      <- may not exist
    turn 4  which channel, and when?
    turn 5  which template, filled how?
    turn 6  emit the intent

Turn 2's answer determines whether turn 3 exists at all. A bank-wide
outage has no alternative rail to pick, so the graph goes straight to a
wait-template. Flattening that into a fixed pipeline means enumerating
the tree, which is exactly the branch-explosion wall Bumblebee's n8n
prototype hit at ~40 nodes.

Two things the model cannot do here, by construction:

* It cannot choose *whether* to act. `action` is set by this module, not
  by the model - the model only selects how. That is I7 in code: narrow
  the permitted action space, never widen it.
* It cannot write a message. It selects a DLT-registered template and
  fills at most 5 slots of at most 30 characters. There is no code path
  from model output to free text.
"""

from __future__ import annotations

import time
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field, conlist, constr

from core.config import Settings, settings
from core.intents import (
    MAX_VARIABLE_LEN,
    MAX_VARIABLES,
    TEMPLATE_REGISTRY,
    Assessment,
    Category,
    Channel,
    RecoveryIntent,
)
from core.llm import LLM, LLMUnavailable, NullLLM, Usage, render_untrusted_block
from core.verdicts import Action, Evidence, VerdictResult

#: Hard bounds. Enforced in the router, never in the prompt - a bound in
#: a prompt is a suggestion, a bound in the router is a bound.
MAX_TURNS = 6
MAX_TOKENS = 8_000
MAX_LATENCY_S = 15.0


class Composition(BaseModel):
    """The model's only creative act: pick a template and fill its slots.

    Note what is absent - there is no `action` field. Whether to act was
    decided by the fold and is re-checked by the gate. The model chooses
    the shape of an intervention it has already been authorised to make.
    """

    template_id: Literal["RCV_UPI_ALT", "RCV_RETRY", "RCV_DOWNTIME_WAIT"]
    variables: conlist(constr(max_length=MAX_VARIABLE_LEN), max_length=MAX_VARIABLES)
    channel: Literal["SMS", "WHATSAPP", "EMAIL"]
    method_hint: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str


def _merge(a: tuple, b: tuple) -> tuple:
    return tuple(a) + tuple(b)


class StrategyState(TypedDict, total=False):
    order_id: str
    now: int
    verdict: VerdictResult
    evidence: Annotated[tuple[Evidence, ...], _merge]
    turns: int
    trace: list[str]
    intent: RecoveryIntent | None
    llm_calls: int
    degraded: list[str]
    merchant: str
    amount: int
    ctx: dict[str, Any]


ASSESS_SYSTEM = """You are choosing the next probe for a payment-recovery
strategist. The payment has already been confirmed failed by deterministic
rules - do not revisit that.

Your only decision is what to find out next, or whether you know enough:

- probe_downtime: is this payment method in a known outage? Choose this
  first when the failure was bank- or gateway-attributed. It decides
  whether re-sending a link on the same method would just fail again.
- probe_history: which methods has this customer successfully paid with
  before? Only useful if you already know an alternative rail is needed -
  i.e. the outage is scoped to one method rather than the whole bank.
- compose: you know enough to choose a template and channel.

Choose compose as soon as further probing would not change the message.
An outage that is bank-wide leaves no alternative rail, so there is
nothing history could tell you."""

COMPOSE_SYSTEM = f"""You are selecting a pre-registered DLT template for a
payment-recovery message and filling its variables.

You are NOT writing a message. You choose one registered template and
supply its variables. TRAI/DLT rules cap this at {MAX_VARIABLES} variables of
{MAX_VARIABLE_LEN} characters each, and the gate will reject anything else.

Templates:
- RCV_RETRY         vars (amount, merchant, link).  Generic retry. Use when
                    nothing suggests the original method is broken.
- RCV_UPI_ALT       vars (method, amount, merchant, link).  Steers the
                    customer to UPI. Use when the original method is in a
                    scoped outage AND UPI is a rail they have used before.
- RCV_DOWNTIME_WAIT vars (amount, merchant, window).  Tells them not to
                    retry yet. Use for a bank-wide outage with no
                    alternative rail - reassures them they were not charged.

Channel: prefer a channel the customer actually engages with. WHATSAPP and
SMS carry recovery links; EMAIL is the fallback.

Amounts are rupees as plain digits, no currency symbol."""


class Strategist:
    def __init__(
        self,
        llm: LLM | None = None,
        cfg: Settings | None = None,
        checkpointer: Any = None,
        probes: dict | None = None,
    ):
        self.cfg = cfg or settings()
        self.llm = llm or NullLLM()
        self.usage = getattr(self.llm, "usage", Usage())
        self.probes = probes or {}
        self.app = self._build(checkpointer)

    # ------------------------------ nodes ------------------------------

    async def assess(self, state: StrategyState) -> StrategyState:
        """Temperature 0 - this is routing, not writing."""
        turns = state.get("turns", 0) + 1
        known = {e.source: e.value for e in state.get("evidence", ()) if e.available}

        try:
            user = (
                f"<verdict>{state['verdict'].verdict.value} "
                f"confidence={state['verdict'].confidence:.2f}</verdict>\n"
                f"<turn>{turns} of {MAX_TURNS}</turn>\n"
                f"<evidence_so_far>{known}</evidence_so_far>\n"
                f"<already_probed>{sorted(known)}</already_probed>"
            )
            a: Assessment = await self.llm.structured(
                Assessment, ASSESS_SYSTEM, user, node="assess", temperature=0.0
            )
            return {
                "turns": turns,
                "trace": state.get("trace", []) + [f"turn {turns}: {a.next_probe} - {a.reasoning}"],
                "ctx": {**state.get("ctx", {}), "next_probe": a.next_probe},
                "llm_calls": state.get("llm_calls", 0) + 1,
            }
        except LLMUnavailable as e:
            # No model: take the safe generic path rather than guessing at
            # a clever intervention.
            return {
                "turns": turns,
                "trace": state.get("trace", []) + [f"turn {turns}: compose (LLM unavailable)"],
                "ctx": {**state.get("ctx", {}), "next_probe": "compose"},
                "degraded": state.get("degraded", []) + [f"assess: {e}"],
            }

    async def probe_downtime(self, state: StrategyState) -> StrategyState:
        fn = self.probes.get("downtime")
        if fn is None:
            return {"evidence": (Evidence.unavailable("downtime", "no probe configured"),)}
        return {"evidence": (await fn(state),)}

    async def probe_history(self, state: StrategyState) -> StrategyState:
        fn = self.probes.get("history")
        if fn is None:
            return {"evidence": (Evidence.unavailable("history", "stub: no history source"),)}
        return {"evidence": (await fn(state),)}

    async def compose(self, state: StrategyState) -> StrategyState:
        """Temperature 0.4 - wording, within a schema that bounds it."""
        ev = {e.source: e.value for e in state.get("evidence", ()) if e.available}
        rupees = state.get("amount", 0) // 100
        merchant = state.get("merchant", "Acme Store")

        downtime = ev.get("downtime") or {}
        history = ev.get("history") or {}

        try:
            user = (
                f"<amount_rupees>{rupees}</amount_rupees>\n"
                f"<merchant>{merchant}</merchant>\n"
                f"<failed_method>{state.get('ctx', {}).get('method')}</failed_method>\n"
                f"<downtime>{downtime}</downtime>\n"
                f"<customer_history>{history}</customer_history>\n"
                f"<available_templates>{sorted(TEMPLATE_REGISTRY)}</available_templates>\n"
                + render_untrusted_block(state.get("ctx", {}).get("notes", {}))
            )
            c: Composition = await self.llm.structured(
                Composition, COMPOSE_SYSTEM, user, node="compose", temperature=0.4
            )
            intent = self._to_intent(c, state)
            return {
                "intent": intent,
                "trace": state.get("trace", []) + [f"compose: {c.template_id} over {c.channel}"],
                "llm_calls": state.get("llm_calls", 0) + 1,
            }
        except LLMUnavailable as e:
            return {
                "intent": self._fallback_intent(state),
                "trace": state.get("trace", []) + ["compose: deterministic fallback"],
                "degraded": state.get("degraded", []) + [f"compose: {e}"],
            }

    # ---------------------------- assembly ----------------------------

    def _to_intent(self, c: Composition, state: StrategyState) -> RecoveryIntent:
        """Model output -> intent. `action` is ours, not the model's (I7).

        The confidence carried forward is the *lower* of the model's and
        the fold's. A confident model cannot lift a shaky verdict over
        the gate's floor; a confident verdict cannot rescue a model that
        is unsure which template fits.
        """
        derived = state["verdict"].confidence
        return RecoveryIntent(
            action=Action.SEND_RECOVERY_LINK,
            template_id=c.template_id,
            variables=list(c.variables),
            channel=Channel(c.channel),
            category=Category.SERVICE_IMPLICIT,
            method_hint=c.method_hint,
            confidence=min(c.confidence, derived),
            reasoning=c.reasoning,
            evidence_version=state.get("ctx", {}).get("evidence_version", ""),
        )

    def _fallback_intent(self, state: StrategyState) -> RecoveryIntent:
        rupees = state.get("amount", 0) // 100
        merchant = state.get("merchant", "Acme Store")[:MAX_VARIABLE_LEN]
        return RecoveryIntent(
            action=Action.SEND_RECOVERY_LINK,
            template_id="RCV_RETRY",
            variables=[str(rupees), merchant, "link"],
            channel=Channel.SMS,
            category=Category.SERVICE_IMPLICIT,
            confidence=state["verdict"].confidence,
            reasoning="deterministic fallback: generic retry template",
            evidence_version=state.get("ctx", {}).get("evidence_version", ""),
        )

    # ----------------------------- routing -----------------------------

    def route(self, state: StrategyState) -> str:
        """The bound lives here, not in the prompt."""
        if state.get("turns", 0) >= MAX_TURNS:
            return "compose"
        if self.usage.total > MAX_TOKENS:
            return "compose"

        nxt = state.get("ctx", {}).get("next_probe", "compose")
        probed = {e.source for e in state.get("evidence", ())}

        if nxt == "probe_downtime" and "downtime" not in probed:
            return "downtime"
        if nxt == "probe_history":
            if "history" in probed:
                return "compose"
            # Turn 2 decides whether turn 3 exists: a bank-wide outage
            # leaves no alternative rail, so history cannot help.
            dt = next(
                (e.value for e in state.get("evidence", ())
                 if e.source == "downtime" and e.available and isinstance(e.value, dict)),
                None,
            )
            if dt and dt.get("active") and dt.get("scope") != "method":
                return "compose"
            return "history"
        return "compose"

    def _build(self, checkpointer: Any):
        from langgraph.graph import END, StateGraph

        g = StateGraph(StrategyState)
        g.add_node("assess", self.assess)
        g.add_node("downtime", self.probe_downtime)
        g.add_node("history", self.probe_history)
        g.add_node("compose", self.compose)

        g.set_entry_point("assess")
        g.add_conditional_edges(
            "assess", self.route,
            {"downtime": "downtime", "history": "history", "compose": "compose"},
        )
        g.add_edge("downtime", "assess")
        g.add_edge("history", "assess")
        g.add_edge("compose", END)
        return g.compile(checkpointer=checkpointer)

    async def run(
        self,
        verdict: VerdictResult,
        now: int,
        amount: int,
        evidence: tuple[Evidence, ...] = (),
        merchant: str = "Acme Store",
        ctx: dict | None = None,
        thread_id: str | None = None,
    ) -> StrategyState:
        state: StrategyState = {
            "order_id": verdict.order_id,
            "now": now,
            "verdict": verdict,
            "evidence": evidence,
            "turns": 0,
            "trace": [],
            "llm_calls": 0,
            "degraded": [],
            "amount": amount,
            "merchant": merchant,
            "ctx": ctx or {},
        }
        started = time.monotonic()
        out = await self.app.ainvoke(
            state,
            config={"configurable": {"thread_id": thread_id or verdict.order_id},
                    "recursion_limit": 2 * MAX_TURNS + 4},
        )
        out["latency_s"] = round(time.monotonic() - started, 4)
        return out
