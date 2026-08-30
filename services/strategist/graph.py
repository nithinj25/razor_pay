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
from core.trace import AgentStep, merge_steps
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
    steps: Annotated[tuple[AgentStep, ...], merge_steps]
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

Templates, with their variables IN ORDER. Supply them in that order.

- RCV_RETRY         (amount, merchant, link)
                    "Your payment of Rs {{amount}} to {{merchant}} did not go
                    through. Complete it here: {{link}}"
                    Generic retry. Use when nothing suggests the original
                    method is broken.

- RCV_UPI_ALT       (method, amount, merchant, link)
                    "Your {{method}} payment of Rs {{amount}} to {{merchant}}
                    could not be processed. Pay by UPI instead: {{link}}"
                    IMPORTANT: {{method}} is the method that FAILED - the one
                    the customer tried and could not complete, e.g.
                    "netbanking". It is NOT the method you are steering them
                    towards. Putting "upi" there produces "Your UPI payment
                    could not be processed. Pay by UPI instead", which is
                    nonsense to the customer.
                    Use when the original method is in a scoped outage AND
                    UPI is a rail they have used before.

- RCV_DOWNTIME_WAIT (amount, merchant, window)
                    "Your bank is temporarily unavailable. Your payment of
                    Rs {{amount}} to {{merchant}} has not been charged. Please
                    retry after {{window}}."
                    Use for a bank-wide outage with no alternative rail -
                    it reassures them they were not charged.

`method_hint` is separate from the template variables: it is the rail you
want the payment link to open on, so for RCV_UPI_ALT it is "upi".

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
        on_step=None,
    ):
        self.cfg = cfg or settings()
        self.llm = llm or NullLLM()
        self.usage = getattr(self.llm, "usage", Usage())
        self.on_step = on_step
        self.probes = probes or {}
        self._source = "scripted" if type(self.llm).__name__ == "ScriptedLLM" else "model"
        #: Steps from calls made outside the graph, so the trace still sees them.
        self.last_steps: tuple = ()
        self.app = self._build(checkpointer)


    def _answering_model(self) -> str:
        """The model that actually replied, not the one we hoped would.

        With a fallback chain the primary can be rate-limited and a
        different provider answers. Reporting the configured model would
        make the trace quietly wrong about where the reasoning came from.
        """
        name = getattr(self.llm, "last_provider", "") or self.cfg.provider
        return self.cfg.model_for(name) or name


    def _wrap(self, fn):
        """Emit each step the moment its node finishes.

        The graph already records steps in state, but state is only
        readable once the whole run completes - which is fine for an audit
        row and useless for watching an agent think. Wrapping the node is
        the least invasive place to hook: the nodes stay unaware, and a
        caller that supplies no `on_step` pays nothing.
        """
        async def inner(state):
            out = await fn(state)
            if self.on_step:
                for step in (out or {}).get("steps", ()):
                    await self.on_step(step)
            return out

        inner.__name__ = getattr(fn, "__name__", "node")
        return inner

    # ------------------------------ nodes ------------------------------

    async def assess(self, state: StrategyState) -> StrategyState:
        """Temperature 0 - this is routing, not writing."""
        turns = state.get("turns", 0) + 1
        known = {e.source: e.value for e in state.get("evidence", ()) if e.available}
        t0 = time.monotonic()
        user = ""

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
            step = AgentStep(
                agent="strategist", node=f"assess (turn {turns})", source=self._source,
                model=self._answering_model(), summary=a.reasoning,
                latency_ms=int((time.monotonic() - t0) * 1000),
                prompt_chars=len(ASSESS_SYSTEM) + len(user),
                tokens_in=len(ASSESS_SYSTEM + user) // 4,
                output={"next_probe": a.next_probe, "confidence": a.confidence,
                        "turn": turns, "max_turns": MAX_TURNS,
                        "evidence_so_far": sorted(known)},
            )
            return {
                "turns": turns,
                "trace": state.get("trace", []) + [f"turn {turns}: {a.next_probe} - {a.reasoning}"],
                "ctx": {**state.get("ctx", {}), "next_probe": a.next_probe},
                "llm_calls": state.get("llm_calls", 0) + 1,
                "steps": (step,),
            }
        except LLMUnavailable as e:
            # No model: take the safe generic path rather than guessing at
            # a clever intervention.
            step = AgentStep(
                agent="strategist", node=f"assess (turn {turns})", source="fallback",
                summary="model unavailable - skipping straight to a generic template",
                latency_ms=int((time.monotonic() - t0) * 1000),
                output={"next_probe": "compose", "turn": turns}, error=str(e),
            )
            return {
                "turns": turns,
                "trace": state.get("trace", []) + [f"turn {turns}: compose (LLM unavailable)"],
                "ctx": {**state.get("ctx", {}), "next_probe": "compose"},
                "degraded": state.get("degraded", []) + [f"assess: {e}"],
                "steps": (step,),
            }

    async def probe_downtime(self, state: StrategyState) -> StrategyState:
        return await self._probe(state, "downtime", "no probe configured")

    async def probe_history(self, state: StrategyState) -> StrategyState:
        return await self._probe(state, "history", "stub: no history source")

    async def _probe(self, state: StrategyState, name: str, why: str) -> StrategyState:
        t0 = time.monotonic()
        fn = self.probes.get(name)
        ev = Evidence.unavailable(name, why) if fn is None else await fn(state)
        step = AgentStep(
            agent="strategist", node=f"probe_{name}", source="rules",
            summary=(f"{name}: {ev.value}" if ev.available else f"{name} unavailable - {ev.provenance}"),
            latency_ms=int((time.monotonic() - t0) * 1000),
            output={"available": ev.available, "value": ev.value,
                    "confidence": ev.confidence, "provenance": ev.provenance},
        )
        return {"evidence": (ev,), "steps": (step,)}

    async def compose(self, state: StrategyState) -> StrategyState:
        """Temperature 0.4 - wording, within a schema that bounds it."""
        ev = {e.source: e.value for e in state.get("evidence", ()) if e.available}
        rupees = state.get("amount", 0) // 100
        merchant = state.get("merchant", "Acme Store")

        downtime = ev.get("downtime") or {}
        history = ev.get("history") or {}
        t0 = time.monotonic()
        user = ""

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
            step = AgentStep(
                agent="strategist", node="compose", source=self._source,
                model=self._answering_model(), summary=c.reasoning,
                latency_ms=int((time.monotonic() - t0) * 1000),
                prompt_chars=len(COMPOSE_SYSTEM) + len(user),
                tokens_in=len(COMPOSE_SYSTEM + user) // 4,
                output={
                    "template_id": c.template_id, "channel": c.channel,
                    "method_hint": c.method_hint, "variables": list(c.variables),
                    "model_confidence": c.confidence,
                    "fold_confidence": round(state["verdict"].confidence, 3),
                    # The carried confidence is the lower of the two: a
                    # confident model cannot lift a shaky verdict over the
                    # gate's floor.
                    "carried_confidence": round(intent.confidence, 3),
                    "rendered": TEMPLATE_REGISTRY[c.template_id].render(list(c.variables)),
                },
            )
            return {
                "intent": intent,
                "trace": state.get("trace", []) + [f"compose: {c.template_id} over {c.channel}"],
                "llm_calls": state.get("llm_calls", 0) + 1,
                "steps": (step,),
            }
        except LLMUnavailable as e:
            fb = self._fallback_intent(state)
            step = AgentStep(
                agent="strategist", node="compose", source="fallback",
                summary="model unavailable - generic RCV_RETRY over SMS",
                latency_ms=int((time.monotonic() - t0) * 1000),
                output={"template_id": fb.template_id, "channel": fb.channel.value,
                        "variables": list(fb.variables)},
                error=str(e),
            )
            return {
                "intent": fb,
                "trace": state.get("trace", []) + ["compose: deterministic fallback"],
                "degraded": state.get("degraded", []) + [f"compose: {e}"],
                "steps": (step,),
            }

    async def compose_voice_brief(self, verdict, reference, gaps):
        """What to ask a customer whose evidence no API can supply.

        Kept off the LangGraph loop deliberately: there is nothing to
        branch on. The decision to call was already made by a rule, and
        this is a single piece of writing - questions that elicit
        checkable facts rather than recollection.
        """
        from services.strategist.voice import (
            VOICE_SYSTEM, VoiceBrief, fallback_brief,
        )

        t0 = time.monotonic()
        user = (
            f"<verdict>{verdict.verdict.value} @ {verdict.confidence:.2f}</verdict>\n"
            f"<reference_we_hold>{reference or 'none'}</reference_we_hold>\n"
            f"<what_we_could_not_establish>{gaps}</what_we_could_not_establish>\n"
            f"<amount_rupees>{verdict.amount_due // 100}</amount_rupees>"
        )
        try:
            brief: VoiceBrief = await self.llm.structured(
                VoiceBrief, VOICE_SYSTEM, user, node="voice_brief", temperature=0.3
            )
            self.last_steps = (AgentStep(
                agent="strategist", node="voice_brief", source=self._source,
                model=self._answering_model(), summary=brief.objective,
                latency_ms=int((time.monotonic() - t0) * 1000),
                prompt_chars=len(VOICE_SYSTEM) + len(user),
                tokens_in=len(VOICE_SYSTEM + user) // 4,
                output={"questions": brief.questions,
                        "reference_to_confirm": brief.reference_to_confirm,
                        "do_not_say": brief.do_not_say},
            ),)
            return brief
        except LLMUnavailable as e:
            # A call that happens without a model must still ask the right
            # questions - degrading to a worse conversation with a real
            # person is not an acceptable fallback.
            self.last_steps = (AgentStep(
                agent="strategist", node="voice_brief", source="fallback",
                summary="model unavailable - standard evidence-elicitation questions",
                latency_ms=int((time.monotonic() - t0) * 1000), error=str(e),
            ),)
            return fallback_brief(verdict, reference, gaps)

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
        # MAX_LATENCY_S was declared as a hard bound but never checked -
        # a documented guarantee that nothing enforced. A slow provider
        # (Nemotron's free tier runs 4-9s a call) can spend the whole
        # budget on probing and still owe the customer a message, so cut
        # to compose while there is time left to produce one.
        started = state.get("ctx", {}).get("started_at")
        if started is not None and (time.monotonic() - started) > MAX_LATENCY_S:
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
        g.add_node("assess", self._wrap(self.assess))
        g.add_node("downtime", self._wrap(self.probe_downtime))
        g.add_node("history", self._wrap(self.probe_history))
        g.add_node("compose", self._wrap(self.compose))

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
            "steps": (),
            "amount": amount,
            "merchant": merchant,
            "ctx": {**(ctx or {}), "started_at": time.monotonic()},
        }
        started = time.monotonic()
        out = await self.app.ainvoke(
            state,
            config={"configurable": {"thread_id": thread_id or verdict.order_id},
                    "recursion_limit": 2 * MAX_TURNS + 4},
        )
        out["latency_s"] = round(time.monotonic() - started, 4)
        return out
