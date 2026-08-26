"""Canned agent responses, so the loop is observable without an API key.

The two agents only run when `ANTHROPIC_API_KEY` is set. Without one they
take their deterministic fallbacks, which is correct behaviour but means
a reviewer cannot see the reasoning loop at all - the console shows
`llm_calls: 0` everywhere and there is nothing to inspect.

This module supplies plausible, hand-written responses so the graphs
execute end to end. Every step they produce is tagged `scripted`, never
`model`: it demonstrates that the wiring works, and claims nothing about
what a real model would say.
"""

from __future__ import annotations

from core.intents import Assessment
from core.llm import ScriptedLLM
from services.resolver.graph import Narrative, Plan
from services.strategist.graph import Composition


def scripted_llm(scenario_key: str) -> ScriptedLLM:
    """A ScriptedLLM primed for one scenario's expected path."""
    llm = ScriptedLLM()
    key = scenario_key.upper()

    if key == "E":
        # The resolver plans too - without this its plan node falls back
        # and the trace shows a gap where the first agent call should be.
        llm.queue(
            Plan,
            Plan(
                reasoning="Bank-attributed failure at payment_initiation. Check "
                          "for a known outage on this method before doing anything "
                          "else - it decides whether a link would just fail again.",
                fetchers=["downtime", "attempts"], confidence=0.8,
            ),
        )

    if key == "D":
        llm.queue(
            Plan,
            Plan(
                reasoning="Bank-attributed failure past T+1. Check for a sibling "
                          "attempt and whether the money reached settlement.",
                fetchers=["attempts", "settlement"], confidence=0.7,
            ),
            Plan(
                reasoning="Nothing conclusive. Check this bank's baseline decline "
                          "rate before escalating.",
                fetchers=["bank_prior", "history"], confidence=0.4,
            ),
        )
        llm.queue(
            Narrative,
            Narrative(
                summary="A netbanking payment failed at payment_response on Friday "
                        "evening with no sibling attempt, no authorisation and no "
                        "refund. The RBI T+1 window closed on Tuesday without a "
                        "resolving event.",
                what_was_checked=[
                    "sibling attempts on the order - none found",
                    "settlement report - payment absent",
                    "refund events - none",
                    "T+1 banking window - closed Tue 27 Jan 23:59 IST",
                ],
                what_is_missing=[
                    "confirmation of whether the customer's account was debited",
                    "the bank's own reference against this RRN",
                ],
                suggested_next_step=(
                    "Ask the customer for their bank reference against RRN "
                    "230901495295. Razorpay documents that a bank can auto-refund "
                    "without changing payment status, so this cannot be settled "
                    "from the API alone."
                ),
            ),
        )

    elif key == "E":
        # The resolver already established the outage, and its evidence is
        # handed forward - so the strategist does not re-probe it. What it
        # still lacks is which rail this customer actually succeeds on.
        # That is the conditional turn: it exists only because the outage
        # turned out to be method-scoped rather than bank-wide.
        llm.queue(
            Assessment,
            Assessment(
                reasoning="The outage is already established and it is scoped to "
                          "netbanking, not the whole bank - so an alternative rail "
                          "exists. Which ones has this customer succeeded with?",
                next_probe="probe_history", confidence=0.8,
            ),
            Assessment(
                reasoning="Two prior successful UPI payments and a WhatsApp "
                          "preference. That is enough to compose.",
                next_probe="compose", confidence=0.9,
            ),
        )
        llm.queue(
            Composition,
            Composition(
                template_id="RCV_UPI_ALT",
                variables=["netbanking", "2340", "Acme Store", "rzp.io/i/nsh"],
                channel="WHATSAPP", method_hint="upi", confidence=0.93,
                reasoning="Netbanking is in a high-severity scoped outage, so "
                          "re-sending on that rail fails again. The customer has "
                          "paid by UPI twice and opens WhatsApp.",
            ),
        )

    else:
        # A, B, C and F resolve on deterministic rules alone, so nothing is
        # queued. If the graph reaches the model on these, the ScriptedLLM
        # raises - which is the assertion, not an accident.
        pass

    return llm


def scripted_fetchers(scenario) -> dict:
    """Resolver fetchers that return the scenario's labelled evidence.

    The live downtime endpoint needs a Razorpay support enablement that
    has not come through, so the demo path serves the same evidence from
    the fixture. It travels the real fetcher interface - pruned Evidence
    with a source, a confidence and a provenance - so the graph is
    exercised exactly as it would be against the API.
    """
    from core.verdicts import Evidence

    by_source = {e.source: e for e in scenario.evidence}

    def make(name):
        async def fetch(ctx):
            ev = by_source.get(name)
            if ev is None:
                return Evidence.unavailable(name, "not present in this fixture")
            return ev.model_copy(update={"provenance": f"fixture: {ev.provenance}"})
        return fetch

    return {name: make(name) for name in by_source}


def scripted_probes(scenario) -> dict:
    """Strategist probes, same idea - the strategist gathers for itself."""
    from core.verdicts import Evidence

    by_source = {e.source: e for e in scenario.evidence}

    def make(name):
        async def probe(state):
            ev = by_source.get(name)
            if ev is None:
                return Evidence.unavailable(name, "not present in this fixture")
            return ev.model_copy(update={"provenance": f"fixture: {ev.provenance}"})
        return probe

    return {name: make(name) for name in by_source}
