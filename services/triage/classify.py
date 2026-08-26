"""Triage: a pure classifier over the error triple. No LLM. No I/O.

Its only job is to decide whether an event can be settled by the
deterministic fold alone, or whether it needs the resolver to go and
gather evidence. Getting this right is what keeps scenarios A, B and C at
zero LLM calls - which is, in turn, what makes the claim "AI was applied
where it earns its place" checkable rather than rhetorical.

Bias: when in doubt, return AMBIGUOUS. Over-routing to the resolver costs
tokens; under-routing costs a wrong verdict on real money.
"""

from __future__ import annotations

from enum import StrEnum

from core.events import Observation
from core.fold import POST_DEBIT_STEPS, PRE_DEBIT_STEPS
from core.verdicts import AMBIGUOUS_SOURCES, TERMINAL_CUSTOMER_REASONS, ErrorSource


class Route(StrEnum):
    #: The fold's deterministic rules are sufficient. No fetchers, no LLM.
    FOLD_ONLY = "FOLD_ONLY"
    #: Evidence is missing or conflicting - hand to the resolver graph.
    AMBIGUOUS = "AMBIGUOUS"
    #: Not a payment-lifecycle event; store and move on.
    IGNORE = "IGNORE"


class Decision(StrEnum):
    SETTLED = "SETTLED"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"


#: Events that state an outcome outright. Nothing to resolve.
SETTLING_EVENTS = frozenset({"payment.captured", "order.paid", "settlement.processed"})

#: Downtime events are global, not order-scoped. They feed the resolver's
#: downtime evidence but never trigger a resolution on their own.
DOWNTIME_EVENTS = frozenset(
    {"payment.downtime.started", "payment.downtime.updated", "payment.downtime.resolved"}
)


def triage(obs: Observation) -> tuple[Route, Decision, str]:
    """Classify one observation. Returns (route, decision, why).

    `why` is kept because it lands in the audit trail: a reviewer asking
    "why did this one skip the LLM?" gets an answer without reading code.
    """
    et = obs.event_type

    if et in DOWNTIME_EVENTS:
        return Route.IGNORE, Decision.UNKNOWN, "downtime signal, not order-scoped"

    if et in SETTLING_EVENTS:
        return Route.FOLD_ONLY, Decision.SETTLED, f"{et} states the outcome"

    if et == "payment.authorized":
        # Authorised is a fact, not an ambiguity: the fold turns it into
        # UNCAPTURED_AUTH and the gate decides whether to capture.
        return Route.FOLD_ONLY, Decision.UNKNOWN, "authorised; capture decision is deterministic"

    if et == "refund.processed":
        return Route.FOLD_ONLY, Decision.FAILED, "refund processed; debit reversed"

    if et == "refund.failed":
        # Money is stranded. Never automate around it (E13).
        return Route.AMBIGUOUS, Decision.UNKNOWN, "refund failed; funds stranded"

    if et != "payment.failed":
        return Route.IGNORE, Decision.UNKNOWN, f"{et} not part of the resolution lattice"

    # -- payment.failed: the whole question lives in the error triple. --
    try:
        source = ErrorSource(obs.error_source or "unknown")
    except ValueError:
        source = ErrorSource.UNKNOWN
    step = obs.error_step or ""
    reason = obs.error_reason or ""

    if source == ErrorSource.CUSTOMER and reason in TERMINAL_CUSTOMER_REASONS:
        # RBI's attributable-to-customer test: never a debit, so there is
        # nothing in flight and nothing to wait for. Scenario B.
        return Route.FOLD_ONLY, Decision.FAILED, f"customer-attributable: {reason}"

    if source == ErrorSource.CUSTOMER:
        # Customer source but an unfamiliar reason. Plausibly terminal,
        # not provably - let the resolver look before we create an order.
        return Route.AMBIGUOUS, Decision.UNKNOWN, f"customer source, unclassified reason: {reason!r}"

    if source in AMBIGUOUS_SOURCES:
        if step in PRE_DEBIT_STEPS:
            # No debit was possible, but whether it was an outage or a
            # one-off decides the *intervention*, not just the verdict.
            return Route.AMBIGUOUS, Decision.FAILED, f"pre-debit failure at {step}; probe downtime"
        if step in POST_DEBIT_STEPS:
            # The expensive case: a debit may exist and only the response
            # was lost. Scenarios C and D both live here.
            return Route.AMBIGUOUS, Decision.UNKNOWN, f"post-debit failure at {step}; debit may exist"
        return Route.AMBIGUOUS, Decision.UNKNOWN, f"{source} failure, step {step!r} unknown"

    if source == ErrorSource.BUSINESS:
        return Route.FOLD_ONLY, Decision.FAILED, "merchant-side rejection; no debit"

    return Route.AMBIGUOUS, Decision.UNKNOWN, f"unclassified source {source!r}"


def needs_resolver(obs: Observation) -> bool:
    return triage(obs)[0] is Route.AMBIGUOUS
