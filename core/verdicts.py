"""Verdict vocabulary, confidence floors, and the money type.

Everything downstream — resolver, strategist, gate, executor — speaks in
these enums. Keep this module free of I/O so `fold` stays pure.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Any

from pydantic import BaseModel, Field, PlainValidator


# ── I1: all money is int paise. Reject floats at the schema boundary. ──

def _paise(v: Any) -> int:
    """Accept int-valued money only.

    A float here is a bug, not a rounding inconvenience: 0.1 + 0.2 paise
    is how a reconciliation drifts. Reject at the boundary (I1, E11).
    """
    if isinstance(v, bool):
        raise ValueError("bool is not an amount")
    if isinstance(v, float):
        raise ValueError(f"money must be int paise, got float {v!r} (I1)")
    if isinstance(v, str):
        if not v.lstrip("-").isdigit():
            raise ValueError(f"money must be int paise, got {v!r} (I1)")
        v = int(v)
    if not isinstance(v, int):
        raise ValueError(f"money must be int paise, got {type(v).__name__} (I1)")
    if v < 0:
        raise ValueError("money must be non-negative")
    return v


Paise = Annotated[int, PlainValidator(_paise)]


class PaymentStatus(StrEnum):
    """Razorpay's payment.status. `failed` is NOT a sink — see fold.py."""

    CREATED = "created"
    AUTHORIZED = "authorized"
    CAPTURED = "captured"
    REFUNDED = "refunded"
    FAILED = "failed"


class ErrorSource(StrEnum):
    """`error.source` on a failed payment.

    RBI's harmonised-TAT test is "not attributable to the customer".
    That maps onto this field: CUSTOMER means no debit occurred, so the
    T+1 wait does not apply. The rest are ambiguous and must wait.
    """

    CUSTOMER = "customer"
    BUSINESS = "business"
    NETWORK = "network"
    GATEWAY = "gateway"
    BANK = "bank"
    ISSUER = "issuer"
    UNKNOWN = "unknown"


#: Sources where a debit may have occurred → must wait out the T+1 window.
AMBIGUOUS_SOURCES: frozenset[ErrorSource] = frozenset(
    {ErrorSource.NETWORK, ErrorSource.GATEWAY, ErrorSource.BANK, ErrorSource.ISSUER}
)

#: Customer-attributable reasons that are terminal on sight: the customer
#: never authorised a debit, so there is nothing to reverse and nothing to
#: wait for. Scenario B depends on this set.
TERMINAL_CUSTOMER_REASONS: frozenset[str] = frozenset(
    {
        "payment_cancelled",
        "payment_failed_by_user",
        "user_cancelled",
        "payment_timeout",
        "invalid_vpa",
        "card_declined_by_user",
    }
)


class Verdict(StrEnum):
    """The six terminal answers `fold` can give. See ARCHITECTURE §4."""

    ORDER_SETTLED = "ORDER_SETTLED"          # order fully paid → NOOP
    CONFIRMED_FAILED = "CONFIRMED_FAILED"    # no debit, or reversed → strategist
    UNCAPTURED_AUTH = "UNCAPTURED_AUTH"      # authorised, not captured → CAPTURE
    PENDING_TAT = "PENDING_TAT"              # inside T+1 → NOOP + recheck
    DUPLICATE_RISK = "DUPLICATE_RISK"        # evidence conflicts → HOLD
    UNRESOLVED = "UNRESOLVED"                # insufficient evidence → ESCALATE


class Action(StrEnum):
    NOOP = "NOOP"
    CAPTURE = "CAPTURE"
    SEND_RECOVERY_LINK = "SEND_RECOVERY_LINK"
    REFUND = "REFUND"
    ESCALATE = "ESCALATE"
    NOTIFY_MERCHANT = "NOTIFY_MERCHANT"
    VOICE_CALL = "VOICE_CALL"
    HOLD = "HOLD"


#: Actions that touch money. Every one needs an idempotency key (I5).
MONEY_MOVING: frozenset[Action] = frozenset(
    {Action.CAPTURE, Action.SEND_RECOVERY_LINK, Action.REFUND}
)

#: GUARDRAILS §3.2 — floors scale with irreversibility.
#: SEND_RECOVERY_LINK is the action that creates order_B. Strictest.
CONFIDENCE_FLOOR: dict[Action, float] = {
    Action.NOOP: 0.00,
    Action.ESCALATE: 0.00,
    Action.HOLD: 0.00,
    Action.NOTIFY_MERCHANT: 0.60,
    Action.CAPTURE: 0.80,
    Action.VOICE_CALL: 0.85,
    Action.SEND_RECOVERY_LINK: 0.90,
    Action.REFUND: 0.95,
}

#: The action each verdict proposes. The gate may still veto it (I8).
PROPOSED_ACTION: dict[Verdict, Action] = {
    Verdict.ORDER_SETTLED: Action.NOOP,
    Verdict.CONFIRMED_FAILED: Action.SEND_RECOVERY_LINK,
    Verdict.UNCAPTURED_AUTH: Action.CAPTURE,
    Verdict.PENDING_TAT: Action.NOOP,
    Verdict.DUPLICATE_RISK: Action.HOLD,
    Verdict.UNRESOLVED: Action.ESCALATE,
}


class Evidence(BaseModel):
    """One pruned fact with provenance. Fetchers return these, never raw JSON.

    Pruning happens at the fetcher (ARCHITECTURE §6): a ~40KB downtime
    response becomes ~180 bytes here.
    """

    model_config = {"frozen": True}

    source: str
    value: Any = None
    confidence: float = Field(ge=0.0, le=1.0)
    provenance: str = ""
    available: bool = True

    @classmethod
    def unavailable(cls, source: str, why: str) -> "Evidence":
        """A dead fetcher yields this, never a raise (I9)."""
        return cls(
            source=source, value=None, confidence=0.0,
            provenance=f"unavailable: {why}", available=False,
        )


class VerdictResult(BaseModel):
    """What `fold` returns. Carries its own audit trail."""

    model_config = {"frozen": True}

    order_id: str
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    proposed_action: Action
    rules_fired: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    amount_due: Paise = 0
    amount_paid: Paise = 0
    #: Set when the verdict is time-dependent — scheduler re-folds at this ts.
    recheck_at: int | None = None
    #: True while ANY sibling attempt on the order is non-terminal. The gate
    #: re-derives this itself; it never trusts this field (I8).
    any_sibling_non_terminal: bool = False
    narrative: str = ""

    def __eq__(self, other: object) -> bool:
        """Verdict equality ignores narrative — prose is not a conclusion.

        Chaos 5 asserts a re-folded verdict is *identical* after a crash;
        that must not turn on LLM wording.
        """
        if not isinstance(other, VerdictResult):
            return NotImplemented
        return (
            self.order_id == other.order_id
            and self.verdict == other.verdict
            and round(self.confidence, 6) == round(other.confidence, 6)
            and self.proposed_action == other.proposed_action
            and sorted(self.rules_fired) == sorted(other.rules_fired)
            and self.amount_due == other.amount_due
            and self.amount_paid == other.amount_paid
            and self.recheck_at == other.recheck_at
            and self.any_sibling_non_terminal == other.any_sibling_non_terminal
        )

    def __hash__(self) -> int:
        return hash((self.order_id, self.verdict, self.proposed_action))
