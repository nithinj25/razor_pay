"""Customer claims: turning what a person wrote into evidence.

This is the one place in the system where a model is genuinely
irreplaceable. Everything else - siblings, statuses, banking days,
template selection - is structured data over which a rule is better than
a model in every respect. But a customer writes:

    "money got deducted twice from my hdfc acc, one shows in statement
     ref 230901495295, the other one payment failed but amount gone"

No rule extracts a reference number from that, works out which of two
debits they mean, or notices they are describing a *pending* debit rather
than a settled one. That is language work.

The split that keeps it safe is the same one used everywhere else here:

    the model EXTRACTS      -> a structured claim, schema-bound
    the rules DECIDE        -> by comparing the claim to hard identifiers

A customer claiming a debit proves nothing on its own; people
misremember, and a support inbox is attacker-reachable. What makes the
claim evidence is whether the reference they quote matches the RRN on
*our* payment. That comparison is deterministic, and the model never
performs it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, Field


class ClaimMatch(StrEnum):
    """How a customer's quoted reference relates to our payment."""

    #: Their reference matches this payment's RRN. They are talking about
    #: this attempt, and they say money left. Strongest signal available.
    CONFIRMS = "CONFIRMS"
    #: They quoted a reference and it belongs to a different payment. They
    #: are describing something else - do not act on this order because of it.
    CONTRADICTS = "CONTRADICTS"
    #: A debit is claimed but no checkable reference was given. Suggestive,
    #: not evidence. It lowers confidence; it never raises it.
    UNVERIFIED = "UNVERIFIED"
    #: No debit claimed, or nothing to compare.
    NONE = "NONE"


class CustomerClaim(BaseModel):
    """What the customer asserts, extracted from free text.

    Deliberately narrow. The model reports what was *said*, never what is
    *true* - there is no `verdict` field here for the same reason the
    strategist's schema has no `action` field. Establishing truth is the
    fold's job, and it does it by comparing these fields to the payment.
    """

    model_config = {"frozen": True}

    claims_debited: bool = Field(
        description="Does the customer state money actually left their account? "
                    "A complaint that a payment failed is NOT a debit claim."
    )
    reference: str | None = Field(
        default=None,
        description="Any transaction reference, RRN, UTR or UPI id they quote. "
                    "Digits only, exactly as written. Null if none.",
    )
    claimed_amount_rupees: float | None = Field(
        default=None, description="Amount they say was taken, in rupees. Null if unstated."
    )
    claims_multiple_debits: bool = Field(
        default=False, description="Do they say they were charged more than once?"
    )
    quotes: list[str] = Field(
        default_factory=list,
        description="Short verbatim spans supporting the extraction. For the "
                    "human reviewer - never paraphrase here.",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = ""


#: RRNs are 12 digits; UTRs and UPI transaction ids vary. Matching is done
#: on digit runs long enough not to collide with amounts or dates.
_REF = re.compile(r"\d{9,22}")


def normalise_reference(ref: str | None) -> str:
    """Strip everything but digits.

    Customers write "ref: 2309-0149-5295" or "RRN 230901495295." - the
    same number, three ways. Comparison happens on digits alone.
    """
    return "".join(ch for ch in (ref or "") if ch.isdigit())


def references_in(text: str) -> list[str]:
    """Every plausible reference in a blob of text, as a fallback.

    Used when the model returns no reference but the text plainly
    contains one - a deterministic backstop so a extraction miss does not
    lose the single most actionable identifier in the message.
    """
    return _REF.findall(text or "")


@dataclass(frozen=True)
class ClaimAssessment:
    """The deterministic verdict on a claim. Computed, never generated."""

    match: ClaimMatch
    claimed_reference: str = ""
    payment_reference: str = ""
    detail: str = ""

    @property
    def debit_confirmed(self) -> bool:
        return self.match is ClaimMatch.CONFIRMS


def assess_claim(
    claim: CustomerClaim | None,
    payment_rrn: str | None,
    payment_upi_id: str | None = None,
    raw_text: str = "",
) -> ClaimAssessment:
    """Compare what the customer said against what the payment carries.

    This function contains no model call and no judgement. It answers one
    question - does their reference match ours? - and that is what turns
    an assertion into evidence.
    """
    if claim is None or not claim.claims_debited:
        return ClaimAssessment(ClaimMatch.NONE, detail="no debit claimed")

    ours = {normalise_reference(payment_rrn), normalise_reference(payment_upi_id)}
    ours.discard("")

    theirs = normalise_reference(claim.reference)
    if not theirs:
        # The model missed it; look again deterministically before giving up.
        for candidate in references_in(raw_text):
            if candidate in ours:
                theirs = candidate
                break

    if not theirs:
        return ClaimAssessment(
            ClaimMatch.UNVERIFIED,
            detail="debit claimed with no checkable reference",
        )

    if not ours:
        return ClaimAssessment(
            ClaimMatch.UNVERIFIED, claimed_reference=theirs,
            detail="customer quoted a reference but the payment carries none",
        )

    if theirs in ours:
        return ClaimAssessment(
            ClaimMatch.CONFIRMS, claimed_reference=theirs,
            payment_reference=theirs,
            detail=f"customer reference {theirs} matches this payment",
        )

    return ClaimAssessment(
        ClaimMatch.CONTRADICTS, claimed_reference=theirs,
        payment_reference=sorted(ours)[0],
        detail=(f"customer reference {theirs} does not match this payment "
                f"({sorted(ours)[0]}) - they are describing a different transaction"),
    )
