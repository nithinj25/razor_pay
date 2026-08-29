"""Voice: designed, gated, telephony stubbed. GUARDRAILS section 6.

The trigger is evidence-type, not token count. When the resolver returns
UNRESOLVED, the missing evidence is not in any API we can call - it is in
the customer's head. Did money actually leave your account? What does
your bank app show right now? What reference is next to it?

That needs a *conversation* to elicit information. Every other verdict
just needs a link, and a link is cheaper, quieter and auditable. So voice
is reserved for the one case where the system has run out of things to
look up and a person is the only remaining source.

What is built here: the decision to call, the brief that says what to ask,
and the compliance gate around both. What is stubbed: the telephony.
Working Hinglish voice is roughly three days, is commodity, and Razorpay's
own Subscription Recovery agent already ships it. The reasoning is the
part worth building; the dialer is not.

Review answer: *"Voice is right for UNRESOLVED because you are eliciting
evidence, not pushing an action. Here is the compliance gate. I stubbed
telephony, not the reasoning."*
"""

from __future__ import annotations


from pydantic import BaseModel, Field

from core.verdicts import Evidence, VerdictResult

#: TRAI caps explicit consent at seven days (2025 amendment), and the gate
#: enforces it. Named here because the *decision* to call should not even
#: be proposed when consent has lapsed - refusing early is cheaper than
#: being vetoed late, and it keeps the veto log meaningful.
CONSENT_MAX_DAYS = 7


class VoiceBrief(BaseModel):
    """What to ask, and why. The model's only job on this path.

    Note what is absent: no phone number, no script to read verbatim, no
    decision about whether to call. The brief is questions for a human or
    a TTS agent to work from - it cannot dial anything, and the gate
    decides whether a call is permitted at all.
    """

    model_config = {"frozen": True}

    objective: str = Field(
        description="One sentence: the specific fact this call exists to establish."
    )
    questions: list[str] = Field(
        description="Three to five short questions, in the order to ask them. "
                    "Each must elicit a checkable fact, not an opinion."
    )
    reference_to_confirm: str | None = Field(
        default=None,
        description="The RRN or UTR to read back to the customer for "
                    "confirmation, if we hold one. Null if we do not.",
    )
    do_not_say: list[str] = Field(
        default_factory=list,
        description="Things the caller must not assert - anything we have "
                    "not established, especially that they were or were not charged.",
    )
    confidence: float = Field(ge=0.0, le=1.0)


VOICE_SYSTEM = """You are briefing a caller who will phone a customer about a
payment that could not be resolved automatically.

The call exists to ELICIT EVIDENCE, not to push an action. Nothing is being
sold and no payment is being requested. By the end of the call we need to
know whether money actually left the customer's account, and if so, what
reference their bank shows against it.

Write questions that produce checkable facts:
  good - "What reference number is shown next to that debit?"
  bad  - "Are you sure you were charged?"

Ask them to look at their bank app or statement rather than recall from
memory. People misremember, and a wrong answer here is worse than no answer.

In `do_not_say`, list what the caller must not assert. We do not know
whether they were charged - that is the entire reason for the call - so
the caller must never state or imply it either way, and must never promise
a refund."""


def should_offer_voice(
    verdict: VerdictResult,
    consent_age_days: int,
    has_reference: bool,
) -> tuple[bool, str]:
    """Is a call the right instrument here? Deterministic.

    Not a model decision. The trigger is the *type* of missing evidence,
    which the verdict already encodes - so this is a rule, and keeping it
    one means the model can never talk the system into phoning someone.
    """
    from core.verdicts import Verdict

    if verdict.verdict not in (Verdict.UNRESOLVED, Verdict.DUPLICATE_RISK):
        return False, (
            f"{verdict.verdict.value} does not need a conversation - "
            "the evidence it lacks is not in the customer's head"
        )
    if consent_age_days > CONSENT_MAX_DAYS:
        return False, (
            f"explicit consent is {consent_age_days}d old, over the "
            f"{CONSENT_MAX_DAYS}d TRAI cap"
        )
    if not has_reference:
        # Without an RRN there is nothing to read back, and the call
        # degenerates into "were you charged?" - which is exactly the
        # question people answer wrongly.
        return False, "no reference to confirm; a call would elicit only recollection"
    return True, "customer holds evidence no API can supply"


def evidence_gaps(verdict: VerdictResult, evidence: tuple[Evidence, ...]) -> list[str]:
    """What we could not establish. Feeds the brief and the exception card."""
    gaps: list[str] = []
    unavailable = [e.source for e in evidence if not e.available]
    if unavailable:
        gaps.append(f"probes returned nothing: {', '.join(sorted(unavailable))}")

    claim_ev = next(
        (e for e in evidence if e.source == "customer_claim" and e.available), None
    )
    if claim_ev is None:
        gaps.append("the customer has not told us whether money left their account")
    else:
        value = claim_ev.value if isinstance(claim_ev.value, dict) else {}
        claim = value.get("claim")
        # Whether the reference *matches* is the fold's business - it holds
        # the payment. What is visible here is whether they gave one at
        # all, which is the gap a call can actually close.
        claims_debit = getattr(claim, "claims_debited", value.get("claims_debited"))
        reference = getattr(claim, "reference", value.get("reference"))
        if claims_debit and not reference:
            gaps.append("customer claims a debit but quoted no checkable reference")

    if verdict.verdict.value == "DUPLICATE_RISK":
        gaps.append("their bank and our payment record disagree about whether money moved")
    return gaps


def fallback_brief(
    verdict: VerdictResult, reference: str | None, gaps: list[str]
) -> VoiceBrief:
    """Deterministic brief for when the model is unavailable (chaos 4).

    Deliberately good enough to actually use. A call that happens without
    a model should still ask the right questions - degrading to a worse
    conversation with a real customer is not an acceptable fallback.
    """
    questions = [
        "Could you open your bank app and check whether this amount was debited?",
        "If it was, what reference number is shown next to it?",
        "What date and time does the statement show for that debit?",
    ]
    if reference:
        questions.append(
            f"Does the reference {reference} match what you are seeing?"
        )
    return VoiceBrief(
        objective=(
            "Establish whether the customer's account was actually debited, "
            "and obtain the bank reference if it was."
        ),
        questions=questions,
        reference_to_confirm=reference,
        do_not_say=[
            "that they were charged - we do not know",
            "that they were not charged - we do not know that either",
            "any promise of a refund or a timeline",
        ],
        confidence=verdict.confidence,
    )
