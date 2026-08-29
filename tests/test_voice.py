"""Voice: designed, gated, telephony stubbed. GUARDRAILS section 6.

The claim under test is narrow: a call is offered only where the missing
evidence is genuinely in the customer's head, the compliance rules around
it actually bind, and the brief is good enough to work from even with no
model available.
"""

from __future__ import annotations

import pytest

from core.intents import Category, Channel, RecoveryIntent
from core.verdicts import Action, Evidence, Verdict, VerdictResult
from harness import scenarios as sc
from services.executor.main import Executor
from services.gate.rules import CustomerContext, evaluate
from services.strategist.voice import (
    CONSENT_MAX_DAYS,
    evidence_gaps,
    fallback_brief,
    should_offer_voice,
)


def verdict(v: Verdict, confidence: float = 0.45) -> VerdictResult:
    from core.verdicts import PROPOSED_ACTION

    return VerdictResult(
        order_id="order_D4nishchay04", verdict=v, confidence=confidence,
        proposed_action=PROPOSED_ACTION[v], amount_due=234000,
    )


# ------------------------------------------------ when to call --

def test_voice_is_offered_only_where_a_person_is_the_last_source():
    """The trigger is evidence-type, not token count or desperation."""
    ok, why = should_offer_voice(verdict(Verdict.UNRESOLVED), 2, has_reference=True)
    assert ok, why

    ok, why = should_offer_voice(verdict(Verdict.DUPLICATE_RISK, 0.9), 2, True)
    assert ok, why


@pytest.mark.parametrize(
    "v", [Verdict.ORDER_SETTLED, Verdict.CONFIRMED_FAILED,
          Verdict.UNCAPTURED_AUTH, Verdict.PENDING_TAT],
)
def test_settled_verdicts_never_warrant_a_call(v):
    """Every other verdict needs a link, and a link is cheaper and quieter."""
    ok, why = should_offer_voice(verdict(v, 0.95), 1, has_reference=True)
    assert not ok
    assert "does not need a conversation" in why


def test_expired_explicit_consent_refuses_before_the_gate_does():
    """Refusing early keeps the veto log meaningful.

    The gate would catch this anyway, but a veto log full of calls we
    should never have proposed is a worse audit trail than one that only
    records genuine refusals.
    """
    ok, why = should_offer_voice(
        verdict(Verdict.UNRESOLVED), CONSENT_MAX_DAYS + 1, has_reference=True
    )
    assert not ok
    assert "TRAI cap" in why


def test_no_reference_means_no_call():
    """Without an RRN the call degenerates into 'were you charged?' -
    which is exactly the question people answer wrongly."""
    ok, why = should_offer_voice(verdict(Verdict.UNRESOLVED), 1, has_reference=False)
    assert not ok
    assert "recollection" in why


# ---------------------------------------------------- the brief --

def test_fallback_brief_is_usable_without_a_model():
    """Chaos 4 reaches this. A call that happens anyway must still ask the
    right questions - a worse conversation with a real customer is not an
    acceptable degradation."""
    b = fallback_brief(verdict(Verdict.UNRESOLVED), "230901495295", [])

    assert len(b.questions) >= 3
    assert b.reference_to_confirm == "230901495295"
    assert any("230901495295" in q for q in b.questions)
    # It must ask them to look, not to remember.
    assert any("bank app" in q.lower() or "statement" in q.lower() for q in b.questions)
    # And it must forbid asserting either direction.
    joined = " ".join(b.do_not_say).lower()
    assert "were charged" in joined and "not charged" in joined
    assert "refund" in joined


def test_evidence_gaps_name_what_is_missing():
    v = verdict(Verdict.DUPLICATE_RISK, 0.9)
    ev = (Evidence.unavailable("settlement", "stub"),)
    gaps = evidence_gaps(v, ev)

    assert any("settlement" in g for g in gaps)
    assert any("disagree" in g for g in gaps)


def test_unverified_claim_shows_up_as_a_gap():
    """A debit claimed with no reference is precisely what a call can fix."""
    from core.claims import CustomerClaim

    v = verdict(Verdict.UNRESOLVED)
    claim = CustomerClaim(claims_debited=True, reference=None, confidence=0.6)
    ev = (Evidence(source="customer_claim", confidence=0.6, provenance="t",
                   value={"claim": claim, "raw_text": "money gone"}),)
    assert any("no checkable reference" in g for g in evidence_gaps(v, ev))


def test_a_claim_with_a_reference_is_not_reported_as_a_gap():
    from core.claims import CustomerClaim

    v = verdict(Verdict.UNRESOLVED)
    claim = CustomerClaim(claims_debited=True, reference="230901495295", confidence=0.9)
    ev = (Evidence(source="customer_claim", confidence=0.9, provenance="t",
                   value={"claim": claim, "raw_text": ""}),)
    assert not any("no checkable reference" in g for g in evidence_gaps(v, ev))


# ----------------------------------------------------- the gate --

def voice_intent(confidence: float = 0.9) -> RecoveryIntent:
    return RecoveryIntent(
        action=Action.VOICE_CALL, template_id=None, variables=[],
        channel=Channel.VOICE, category=Category.SERVICE_EXPLICIT,
        confidence=confidence, reasoning="establish whether money left the account",
    )


def test_a_voice_call_is_not_vetoed_for_lacking_a_dlt_template():
    """DLT registration governs messages. A call is a conversation whose
    questions are composed per case - demanding a template vetoed every
    call, which is how this bug was found."""
    s = sc.SCENARIO_D
    d = evaluate(voice_intent(), s.observations(), s.evaluate_at,
                 CustomerContext(consent_age_days=2))
    assert "DLT" not in {v.rule for v in d.vetoes}, d.reason


def test_a_voice_call_is_not_vetoed_by_the_implicit_consent_window():
    """That window governs SERVICE_IMPLICIT messages. A call runs on
    explicit consent, which has its own stricter rule."""
    s = sc.SCENARIO_D
    d = evaluate(voice_intent(), s.observations(), s.evaluate_at,
                 CustomerContext(consent_age_days=2))
    assert d.allowed, d.reason


def test_stale_explicit_consent_is_still_vetoed():
    s = sc.SCENARIO_D
    d = evaluate(voice_intent(), s.observations(), s.evaluate_at,
                 CustomerContext(consent_age_days=9))
    assert not d.allowed
    assert "CONSENT" in {v.rule for v in d.vetoes}


def test_calling_hours_still_bind():
    """21:00-09:00 IST is out of bounds regardless of consent."""
    from datetime import datetime

    from core.banking import IST

    night = int(datetime(2026, 1, 28, 23, 30, tzinfo=IST).timestamp())
    s = sc.SCENARIO_D
    d = evaluate(voice_intent(), s.observations(), night,
                 CustomerContext(consent_age_days=1))
    assert not d.allowed
    assert "HOURS" in {v.rule for v in d.vetoes}


def test_a_low_confidence_call_is_still_refused():
    """The floor still binds - it just bounds confidence that a call is
    the right instrument, not confidence about what happened."""
    s = sc.SCENARIO_D
    d = evaluate(voice_intent(confidence=0.5), s.observations(), s.evaluate_at,
                 CustomerContext(consent_age_days=1))
    assert not d.allowed
    assert "FLOOR" in {v.rule for v in d.vetoes}


# -------------------------------------------------- the executor --

async def test_executor_stubs_telephony_and_keeps_the_brief():
    """The brief is the deliverable. A human can work the call from it."""
    s = sc.SCENARIO_D
    ex = Executor(dry_run=True)
    intent = voice_intent()
    d = evaluate(intent, s.observations(), s.evaluate_at,
                 CustomerContext(consent_age_days=2))
    assert d.allowed, d.reason

    brief = fallback_brief(verdict(Verdict.UNRESOLVED), "230901495295", [])
    out = await ex.execute(d, intent, s.order_id, "pay_D1ambiguous", 234000,
                           voice_brief=brief)

    assert out.status == "STUBBED"
    assert "telephony not implemented" in out.detail
    assert out.request["questions"] == list(brief.questions)
    assert out.request["reference_to_confirm"] == "230901495295"

    # And it lands where a human will actually see it.
    assert ex.exception_queue
    assert ex.exception_queue[-1]["voice_brief"]["questions"]
