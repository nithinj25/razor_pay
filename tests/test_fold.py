"""Day 2 acceptance. The property test IS the point.

If `fold` is not order-independent, every downstream service is built on
sand: replay is meaningless, crash recovery cannot be verified, and the
duplicate-order invariant (I3) is unenforceable. So this file asserts the
property directly rather than testing a handful of happy paths.
"""

from __future__ import annotations

import pytest
from datetime import date, datetime
from hypothesis import given, settings
from hypothesis import strategies as st

from core.banking import IST, is_banking_day, naive_deadline, tat_deadline, ist_date
from core.events import InMemoryStore, canonical_order
from core.fold import fold, naive_lww
from core.verdicts import Verdict, Evidence, Paise
from harness import scenarios as sc
from pydantic import BaseModel, ValidationError


# ---------------------------------------------------------------- I4 --

@settings(max_examples=200, deadline=None)
@given(perm=st.permutations(sc.SCENARIO_A.observations()))
def test_fold_is_order_independent(perm):
    """Arrival order must not change the verdict. This is I4.

    Hypothesis shuffles the delivery sequence; the fold sorts by
    event_time internally, so every permutation must agree with the
    canonical ordering.
    """
    canonical = canonical_order(sc.SCENARIO_A.observations())
    expected = fold(canonical, sc.SCENARIO_A.evaluate_at)
    assert fold(list(perm), sc.SCENARIO_A.evaluate_at) == expected


@settings(max_examples=100, deadline=None)
@given(data=st.data())
def test_order_independence_holds_for_every_scenario(data):
    for s in sc.ALL:
        obs = s.observations()
        if len(obs) < 2:
            continue
        perm = data.draw(st.permutations(obs))
        assert fold(list(perm), s.evaluate_at, evidence=s.evidence) == fold(
            canonical_order(obs), s.evaluate_at, evidence=s.evidence
        )


def test_fold_is_idempotent_under_duplicate_delivery():
    """At-least-once delivery must not shift the verdict (E9, chaos 1)."""
    obs = sc.SCENARIO_A.observations()
    once = fold(obs, sc.SCENARIO_A.evaluate_at)
    five_times = fold(obs * 5, sc.SCENARIO_A.evaluate_at)
    assert once == five_times


def test_lww_would_fail():
    """Documents why we do not use a mutable status field.

    Kept executable so the justification for the whole architecture lives
    in the test suite, not in a comment someone can delete.
    """
    obs = sc.SCENARIO_A.observations()          # arrival order: captured, then failed
    assert naive_lww(obs) == "FAILED"           # the baseline's answer - wrong
    assert fold(obs, sc.SCENARIO_A.evaluate_at).verdict == Verdict.ORDER_SETTLED


# ------------------------------------------------------- ground truth --

@pytest.mark.parametrize("s", sc.ALL, ids=[s.key for s in sc.ALL])
def test_scenario_matches_ground_truth(s):
    v = fold(s.observations(), s.evaluate_at, order_id=s.order_id, evidence=s.evidence)
    assert v.verdict == s.ground_truth, f"{s.key}: {s.title}\nrules={v.rules_fired}"


@pytest.mark.parametrize("s", [x for x in sc.ALL if x.checkpoints], ids=lambda x: x.key)
def test_scenario_checkpoints(s):
    """Some verdicts are only correct at a given `now`.

    C is PENDING_TAT at +60s and UNCAPTURED_AUTH once the late
    authorisation lands. D stays PENDING_TAT across the whole long
    weekend and only becomes UNRESOLVED on the Wednesday.
    """
    for at, expected in s.checkpoints:
        v = fold(s.observations(until=at), at, order_id=s.order_id, evidence=s.evidence)
        when = datetime.fromtimestamp(at, tz=IST).strftime("%a %d %b %H:%M")
        assert v.verdict == expected, f"{s.key} at {when}: got {v.verdict} ({v.rules_fired})"


def test_scenario_A_needs_no_llm():
    """A, B and C must resolve on deterministic rules alone."""
    for key in ("A", "B", "C"):
        s = sc.BY_KEY[key]
        v = fold(s.observations(), s.evaluate_at, order_id=s.order_id, evidence=s.evidence)
        assert v.rules_fired, f"{key} produced no rule trace"
        assert v.confidence >= 0.75


def test_confirmed_failed_clears_the_link_floor():
    """Scenario B must clear the 0.90 floor or the gate vetoes the link
    and the system is merely cautious rather than correct."""
    s = sc.SCENARIO_B
    v = fold(s.observations(), s.evaluate_at, order_id=s.order_id)
    assert v.verdict == Verdict.CONFIRMED_FAILED
    assert v.confidence >= 0.90, "B would be vetoed by its own confidence floor"


# ------------------------------------------------------------ I3 / I9 --

def test_sibling_non_terminal_flag_blocks_the_link():
    """I3: while any sibling is unresolved, no recovery order may spawn."""
    s = sc.SCENARIO_C
    v = fold(s.observations(until=s.start + 60), s.start + 60, order_id=s.order_id)
    assert v.any_sibling_non_terminal is True


def test_missing_evidence_lowers_confidence():
    """I9 / chaos 3: a dead fetcher degrades, it never raises."""
    s = sc.SCENARIO_E
    healthy = fold(s.observations(), s.evaluate_at, evidence=s.evidence)
    degraded = fold(
        s.observations(),
        s.evaluate_at,
        evidence=(Evidence.unavailable("downtime", "circuit open"),),
    )
    assert degraded.confidence < healthy.confidence
    # Without downtime evidence the outage cannot be attributed, so the
    # failure falls back into the ambiguous wait rather than acting.
    assert degraded.verdict == Verdict.PENDING_TAT


def test_scenario_F_injection_does_not_move_the_verdict():
    """The notes field screams CONFIRMED_FAILED. The sibling says otherwise."""
    s = sc.SCENARIO_F
    v = fold(s.observations(), s.evaluate_at, order_id=s.order_id)
    assert v.verdict == Verdict.ORDER_SETTLED
    assert "R2_sibling_captured_full" in v.rules_fired


# ------------------------------------------------------------- E14 --

def test_banking_day_arithmetic():
    """Friday 19:40 IST, 4th Saturday, Sunday, Republic Day -> Tuesday."""
    dl = tat_deadline(sc.FRIDAY_EVENING)
    assert ist_date(dl) == date(2026, 1, 27)
    assert ist_date(naive_deadline(sc.FRIDAY_EVENING)) == date(2026, 1, 24)
    assert (dl - sc.FRIDAY_EVENING) / 86_400 > 3.5


def test_saturday_rule():
    """RBI closes the 2nd and 4th Saturday only - not every Saturday."""
    assert is_banking_day(date(2026, 1, 3)) is True     # 1st Saturday
    assert is_banking_day(date(2026, 1, 10)) is False   # 2nd
    assert is_banking_day(date(2026, 1, 17)) is True    # 3rd
    assert is_banking_day(date(2026, 1, 24)) is False   # 4th
    assert is_banking_day(date(2026, 1, 25)) is False   # Sunday


# -------------------------------------------------------------- I1 --

def test_money_rejects_floats():
    class M(BaseModel):
        amount: Paise

    assert M(amount=234000).amount == 234000
    assert M(amount="234000").amount == 234000
    for bad in (2340.00, 0.1, -5, True):
        with pytest.raises(ValidationError):
            M(amount=bad)


# -------------------------------------------------------------- I2 --

async def test_store_dedupes_on_event_id():
    """Chaos 1: the same webhook five times is one observation."""
    store = InMemoryStore()
    o = sc.SCENARIO_B.observations()[0]
    results = [await store.append(o) for _ in range(5)]
    assert results == [True, False, False, False, False]
    assert len(await store.load(o.order_id)) == 1


def test_observations_are_frozen():
    o = sc.SCENARIO_B.observations()[0]
    with pytest.raises(ValidationError):
        o.status = "captured"


def test_event_time_not_received_at_drives_order():
    """Pitfall #5 in executable form."""
    obs = sc.SCENARIO_A.observations()
    by_arrival = sorted(obs, key=lambda o: o.received_at)
    by_event_time = canonical_order(obs)
    assert by_arrival[0].event_type == "payment.captured"
    assert by_event_time[0].event_type == "payment.failed"
    assert by_arrival != by_event_time, "fixture no longer exercises the inversion"
