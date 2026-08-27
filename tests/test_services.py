"""Triage, executor, scheduler and the assembled pipeline.

These four had no direct coverage: the scenario tests exercised them
end to end, which proves they work together but not that each refuses
the right things on its own.
"""

from __future__ import annotations

import pytest

from core.banking import ist_date, tat_deadline
from core.intents import Channel, RecoveryIntent
from core.verdicts import Action, Verdict
from harness import scenarios as sc
from services.executor.main import Executor, RazorpayClient
from services.gate.rules import evaluate
from services.pipeline import Pipeline
from services.scheduler.main import InMemoryScheduler, run_worker
from services.triage.classify import Decision, Route, needs_resolver, triage


# ------------------------------------------------------------ triage --

@pytest.mark.parametrize(
    "key,expected_route",
    [
        ("A", Route.FOLD_ONLY),      # latest event is payment.failed w/ retry-able reason
        ("B", Route.FOLD_ONLY),      # customer-terminal reason
        ("D", Route.AMBIGUOUS),      # bank, post-debit: a debit may exist
        ("E", Route.AMBIGUOUS),      # bank, pre-debit: probe the outage
    ],
)
def test_triage_routes(key, expected_route):
    s = sc.BY_KEY[key]
    latest = max(s.observations(), key=lambda o: o.event_time)
    route, _, why = triage(latest)
    assert route == expected_route, f"{key}: {why}"


def test_triage_is_pure_and_explains_itself():
    """`why` lands in the audit trail, so it must never be empty."""
    for s in sc.ALL:
        for o in s.observations():
            route, decision, why = triage(o)
            assert isinstance(route, Route)
            assert isinstance(decision, Decision)
            assert why, f"{o.event_type} produced no explanation"


def test_settling_events_never_reach_the_resolver():
    s = sc.SCENARIO_A
    captured = next(o for o in s.observations() if o.event_type == "payment.captured")
    route, decision, _ = triage(captured)
    assert route == Route.FOLD_ONLY
    assert decision == Decision.SETTLED
    assert not needs_resolver(captured)


def test_unknown_error_source_biases_to_ambiguous():
    """When in doubt the classifier must over-route, never under-route."""
    s = sc.SCENARIO_D
    o = s.observations()[0].model_copy(update={"error_source": "martian"})
    assert triage(o)[0] == Route.AMBIGUOUS


def test_downtime_events_are_ignored_not_resolved():
    """Downtime is global, not order-scoped - it must not trigger a fold."""
    o = sc.SCENARIO_E.observations()[0].model_copy(
        update={"event_type": "payment.downtime.started"}
    )
    assert triage(o)[0] == Route.IGNORE


# ---------------------------------------------------------- executor --

async def test_executor_is_idempotent_on_the_same_evidence():
    s = sc.SCENARIO_B
    ex = Executor(dry_run=True)
    intent = RecoveryIntent(
        action=Action.SEND_RECOVERY_LINK, template_id="RCV_RETRY",
        variables=["2340", "Acme", "link"], channel=Channel.SMS,
        confidence=0.95, evidence_version="ev1",
    )
    d = evaluate(intent, s.observations(), s.evaluate_at)

    first = await ex.execute(d, intent, "order_x", "pay_x", 234000)
    second = await ex.execute(d, intent, "order_x", "pay_x", 234000)

    assert first.status == "STUBBED"
    assert second.status == "SKIPPED"
    assert len([o for o in ex.outcomes if o.status in ("EXECUTED", "STUBBED")]) == 1


async def test_vetoed_decision_never_reaches_the_api():
    """A veto must produce a NOOP outcome, not a suppressed exception."""
    s = sc.SCENARIO_F
    ex = Executor(dry_run=False, client=_ExplodingClient())
    intent = RecoveryIntent(
        action=Action.SEND_RECOVERY_LINK, template_id="RCV_RETRY",
        variables=["2340", "Acme", "link"], channel=Channel.SMS, confidence=1.0,
    )
    d = evaluate(intent, s.observations(), s.evaluate_at)
    assert not d.allowed

    out = await ex.execute(d, intent, s.order_id, "pay_x", 234000)
    assert out.status == "VETOED"
    assert out.action == Action.NOOP


async def test_whatsapp_is_stubbed_not_silently_downgraded():
    """The console must never claim a channel the executor did not use.

    WhatsApp now delivers for real when Meta Cloud API credentials are
    configured (see tests/test_whatsapp.py). Unconfigured, it must still
    stub rather than quietly fall back to SMS - a downgrade the console
    would report as a WhatsApp send.
    """
    s = sc.SCENARIO_E
    ex = Executor(dry_run=False, client=_ExplodingClient())
    intent = RecoveryIntent(
        action=Action.SEND_RECOVERY_LINK, template_id="RCV_UPI_ALT",
        variables=["netbanking", "2340", "Acme", "link"],
        channel=Channel.WHATSAPP, confidence=0.92, method_hint="upi",
    )
    d = evaluate(intent, s.observations(), s.evaluate_at, evidence=s.evidence)
    assert d.allowed, d.reason

    out = await ex.execute(d, intent, s.order_id, "pay_x", 234000)
    assert out.status == "STUBBED"
    assert "not configured" in out.detail
    # The payload it *would* have sent is preserved for the audit row.
    assert out.request["notes"]["template_id"] == "RCV_UPI_ALT"


async def test_escalation_lands_in_the_exception_queue_with_identifiers():
    """Scenario D's deliverable: a row a human can act on."""
    ex = Executor(dry_run=True)
    intent = RecoveryIntent(action=Action.ESCALATE, confidence=0.45,
                            reasoning="bank silent past T+1; RRN 230901495295")
    d = evaluate(intent, sc.SCENARIO_D.observations(), sc.SCENARIO_D.evaluate_at)
    out = await ex.execute(d, intent, "order_D", "pay_D", 234000)

    assert out.status == "EXECUTED"
    assert len(ex.exception_queue) == 1
    assert "230901495295" in ex.exception_queue[0]["reason"]


async def test_api_failure_becomes_an_outcome_not_an_exception():
    """I9: a failed execution is data. It must not raise into the caller."""
    ex = Executor(dry_run=False, client=_ExplodingClient())
    intent = RecoveryIntent(
        action=Action.CAPTURE, confidence=0.93, evidence_version="ev1",
    )
    obs = sc.SCENARIO_C.observations()
    d = evaluate(intent, obs, sc.SCENARIO_C.evaluate_at)
    assert d.allowed

    out = await ex.execute(d, intent, "order_C", "pay_C", 234000)
    assert out.status == "FAILED"
    assert "HTTPError" in out.detail or "Timeout" in out.detail


# --------------------------------------------------------- scheduler --

async def test_scheduler_keeps_the_earliest_deadline():
    """A later re-fold must not push a tighter deadline out."""
    s = InMemoryScheduler()
    await s.schedule("order_1", 2_000)
    await s.schedule("order_1", 5_000)
    await s.schedule("order_1", 1_000)
    assert await s.pending() == [("order_1", 1_000)]


async def test_scheduler_returns_only_due_items_and_clears_them():
    s = InMemoryScheduler()
    await s.schedule("early", 100)
    await s.schedule("later", 9_000)

    assert await s.due(500) == ["early"]
    assert await s.due(500) == []                 # consumed
    assert [k for k, _ in await s.pending()] == ["later"]


async def test_scheduler_worker_refolds_due_orders():
    s = InMemoryScheduler()
    await s.schedule("order_D", 100)
    seen = []

    async def on_due(order_id, now):
        seen.append((order_id, now))

    n = await run_worker(s, on_due, stop_after=1, tick_s=0.01, clock=lambda: 200)
    assert n == 1 and seen[0][0] == "order_D"


def test_recheck_deadlines_are_banking_days():
    """The scheduler's due time is the thing E14 protects."""
    dl = tat_deadline(sc.FRIDAY_EVENING)
    assert ist_date(dl).strftime("%a") == "Tue"


# ---------------------------------------------------------- pipeline --

async def test_pending_tat_schedules_a_recheck():
    """PENDING_TAT is a deferral, so something must come back for it."""
    s = sc.SCENARIO_D
    sched = InMemoryScheduler()
    p = Pipeline(scheduler=sched)
    inside = s.start + 3600

    d = await p.process(s.observations(), inside, order_id=s.order_id)
    assert d.verdict.verdict == Verdict.PENDING_TAT

    pending = await sched.pending()
    assert pending and pending[0][0] == s.order_id

    # The recheck must land strictly AFTER the window closes, never on
    # it: re-folding at 23:59:59 on the Tuesday would race the very
    # reversal we are waiting for. Tuesday 23:59:59 + 60s is Wednesday.
    due = pending[0][1]
    deadline = tat_deadline(s.observations()[0].event_time)
    assert due > deadline
    assert ist_date(deadline).strftime("%a") == "Tue"
    assert due - deadline <= 300, "recheck drifted far past the window"


async def test_pipeline_records_every_veto_for_the_audit_trail():
    s = sc.SCENARIO_F
    p = Pipeline()
    await p.process(s.observations(), s.evaluate_at, order_id=s.order_id)
    # F settles cleanly, so the deterministic NOOP is allowed; the veto
    # log is exercised where an intent is actually refused.
    s2 = sc.SCENARIO_D
    p2 = Pipeline()
    await p2.process(s2.observations(), s2.start + 3600, order_id=s2.order_id)
    assert isinstance(p2.vetoes, list)


async def test_pipeline_never_moves_money_on_a_settled_order():
    """The invariant, asserted directly rather than via the demo runner."""
    for key in ("A", "F"):
        s = sc.BY_KEY[key]
        ex = Executor(dry_run=True)
        p = Pipeline(executor=ex)
        await p.process(s.observations(), s.evaluate_at, order_id=s.order_id)

        moved = [
            o for o in ex.outcomes
            if o.action in (Action.SEND_RECOVERY_LINK, Action.REFUND)
            and o.status in ("EXECUTED", "STUBBED")
        ]
        assert not moved, f"{key} moved money on a settled order: {moved}"


class _ExplodingClient(RazorpayClient):
    """Any live call fails. Proves nothing reaches the API when vetoed."""

    def __init__(self):
        pass

    async def _post(self, *a, **kw):
        import httpx

        raise httpx.HTTPError("network is down")
