"""resolve() - the pure function. I4.

Status is never stored. It is `fold(observations, now)`: a deterministic
reduction over every observation on an order, ordered by `event_time`.

Why this and not a status column: webhooks are at-least-once AND
unordered, so last-write-wins is provably wrong. Scenario A is the proof
- `captured` (event_time +47) arrives before `failed` (event_time +18),
and a mutable field lands on FAILED while the customer has already paid.

`fold` performs no I/O, reads no clock, and calls no LLM. `now` and any
fetched `evidence` are parameters. That purity is what buys replay,
property tests, order-independence, and identical verdicts after a crash
(chaos 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from core.banking import (
    DEFAULT_HOLIDAYS,
    add_banking_days,
    end_of_day_ist,
    tat_deadline,
)
from core.claims import ClaimMatch, assess_claim
from core.events import Observation, canonical_order
from core.verdicts import (
    AMBIGUOUS_SOURCES,
    PROPOSED_ACTION,
    TERMINAL_CUSTOMER_REASONS,
    ErrorSource,
    Evidence,
    PaymentStatus,
    Verdict,
    VerdictResult,
)

#: Failure steps at which money cannot yet have left the customer.
#: The bank was never asked to debit, so there is nothing to reverse and
#: the RBI T+1 wait does not apply.
PRE_DEBIT_STEPS: frozenset[str] = frozenset(
    {"payment_initiation", "payment_authentication"}
)

#: Failure steps at which a debit may already have happened and only the
#: *response* was lost. This is the genuinely ambiguous zone - it is what
#: separates scenario D (escalate) from scenario E (act).
POST_DEBIT_STEPS: frozenset[str] = frozenset(
    {"payment_response", "payment_capture"}
)


@dataclass(frozen=True)
class FoldConfig:
    tat_banking_days: int = 1
    settle_horizon_days: int = 3
    holidays: frozenset[date] = DEFAULT_HOLIDAYS


DEFAULT_CONFIG = FoldConfig()


@dataclass
class PaymentState:
    """Folded state of one attempt. Derived, never stored."""

    payment_id: str
    status: PaymentStatus | None = None
    amount: int = 0
    method: str | None = None
    error_source: str | None = None
    error_step: str | None = None
    error_reason: str | None = None
    rrn: str | None = None
    upi_transaction_id: str | None = None
    failed_at: int | None = None
    authorized_at: int | None = None
    captured_at: int | None = None
    refund_processed: bool = False
    refund_failed: bool = False
    last_event_time: int = 0

    def is_terminal(self, now: int, cfg: FoldConfig) -> bool:
        """ARCHITECTURE section 4, by the settlement horizon alone."""
        return not is_flippable(self, now, cfg, ())


def _apply(ps: PaymentState, o: Observation) -> None:
    """Apply one observation to one payment's state.

    Observations arrive pre-sorted by event_time, so a later event simply
    supersedes an earlier one. That is what makes `failed -> authorized`
    (late authorisation) and `failed -> captured` (in-app retry) fall out
    for free rather than needing special cases.
    """
    ps.last_event_time = max(ps.last_event_time, o.event_time)
    ps.amount = ps.amount or o.amount
    ps.method = ps.method or o.method
    ps.rrn = ps.rrn or o.rrn
    ps.upi_transaction_id = ps.upi_transaction_id or o.upi_transaction_id

    match o.event_type:
        case "payment.failed":
            ps.status = PaymentStatus.FAILED
            ps.failed_at = o.event_time
            # Error detail belongs to the failure, so it is only recorded
            # here - a later success must not inherit it.
            ps.error_source = o.error_source
            ps.error_step = o.error_step
            ps.error_reason = o.error_reason
        case "payment.authorized":
            ps.status = PaymentStatus.AUTHORIZED
            ps.authorized_at = o.event_time
        case "payment.captured":
            ps.status = PaymentStatus.CAPTURED
            ps.captured_at = o.event_time
        case "refund.processed":
            ps.status = PaymentStatus.REFUNDED
            ps.refund_processed = True
        case "refund.failed":
            # Money is stranded: the refund itself failed (E13).
            ps.refund_failed = True
        case "payment.created" | "order.paid" | _:
            if ps.status is None and o.status:
                try:
                    ps.status = PaymentStatus(o.status)
                except ValueError:
                    pass


def is_flippable(
    ps: PaymentState,
    now: int,
    cfg: FoldConfig,
    evidence: tuple[Evidence, ...] = (),
) -> bool:
    """Could this attempt still become `captured` or `authorized`?

    This, not raw status, is the question I3 actually asks. A recovery
    link is dangerous precisely when some other attempt on the order may
    still succeed - because the link creates a *second order*, and
    Razorpay's clubbing cannot span two orders.

    Read literally, "failed is non-terminal until the settlement horizon"
    would block every recovery for three days, including a payment the
    customer explicitly cancelled. That is not caution, it is paralysis.
    So a failure is flippable only while it genuinely could flip:

      * a customer-attributable cancellation never reaches the bank, so
        there is nothing to arrive late;
      * a pre-debit failure during a *known, method-scoped* outage never
        reached the bank either;
      * everything else can still come back as a late authorisation or an
        in-app retry until the horizon passes.
    """
    if ps.status in (PaymentStatus.CAPTURED, PaymentStatus.REFUNDED):
        return False
    if ps.status in (PaymentStatus.AUTHORIZED, PaymentStatus.CREATED, None):
        # Authorised money is recoverable by CAPTURE, so a link would
        # duplicate it. Still "in play".
        return True
    if ps.status == PaymentStatus.FAILED:
        if ps.refund_processed:
            return False
        src = _source_of(ps)
        if src == ErrorSource.CUSTOMER and (ps.error_reason or "") in TERMINAL_CUSTOMER_REASONS:
            return False
        if src == ErrorSource.BUSINESS:
            return False
        if (ps.error_step or "") in PRE_DEBIT_STEPS and _downtime_matches(evidence, ps.method):
            return False
        if ps.failed_at is None:
            return True
        horizon = end_of_day_ist(
            add_banking_days(ps.failed_at, cfg.settle_horizon_days, cfg.holidays)
        )
        return now <= horizon
    return True


@dataclass
class OrderState:
    order_id: str
    payments: dict[str, PaymentState] = field(default_factory=dict)
    amount_due: int = 0
    amount_paid_reported: int = 0
    order_paid_seen: bool = False
    settled_in_report: bool = False
    first_failure_at: int | None = None

    @property
    def captured(self) -> list[PaymentState]:
        return [p for p in self.payments.values() if p.status == PaymentStatus.CAPTURED]

    @property
    def authorized(self) -> list[PaymentState]:
        return [
            p for p in self.payments.values() if p.status == PaymentStatus.AUTHORIZED
        ]

    @property
    def failed(self) -> list[PaymentState]:
        return [p for p in self.payments.values() if p.status == PaymentStatus.FAILED]

    @property
    def amount_paid(self) -> int:
        """Prefer the order entity's own figure; fall back to captures.

        Razorpay reports `amount_paid` on the order, which already accounts
        for partial capture and refunds (E12). Summing captures is only a
        fallback for streams where no order entity was ever delivered.
        """
        if self.amount_paid_reported:
            return self.amount_paid_reported
        return sum(p.amount for p in self.captured)


def build_state(obs: list[Observation], order_id: str) -> OrderState:
    """Reduce observations into per-payment and order-level state."""
    st = OrderState(order_id=order_id)
    for o in canonical_order(obs):
        st.amount_due = max(st.amount_due, o.amount_due)
        st.amount_paid_reported = max(st.amount_paid_reported, o.amount_paid)
        if o.event_type == "order.paid":
            st.order_paid_seen = True
        if o.event_type == "settlement.processed":
            st.settled_in_report = True
        pid = o.payment_id
        if not pid:
            continue
        ps = st.payments.setdefault(pid, PaymentState(payment_id=pid))
        _apply(ps, o)

    # An order's due amount is not always delivered (a stream of payment
    # events only). The largest attempt is the best available proxy.
    if not st.amount_due and st.payments:
        st.amount_due = max(p.amount for p in st.payments.values())

    failures = [p.failed_at for p in st.payments.values() if p.failed_at is not None]
    st.first_failure_at = min(failures) if failures else None
    return st


#: Ceiling on how far absent evidence can drag a verdict down. Without a
#: cap, a handful of unconfigured stubs would sink every verdict below
#: every floor, and the system would be unable to act on anything - which
#: is not caution, it is a different failure.
MAX_DEGRADATION = 0.30


def _degrade(confidence: float, evidence: tuple[Evidence, ...]) -> float:
    """I9 - missing evidence lowers confidence, which fails the gate floor,
    which yields NOOP. Degradation biases toward inaction, never toward
    acting on a guess. Chaos 3 depends on this being visible.

    Only probes that were *attempted and failed* count. A probe that was
    never configured is a known unknown, not a new one, and is reported
    on the decision instead of silently taxing its confidence.
    """
    missing = sum(1 for e in evidence if not e.available)
    if not missing:
        return confidence
    penalty = min(0.12 * missing, MAX_DEGRADATION)
    return round(max(0.0, confidence - penalty), 4)


def fold(
    obs: list[Observation],
    now: int,
    order_id: str | None = None,
    evidence: tuple[Evidence, ...] = (),
    cfg: FoldConfig = DEFAULT_CONFIG,
) -> VerdictResult:
    """The pure fold. Deterministic in (obs, now, evidence, cfg).

    Rules are evaluated in strict priority order and the first match wins.
    Ordering is deliberate: positive proof of settlement outranks
    everything, and the ambiguous-wait rule sits below every rule that can
    establish a fact.
    """
    oid = order_id or (obs[0].order_id if obs else "")
    st = build_state(obs, oid)
    fired: list[str] = []

    def result(
        verdict: Verdict, confidence: float, recheck_at: int | None = None
    ) -> VerdictResult:
        return VerdictResult(
            order_id=oid,
            verdict=verdict,
            confidence=_degrade(confidence, evidence),
            proposed_action=PROPOSED_ACTION[verdict],
            rules_fired=tuple(fired),
            evidence=evidence,
            amount_due=st.amount_due,
            amount_paid=st.amount_paid,
            recheck_at=recheck_at,
            any_sibling_non_terminal=any(
                is_flippable(p, now, cfg, evidence) for p in st.payments.values()
            ),
        )

    if not obs:
        fired.append("R0_no_observations")
        return result(Verdict.UNRESOLVED, 0.0)

    # -- R1  Settlement report is ground truth: the money reached the bank.
    settled_ids = {
        e.value for e in evidence if e.source == "settlement" and e.available and e.value
    }
    if st.settled_in_report or (settled_ids & set(st.payments)):
        fired.append("R1_settlement_report")
        return result(Verdict.ORDER_SETTLED, 0.99)

    # -- R2  A captured sibling covering the full amount. Scenario A.
    #        This is the rule that stops the duplicate order (I3).
    if st.captured and st.amount_due and st.amount_paid >= st.amount_due:
        fired.append("R2_sibling_captured_full")
        return result(Verdict.ORDER_SETTLED, 0.99)

    if st.order_paid_seen:
        fired.append("R3_order_paid")
        return result(Verdict.ORDER_SETTLED, 0.99)

    # -- R4  Two captures on one order: the double charge already happened.
    #        Not our doing, but never report it as a recovery.
    if len(st.captured) > 1:
        fired.append("R4_multiple_captures")
        return result(Verdict.DUPLICATE_RISK, 0.9)

    # -- R5  Authorised but never captured. Revenue sitting on the table,
    #        and Razorpay auto-refunds it at 3 days (E3).
    if st.authorized and not st.captured:
        fired.append("R5_uncaptured_auth")
        auth_at = min(p.authorized_at or now for p in st.authorized)
        deadline = end_of_day_ist(
            add_banking_days(auth_at, cfg.settle_horizon_days, cfg.holidays)
        )
        return result(Verdict.UNCAPTURED_AUTH, 0.93, recheck_at=deadline)

    # -- R6  A processed refund means the debit, if any, is reversed.
    if any(p.refund_processed for p in st.payments.values()):
        if st.captured:
            # Captured then refunded with the order still unpaid: money
            # went out and came back. Nothing to recover.
            fired.append("R6b_captured_then_refunded")
            return result(Verdict.CONFIRMED_FAILED, 0.93)
        fired.append("R6_refund_processed")
        return result(Verdict.CONFIRMED_FAILED, 0.95)

    # -- E13  A refund that itself failed strands money. Never automate.
    if any(p.refund_failed for p in st.payments.values()):
        fired.append("R7_refund_failed_stranded")
        return result(Verdict.UNRESOLVED, 0.5)

    failed = st.failed
    if not failed:
        fired.append("R8_no_terminal_evidence")
        return result(Verdict.UNRESOLVED, 0.3)

    latest = max(failed, key=lambda p: p.failed_at or 0)
    src = _source_of(latest)
    step = latest.error_step or ""

    # -- R9  Customer-attributable failure. RBI's own test: not completed
    #        for reasons attributable to the customer is not a "failed
    #        transaction", so no debit occurred and no T+1 wait applies.
    #        Scenario B - this is what stops the system being merely cautious.
    if src == ErrorSource.CUSTOMER and (latest.error_reason or "") in TERMINAL_CUSTOMER_REASONS:
        fired.append("R9_customer_terminal")
        return result(Verdict.CONFIRMED_FAILED, 0.95)

    # -- R10  A known, active, method-scoped outage at a pre-debit step.
    #         The bank was never asked to move money, so there is nothing
    #         in flight to wait for. Scenario E.
    if step in PRE_DEBIT_STEPS and _downtime_matches(evidence, latest.method):
        fired.append("R10_downtime_pre_debit")
        return result(Verdict.CONFIRMED_FAILED, 0.92)

    # -- R10b  The customer says money left their account, and the
    #          reference they quoted is the RRN on this payment. Status
    #          says failed; the customer's bank says otherwise. That is
    #          conflicting evidence about whether money moved, which is
    #          exactly what DUPLICATE_RISK is for - and it bars a recovery
    #          link outright, because sending one would charge them twice.
    #
    #          Razorpay documents that a bank can auto-refund without
    #          changing payment status, so the API alone cannot settle
    #          this. The customer's own reference can.
    claim = _claim_assessment(evidence, latest)
    if claim is not None and claim.match is ClaimMatch.CONFIRMS:
        fired.append("R10b_customer_confirms_debit")
        return result(Verdict.DUPLICATE_RISK, 0.9)

    # -- R10c  They quoted a reference and it belongs to a different
    #          payment. Their complaint is real but not about this order,
    #          so it must not change this verdict - only be recorded.
    if claim is not None and claim.match is ClaimMatch.CONTRADICTS:
        fired.append("R10c_customer_reference_mismatch")

    # -- R11  The ambiguous zone: a debit may exist and the bank may still
    #         reverse it unaided. Wait out the RBI window (E14, I10).
    if src in AMBIGUOUS_SOURCES and latest.failed_at is not None:
        deadline = tat_deadline(latest.failed_at, cfg.tat_banking_days, cfg.holidays)
        if now <= deadline:
            fired.append("R11_pending_tat")
            return result(Verdict.PENDING_TAT, 0.75, recheck_at=deadline + 60)

        # Window closed with no resolving event. Razorpay documents that a
        # bank can auto-refund without changing status (E6), so this is
        # genuinely unresolvable from the API. Scenario D.
        if step in POST_DEBIT_STEPS:
            fired.append("R12_tat_expired_post_debit")
            # An unverifiable debit claim is doubt, not evidence. It can
            # lower confidence - never raise it - so a human sees the
            # complaint without the system acting on an assertion.
            if claim is not None and claim.match is ClaimMatch.UNVERIFIED:
                fired.append("R12b_unverified_debit_claim")
                return result(Verdict.UNRESOLVED, 0.35)
            return result(Verdict.UNRESOLVED, 0.45)
        fired.append("R13_tat_expired_pre_debit")
        return result(Verdict.CONFIRMED_FAILED, 0.9)

    # -- R14  Customer source but a reason we do not have in the terminal
    #         set. Plausibly failed, but not provably - let the resolver's
    #         LLM look at the unstructured evidence rather than guessing.
    if src == ErrorSource.CUSTOMER:
        fired.append("R14_customer_unclassified_reason")
        return result(Verdict.CONFIRMED_FAILED, 0.82)

    fired.append("R15_insufficient_evidence")
    return result(Verdict.UNRESOLVED, 0.4)


def _claim_assessment(evidence: tuple[Evidence, ...], ps: PaymentState):
    """Compare the customer's claim against this payment's identifiers.

    The model extracted the claim; this comparison is deterministic. A
    customer asserting a debit proves nothing - what makes it evidence is
    whether the reference they quoted is the RRN on our payment.
    """
    for e in evidence:
        if e.source != "customer_claim" or not e.available:
            continue
        value = e.value if isinstance(e.value, dict) else {}
        claim = value.get("claim")
        if claim is None:
            continue
        return assess_claim(
            claim, ps.rrn, ps.upi_transaction_id, value.get("raw_text", "")
        )
    return None


def _source_of(ps: PaymentState) -> ErrorSource:
    try:
        return ErrorSource(ps.error_source or "unknown")
    except ValueError:
        return ErrorSource.UNKNOWN


def _downtime_matches(evidence: tuple[Evidence, ...], method: str | None) -> bool:
    """True when downtime evidence is active for this payment's method.

    Method scoping matters: a UPI outage says nothing about a netbanking
    failure, and treating it as though it did would licence acting on a
    payment that may still be in flight.
    """
    for e in evidence:
        if e.source != "downtime" or not e.available or not isinstance(e.value, dict):
            continue
        if not e.value.get("active"):
            continue
        dm = e.value.get("method")
        if dm is None or method is None or dm == method:
            return True
    return False


def naive_lww(obs: list[Observation]) -> str:
    """The baseline's status model, kept as an executable counter-example.

    Mutable field, last write in *arrival* order wins. On scenario A this
    returns FAILED while the customer has already paid. Day 2's test
    asserts this is wrong, so the reason for the whole architecture stays
    in the test suite rather than in a comment.
    """
    status = "created"
    for o in sorted(obs, key=lambda o: o.received_at):
        if o.event_type.startswith("payment.") and o.status:
            status = o.status
    return status.upper()


async def resolve(store, order_id: str, now: int, **kw) -> VerdictResult:
    """Load every sibling on the order, then fold. The only I/O is the load."""
    obs = await store.load(order_id)
    return fold(obs, now, order_id=order_id, **kw)
