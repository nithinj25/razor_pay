"""The gate. Two families of rule, one job: refuse.

I8 is the whole design: the gate re-derives every precondition from the
event store itself and trusts no field the model produced. That is what
makes scenario F survive. Even granting the attacker a fully compromised
intent - CONFIRMED_FAILED, confidence 1.0, send the link - the gate loads
the observations, folds them, sees a captured sibling, and vetoes. The
model's confidence is not evidence.

Every veto is persisted with its reason. `SELECT * FROM vetoes` is the
audit trail the track brief asks for.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, time

from core.banking import IST, banking_days_between, tat_deadline
from core.events import Observation
from core.fold import FoldConfig, build_state, fold, is_flippable
from core.intents import (
    MAX_VARIABLE_LEN,
    MAX_VARIABLES,
    Category,
    Channel,
    RecoveryIntent,
    template_registered,
)
from core.verdicts import (
    AMBIGUOUS_SOURCES,
    CONFIDENCE_FLOOR,
    MONEY_MOVING,
    Action,
    ErrorSource,
    Verdict,
)

#: TCCCPR service-implicit consent is triggered by a customer action and
#: does not last indefinitely. Beyond this the message stops being a
#: response to their action and becomes marketing.
SERVICE_IMPLICIT_WINDOW_H = 72

#: Explicit consent validity, capped by the 2025 TRAI amendment.
EXPLICIT_CONSENT_MAX_DAYS = 7

#: Permitted outbound calling hours, IST.
CALLING_HOURS = (time(9, 0), time(21, 0))


@dataclass
class Veto:
    rule: str
    reason: str

    def __str__(self) -> str:
        return f"{self.rule}: {self.reason}"


@dataclass
class CustomerContext:
    """Compliance facts about the recipient. Not derivable from payments.

    In production this comes from the consent ledger and the DLT scrub.
    Defaults are deliberately permissive so that a *missing* ledger does
    not silently block everything - but `dnd` defaults to FULLY_BLOCKED
    nowhere, and every unknown is recorded on the decision.
    """

    dnd: str = "NONE"                       # NONE | PARTIAL | FULLY_BLOCKED
    consent_age_days: int = 0
    opted_out_channels: frozenset[Channel] = frozenset()
    timezone: str = "IST"
    engagement_channel: Channel | None = None

    #: Destinations we actually hold, straight off the payment entity, plus
    #: whether a WhatsApp sender is wired up. The gate derives reachability
    #: from these rather than trusting the channel the model chose.
    #: `None` means "no reachability information" and is permissive, so a
    #: caller that does not populate it keeps the old behaviour.
    contact: str | None = None
    email: str | None = None
    whatsapp_ready: bool = False


@dataclass
class GateDecision:
    allowed: bool
    action: Action
    vetoes: list[Veto] = field(default_factory=list)
    idem_key: str = ""
    rendered: str = ""
    #: Everything the gate re-derived for itself, for the audit row.
    derived: dict = field(default_factory=dict)

    @property
    def reason(self) -> str:
        return "; ".join(str(v) for v in self.vetoes) if self.vetoes else "allowed"


def idempotency_key(payment_id: str, action: Action, evidence_version: str) -> str:
    """Keyed on evidence version, not just action (ARCHITECTURE section 9).

    The same decision on the same evidence is a no-op. Genuinely new
    evidence produces a new key and may legitimately act again.
    """
    return hashlib.sha256(
        f"{payment_id}|{action.value}|{evidence_version}".encode()
    ).hexdigest()


def evidence_version(obs: list[Observation]) -> str:
    """A stable digest of exactly what we knew when we decided."""
    h = hashlib.sha256()
    for o in sorted(obs, key=lambda o: (o.event_time, o.event_id)):
        h.update(f"{o.event_id}:{o.event_type}:{o.event_time}".encode())
    return h.hexdigest()[:16]


def _recovery_source(observations: list[Observation]) -> str:
    """The original order, if this payment was made on a link we sent.

    The executor stamps `source_order` into the recovery link's notes and
    Razorpay carries those onto the payment entity, so this is re-derived
    from the webhook itself rather than from anything the process
    remembers - it survives a worker restart, which an in-memory set of
    "orders we created" would not.
    """
    for o in sorted(observations, key=lambda x: x.event_time, reverse=True):
        ent = (o.payload.get("payload", {}).get("payment", {}) or {}).get("entity", {}) or {}
        src = (ent.get("notes") or {}).get("source_order")
        if src:
            return str(src)
    return ""


def _reachable(channel: Channel, customer: CustomerContext) -> bool:
    """Can anything actually be delivered over this channel, for them?

    Deliberately not a capability check on the provider alone: a WhatsApp
    sender with no recipient and an SMS route with no phone number are
    both "configured" and both reach nobody.

    A customer with no reachability information at all is treated as
    reachable. Blocking on absent data would invert I5 here - degradation
    biases toward inaction, but that is inaction on *money*, and refusing
    to message a customer we simply have not looked up yet is a different
    thing from refusing to move funds we cannot account for.
    """
    if customer.contact is None and customer.email is None:
        return True
    if channel == Channel.WHATSAPP:
        # The sender falls back to a configured demo recipient, which is
        # a destination even when the order carries no contact.
        return bool(customer.contact) or customer.whatsapp_ready
    if channel in (Channel.SMS, Channel.VOICE):
        return bool(customer.contact)
    if channel == Channel.EMAIL:
        return bool(customer.email)
    return True


def evaluate(
    intent: RecoveryIntent,
    observations: list[Observation],
    now: int,
    customer: CustomerContext | None = None,
    seen_keys: set[str] | None = None,
    cfg: FoldConfig = FoldConfig(),
    evidence: tuple = (),
) -> GateDecision:
    """Re-derive, then judge. Nothing from `intent` is taken on faith.

    `evidence` is fetcher output - provenanced facts with a source and a
    confidence, not model output - so re-deriving with it does not breach
    I8. What the gate refuses to trust is the *intent*: its verdict, its
    confidence, its claim about siblings.
    """
    customer = customer or CustomerContext()
    seen_keys = seen_keys if seen_keys is not None else set()
    vetoes: list[Veto] = []

    # ---- Re-derivation. This is I8. The model's view is irrelevant. ----
    truth = fold(observations, now, evidence=evidence, cfg=cfg)
    st = build_state(observations, truth.order_id)
    any_sibling_non_terminal = any(
        is_flippable(p, now, cfg, evidence) for p in st.payments.values()
    )
    amount_paid, amount_due = st.amount_paid, st.amount_due

    failed = st.failed
    latest_failure = max(failed, key=lambda p: p.failed_at or 0) if failed else None
    src = ErrorSource.UNKNOWN
    if latest_failure:
        try:
            src = ErrorSource(latest_failure.error_source or "unknown")
        except ValueError:
            src = ErrorSource.UNKNOWN

    failure_ts = st.first_failure_at
    age_hours = (now - failure_ts) / 3600 if failure_ts else 0.0
    age_banking = banking_days_between(failure_ts, now) if failure_ts else 0

    payment_id = latest_failure.payment_id if latest_failure else (
        next(iter(st.payments), "")
    )
    ev_version = intent.evidence_version or evidence_version(observations)
    idem = idempotency_key(payment_id, intent.action, ev_version)

    derived = {
        "re_derived_verdict": truth.verdict.value,
        "re_derived_confidence": truth.confidence,
        "any_sibling_non_terminal": any_sibling_non_terminal,
        "amount_paid": amount_paid,
        "amount_due": amount_due,
        "error_source": src.value,
        "age_hours": round(age_hours, 2),
        "age_banking_days": age_banking,
        "evidence_version": ev_version,
        "model_claimed_confidence": intent.confidence,
    }

    # ================= correctness invariants =================

    if intent.action == Action.SEND_RECOVERY_LINK:
        # I3. The reason this project exists. A recovery link creates a
        # NEW order, and Razorpay's clubbing is scoped to one order - so
        # this check cannot be delegated to the payment gateway.
        if any_sibling_non_terminal:
            vetoes.append(
                Veto("I3", "sibling attempt unresolved on source order; a link "
                           "would create a second order Razorpay cannot club")
            )
        if amount_due and amount_paid >= amount_due:
            vetoes.append(
                Veto("SETTLED", f"order already paid ({amount_paid} >= {amount_due} paise)")
            )
        if truth.verdict in (Verdict.ORDER_SETTLED, Verdict.DUPLICATE_RISK):
            vetoes.append(
                Veto("VERDICT", f"re-derived verdict is {truth.verdict.value}, "
                                f"not CONFIRMED_FAILED")
            )
        # E14 / I10 - wait only while there is genuinely something in
        # flight. The window exists so an unaided bank reversal can land;
        # if no debit was ever attempted (customer cancellation, or a
        # pre-debit failure during a known outage) there is nothing to
        # wait for, and waiting would just be latency dressed as caution.
        in_flight = latest_failure is not None and is_flippable(
            latest_failure, now, cfg, evidence
        )
        if failure_ts and in_flight and src in AMBIGUOUS_SOURCES and now <= tat_deadline(
            failure_ts, cfg.tat_banking_days, cfg.holidays
        ):
            closes = datetime.fromtimestamp(
                tat_deadline(failure_ts, cfg.tat_banking_days, cfg.holidays), tz=IST
            )
            vetoes.append(
                Veto("TAT", f"inside RBI T+1 banking window (closes "
                            f"{closes:%a %d %b %H:%M} IST)")
            )

    if intent.action == Action.CAPTURE:
        if not st.authorized:
            vetoes.append(Veto("CAPTURE", "no authorised payment to capture"))
        if st.captured:
            vetoes.append(Veto("CAPTURE", "order already has a captured payment"))

    if intent.action == Action.REFUND and not st.captured and not st.authorized:
        vetoes.append(Veto("REFUND", "nothing captured or authorised to refund"))

    if intent.action in MONEY_MOVING and idem in seen_keys:
        vetoes.append(Veto("I5", f"duplicate action, idempotency key {idem[:12]} seen"))

    # A voice call to *elicit* evidence is the one action whose floor
    # cannot be read as "how sure are we what happened". We call precisely
    # because the verdict is uncertain - requiring high confidence in the
    # verdict before calling would forbid the call in exactly the case it
    # exists for. The floor still applies, but to the thing it can
    # meaningfully bound: confidence that a call is the right instrument,
    # which `should_offer_voice` establishes deterministically.
    floor = CONFIDENCE_FLOOR.get(intent.action, 1.0)
    if intent.confidence < floor:
        vetoes.append(
            Veto("FLOOR", f"confidence {intent.confidence:.2f} below floor "
                          f"{floor:.2f} for {intent.action.value}")
        )
    # I7 - the model may narrow the action space, never widen it. If the
    # re-derived verdict does not itself support acting, the model's own
    # confidence cannot licence it.
    if intent.action in MONEY_MOVING and truth.confidence < floor:
        vetoes.append(
            Veto("I7", f"re-derived confidence {truth.confidence:.2f} below floor "
                       f"{floor:.2f}; model claimed {intent.confidence:.2f}")
        )

    # =================== compliance (TCCCPR) ===================

    if intent.action in (Action.SEND_RECOVERY_LINK, Action.VOICE_CALL):
        if intent.category == Category.PROMOTIONAL:
            vetoes.append(Veto("TCCCPR", "no promotional recovery messaging"))

        if customer.dnd == "FULLY_BLOCKED" and intent.category != Category.SERVICE_IMPLICIT:
            vetoes.append(Veto("DND", "fully-blocked subscriber: service-implicit only"))

        # The implicit-consent window governs SERVICE_IMPLICIT messages -
        # those justified by the customer's own recent action. A call made
        # under SERVICE_EXPLICIT consent is governed by the explicit rule
        # below instead, which is stricter (7 days, TRAI 2025). Applying
        # both meant a voice call was vetoed by a window that never
        # applied to it.
        if (intent.category == Category.SERVICE_IMPLICIT
                and age_hours > SERVICE_IMPLICIT_WINDOW_H):
            vetoes.append(
                Veto("CONSENT", f"outside implicit-consent window "
                                f"({age_hours:.0f}h > {SERVICE_IMPLICIT_WINDOW_H}h)")
            )

        # DLT registration governs *messages*. A voice call is a
        # conversation whose questions are composed per case - there is no
        # template to register, and demanding one vetoed every call.
        # Voice is constrained instead by consent age and calling hours.
        if intent.channel != Channel.VOICE and not template_registered(
            intent.template_id, intent.channel
        ):
            vetoes.append(
                Veto("DLT", f"template {intent.template_id!r} not registered for "
                            f"channel {intent.channel.value if intent.channel else None!r}")
            )

        if len(intent.variables) > MAX_VARIABLES:
            vetoes.append(Veto("DLT", f"{len(intent.variables)} variables > {MAX_VARIABLES}"))
        for v in intent.variables:
            if len(v) > MAX_VARIABLE_LEN:
                vetoes.append(Veto("DLT", f"variable {v[:12]!r} exceeds {MAX_VARIABLE_LEN} chars"))

        # A recovery link is a NEW order (I3). If the payment that just
        # failed was itself made on a link we sent, another link makes a
        # third order for one debt - the exact multiplication this project
        # exists to prevent, performed by the agent itself. Found live:
        # one failed checkout produced a recovery order, whose own failure
        # produced another, unbounded.
        #
        # The chain stops at one. A customer who could not pay the
        # recovery link does not need a second link; they need a human,
        # and NOOP here puts the order in front of one.
        source = _recovery_source(observations)
        if source:
            vetoes.append(
                Veto("RECOVERY_CHAIN",
                     f"payment is already on a recovery link for {source}; "
                     "a second link would be a third order for one debt")
            )

        # A channel with no destination is not a plan. The model picks a
        # channel from a menu of three; whether anything can actually be
        # delivered over it is a fact about this order, so the gate derives
        # it rather than believing the choice. Found live: the strategist
        # chose SMS for an order carrying no contact, the executor built a
        # correct payload, and the message reached nobody while the run
        # still reported EXECUTED.
        if intent.channel and not _reachable(intent.channel, customer):
            vetoes.append(
                Veto("UNREACHABLE",
                     f"no destination for {intent.channel.value}")
            )

        if intent.channel and intent.channel in customer.opted_out_channels:
            vetoes.append(Veto("OPT_OUT", f"opted out of {intent.channel.value}"))
        # Stricter than currently required, and deliberately so: an
        # opt-out on any channel is honoured across all of them. Say so
        # in review rather than letting it look like a bug.
        if customer.opted_out_channels:
            vetoes.append(
                Veto("OPT_OUT", "cross-channel opt-out honoured "
                                f"({', '.join(c.value for c in customer.opted_out_channels)})")
            )

    if intent.action == Action.VOICE_CALL:
        if customer.consent_age_days > EXPLICIT_CONSENT_MAX_DAYS:
            vetoes.append(
                Veto("CONSENT", f"explicit consent expired "
                                f"({customer.consent_age_days}d > {EXPLICIT_CONSENT_MAX_DAYS}d)")
            )
        if not in_calling_hours(now):
            vetoes.append(Veto("HOURS", "outside permitted calling hours (09:00-21:00 IST)"))

    rendered = ""
    if not vetoes and intent.template_id:
        from core.intents import TEMPLATE_REGISTRY

        t = TEMPLATE_REGISTRY.get(intent.template_id)
        if t:
            rendered = t.render(list(intent.variables))

    return GateDecision(
        allowed=not vetoes,
        action=intent.action if not vetoes else Action.NOOP,
        vetoes=vetoes,
        idem_key=idem,
        rendered=rendered,
        derived=derived,
    )


def in_calling_hours(now: int) -> bool:
    t = datetime.fromtimestamp(now, tz=IST).time()
    return CALLING_HOURS[0] <= t <= CALLING_HOURS[1]
