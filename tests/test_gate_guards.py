"""Two preconditions the gate learned the hard way, on live payments.

Both were invisible from reading the code, and both reported success
while doing nothing useful:

* the strategist chose a channel with no destination, so a correct
  message was handed to a rail that reached nobody;
* a recovery link's own failure produced another recovery link, and so
  on - the agent multiplying orders for one debt, which is the exact
  exposure the project exists to close.
"""

from __future__ import annotations

import json

from core.intents import Channel
from harness import scenarios as sc
from services.gate.rules import CustomerContext, evaluate

from tests.test_gate import link_intent


def _with_payment_notes(obs, notes: dict):
    """Stamp notes where Razorpay actually carries them: on the payment."""
    out = []
    for o in obs:
        p = json.loads(json.dumps(o.payload))
        ent = (
            p.setdefault("payload", {})
            .setdefault("payment", {})
            .setdefault("entity", {})
        )
        ent["notes"] = notes
        out.append(o.model_copy(update={"payload": p}))
    return out


# ------------------------------------------------ recovery chains --

def test_a_recovery_link_does_not_beget_another():
    """One debt, one link. The second would be a third order.

    A recovery link creates a NEW order (I3). If the payment that just
    failed was itself made on a link we sent, issuing another multiplies
    orders for a single debt - the duplicate-charge exposure this system
    exists to prevent, performed by the system.
    """
    s = sc.SCENARIO_B
    tainted = _with_payment_notes(
        s.observations(),
        {"source_order": "order_original01", "template_id": "RCV_RETRY"},
    )

    d = evaluate(link_intent(confidence=0.95), tainted, s.evaluate_at)

    assert not d.allowed
    assert any(v.rule == "RECOVERY_CHAIN" for v in d.vetoes), d.reason
    # The veto has to name the original, or a human cannot pick it up.
    assert "order_original01" in d.reason


def test_an_ordinary_order_is_not_mistaken_for_a_recovery_one():
    """The guard must not veto the first link, which is the whole point."""
    s = sc.SCENARIO_B
    ordinary = _with_payment_notes(s.observations(), {"merchant_ref": "INV-4471"})

    d = evaluate(link_intent(confidence=0.95), ordinary, s.evaluate_at)

    assert d.allowed, f"the first recovery link was vetoed: {d.reason}"


# ---------------------------------------------------- reachability --

def test_a_channel_with_no_destination_is_vetoed():
    """Found live: SMS chosen for an order carrying no phone number."""
    s = sc.SCENARIO_B
    nowhere = CustomerContext(contact="", email="", whatsapp_ready=False)

    d = evaluate(
        link_intent(channel=Channel.SMS), s.observations(), s.evaluate_at,
        customer=nowhere,
    )

    assert not d.allowed
    assert any(v.rule == "UNREACHABLE" for v in d.vetoes), d.reason


def test_whatsapp_is_reachable_through_a_configured_sender():
    """A configured sender with a fallback recipient is a destination."""
    s = sc.SCENARIO_B
    ctx = CustomerContext(contact="", email="", whatsapp_ready=True)

    d = evaluate(
        link_intent(channel=Channel.WHATSAPP), s.observations(), s.evaluate_at,
        customer=ctx,
    )

    assert d.allowed, d.reason


def test_absent_reachability_information_does_not_block():
    """Unknown is not the same as unreachable.

    I5 biases toward inaction on *money*. Refusing to message a customer
    we simply have not looked up yet is a different thing, and treating
    it the same would veto every message in any deployment without a
    contact ledger.
    """
    s = sc.SCENARIO_B

    d = evaluate(
        link_intent(channel=Channel.SMS), s.observations(), s.evaluate_at,
        customer=CustomerContext(),
    )

    assert d.allowed, d.reason
