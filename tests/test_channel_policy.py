"""Delivery has exactly one owner, chosen the same way every time.

Two arrangements exist and no others: WhatsApp, which we send ourselves
over Meta's API with the payment link's `notify` off; or SMS/email,
where the link is created with `notify` on and Razorpay sends it.

Left to the model, which one applied varied per run on identical inputs
- SMS once, WhatsApp the next time - so one customer with one debt was
messaged on both rails across a retry. These tests pin the choice to
facts about the deployment instead.
"""

from __future__ import annotations

from core.intents import Category, Channel, RecoveryIntent
from core.verdicts import Action
from services.gate.rules import CustomerContext
from services.pipeline import _choose_channel


def intent(channel: Channel, template_id: str = "RCV_RETRY") -> RecoveryIntent:
    return RecoveryIntent(
        action=Action.SEND_RECOVERY_LINK,
        template_id=template_id,
        variables=["2340", "Acme Store", "https://rzp.io/i/x"],
        channel=channel,
        category=Category.SERVICE_IMPLICIT,
        confidence=0.95,
        reasoning="payment failed",
    )


def test_the_same_inputs_always_choose_the_same_rail():
    """The bug this exists for: the pick must not vary per invocation."""
    who = CustomerContext(contact="919902740794", email="a@b.c", whatsapp_ready=True)

    picks = {
        _choose_channel(intent(guess), who, []).channel
        for guess in (Channel.SMS, Channel.WHATSAPP, Channel.EMAIL)
    }

    assert len(picks) == 1, f"the model's guess still leaks through: {picks}"


def test_whatsapp_wins_when_we_can_deliver_it_ourselves():
    """Preferred because we send it and get a delivery receipt back."""
    who = CustomerContext(contact="919902740794", email="a@b.c", whatsapp_ready=True)

    assert _choose_channel(intent(Channel.SMS), who, []).channel == Channel.WHATSAPP


def test_sms_carries_it_when_whatsapp_is_not_configured():
    """Without a sender there is no WhatsApp rail, so Razorpay notifies."""
    who = CustomerContext(contact="919902740794", email="a@b.c", whatsapp_ready=False)

    assert _choose_channel(intent(Channel.WHATSAPP), who, []).channel == Channel.SMS


def test_a_channel_the_customer_engages_with_beats_our_preference():
    who = CustomerContext(
        contact="919902740794", email="a@b.c", whatsapp_ready=True,
        engagement_channel=Channel.SMS,
    )

    assert _choose_channel(intent(Channel.WHATSAPP), who, []).channel == Channel.SMS


def test_the_choice_stays_inside_the_templates_registered_channels():
    """RCV_UPI_ALT is registered for SMS and WhatsApp, never email.

    Routing must not move a message onto a rail its DLT registration
    does not cover, even when that rail is the only reachable one.
    """
    email_only = CustomerContext(contact="", email="a@b.c", whatsapp_ready=False)

    chosen = _choose_channel(intent(Channel.SMS, "RCV_UPI_ALT"), email_only, [])

    assert chosen.channel != Channel.EMAIL


def test_an_unreachable_customer_is_left_for_the_gate_to_veto():
    """Silence beats a false success, and only the gate may say so."""
    nowhere = CustomerContext(contact="", email="", whatsapp_ready=False)
    trace: list[str] = []

    chosen = _choose_channel(intent(Channel.SMS), nowhere, trace)

    assert chosen.channel == Channel.SMS, "must not invent a deliverable rail"
    assert any("unreachable" in t for t in trace)
