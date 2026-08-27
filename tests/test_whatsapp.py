"""WhatsApp delivery. No network.

The gate has already decided this message may be sent; these tests cover
what the delivery layer itself can get wrong - messaging the wrong person,
swallowing a failure, or reporting a stub as a send.
"""

from __future__ import annotations


from core.config import Settings
from core.intents import Category, Channel, RecoveryIntent
from core.verdicts import Action
from harness import scenarios as sc
from services.executor.main import Executor
from services.executor.whatsapp import NOT_ALLOWED, WINDOW_CLOSED, WhatsAppSender


def cfg(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def configured(**kw) -> Settings:
    return cfg(whatsapp_access_token="t", whatsapp_phone_number_id="123", **kw)


class FakeGraph:
    """Stands in for Meta's Graph API."""

    def __init__(self, status=200, body=None):
        self.status, self.body, self.calls = status, body or {}, []

    async def post(self, path, json=None, **kw):
        self.calls.append((path, json))
        outer = self

        class R:
            status_code = outer.status
            content = b"x"

            def json(self):
                return outer.body

        return R()


# ------------------------------------------------------ recipient --

def test_demo_recipient_overrides_the_customer():
    """Meta test numbers only reach an allow-list.

    Sending to the customer's real number would just fail, so the demo
    recipient wins - but the executor has to say so, not imply we messaged
    the actual customer.
    """
    w = WhatsAppSender(configured(demo_whatsapp_to="+91 90000 00000"))
    assert w.recipient_for("+919999999999") == "919000000000"


def test_customer_contact_used_when_no_demo_override():
    w = WhatsAppSender(configured())
    assert w.recipient_for("+91 99999-99999") == "919999999999"


def test_unconfigured_sender_reports_rather_than_raising():
    assert WhatsAppSender(cfg()).configured is False


async def test_unconfigured_send_is_a_result_not_an_exception():
    res = await WhatsAppSender(cfg()).send_text("919000000000", "hi")
    assert res.ok is False and "not configured" in res.detail


# --------------------------------------------------------- errors --

async def test_closed_session_window_explains_the_fix():
    """The single most common silent failure in a WhatsApp integration."""
    graph = FakeGraph(400, {"error": {"code": WINDOW_CLOSED, "message": "..."}})
    res = await WhatsAppSender(configured(), client=graph).send_text("91900", "hi")
    assert res.ok is False
    assert "reply to the WhatsApp thread" in res.detail


async def test_not_allow_listed_explains_the_fix():
    graph = FakeGraph(400, {"error": {"code": NOT_ALLOWED, "message": "..."}})
    res = await WhatsAppSender(configured(), client=graph).send_text("91900", "hi")
    assert "allow-list" in res.detail


async def test_successful_send_returns_the_message_id():
    graph = FakeGraph(200, {"messages": [{"id": "wamid.ABC"}]})
    res = await WhatsAppSender(configured(), client=graph).send_text("919000000000", "hello")
    assert res.ok and res.message_id == "wamid.ABC"
    _, body = graph.calls[0]
    assert body["to"] == "919000000000"
    assert body["type"] == "text"
    # A preview card of the checkout page adds noise to a payment instruction.
    assert body["text"]["preview_url"] is False


async def test_number_is_normalised_before_sending():
    graph = FakeGraph(200, {"messages": [{"id": "m"}]})
    await WhatsAppSender(configured(), client=graph).send_text("+91 (90000) 00-000", "x")
    assert graph.calls[0][1]["to"] == "919000000000"


# ------------------------------------------------------- executor --

def _intent() -> RecoveryIntent:
    return RecoveryIntent(
        action=Action.SEND_RECOVERY_LINK, template_id="RCV_UPI_ALT",
        variables=["netbanking", "2340", "Acme Store", "link"],
        channel=Channel.WHATSAPP, category=Category.SERVICE_IMPLICIT,
        confidence=0.92, method_hint="upi",
    )


async def test_executor_sends_over_whatsapp_when_configured():
    from services.gate.rules import evaluate

    s = sc.SCENARIO_E
    graph = FakeGraph(200, {"messages": [{"id": "wamid.X"}]})
    sender = WhatsAppSender(configured(demo_whatsapp_to="919000000000"), client=graph)
    ex = Executor(dry_run=True, whatsapp=sender)

    d = evaluate(_intent(), s.observations(), s.evaluate_at, evidence=s.evidence)
    assert d.allowed, d.reason

    out = await ex.execute(d, _intent(), s.order_id, "pay_E1downtime", 234000)
    assert out.status == "EXECUTED"
    assert "919000000000" in out.detail
    assert "demo recipient" in out.detail, "must not imply we messaged the customer"

    body = graph.calls[0][1]["text"]["body"]
    assert "netbanking" in body and "Acme Store" in body


async def test_executor_stubs_whatsapp_when_not_configured():
    """Never silently downgrade to SMS - the console would be lying."""
    from services.gate.rules import evaluate

    s = sc.SCENARIO_E
    ex = Executor(dry_run=True, whatsapp=WhatsAppSender(cfg()))
    d = evaluate(_intent(), s.observations(), s.evaluate_at, evidence=s.evidence)
    out = await ex.execute(d, _intent(), s.order_id, "pay_E1downtime", 234000)

    assert out.status == "STUBBED"
    assert "not configured" in out.detail


async def test_send_failure_is_recorded_not_raised():
    from services.gate.rules import evaluate

    s = sc.SCENARIO_E
    graph = FakeGraph(400, {"error": {"code": WINDOW_CLOSED, "message": "x"}})
    ex = Executor(dry_run=True, whatsapp=WhatsAppSender(configured(), client=graph))
    d = evaluate(_intent(), s.observations(), s.evaluate_at, evidence=s.evidence)

    out = await ex.execute(d, _intent(), s.order_id, "pay_E1downtime", 234000)
    assert out.status == "FAILED"
    assert "session window" in out.detail


async def test_vetoed_intent_never_reaches_whatsapp():
    """The gate is upstream of delivery, and stays that way."""
    from services.gate.rules import evaluate

    s = sc.SCENARIO_F                      # settled order - I3 vetoes
    graph = FakeGraph(200, {"messages": [{"id": "m"}]})
    ex = Executor(dry_run=True, whatsapp=WhatsAppSender(configured(), client=graph))

    d = evaluate(_intent(), s.observations(), s.evaluate_at)
    assert not d.allowed

    out = await ex.execute(d, _intent(), s.order_id, "pay_F1injected", 234000)
    assert out.status == "VETOED"
    assert graph.calls == [], "a vetoed intent reached the network"


# ------------------------------------------------------ link slot --

def test_link_replaces_whatever_the_model_invented():
    """Asked for a link variable, a model will confidently invent a URL.

    Matching only the literal "{link}" left the invented one in place and
    appended the real one, so the customer got two links and the first
    404s.
    """
    from services.executor.main import _with_link

    real = "https://rzp.io/rzp/REAL"
    assert _with_link("Pay by UPI instead: https://pay.acme/alt", real).endswith(real)
    assert "pay.acme" not in _with_link("Pay by UPI instead: https://pay.acme/alt", real)
    assert _with_link("Complete it here: {link}", real).endswith(real)
    assert _with_link("Complete it here: link", real).endswith(real)
    # Idempotent - a message that already carries the link is left alone.
    assert _with_link(f"here: {real}", real).count(real) == 1


async def test_wait_template_gets_no_payment_link():
    """RCV_DOWNTIME_WAIT says "you were not charged, retry later".

    A pay-now link in that message contradicts it, so a template with no
    link variable must not receive one.
    """
    from services.gate.rules import evaluate

    s = sc.SCENARIO_E
    graph = FakeGraph(200, {"messages": [{"id": "m"}]})
    sender = WhatsAppSender(configured(demo_whatsapp_to="919000000000"), client=graph)
    ex = Executor(dry_run=True, whatsapp=sender)

    wait = RecoveryIntent(
        action=Action.SEND_RECOVERY_LINK, template_id="RCV_DOWNTIME_WAIT",
        variables=["2340", "Acme Store", "2 hours"],
        channel=Channel.WHATSAPP, category=Category.SERVICE_IMPLICIT, confidence=0.92,
    )
    d = evaluate(wait, s.observations(), s.evaluate_at, evidence=s.evidence)
    assert d.allowed, d.reason

    await ex.execute(d, wait, s.order_id, "pay_E1downtime", 234000)
    body = graph.calls[0][1]["text"]["body"]
    assert "http" not in body, f"a wait message carried a payment link: {body}"
    assert "not been charged" in body
