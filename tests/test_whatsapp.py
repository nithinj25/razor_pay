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

    # Registered templates send as type=template, so the content lives in
    # positional parameters rather than a text body.
    params = [p["text"] for p in
              graph.calls[0][1]["template"]["components"][0]["parameters"]]
    assert "netbanking" in params and "Acme Store" in params


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
    sent = graph.calls[0][1]
    assert sent["template"]["name"] == "rcv_downtime_wait"
    params = [p["text"] for p in sent["template"]["components"][0]["parameters"]]
    assert not any("http" in p for p in params), (
        f"a wait message carried a payment link: {params}")
    assert params == ["2340", "Acme Store", "2 hours"]


# --------------------------------------------------- template path --

async def test_registered_template_is_preferred_over_freeform():
    """WhatsApp forbids business-initiated freeform outside a 24h window
    the customer opens. A customer who failed a payment and left has not
    opened one - so freeform works in testing and fails for every real
    customer, returning a success id either way."""
    from services.gate.rules import evaluate

    s = sc.SCENARIO_E
    graph = FakeGraph(200, {"messages": [{"id": "wamid.T"}]})
    sender = WhatsAppSender(configured(demo_whatsapp_to="919000000000"), client=graph)
    ex = Executor(dry_run=True, whatsapp=sender)

    d = evaluate(_intent(), s.observations(), s.evaluate_at, evidence=s.evidence)
    out = await ex.execute(d, _intent(), s.order_id, "pay_E1downtime", 234000)

    assert out.status == "EXECUTED"
    body = graph.calls[0][1]
    assert body["type"] == "template", "sent freeform when a template was registered"
    assert body["template"]["name"] == "rcv_upi_alt"
    params = [p["text"] for p in body["template"]["components"][0]["parameters"]]
    assert params[0] == "netbanking", "parameter order must follow the DLT definition"
    assert "template rcv_upi_alt" in out.detail


def test_template_params_follow_the_dlt_order_and_take_the_real_link():
    from core.intents import TEMPLATE_REGISTRY
    from services.executor.main import _template_params

    t = TEMPLATE_REGISTRY["RCV_UPI_ALT"]
    # The model invents a URL because it cannot know one that does not
    # exist yet; the executor overwrites that slot.
    out = _template_params(
        t, ["netbanking", "2340", "Acme", "https://pay.acme/invented"],
        "https://rzp.io/rzp/REAL",
    )
    assert out == ["netbanking", "2340", "Acme", "https://rzp.io/rzp/REAL"]


def test_template_params_pad_rather_than_shift():
    """Fewer variables than the template declares must not shift slots left."""
    from core.intents import TEMPLATE_REGISTRY
    from services.executor.main import _template_params

    t = TEMPLATE_REGISTRY["RCV_UPI_ALT"]
    out = _template_params(t, ["netbanking"], "")
    assert len(out) == len(t.variables)
    assert out[0] == "netbanking" and out[1] == ""


# ------------------------------------------------ delivery receipts --

def test_receipts_extracted_from_metas_envelope():
    from services.ingress.whatsapp_hook import extract_statuses

    rows = extract_statuses({"entry": [{"changes": [{"value": {"statuses": [
        {"id": "wamid.A", "status": "delivered", "timestamp": "1700000000",
         "recipient_id": "919902740794"},
        {"id": "wamid.B", "status": "failed", "timestamp": "1700000001",
         "recipient_id": "919902740794",
         "errors": [{"code": 131047, "title": "Re-engagement message"}]},
    ]}}]}]})
    assert [r["status"] for r in rows] == ["delivered", "failed"]
    assert rows[1]["error_code"] == 131047


def test_inbound_customer_messages_are_ignored():
    """The same webhook carries inbound messages; only statuses matter."""
    from services.ingress.whatsapp_hook import extract_statuses

    assert extract_statuses({"entry": [{"changes": [{"value": {
        "messages": [{"id": "wamid.IN", "from": "919902740794"}]
    }}]}]}) == []


def test_receipt_signature_verification():
    import hashlib
    import hmac
    import json as _json

    from services.ingress.whatsapp_hook import verify_signature

    raw = _json.dumps({"entry": []}).encode()
    good = "sha256=" + hmac.new(b"appsecret", raw, hashlib.sha256).hexdigest()

    assert verify_signature(raw, good, "appsecret")
    assert not verify_signature(raw, "sha256=deadbeef", "appsecret")
    assert not verify_signature(raw, None, "appsecret")
    # No app secret: accept but the caller marks it unverified. Dropping
    # every receipt would be worse than recording that we could not check.
    assert verify_signature(raw, None, "")


def test_receipt_summary_counts_by_status():
    from services.ingress import whatsapp_hook as h

    h.RECEIPTS.clear()
    h.RECEIPTS["a"] = {"message_id": "a", "status": "delivered", "ts": 1}
    h.RECEIPTS["b"] = {"message_id": "b", "status": "failed", "ts": 2}
    s = h.summary()
    assert s["total"] == 2
    assert s["by_status"] == {"delivered": 1, "failed": 1}
    assert len(s["failures"]) == 1
    h.RECEIPTS.clear()
