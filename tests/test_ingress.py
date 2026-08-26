"""Day 1 acceptance: signature, dedupe, append, ordering key."""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from core.config import settings
from core.events import order_id_of
from harness import scenarios as sc
from services.ingress.main import app, stats
from services.ingress.signing import sign, verify, verify_any

SECRET = settings().rzp_webhook_secret


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def post(client, body: dict, event_id: str, secret: str = SECRET, sig: str | None = None):
    """Sign the exact bytes we send - the same discipline as the server."""
    raw = json.dumps(body).encode()
    return client.post(
        "/webhook/razorpay",
        content=raw,
        headers={
            "X-Razorpay-Signature": sig if sig is not None else sign(raw, secret),
            "X-Razorpay-Event-Id": event_id,
            "Content-Type": "application/json",
        },
    )


def test_bad_signature_is_rejected(client):
    body = sc.SCENARIO_B.deliveries[0].body
    assert post(client, body, "evt_bad", sig="deadbeef").status_code == 400


def test_missing_event_id_is_rejected(client):
    raw = json.dumps(sc.SCENARIO_B.deliveries[0].body).encode()
    r = client.post(
        "/webhook/razorpay",
        content=raw,
        headers={"X-Razorpay-Signature": sign(raw, SECRET)},
    )
    assert r.status_code == 400


def test_valid_webhook_is_stored_once(client):
    d = sc.SCENARIO_B.deliveries[0]
    before = stats["received"]

    assert post(client, d.body, d.event_id).status_code == 200
    assert post(client, d.body, d.event_id).status_code == 200   # chaos 1

    assert stats["received"] == before + 1, "duplicate was stored twice"


def test_signature_covers_raw_bytes_not_the_parsed_dict():
    """Pitfall #2 in executable form.

    Re-serialising changes separators and key order, so a signature over
    the re-serialised form does not match the original bytes.
    """
    original = b'{"event":"payment.failed","created_at":1}'
    reserialised = json.dumps(json.loads(original)).encode()
    assert original != reserialised
    assert verify(original, sign(original, SECRET), SECRET)
    assert not verify(reserialised, sign(original, SECRET), SECRET)


def test_dual_secret_window():
    """E8: retries signed with the rotated-out secret must still validate."""
    raw = b'{"event":"payment.failed"}'
    old_sig = sign(raw, "old_secret")
    assert verify_any(raw, old_sig, "new_secret", "old_secret")
    assert not verify_any(raw, old_sig, "new_secret", "")
    assert not verify_any(raw, "", "new_secret", "old_secret")


def test_partition_key_is_order_id_never_payment_id():
    """Pitfall #6. Siblings must share a partition or I3 is unenforceable."""
    for d in sc.SCENARIO_A.deliveries:
        raw = json.dumps(d.body)
        assert order_id_of(raw) == sc.SCENARIO_A.order_id
        assert order_id_of(raw) != d.body["payload"]["payment"]["entity"]["id"]


def test_health_reports_degradation(client):
    h = client.get("/health").json()
    assert h["ok"] is True
    assert h["mode"] == "test"
    assert isinstance(h["degraded"], list)
