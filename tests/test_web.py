"""Console endpoints. Smoke-level, but they guard two real claims.

Screen 2's whole argument is that the verdict is *derived*: the same
observations at a later clock must be able to produce a different
verdict. Screen 3's argument is that an escalation is actionable, which
means the RRN has to actually be in the payload. Both are asserted here
rather than left to the eye.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from web.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_scenarios_listed(client):
    rows = client.get("/api/scenarios").json()
    # A-F are GUARDRAILS' six. G is the unstructured-evidence case, split
    # out so the specified six stay exactly as specified.
    assert {r["key"] for r in rows} == {"A", "B", "C", "D", "E", "F", "G"}
    assert all(r["note"] for r in rows), "a scenario with no note explains nothing"


def test_timeline_shows_the_verdict_changing_with_time_alone(client):
    """Scenario D: no new event, later clock, different verdict.

    This is the strongest single piece of evidence that status is a fold
    rather than a stored field.
    """
    d = client.get("/api/timeline?scenario=D").json()
    verdicts = [r["verdict"] for r in d["rows"]]

    assert verdicts[0] == "PENDING_TAT"
    assert verdicts[-1] == "UNRESOLVED"
    assert d["rows"][-1]["event_type"].startswith("(re-fold")
    assert d["rows"][-1]["payment_id"] is None, "the last row must add no new evidence"


def test_timeline_exposes_the_banking_day_gap(client):
    d = client.get("/api/timeline?scenario=D").json()
    c = d["clocks"]
    assert "Sat 24 Jan" in c["naive_deadline"]
    assert "Tue 27 Jan" in c["banking_deadline"]
    assert c["ratio"] > 3.5


def test_timeline_marks_the_delivery_inversion(client):
    """Scenario A: event_time and received_at disagree, and it is shown."""
    d = client.get("/api/timeline?scenario=A").json()
    assert any(r["inverted"] for r in d["rows"])
    assert d["rows"][-1]["verdict"] == "ORDER_SETTLED"


def test_exception_queue_carries_the_rrn(client):
    """An escalation without identifiers is a shrug, not a handoff."""
    d = client.get("/api/exceptions").json()
    assert d["count"] >= 1

    card = next(c for c in d["cards"] if c["scenario"] == "D")
    assert card["rrn"] == "230901495295"
    assert card["rrn"] in card["suggested_next_step"]
    assert card["what_was_checked"] and card["what_is_missing"]
    assert "Tue 27 Jan" in card["window_closed"]


def test_settled_scenarios_never_appear_in_the_exception_queue(client):
    d = client.get("/api/exceptions").json()
    assert not [c for c in d["cards"] if c["scenario"] in ("A", "F")]


def test_unknown_scenario_is_a_404_not_a_500(client):
    assert client.get("/api/timeline?scenario=Z").status_code == 404


def test_metrics_endpoint(client):
    """404 with instructions before the demo runs, data afterwards."""
    r = client.get("/api/metrics")
    assert r.status_code in (200, 404)
    if r.status_code == 404:
        assert "harness.demo" in r.json()["error"]
    else:
        h = r.json()["headline"]
        assert h["nishchay_duplicate_orders"] == 0
        assert h["nishchay_false_positives"] == 0


# ----------------------------------------- merchant dashboard --

def test_orders_list_covers_every_scenario(client):
    """The merchant view answers 'what happened to my orders?'"""
    d = client.get("/api/orders").json()
    assert d["count"] == 7
    assert d["source"] in ("live", "scenarios")

    by_id = {o["order_id"]: o for o in d["orders"]}
    assert by_id["order_A1nishchay01"]["verdict"] == "ORDER_SETTLED"
    assert by_id["order_A1nishchay01"]["action"] == "NOOP"
    # Every row carries what a merchant needs to triage at a glance.
    for o in d["orders"]:
        assert o["verdict"] and o["action"]
        assert "health" in (o["agents"] or {})


def test_order_detail_carries_the_agent_trace(client):
    """A merchant asking 'what did the agent do' needs the steps, not
    just the conclusion."""
    d = client.get("/api/orders/order_G7nishchay07").json()
    assert d["order_id"] == "order_G7nishchay07"
    assert d["observations"], "no events to show"
    assert d["customer_messages"], "G's whole point is the customer's email"

    latest = d["latest"]
    assert latest["verdict"]
    assert "steps" in latest and "agents" in latest
    assert latest["triage"]["route"]


def test_order_detail_404s_for_an_unknown_order(client):
    assert client.get("/api/orders/order_does_not_exist").status_code == 404


def test_settled_order_shows_no_agent_steps(client):
    """A rules-only order should say so rather than showing an empty
    trace that looks like a failure."""
    d = client.get("/api/orders/order_A1nishchay01").json()
    assert d["latest"]["agents"]["health"] == "rules-only"
    assert d["latest"]["agents"]["model_calls"] == 0


# ------------------------------------------- live agent stream --

def test_agent_stream_emits_steps_then_the_outcome(client):
    """The demo view needs steps as they happen, not a finished table.

    Scripted mode so this never touches a provider.
    """
    with client.stream("GET", "/api/agents/stream?scenario=G&mode=scripted") as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())

    events = [ln[7:] for ln in body.splitlines() if ln.startswith("event: ")]
    assert events[0] == "start"
    assert "step" in events, "no steps streamed"
    assert events[-1] == "done"
    # The order matters: a verdict must not precede the steps that produced it.
    assert events.index("verdict") > events.index("step")


def test_agent_stream_rejects_an_unknown_scenario(client):
    with client.stream("GET", "/api/agents/stream?scenario=Z") as r:
        body = "".join(r.iter_text())
    assert "unknown scenario" in body


def test_a_rules_only_case_still_streams_its_single_step(client):
    """A finishes in one node. The stream must show that rather than
    looking like it failed to start."""
    with client.stream("GET", "/api/agents/stream?scenario=A&mode=scripted") as r:
        body = "".join(r.iter_text())
    assert body.count("event: step") == 1
    assert "ORDER_SETTLED" in body
