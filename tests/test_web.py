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
    assert {r["key"] for r in rows} == {"A", "B", "C", "D", "E", "F"}
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
