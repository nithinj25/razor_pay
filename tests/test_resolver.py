"""Day 6 acceptance. ScriptedLLM only - the real API is never called here.

The load-bearing assertion is `llm_calls == 0` for A, B and C. If that
ever goes non-zero, the claim that the model is used only where it earns
its place has quietly stopped being true, and this is where we find out.
"""

from __future__ import annotations

import pytest

from core.llm import LLMUnavailable, NullLLM, ScriptedLLM
from core.verdicts import Evidence, Verdict
from harness import scenarios as sc
from services.resolver.fetchers import CircuitBreaker, FetchContext, guarded
from services.resolver.graph import Narrative, Plan, Resolver


def scripted(*plans, narratives=()):
    llm = ScriptedLLM()
    for p in plans:
        llm.queue(Plan, p)
    for n in narratives:
        llm.queue(Narrative, n)
    return llm


# ------------------------------------------------ zero-LLM fast path --

@pytest.mark.parametrize("key", ["A", "B", "C"])
async def test_scenarios_abc_use_no_llm(key):
    s = sc.BY_KEY[key]
    # A ScriptedLLM with nothing queued raises if touched, so any call
    # here fails the test loudly rather than silently.
    llm = ScriptedLLM()
    r = Resolver(llm=llm)
    out = await r.resolve(s.observations(), s.evaluate_at, order_id=s.order_id)

    assert out["llm_calls"] == 0, f"{key} reached the model"
    assert llm.usage.calls == 0
    assert out["verdict"].verdict == s.ground_truth


async def test_precheck_ends_the_graph_immediately():
    s = sc.SCENARIO_A
    out = await Resolver(llm=ScriptedLLM()).resolve(s.observations(), s.evaluate_at)
    assert out["rounds"] == 0, "fetchers ran on a case the rules already settled"
    assert out["verdict"].verdict == Verdict.ORDER_SETTLED


# ---------------------------------------------------- planned fetch --

async def test_ambiguous_case_plans_and_fetches():
    """D is genuinely ambiguous, so the graph plans, probes and narrates."""
    s = sc.SCENARIO_D
    llm = scripted(
        Plan(reasoning="check siblings and settlement", fetchers=["attempts", "settlement"], confidence=0.7),
        narratives=[
            Narrative(
                summary="Bank-side failure with no resolving event.",
                what_was_checked=["sibling attempts", "settlement report"],
                what_is_missing=["bank statement confirmation"],
                suggested_next_step="Ask the customer for their bank reference against RRN 230901495295.",
            )
        ],
    )
    r = Resolver(llm=llm, fetchers={"attempts": _stub_attempts, "settlement": _stub_settlement})
    out = await r.resolve(s.observations(), s.evaluate_at, order_id=s.order_id)

    assert out["verdict"].verdict == Verdict.UNRESOLVED
    assert out["rounds"] >= 1
    assert out["llm_calls"] >= 1
    assert out["narrative"] is not None
    assert "230901495295" in out["narrative"].suggested_next_step


async def test_downtime_evidence_flips_scenario_E():
    """E: with the outage confirmed, the failure is attributable."""
    s = sc.SCENARIO_E
    llm = scripted(Plan(reasoning="probe the outage", fetchers=["downtime"], confidence=0.8))
    r = Resolver(llm=llm, fetchers={"downtime": _stub_downtime})
    out = await r.resolve(s.observations(), s.evaluate_at, order_id=s.order_id)
    assert out["verdict"].verdict == Verdict.CONFIRMED_FAILED
    assert "R10_downtime_pre_debit" in out["verdict"].rules_fired


# ---------------------------------------------------------- chaos 4 --

async def test_no_llm_still_produces_a_safe_verdict():
    """Kill the model: rules only, UNRESOLVED, and therefore NOOP."""
    s = sc.SCENARIO_D
    r = Resolver(llm=NullLLM(), fetchers={"attempts": _stub_attempts})
    out = await r.resolve(s.observations(), s.evaluate_at, order_id=s.order_id)

    assert out["verdict"].verdict == Verdict.UNRESOLVED
    assert out["narrative"] is None
    assert any("plan" in d for d in out["degraded"])
    # The verdict is unchanged by the model's absence, because the
    # verdict never came from the model.
    assert out["verdict"].proposed_action.value == "ESCALATE"


async def test_llm_failure_mid_graph_does_not_raise():
    s = sc.SCENARIO_D
    llm = ScriptedLLM().queue(Plan, LLMUnavailable("429 from provider"))
    r = Resolver(llm=llm, fetchers={"attempts": _stub_attempts})
    out = await r.resolve(s.observations(), s.evaluate_at, order_id=s.order_id)
    assert out["verdict"] is not None


# ---------------------------------------------------------- chaos 3 --

async def test_dead_fetcher_degrades_rather_than_raises():
    async def explode(ctx):
        raise ConnectionError("downtime endpoint not enabled")

    ev = await guarded("downtime", explode, FetchContext(order_id="o"), CircuitBreaker("downtime"))
    assert ev.available is False
    assert ev.confidence == 0.0
    assert "ConnectionError" in ev.provenance


async def test_fetcher_timeout_is_bounded():
    import asyncio

    async def slow(ctx):
        await asyncio.sleep(5)

    ev = await guarded("slow", slow, FetchContext(order_id="o"), CircuitBreaker("slow"), timeout_s=0.05)
    assert ev.available is False
    assert "timeout" in ev.provenance


async def test_circuit_breaker_opens_and_stops_calling():
    calls = {"n": 0}

    async def failing(ctx):
        calls["n"] += 1
        raise ConnectionError("down")

    b = CircuitBreaker("x", threshold=2, cooldown_s=999)
    ctx = FetchContext(order_id="o")
    for _ in range(5):
        await guarded("x", failing, ctx, b)

    assert calls["n"] == 2, "breaker did not stop the calls"


# ------------------------------------------------------------ stubs --

async def _stub_attempts(ctx: FetchContext) -> Evidence:
    return Evidence(
        source="attempts",
        value={"count": 1, "statuses": ["failed"], "any_captured": False, "any_authorized": False},
        confidence=0.97,
        provenance="stub",
    )


async def _stub_settlement(ctx: FetchContext) -> Evidence:
    return Evidence(source="settlement", value=None, confidence=0.6, provenance="stub")


async def _stub_downtime(ctx: FetchContext) -> Evidence:
    return Evidence(
        source="downtime",
        value={"active": True, "method": "netbanking", "bank": "HDFC",
               "severity": "high", "scope": "method"},
        confidence=0.9,
        provenance="stub",
    )
