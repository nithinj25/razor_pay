"""Provider plumbing and agent tracing.

No network. The NVIDIA path has two pieces of pure logic that would fail
silently if wrong - schema flattening and JSON extraction - and silence is
the problem: a mis-flattened schema still returns 200, just unconstrained.
"""

from __future__ import annotations

import json

import pytest
from core.config import Settings
from core.intents import Assessment
from core.llm import (
    LLMUnavailable,
    NullLLM,
    NvidiaLLM,
    build_llm,
    extract_json,
    to_nim_schema,
)
from core.trace import AgentStep, AgentSummary
from services.strategist.graph import Composition


def cfg(**kw) -> Settings:
    """Settings isolated from the developer's real .env.

    Without `_env_file=None` these assertions read whatever key happens to
    be configured locally, so they pass or fail depending on the machine.
    """
    return Settings(_env_file=None, **kw)


# ------------------------------------------------------- provider --

def test_provider_switch():
    assert cfg(anthropic_api_key="k").provider == "anthropic"
    assert cfg(nvidia_api_key="k").provider == "nvidia"
    # Anthropic wins when both are present.
    assert cfg(anthropic_api_key="a", nvidia_api_key="n").provider == "anthropic"
    assert cfg().provider == "none"
    # An explicit choice is honoured, and refuses to fall through.
    assert cfg(llm_provider="nvidia", nvidia_api_key="n").provider == "nvidia"
    assert cfg(llm_provider="nvidia", anthropic_api_key="a").provider == "none"


def test_build_llm_returns_the_right_client():
    assert isinstance(build_llm(cfg()), NullLLM)
    assert isinstance(build_llm(cfg(nvidia_api_key="k")), NvidiaLLM)


def test_model_name_tracks_the_provider():
    assert cfg(nvidia_api_key="k", nvidia_model="m").model_name == "m"
    assert cfg().model_name == ""


# --------------------------------------------------------- schema --

def test_nim_schema_inlines_refs():
    """Pydantic emits $defs/$ref; NIM strict mode does not resolve them.

    An unresolved $ref does not error - it just stops constraining, so the
    enum silently becomes 'any string'. That is the failure this guards.
    """
    s = to_nim_schema(Composition)
    dumped = json.dumps(s)
    assert "$ref" not in dumped and "$defs" not in dumped
    assert s["additionalProperties"] is False

    # The constraints that carry the security property must survive.
    assert set(s["properties"]["template_id"]["enum"]) == {
        "RCV_UPI_ALT", "RCV_RETRY", "RCV_DOWNTIME_WAIT"
    }
    assert set(s["properties"]["channel"]["enum"]) == {"SMS", "WHATSAPP", "EMAIL"}
    assert s["properties"]["variables"]["maxItems"] == 5
    assert s["properties"]["variables"]["items"]["maxLength"] == 30

    # I7: there is no `action` field for a model to fill.
    assert "action" not in s["properties"]


def test_optional_becomes_a_nullable_type():
    """`anyOf: [str, null]` is pydantic's Optional; strict mode wants a type."""
    s = to_nim_schema(Composition)
    assert s["properties"]["method_hint"]["type"] == ["string", "null"]


def test_assessment_schema_keeps_its_enum():
    s = to_nim_schema(Assessment)
    assert set(s["properties"]["next_probe"]["enum"]) == {
        "probe_downtime", "probe_history", "compose"
    }


# ----------------------------------------------------- extraction --

def test_extract_json_handles_a_reasoning_preamble():
    """Reasoning models narrate before answering, even under a schema."""
    raw = 'Let me think about this.\n{"next_probe": "compose", "confidence": 0.9}\nDone.'
    assert json.loads(extract_json(raw))["next_probe"] == "compose"


def test_extract_json_handles_nesting_and_braces_in_strings():
    raw = 'prose {"a": {"b": 1}, "c": "a } brace in a string"} trailing'
    d = json.loads(extract_json(raw))
    assert d["a"]["b"] == 1 and "brace" in d["c"]


def test_extract_json_handles_escaped_quotes():
    raw = r'{"reasoning": "he said \"no\" firmly", "confidence": 0.5}'
    assert json.loads(extract_json(raw))["confidence"] == 0.5


@pytest.mark.parametrize("bad", ["", "no json here", "{unbalanced"])
def test_extract_json_rejects_junk(bad):
    with pytest.raises(ValueError):
        extract_json(bad)


# -------------------------------------------------------- client --

async def test_nvidia_client_without_a_key_is_unavailable():
    llm = NvidiaLLM(cfg())
    with pytest.raises(LLMUnavailable, match="NVIDIA_API_KEY"):
        await llm.structured(Assessment, "sys", "user")


async def test_rate_limit_degrades_rather_than_raising_upward():
    """The free tier 429s mid-loop. That must become a fallback, not a crash."""
    class Busy:
        async def post(self, *a, **kw):
            class R:
                status_code = 429
                text = "rate limited"
            return R()

    llm = NvidiaLLM(cfg(nvidia_api_key="k"), client=Busy())
    with pytest.raises(LLMUnavailable, match="provider busy"):
        await llm.structured(Assessment, "sys", "user")


async def test_schema_violation_retries_once_then_gives_up():
    calls = {"n": 0}

    class Wrong:
        async def post(self, *a, **kw):
            calls["n"] += 1

            class R:
                status_code = 200

                def raise_for_status(self): ...

                def json(self):
                    # Valid JSON, wrong shape: next_probe is not in the enum.
                    return {"choices": [{"message": {"content":
                            '{"reasoning":"x","next_probe":"nonsense","confidence":0.5}'}}],
                            "usage": {}}
            return R()

    llm = NvidiaLLM(cfg(nvidia_api_key="k"), client=Wrong())
    with pytest.raises(LLMUnavailable, match="schema violation"):
        await llm.structured(Assessment, "sys", "user")
    assert calls["n"] == 2, "should try exactly twice"


async def test_valid_response_is_parsed():
    class Good:
        async def post(self, *a, **kw):
            class R:
                status_code = 200

                def raise_for_status(self): ...

                def json(self):
                    return {"choices": [{"message": {"content":
                            'Thinking...\n{"reasoning":"outage is scoped",'
                            '"next_probe":"probe_history","confidence":0.8}'}}],
                            "usage": {"prompt_tokens": 120, "completion_tokens": 40}}
            return R()

    llm = NvidiaLLM(cfg(nvidia_api_key="k"), client=Good())
    a = await llm.structured(Assessment, "sys", "user", node="assess")
    assert a.next_probe == "probe_history"
    assert llm.usage.tokens_in == 120 and llm.usage.tokens_out == 40


# --------------------------------------------------------- trace --

def test_health_distinguishes_all_four_states():
    """The whole point: these must never look the same on a dashboard."""
    def h(**kw):
        return AgentSummary(**kw).health

    assert h(model_calls=2) == "live"
    assert h(scripted_calls=2) == "scripted"
    assert h(fallbacks=2) == "degraded"
    assert h(rule_nodes=3) == "rules-only"
    # A scripted run must never be reported as live.
    assert h(scripted_calls=5, rule_nodes=2) == "scripted"


def test_scripted_is_never_counted_as_a_model_call():
    s = AgentSummary.of((
        AgentStep(agent="strategist", node="assess", summary="", source="scripted"),
        AgentStep(agent="resolver", node="precheck", summary="", source="rules"),
    ))
    assert s.model_calls == 0
    assert s.scripted_calls == 1
    assert s.health == "scripted"


def test_summary_rolls_up_tokens_and_latency():
    s = AgentSummary.of((
        AgentStep(agent="resolver", node="plan", summary="", source="model",
                  tokens_in=100, tokens_out=50, latency_ms=1200),
        AgentStep(agent="strategist", node="compose", summary="", source="model",
                  tokens_in=200, tokens_out=80, latency_ms=800),
    ))
    assert s.tokens == 430
    assert s.latency_ms == 2000
    assert s.resolver_ran and s.strategist_ran
