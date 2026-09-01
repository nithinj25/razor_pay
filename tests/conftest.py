"""Test wiring.

Tests run against the in-process fallbacks by default. That is deliberate
rather than a shortcut: the fallbacks are a supported operating mode (the
slip protocol drops Kafka for an asyncio queue), so exercising them in
CI keeps that path honest. Integration tests that need real backends
declare it explicitly and are skipped when Docker is not up.
"""

from __future__ import annotations

import os

import pytest

# Must run before pytest-asyncio creates a loop, or the integration
# tests cannot reach the Postgres checkpointer on Windows.
from core.infra import use_psycopg_compatible_loop  # noqa: E402

use_psycopg_compatible_loop()

# No test may reach a real provider. BUILD.md is explicit: tests run on
# ScriptedLLM, only the accuracy run touches a live API. Once a key lands
# in .env, anything constructing a default client would start making real
# calls - which is slow, rate-limited, and makes the suite depend on the
# network. Blank the keys so build_llm() returns NullLLM.
# Every provider key, not just the ones that existed when this was
# written - a new provider added to .env would otherwise silently put the
# suite back on the network.
#
# RZP_KEY_SECRET is on the list for the same reason, one step further
# out: with a secret present, any test running the executor outside dry
# run would create real payment links on the merchant account. No test
# does that today, which is exactly why it needs to be impossible rather
# than merely avoided.
for _key in ("ANTHROPIC_API_KEY", "NVIDIA_API_KEY", "GEMINI_API_KEY",
             "WHATSAPP_ACCESS_TOKEN", "DEMO_WHATSAPP_TO",
             "RZP_KEY_SECRET"):
    os.environ[_key] = ""
os.environ["LLM_PROVIDER"] = "auto"

# Must precede any import that calls settings() - it is lru_cached.
os.environ.setdefault("ENABLE_KAFKA", "false")
os.environ.setdefault("POSTGRES_DSN", "postgresql://nishchay:nishchay@127.0.0.1:5432/nishchay")


@pytest.fixture(scope="session", autouse=True)
def _fast_local_settings():
    from core.config import settings

    settings.cache_clear()
    cfg = settings()
    cfg.enable_kafka = False
    # Belt and braces: .env is read at construction, so clear the keys on
    # the instance too rather than trusting env precedence.
    cfg.anthropic_api_key = ""
    cfg.nvidia_api_key = ""
    cfg.gemini_api_key = ""
    # No test may send a real WhatsApp message either.
    cfg.whatsapp_access_token = ""
    cfg.demo_whatsapp_to = ""
    yield
    settings.cache_clear()
