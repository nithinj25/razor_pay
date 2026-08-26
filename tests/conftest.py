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
os.environ["ANTHROPIC_API_KEY"] = ""
os.environ["NVIDIA_API_KEY"] = ""
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
    yield
    settings.cache_clear()
