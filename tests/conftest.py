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

# Must precede any import that calls settings() - it is lru_cached.
os.environ.setdefault("ENABLE_KAFKA", "false")
os.environ.setdefault("POSTGRES_DSN", "postgresql://nishchay:nishchay@127.0.0.1:5432/nishchay")


@pytest.fixture(scope="session", autouse=True)
def _fast_local_settings():
    from core.config import settings

    settings.cache_clear()
    cfg = settings()
    cfg.enable_kafka = False
    yield
    settings.cache_clear()
