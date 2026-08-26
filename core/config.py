"""Settings. Loaded once, injected everywhere - no module reads os.environ."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # -- Razorpay. The webhook secret is NOT the API secret (pitfall #1). --
    rzp_key_id: str = "rzp_test_placeholder"
    rzp_key_secret: str = ""
    rzp_webhook_secret: str = "nishchay_test_webhook_secret"
    #: E8 - during a rotation, retries of older events still carry a
    #: signature made with the previous secret. Both must validate.
    rzp_webhook_secret_prev: str = ""
    rzp_api_base: str = "https://api.razorpay.com"

    # -- LLM provider ------------------------------------------------
    #: "anthropic" | "nvidia" | "auto". `auto` prefers whichever key is
    #: present, Anthropic first.
    llm_provider: str = "auto"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-5"

    #: NVIDIA NIM, OpenAI-compatible. Free tier, which is why it is here.
    #: Note it does NOT support forced tool_choice or nvext.guided_json on
    #: this account - structured output is enforced with
    #: response_format={"type":"json_schema", strict:true}, which held the
    #: schema under prompt injection in testing. See core/llm.py.
    nvidia_api_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    #: nemotron-3-nano-30b-a3b answers in ~4s; super-120b-a12b reasons
    #: better but takes ~9s, and the strategist's 6-turn loop has a 15s
    #: budget. Latency is the reason the smaller model is the default.
    nvidia_model: str = "nvidia/nemotron-3-nano-30b-a3b"

    postgres_dsn: str = "postgresql://nishchay:nishchay@localhost:5432/nishchay"
    redis_url: str = "redis://localhost:6379"
    kafka_bootstrap: str = "localhost:19092"
    clickhouse_url: str = "http://localhost:8123"

    tat_window_banking_days: int = 1
    settle_horizon_days: int = 3
    max_turns: int = 6
    max_tokens_per_resolution: int = 8_000
    fetch_timeout_s: float = 3.0
    strategist_max_latency_s: float = 15.0

    #: E15 - test and live keys are separate worlds. Assert the mode on
    #: every event rather than discovering the mix-up in the audit log.
    require_test_mode: bool = True

    #: Degrade to in-process queues when the broker is absent, so the fold
    #: and the console still run on a laptop with no Docker.
    enable_kafka: bool = True

    @property
    def is_test_mode(self) -> bool:
        return self.rzp_key_id.startswith("rzp_test_")

    @property
    def provider(self) -> str:
        """Which provider will actually be used."""
        if self.llm_provider == "anthropic":
            return "anthropic" if self.anthropic_api_key else "none"
        if self.llm_provider == "nvidia":
            return "nvidia" if self.nvidia_api_key else "none"
        if self.anthropic_api_key:
            return "anthropic"
        if self.nvidia_api_key:
            return "nvidia"
        return "none"

    @property
    def model_name(self) -> str:
        """The model id to report in traces, whichever provider is live."""
        return {
            "anthropic": self.anthropic_model,
            "nvidia": self.nvidia_model,
        }.get(self.provider, "")


@lru_cache
def settings() -> Settings:
    return Settings()
