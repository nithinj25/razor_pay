"""Settings. Loaded once, injected everywhere - no module reads os.environ."""

from __future__ import annotations

from functools import lru_cache
from typing import ClassVar

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
    #: "gemini" | "nvidia" | "anthropic" | "auto".
    #: `auto` builds a fallback chain from every key present, in
    #: preference order - so one provider rate-limiting mid-loop does not
    #: force the agents onto their deterministic path.
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

    #: Google AI Studio. Free tier, and the strongest schema enforcement
    #: of the three: `responseSchema` constrains decoding directly.
    #: flash-lite answers in ~1.5s, which matters against the strategist's
    #: 15s budget - Nemotron was spending 5-13s a call.
    gemini_api_key: str = ""
    gemini_base_url: str = "https://generativelanguage.googleapis.com/v1beta"
    gemini_model: str = "gemini-3.1-flash-lite"

    # -- WhatsApp (Meta Cloud API) -----------------------------------
    #: Delivery only. The decision to send, the template and the channel
    #: are already settled by the strategist and validated by the gate -
    #: nothing here re-decides any of it.
    whatsapp_access_token: str = ""
    whatsapp_phone_number_id: str = ""
    whatsapp_api_base: str = "https://graph.facebook.com/v21.0"
    #: Meta test numbers reach only an allow-list, so the demo recipient
    #: overrides the customer's real contact. The outcome records that it
    #: did, rather than implying we messaged the actual customer.
    demo_whatsapp_to: str = ""
    #: Meta app secret, for X-Hub-Signature-256 on delivery receipts.
    #: Unset means receipts are accepted but marked unverified - losing
    #: them silently is worse than recording that we could not check.
    whatsapp_app_secret: str = ""
    #: Echoed back during Meta's subscription handshake.
    whatsapp_verify_token: str = "nishchay_verify_token"
    #: Meta app id. Needed only to subscribe the delivery-receipt webhook;
    #: sending messages does not use it.
    whatsapp_app_id: str = ""

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

    #: Preference order for `auto`. Gemini first: fastest, and its
    #: responseSchema constrains decoding rather than merely requesting it.
    PROVIDER_ORDER: ClassVar[tuple[str, ...]] = ("gemini", "anthropic", "nvidia")

    def _key_for(self, name: str) -> str:
        return {
            "gemini": self.gemini_api_key,
            "anthropic": self.anthropic_api_key,
            "nvidia": self.nvidia_api_key,
        }.get(name, "")

    @property
    def providers(self) -> tuple[str, ...]:
        """Every usable provider, in the order they will be tried.

        More than one is not redundancy for its own sake: free tiers
        rate-limit mid-loop, and a second provider means a 429 costs a
        retry rather than dropping the agents onto their fallback path in
        the middle of a demo.
        """
        if self.llm_provider != "auto":
            name = self.llm_provider
            return (name,) if self._key_for(name) else ()
        return tuple(n for n in self.PROVIDER_ORDER if self._key_for(n))

    @property
    def provider(self) -> str:
        """The primary provider, or "none"."""
        return self.providers[0] if self.providers else "none"

    def model_for(self, name: str) -> str:
        return {
            "gemini": self.gemini_model,
            "anthropic": self.anthropic_model,
            "nvidia": self.nvidia_model,
        }.get(name, "")

    @property
    def model_name(self) -> str:
        """The model id to report in traces, whichever provider is live."""
        return self.model_for(self.provider)


@lru_cache
def settings() -> Settings:
    return Settings()
