"""Agent tracing: what each node did, and whether a model was involved.

Without this the console can show a verdict but not *how* it was reached,
and "LLM calls: 0" is ambiguous — it could mean the deterministic rules
did their job (the claim we are making), or that the model is misconfigured
and everything is silently falling back (a broken system that looks
identical from outside).

`source` is the field that disambiguates it:

    rules     a deterministic node. No model, by design.
    model     a real API call happened.
    scripted  a scripted response, for demonstrating the loop without a key.
    fallback  the model was asked for and was unavailable; the
              deterministic path ran instead.

A screen full of `fallback` means the agents are not working. A screen
full of `rules` on scenarios A/B/C means they are working exactly as
designed. Those must never look the same.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

Source = Literal["rules", "model", "scripted", "fallback"]


@dataclass(frozen=True)
class AgentStep:
    agent: str                       # resolver | strategist
    node: str                        # precheck | plan | fetch | analyze | ...
    summary: str
    source: Source = "rules"
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    prompt_chars: int = 0
    output: dict[str, Any] = field(default_factory=dict)
    error: str = ""

    @property
    def used_model(self) -> bool:
        return self.source in ("model", "scripted")

    def to_row(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "node": self.node,
            "summary": self.summary,
            "source": self.source,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "tokens": self.tokens_in + self.tokens_out,
            "latency_ms": self.latency_ms,
            "prompt_chars": self.prompt_chars,
            "output": self.output,
            "error": self.error,
        }


def merge_steps(a: tuple, b: tuple) -> tuple:
    """LangGraph reducer — nodes append rather than overwrite."""
    return tuple(a) + tuple(b)


@dataclass
class AgentSummary:
    """Rolled up for the console header."""

    resolver_ran: bool = False
    strategist_ran: bool = False
    model_calls: int = 0
    scripted_calls: int = 0
    fallbacks: int = 0
    rule_nodes: int = 0
    tokens: int = 0
    latency_ms: int = 0

    @classmethod
    def of(cls, steps: tuple[AgentStep, ...]) -> "AgentSummary":
        s = cls()
        for st in steps:
            s.resolver_ran |= st.agent == "resolver"
            s.strategist_ran |= st.agent == "strategist"
            if st.source == "model":
                s.model_calls += 1
            elif st.source == "scripted":
                s.scripted_calls += 1
            elif st.source == "fallback":
                s.fallbacks += 1
            else:
                s.rule_nodes += 1
            s.tokens += st.tokens_in + st.tokens_out
            s.latency_ms += st.latency_ms
        return s

    @property
    def health(self) -> str:
        """One word for the console badge.

        Four states, and conflating any two of them hides something:

          live       a real model answered.
          scripted   the loop ran on canned responses. Proves the graph
                     works; proves nothing about the model. Never report
                     this as `live`.
          degraded   the model was wanted and was not there. Verdicts are
                     still correct, which is exactly why this is easy to
                     miss and why the badge shouts.
          rules-only the deterministic rules settled it and no model was
                     ever needed. This is the design working, not a fault.
        """
        if self.model_calls:
            return "live"
        if self.scripted_calls:
            return "scripted"
        if self.fallbacks:
            return "degraded"
        return "rules-only"

    def to_row(self) -> dict[str, Any]:
        return {
            "resolver_ran": self.resolver_ran,
            "strategist_ran": self.strategist_ran,
            "model_calls": self.model_calls,
            "scripted_calls": self.scripted_calls,
            "fallbacks": self.fallbacks,
            "rule_nodes": self.rule_nodes,
            "tokens": self.tokens,
            "latency_ms": self.latency_ms,
            "health": self.health,
        }
