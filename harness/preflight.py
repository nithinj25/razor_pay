"""Pre-flight: is everything actually working, right now?

Run this before recording. Every check hits the real thing rather than
asserting from config - a key that is present but expired reads as
configured everywhere except at the moment you need it, which is exactly
when you find out on camera.

    python -m harness.preflight

Exits non-zero if anything a demo depends on is down. Checks marked
OPTIONAL can fail without stopping the recording, and each one says what
you lose if it does.
"""

from __future__ import annotations

import asyncio
import socket
import sys
import time
from dataclasses import dataclass, field

import httpx


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True
    fix: str = ""


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)

    def add(self, *a, **kw) -> None:
        self.checks.append(Check(*a, **kw))

    @property
    def blocking(self) -> list[Check]:
        return [c for c in self.checks if c.required and not c.ok]

    @property
    def degraded(self) -> list[Check]:
        return [c for c in self.checks if not c.required and not c.ok]


def _port(host: str, port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


async def infrastructure(r: Report) -> None:
    for name, port, required in (
        ("postgres", 5432, True), ("redis", 6379, True),
        ("redpanda", 19092, False), ("clickhouse", 8123, False),
    ):
        up = _port("localhost", port)
        r.add(name, up, f"localhost:{port}" if up else "not listening",
              required=required,
              fix="docker compose up -d" if not up else "")

    for name, url, required in (
        ("ingress", "http://localhost:8000/health", True),
        ("console", "http://localhost:8080/api/scenarios", True),
    ):
        try:
            async with httpx.AsyncClient(timeout=8) as c:
                resp = await c.get(url)
            r.add(name, resp.status_code == 200, f"HTTP {resp.status_code}",
                  required=required,
                  fix="docker compose up -d --build" if resp.status_code != 200 else "")
        except Exception as e:                       # noqa: BLE001
            r.add(name, False, f"{type(e).__name__}", required=required,
                  fix="docker compose up -d --build")


async def razorpay(r: Report) -> None:
    """The one that carries the demo. A real API call, not a key check."""
    from core.config import settings

    cfg = settings()
    if not cfg.rzp_key_secret:
        r.add("razorpay api", False, "RZP_KEY_SECRET empty",
              fix="Dashboard -> API Keys -> Generate Test Key, into .env")
        return
    if not cfg.is_test_mode:
        r.add("razorpay api", False, f"{cfg.rzp_key_id} is not a test key",
              fix="use a rzp_test_ key - live keys are refused at boot (E15)")
        return

    try:
        async with httpx.AsyncClient(
            base_url=cfg.rzp_api_base,
            auth=(cfg.rzp_key_id, cfg.rzp_key_secret), timeout=15,
        ) as c:
            resp = await c.get("/v1/payments?count=1")
        r.add("razorpay api", resp.status_code == 200,
              f"{cfg.rzp_key_id} -> HTTP {resp.status_code}",
              fix="check the key/secret pair" if resp.status_code != 200 else "")
    except Exception as e:                           # noqa: BLE001
        r.add("razorpay api", False, f"{type(e).__name__}: {e}",
              fix="check network and the key/secret pair")

    # Creating an order is the thing the demo actually does, so prove it.
    try:
        from harness.live import create_order

        order = await create_order(100, receipt=f"preflight_{int(time.time())}")
        r.add("razorpay order", True, f"created {order['id']} (Rs 1.00 probe)")
    except SystemExit as e:
        r.add("razorpay order", False, str(e)[:110],
              fix="the key can read but not write - regenerate it")
    except Exception as e:                           # noqa: BLE001
        r.add("razorpay order", False, f"{type(e).__name__}: {e}")


async def model(r: Report) -> None:
    from core.config import settings
    from core.intents import Assessment
    from core.llm import LLMUnavailable, build_llm

    cfg = settings()
    if cfg.provider == "none":
        r.add("llm provider", False, "no key configured", required=False,
              fix="set GEMINI_API_KEY (free) - without it the agents run "
                  "deterministic fallbacks and the demo shows amber steps")
        return

    llm = build_llm(cfg)
    try:
        t0 = time.monotonic()
        a = await llm.structured(
            Assessment, "You choose the next probe for a payment-recovery agent.",
            "A netbanking payment failed at payment_initiation, bank-attributed. "
            "An outage is already confirmed. What next?",
            node="preflight",
        )
        dt = time.monotonic() - t0
        r.add("llm provider", True,
              f"{cfg.model_name} via {cfg.provider} - {dt:.1f}s, "
              f"returned {a.next_probe!r}")
    except LLMUnavailable as e:
        r.add("llm provider", False, str(e)[:110], required=False,
              fix="regenerate the key, or the chain falls back to the next provider")


async def whatsapp(r: Report) -> None:
    from core.config import settings
    from services.executor.whatsapp import WhatsAppSender

    cfg = settings()
    w = WhatsAppSender(cfg)
    if not w.configured:
        r.add("whatsapp", False, "not configured", required=False,
              fix="Meta dashboard -> Step 1 -> copy Phone number ID and token")
        return
    if not cfg.demo_whatsapp_to:
        r.add("whatsapp", False, "DEMO_WHATSAPP_TO not set", required=False,
              fix="your number, country code + digits, no + or spaces")
        return

    # A template send works outside the 24h window, so this isolates
    # "token/number is broken" from "the session window has closed".
    res = await w.send_template(cfg.demo_whatsapp_to, "hello_world")
    r.add("whatsapp", res.ok,
          f"{cfg.demo_whatsapp_to} <- {res.detail[:90]}", required=False,
          fix="" if res.ok else "regenerate the token on the Meta page")

    if res.ok:
        # And a freeform one, which is what a recovery message actually is.
        free = await w.send_text(
            cfg.demo_whatsapp_to,
            "Nishchay pre-flight: if you can read this, the 24h window is open.",
        )
        r.add("whatsapp freeform", free.ok, free.detail[:100], required=False,
              fix="" if free.ok else
                  "reply to the WhatsApp thread from your phone to reopen the window")


async def agents(r: Report) -> None:
    """Resolve one scenario end to end, exactly as the demo will."""
    from core.llm import build_llm
    from harness import scenarios as sc
    from harness.scripted_agents import pipeline_for

    for key in ("A", "G"):
        s = sc.BY_KEY[key]
        try:
            p = pipeline_for(s, build_llm())
            t0 = time.monotonic()
            d = await p.process(
                s.observations(), s.evaluate_at, order_id=s.order_id,
                extra={"customer_messages": list(s.customer_messages)}
                if s.customer_messages else None,
            )
            ok = d.verdict.verdict == s.ground_truth
            r.add(f"agents scenario {key}", ok,
                  f"{d.verdict.verdict.value} @ {d.verdict.confidence:.2f} "
                  f"({d.agents.model_calls} model calls, {time.monotonic()-t0:.1f}s)",
                  fix="" if ok else f"expected {s.ground_truth.value}")
        except Exception as e:                       # noqa: BLE001
            r.add(f"agents scenario {key}", False, f"{type(e).__name__}: {e}")


async def suites(r: Report) -> None:
    """The invariants. Cheap enough to re-check before every recording."""
    import subprocess

    for name, cmd, needle in (
        ("chaos matrix", [sys.executable, "-m", "harness.chaos", "--all"],
         "9/9 chaos faults handled"),
        ("accuracy + invariants", [sys.executable, "-m", "harness.demo", "--all"],
         "duplicate orders=0, false positives=0"),
    ):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            body = out.stdout + out.stderr
            ok = needle in body
            line = next((ln.strip() for ln in body.splitlines() if needle in ln), "")
            r.add(name, ok, line or "expected output not found",
                  fix="" if ok else f"run `{' '.join(cmd[1:])}` and read the failure")
        except Exception as e:                       # noqa: BLE001
            r.add(name, False, f"{type(e).__name__}: {e}")


async def main_async() -> int:
    r = Report()
    print("\nnishchay pre-flight\n")

    for label, fn in (
        ("infrastructure", infrastructure), ("razorpay", razorpay),
        ("model", model), ("whatsapp", whatsapp), ("agents", agents),
        ("suites", suites),
    ):
        print(f"  checking {label} ...")
        await fn(r)

    print()
    width = max(len(c.name) for c in r.checks) + 2
    for c in r.checks:
        mark = "PASS" if c.ok else ("FAIL" if c.required else "WARN")
        print(f"  [{mark}] {c.name:<{width}} {c.detail}")
        if c.fix and not c.ok:
            print(f"         -> {c.fix}")

    print()
    if r.blocking:
        print(f"NOT READY - {len(r.blocking)} blocking: "
              f"{', '.join(c.name for c in r.blocking)}")
        return 1
    if r.degraded:
        print(f"READY, with {len(r.degraded)} degraded: "
              f"{', '.join(c.name for c in r.degraded)}")
        print("The demo will run. Read the -> lines for what you lose.")
        return 0
    print("READY - everything green.")
    return 0


def main() -> None:
    sys.exit(asyncio.run(main_async()))


if __name__ == "__main__":
    main()
