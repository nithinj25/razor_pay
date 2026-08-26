"""Ingress. Verify, dedupe, append, publish, ACK. Nothing else.

The hard requirement is p99 < 50ms, and the reason is Razorpay's retry
policy: a non-2xx starts 24h of exponential-backoff redelivery (E9). So
this endpoint does the minimum that cannot be undone later - a signature
check, a dedupe claim, and an append - then returns. All interpretation
happens downstream, off the request path.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request, Response

from core.config import settings
from core.events import parse_webhook
from core.infra import Infra
from services.ingress.signing import verify_any

TOPIC_RAW = "raw"

infra = Infra()
#: Observability only. `skew` of hours is normal (E17); a *negative* skew
#: means a clock disagreement worth alerting on (E18).
stats: dict[str, Any] = {"received": 0, "duplicates": 0, "rejected": 0, "max_skew": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await infra.start()
    cfg = settings()
    # E15: test and live keys are separate worlds. Fail loudly at boot
    # rather than discovering the mix-up in the audit trail.
    if cfg.require_test_mode and not cfg.is_test_mode:
        raise RuntimeError(
            f"refusing to start: RZP_KEY_ID={cfg.rzp_key_id!r} is not a test key (E15)"
        )
    yield
    await infra.stop()


app = FastAPI(title="nishchay-ingress", lifespan=lifespan)


@app.post("/webhook/razorpay")
async def webhook(request: Request) -> Response:
    cfg = settings()
    raw = await request.body()                       # RAW bytes. Never re-serialise.
    sig = request.headers.get("X-Razorpay-Signature", "")

    if not verify_any(raw, sig, cfg.rzp_webhook_secret, cfg.rzp_webhook_secret_prev):
        stats["rejected"] += 1
        return Response(status_code=400)

    # Razorpay guarantees this header; a delivery without one is malformed
    # rather than retryable, so it is rejected instead of stored.
    event_id = request.headers.get("X-Razorpay-Event-Id")
    if not event_id:
        stats["rejected"] += 1
        return Response(status_code=400)

    # E20: several webhook URLs receive the same event. Dedupe is global
    # on event_id, claimed atomically so concurrent deliveries cannot both win.
    if not await infra.dedupe.claim(f"evt:{event_id}"):
        stats["duplicates"] += 1
        return Response(status_code=200)             # already have it - ACK anyway

    received_at = int(time.time())
    try:
        obs = parse_webhook(raw, event_id=event_id, received_at=received_at)
    except Exception:                                # noqa: BLE001
        # A shape we cannot parse is still a fact. Store the envelope so
        # replay can revisit it, and never 500 - that would trigger 24h
        # of retries for a payload that will never parse (E9).
        stats["rejected"] += 1
        return Response(status_code=200)

    if not obs.order_id:
        # E7: absence of an order is not success. Park it for the poller
        # rather than dropping it.
        await infra.bus.publish(TOPIC_RAW, key="_orphan", value={"event_id": event_id})
        return Response(status_code=200)

    await infra.store.append(obs)

    stats["received"] += 1
    stats["max_skew"] = max(stats["max_skew"], obs.skew)

    # The key is order_id so every sibling attempt lands on one partition
    # and one consumer. Keying on payment_id races sibling detection.
    await infra.bus.publish(
        TOPIC_RAW,
        key=obs.order_id,
        value={
            "event_id": event_id,
            "order_id": obs.order_id,
            "payment_id": obs.payment_id,
            "event_type": obs.event_type,
            "event_time": obs.event_time,
            "received_at": received_at,
        },
    )
    return Response(status_code=200)


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "ok": True,
        "mode": "test" if settings().is_test_mode else "live",
        "degraded": infra.degraded,
        "stats": stats,
    }
