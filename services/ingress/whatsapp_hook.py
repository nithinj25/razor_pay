"""Meta delivery receipts. What `EXECUTED` actually means.

The executor reported EXECUTED for a message that Meta accepted and then
silently never delivered - a message id came back, and nothing arrived.
That is not a hypothetical: it happened during live testing, and the only
reason we caught it was a human checking their phone.

Meta reports the truth asynchronously on this webhook: `sent`,
`delivered`, `read`, or `failed` with a reason. Recording it turns
"we called the API" into "the customer received it", which is the
difference between a claim and an audit trail.

Signature verification uses `X-Hub-Signature-256` over the raw body -
same discipline as the Razorpay ingress, and the same pitfall: hash the
bytes you received, never a re-serialised dict.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
from typing import Any

from fastapi import APIRouter, Request, Response

router = APIRouter()

#: Terminal delivery states. Anything else is still in flight.
TERMINAL = frozenset({"delivered", "read", "failed"})

#: In-memory receipts, keyed by Meta's message id. The console reads this;
#: `PostgresReceipts` persists it when a pool is available.
RECEIPTS: dict[str, dict[str, Any]] = {}

#: Populated at startup by the ingress if a database is reachable.
STORE: Any = None


def verify_signature(raw: bytes, header: str | None, app_secret: str) -> bool:
    """X-Hub-Signature-256: 'sha256=' + HMAC-SHA256(app_secret, raw).

    An unset app secret means we cannot verify. Returning False would
    drop every receipt; returning True would accept forgeries. We treat
    it as unverified-but-accepted and mark the receipt, because a
    delivery receipt is not a money action - and losing them silently is
    worse than recording that we could not check.
    """
    if not app_secret:
        return True                       # unverifiable; caller marks it
    if not header or not header.startswith("sha256="):
        return False
    expected = hmac.new(app_secret.encode(), raw, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header.split("=", 1)[1])


@router.get("/webhook/whatsapp")
async def verify(request: Request) -> Response:
    """Meta's subscription handshake: echo hub.challenge if the token matches."""
    from core.config import settings

    q = request.query_params
    token = settings().whatsapp_verify_token
    if q.get("hub.mode") == "subscribe" and q.get("hub.verify_token") == token:
        return Response(content=q.get("hub.challenge", ""), media_type="text/plain")
    return Response(status_code=403)


@router.post("/webhook/whatsapp")
async def receive(request: Request) -> Response:
    """Record delivery statuses. Always 200 - Meta retries otherwise."""
    from core.config import settings

    cfg = settings()
    raw = await request.body()
    sig = request.headers.get("X-Hub-Signature-256")
    verified = verify_signature(raw, sig, cfg.whatsapp_app_secret)
    if not verified:
        return Response(status_code=403)

    try:
        body = json.loads(raw)
    except Exception:                                 # noqa: BLE001
        return Response(status_code=200)              # malformed; do not retry

    for receipt in extract_statuses(body):
        receipt["verified"] = bool(cfg.whatsapp_app_secret)
        RECEIPTS[receipt["message_id"]] = receipt
        if STORE is not None:
            try:
                await STORE.write_receipt(receipt)
            except Exception:                         # noqa: BLE001
                pass                                  # never fail the ACK
    return Response(status_code=200)


def extract_statuses(body: dict) -> list[dict[str, Any]]:
    """Flatten Meta's nested envelope into receipt rows.

    Shape: entry[] -> changes[] -> value.statuses[]. Inbound customer
    messages arrive on the same webhook under `value.messages` and are
    ignored here - they matter only for the 24h window, which the
    template path removes the need to track.
    """
    out: list[dict[str, Any]] = []
    for entry in body.get("entry", []) or []:
        for change in entry.get("changes", []) or []:
            value = change.get("value") or {}
            for st in value.get("statuses", []) or []:
                errors = st.get("errors") or []
                err = errors[0] if errors else {}
                out.append({
                    "ts": int(st.get("timestamp") or time.time()),
                    "message_id": st.get("id", ""),
                    "recipient": st.get("recipient_id", ""),
                    "status": st.get("status", ""),
                    "error_code": err.get("code", 0),
                    "error_title": err.get("title", "") or err.get("message", ""),
                    "conversation": (st.get("conversation") or {}).get("id", ""),
                })
    return out


def status_of(message_id: str) -> dict[str, Any] | None:
    return RECEIPTS.get(message_id)


def summary() -> dict[str, Any]:
    """Counts by status, for the console."""
    counts: dict[str, int] = {}
    for r in RECEIPTS.values():
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    failed = [r for r in RECEIPTS.values() if r["status"] == "failed"]
    return {
        "total": len(RECEIPTS),
        "by_status": counts,
        "failures": failed[-10:],
        "receipts": sorted(RECEIPTS.values(), key=lambda r: r["ts"], reverse=True)[:25],
    }
