"""The five fetchers. Two are stubs, and say so.

`attempts` is the highest-signal fetcher in the system: it asks Razorpay
for every payment on the order, which is exactly the sibling set I3 turns
on. If only one fetcher survives a degraded run, it should be this one.
"""

from __future__ import annotations

from typing import Any

from core.verdicts import Evidence
from services.resolver.fetchers import FetchContext

#: Loaded from a local table rather than a live call. NPCI publishes
#: per-bank technical-decline rates but not at an interval that would
#: make a live fetch meaningful for a demo. Flagged as a stub rather
#: than dressed up as real data.
BANK_TECHNICAL_DECLINE: dict[str, float] = {
    "HDFC": 0.012, "SBIN": 0.041, "ICIC": 0.009,
    "UTIB": 0.018, "PUNB": 0.055, "BARB": 0.038,
}


async def probe_payment(ctx: FetchContext) -> Evidence:
    """GET /v1/payments/:id - the authoritative status right now.

    Webhook delivery can lag by hours (E17), so the API is what settles
    disagreements about freshness.
    """
    if ctx.client is None or not ctx.payment_id:
        return Evidence.unavailable("payment", "no client or payment_id")
    r = await ctx.client.get(f"/v1/payments/{ctx.payment_id}")
    r.raise_for_status()
    p = r.json()
    acq = p.get("acquirer_data") or {}
    return Evidence(
        source="payment",
        value={
            "status": p.get("status"),
            "captured": bool(p.get("captured")),
            "amount": p.get("amount"),
            "amount_refunded": p.get("amount_refunded", 0),
            "method": p.get("method"),
            "error_source": p.get("error_source"),
            "error_step": p.get("error_step"),
            "error_reason": p.get("error_reason"),
            "rrn": acq.get("rrn"),
        },
        confidence=0.95,
        provenance=f"GET /v1/payments/{ctx.payment_id}",
    )


async def probe_attempts(ctx: FetchContext) -> Evidence:
    """GET /v1/orders/:id/payments - every sibling attempt.

    The highest-signal call we make. A captured sibling here is the fact
    that stops a duplicate order, and it is the one thing a naive agent
    never asks for.
    """
    if ctx.client is None:
        return Evidence.unavailable("attempts", "no client")
    r = await ctx.client.get(f"/v1/orders/{ctx.order_id}/payments")
    r.raise_for_status()
    items = r.json().get("items", [])
    statuses = [i.get("status") for i in items]
    return Evidence(
        source="attempts",
        value={
            "count": len(items),
            "statuses": statuses,
            "any_captured": "captured" in statuses,
            "any_authorized": "authorized" in statuses,
            "captured_total": sum(
                i.get("amount", 0) for i in items if i.get("status") == "captured"
            ),
            "payment_ids": [i.get("id") for i in items],
        },
        confidence=0.97,
        provenance=f"GET /v1/orders/{ctx.order_id}/payments",
    )


async def probe_downtime(ctx: FetchContext) -> Evidence:
    """GET /v1/payments/downtimes - ~40KB in, ~180 bytes out.

    NOTE: this endpoint is not enabled by default; it needs a Razorpay
    support request. Until it is, this fetcher returns unavailable, the
    verdict's confidence drops, and scenario E degrades to PENDING_TAT
    rather than acting. That is the correct failure, but it does mean the
    demo runs E from fixture evidence.
    """
    if ctx.client is None:
        return Evidence.unavailable("downtime", "no client")
    r = await ctx.client.get("/v1/payments/downtimes")
    r.raise_for_status()
    hits = [
        d
        for d in r.json().get("items", [])
        if d.get("method") == ctx.method and d.get("status") in ("started", "updated")
    ]
    if not hits:
        return Evidence(
            source="downtime",
            value={"active": False, "method": ctx.method},
            confidence=0.8,
            provenance="GET /v1/payments/downtimes",
        )
    top = hits[0]
    return Evidence(
        source="downtime",
        value={
            "active": True,
            "method": top.get("method"),
            "bank": (top.get("instrument") or {}).get("bank"),
            "severity": top.get("severity"),
            # Method-scoped vs issuer-wide decides whether an alternative
            # rail exists at all - it is the branch that makes the
            # strategist's turn 3 conditional.
            "scope": "method" if (top.get("instrument") or {}).get("bank") else "network",
        },
        confidence=0.9,
        provenance="GET /v1/payments/downtimes",
    )


async def probe_history(ctx: FetchContext) -> Evidence:
    """Which rails has this customer actually succeeded on before?

    Feeds the strategist's turn 3. STUBBED against a local table: the
    live version is a customer-scoped payments query, which needs a
    customer_id we do not reliably have in test mode.
    """
    hist = ctx.extra.get("history")
    if hist is None:
        return Evidence.unavailable("history", "stub: no customer history source")
    return Evidence(
        source="history",
        value=hist,
        confidence=0.8,
        provenance="stub: local customer history table",
    )


async def probe_settlement(ctx: FetchContext) -> Evidence:
    """Did this payment appear in a settlement report?

    Ground truth when present - money in the merchant's bank account.
    STUBBED: the recon report endpoint path is unverified.
    """
    ids = ctx.extra.get("settled_payment_ids")
    if ids is None:
        return Evidence.unavailable("settlement", "stub: recon endpoint unverified")
    return Evidence(
        source="settlement",
        value=ctx.payment_id if ctx.payment_id in ids else None,
        confidence=0.99 if ctx.payment_id in ids else 0.6,
        provenance="stub: settlement recon report",
    )


async def probe_bank_prior(ctx: FetchContext) -> Evidence:
    """Prior probability that this bank technically declines.

    STUBBED against a local NPCI-derived table. Real per-bank BD/TD data
    would make this a genuine signal; flagged rather than overclaimed.
    """
    bank = ctx.extra.get("bank")
    if not bank:
        return Evidence.unavailable("bank_prior", "stub: no bank identified")
    rate = BANK_TECHNICAL_DECLINE.get(bank)
    if rate is None:
        return Evidence.unavailable("bank_prior", f"stub: no prior for {bank}")
    return Evidence(
        source="bank_prior",
        value={"bank": bank, "technical_decline_rate": rate, "elevated": rate > 0.03},
        confidence=0.5,
        provenance="stub: local NPCI BD/TD table",
    )


FETCHERS: dict[str, Any] = {
    "payment": probe_payment,
    "attempts": probe_attempts,
    "downtime": probe_downtime,
    "history": probe_history,
    "settlement": probe_settlement,
    "bank_prior": probe_bank_prior,
}

#: What the planner may choose from. Kept separate from FETCHERS so the
#: planner's action space is explicit and auditable.
PLANNABLE = tuple(FETCHERS)


#: Fetchers that cannot run without an HTTP client. When no client is
#: configured they are *skipped*, not reported as failed evidence: an
#: unconfigured probe is a known unknown, and taxing the verdict's
#: confidence for it would make every offline run look like an outage.
#: A configured probe that times out or 500s still degrades (chaos 3).
NEEDS_CLIENT = frozenset({"payment", "attempts", "downtime"})
