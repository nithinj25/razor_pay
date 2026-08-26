"""Observations: the append-only substrate. I2 - immutable once written.

Webhooks are at-least-once *and* unordered, so nothing here interprets
anything. We record what arrived, when Razorpay said it happened, and
when we saw it. Meaning is computed later, by `fold`, from the whole set.

The single most important column is `event_time` (Razorpay's created_at),
because that is the only ordering that is causally true. `received_at` is
kept solely so the console can *show* the inversion (E18) - never to sort
by (pitfall #5).
"""

from __future__ import annotations

import json
from typing import Any, Iterable, Literal, Protocol

from pydantic import BaseModel, Field

from core.verdicts import Paise

Source = Literal["webhook", "api_poll", "settlement_report", "fixture"]

#: Webhook events we subscribe to. Anything else is stored but ignored by
#: the fold - storing it costs nothing and replay may need it later (E10).
KNOWN_EVENTS: frozenset[str] = frozenset(
    {
        "payment.authorized", "payment.failed", "payment.captured",
        "order.paid",
        "refund.created", "refund.processed", "refund.failed",
        "settlement.processed",
        "payment.downtime.started", "payment.downtime.updated",
        "payment.downtime.resolved",
    }
)


class Observation(BaseModel):
    """One thing we were told. Immutable (I2)."""

    model_config = {"frozen": True}

    event_id: str                    # X-Razorpay-Event-Id - global dedupe key (E20)
    event_type: str
    order_id: str
    payment_id: str | None = None
    event_time: int                  # Razorpay's created_at - THE ordering key
    received_at: int                 # ours, for observability only
    source: Source = "webhook"
    payload: dict[str, Any] = Field(default_factory=dict)

    # -- Denormalised for the fold. Extracted once, never recomputed. --
    status: str | None = None
    amount: Paise = 0
    amount_paid: Paise = 0
    amount_due: Paise = 0
    method: str | None = None
    error_source: str | None = None
    error_step: str | None = None
    error_reason: str | None = None
    #: The only field tying a payment to the customer's bank statement.
    #: When a customer says "money was debited", this is the evidence.
    rrn: str | None = None
    upi_transaction_id: str | None = None

    @property
    def skew(self) -> int:
        """received_at - event_time. Hours of lag are normal (E17)."""
        return self.received_at - self.event_time


def _entity(body: dict[str, Any], name: str) -> dict[str, Any]:
    return (body.get("payload", {}).get(name, {}) or {}).get("entity", {}) or {}


def parse_webhook(
    raw: bytes | str, event_id: str, received_at: int, source: Source = "webhook"
) -> Observation:
    """Razorpay webhook body -> Observation.

    Tolerant by design: an unknown event shape must not 500 the ingress,
    because a non-2xx puts Razorpay into 24h of retries (E9).
    """
    body = json.loads(raw)
    event_type = body.get("event", "unknown")
    created_at = int(body.get("created_at", received_at))

    payment = _entity(body, "payment")
    order = _entity(body, "order")
    refund = _entity(body, "refund")

    # A refund event names its payment but not always its order; the
    # order_id is recovered by the caller from the sibling payment row.
    payment_id = payment.get("id") or refund.get("payment_id")
    order_id = payment.get("order_id") or order.get("id") or refund.get("order_id") or ""

    acq = payment.get("acquirer_data", {}) or {}

    return Observation(
        event_id=event_id,
        event_type=event_type,
        order_id=order_id or "",
        payment_id=payment_id,
        event_time=int(payment.get("created_at") or order.get("created_at") or created_at),
        received_at=received_at,
        source=source,
        payload=body,
        status=payment.get("status") or order.get("status") or refund.get("status"),
        amount=payment.get("amount") or order.get("amount") or refund.get("amount") or 0,
        amount_paid=order.get("amount_paid") or 0,
        amount_due=order.get("amount_due") or 0,
        method=payment.get("method"),
        error_source=payment.get("error_source"),
        error_step=payment.get("error_step"),
        error_reason=payment.get("error_reason"),
        rrn=acq.get("rrn"),
        upi_transaction_id=acq.get("upi_transaction_id"),
    )


def order_id_of(raw: bytes | str) -> str:
    """Kafka partition key. MUST be order_id, never payment_id (pitfall #6).

    Siblings that land on different partitions cannot be compared, and
    sibling comparison is the entire point of the system (I3).
    """
    body = json.loads(raw)
    return (
        _entity(body, "payment").get("order_id")
        or _entity(body, "order").get("id")
        or _entity(body, "refund").get("order_id")
        or ""
    )


# -------------------------- storage --------------------------

DDL = """
CREATE TABLE IF NOT EXISTS observations (
    id            BIGSERIAL PRIMARY KEY,
    event_id      TEXT        NOT NULL,
    event_type    TEXT        NOT NULL,
    order_id      TEXT        NOT NULL,
    payment_id    TEXT,
    event_time    BIGINT      NOT NULL,
    received_at   BIGINT      NOT NULL,
    source        TEXT        NOT NULL DEFAULT 'webhook',
    status        TEXT,
    amount        BIGINT      NOT NULL DEFAULT 0,
    amount_paid   BIGINT      NOT NULL DEFAULT 0,
    amount_due    BIGINT      NOT NULL DEFAULT 0,
    method        TEXT,
    error_source  TEXT,
    error_step    TEXT,
    error_reason  TEXT,
    rrn           TEXT,
    upi_transaction_id TEXT,
    payload       JSONB       NOT NULL,
    CONSTRAINT observations_unique UNIQUE (order_id, event_id)
);

CREATE INDEX IF NOT EXISTS observations_order_time
    ON observations (order_id, event_time, event_id);
CREATE INDEX IF NOT EXISTS observations_payment
    ON observations (payment_id);
"""

#: I2 enforced in the database, not by convention - a stray UPDATE would
#: silently invalidate every replay and every property test.
IMMUTABILITY_DDL = """
CREATE OR REPLACE FUNCTION observations_immutable() RETURNS trigger AS $fn$
BEGIN
    RAISE EXCEPTION 'observations are append-only (I2)';
END;
$fn$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS observations_no_mutate ON observations;
CREATE TRIGGER observations_no_mutate
    BEFORE UPDATE OR DELETE ON observations
    FOR EACH ROW EXECUTE FUNCTION observations_immutable();
"""


class ObservationStore(Protocol):
    async def append(self, obs: Observation) -> bool: ...
    async def load(self, order_id: str) -> list[Observation]: ...


class InMemoryStore:
    """Test double. Same dedupe semantics as Postgres."""

    def __init__(self) -> None:
        self._rows: list[Observation] = []
        self._seen: set[tuple[str, str]] = set()

    async def append(self, obs: Observation) -> bool:
        key = (obs.order_id, obs.event_id)
        if key in self._seen:
            return False                      # duplicate, absorbed
        self._seen.add(key)
        self._rows.append(obs)
        return True

    async def load(self, order_id: str) -> list[Observation]:
        return [o for o in self._rows if o.order_id == order_id]

    async def all_orders(self) -> list[str]:
        return sorted({o.order_id for o in self._rows})


class PostgresStore:
    """Append-only store over asyncpg."""

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def init(self) -> None:
        async with self.pool.acquire() as c:
            await c.execute(DDL)
            await c.execute(IMMUTABILITY_DDL)

    async def append(self, obs: Observation) -> bool:
        async with self.pool.acquire() as c:
            row = await c.fetchrow(
                """
                INSERT INTO observations (
                    event_id, event_type, order_id, payment_id, event_time,
                    received_at, source, status, amount, amount_paid,
                    amount_due, method, error_source, error_step,
                    error_reason, rrn, upi_transaction_id, payload
                ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,
                          $15,$16,$17,$18)
                ON CONFLICT (order_id, event_id) DO NOTHING
                RETURNING id
                """,
                obs.event_id, obs.event_type, obs.order_id, obs.payment_id,
                obs.event_time, obs.received_at, obs.source, obs.status,
                obs.amount, obs.amount_paid, obs.amount_due, obs.method,
                obs.error_source, obs.error_step, obs.error_reason, obs.rrn,
                obs.upi_transaction_id, json.dumps(obs.payload),
            )
            return row is not None

    async def load(self, order_id: str) -> list[Observation]:
        async with self.pool.acquire() as c:
            rows = await c.fetch(
                "SELECT * FROM observations WHERE order_id = $1", order_id
            )
        return [_row_to_obs(r) for r in rows]


def _row_to_obs(r: Any) -> Observation:
    d = dict(r)
    d.pop("id", None)
    payload = d.get("payload")
    d["payload"] = json.loads(payload) if isinstance(payload, str) else (payload or {})
    return Observation(**d)


def canonical_order(obs: Iterable[Observation]) -> list[Observation]:
    """The one true ordering: event_time, tie-broken by event_id.

    Deterministic regardless of arrival sequence - this is what makes the
    fold order-independent and therefore property-testable.
    """
    return sorted(obs, key=lambda o: (o.event_time, o.event_id))
