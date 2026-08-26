"""Outcome and veto persistence — what `analytics/matrix.sql` reads.

Every decision the pipeline makes lands here, allowed or vetoed. The veto
table is not a debug log: `SELECT * FROM vetoes` is the audit trail the
track brief asks for, and it is the thing that makes "bounded and gated"
checkable by someone who does not trust us.

ClickHouse is the intended home. The slip protocol says drop it for
Postgres if day 2 slips, so both are implemented behind one interface and
`NullOutcomeStore` keeps the demo running when neither is up. Which one
you got is reported, never guessed at.
"""

from __future__ import annotations

from typing import Any, Protocol

CLICKHOUSE_DDL = [
    """
    CREATE TABLE IF NOT EXISTS outcomes (
        ts            DateTime,
        trace_id      String,
        order_id      String,
        payment_id    String,
        scenario      String,
        ground_truth  String,
        verdict       String,
        confidence    Float32,
        action        String,
        status        String,
        gate_allowed  UInt8,
        rules_fired   String,
        llm_calls     UInt16,
        tokens_in     UInt32,
        tokens_out    UInt32,
        latency_ms    UInt32,
        amount_due    Int64,
        amount_paid   Int64
    ) ENGINE = MergeTree ORDER BY (ts, order_id)
    """,
    """
    CREATE TABLE IF NOT EXISTS vetoes (
        ts          DateTime,
        trace_id    String,
        order_id    String,
        action      String,
        rule        String,
        reason      String,
        confidence  Float32,
        evidence    String
    ) ENGINE = MergeTree ORDER BY (ts, order_id)
    """,
]

#: Same shape, ANSI types. Used when the slip protocol drops ClickHouse.
POSTGRES_DDL = [
    """
    CREATE TABLE IF NOT EXISTS outcomes (
        id            BIGSERIAL PRIMARY KEY,
        ts            TIMESTAMPTZ NOT NULL,
        trace_id      TEXT NOT NULL DEFAULT '',
        order_id      TEXT NOT NULL,
        payment_id    TEXT NOT NULL DEFAULT '',
        scenario      TEXT NOT NULL DEFAULT '',
        ground_truth  TEXT NOT NULL DEFAULT '',
        verdict       TEXT NOT NULL,
        confidence    REAL NOT NULL DEFAULT 0,
        action        TEXT NOT NULL,
        status        TEXT NOT NULL DEFAULT '',
        gate_allowed  BOOLEAN NOT NULL DEFAULT FALSE,
        rules_fired   TEXT NOT NULL DEFAULT '',
        llm_calls     INT NOT NULL DEFAULT 0,
        tokens_in     INT NOT NULL DEFAULT 0,
        tokens_out    INT NOT NULL DEFAULT 0,
        latency_ms    INT NOT NULL DEFAULT 0,
        amount_due    BIGINT NOT NULL DEFAULT 0,
        amount_paid   BIGINT NOT NULL DEFAULT 0
    )
    """,
    "CREATE INDEX IF NOT EXISTS outcomes_order ON outcomes (order_id, ts)",
    """
    CREATE TABLE IF NOT EXISTS vetoes (
        id          BIGSERIAL PRIMARY KEY,
        ts          TIMESTAMPTZ NOT NULL,
        trace_id    TEXT NOT NULL DEFAULT '',
        order_id    TEXT NOT NULL,
        action      TEXT NOT NULL,
        rule        TEXT NOT NULL,
        reason      TEXT NOT NULL,
        confidence  REAL NOT NULL DEFAULT 0,
        evidence    TEXT NOT NULL DEFAULT ''
    )
    """,
    "CREATE INDEX IF NOT EXISTS vetoes_rule ON vetoes (rule, ts)",
]

COLUMNS = (
    "ts trace_id order_id payment_id scenario ground_truth verdict confidence "
    "action status gate_allowed rules_fired llm_calls tokens_in tokens_out "
    "latency_ms amount_due amount_paid"
).split()

VETO_COLUMNS = "ts trace_id order_id action rule reason confidence evidence".split()


def outcome_row(
    decision: Any, scenario: str = "", ground_truth: str = "", trace_id: str = ""
) -> dict[str, Any]:
    """Flatten a `Decision` into the analytics schema."""
    v = decision.verdict
    return {
        "ts": decision.now,
        "trace_id": trace_id or f"{decision.order_id}:{decision.now}",
        "order_id": decision.order_id,
        "payment_id": decision.outcome.payment_id if decision.outcome else "",
        "scenario": scenario,
        "ground_truth": ground_truth,
        "verdict": v.verdict.value,
        "confidence": round(v.confidence, 4),
        "action": decision.action.value,
        "status": decision.outcome.status if decision.outcome else "NONE",
        "gate_allowed": 1 if (decision.gate and decision.gate.allowed) else 0,
        "rules_fired": ",".join(v.rules_fired),
        "llm_calls": decision.llm_calls,
        "tokens_in": 0,
        "tokens_out": 0,
        "latency_ms": int(decision.latency_s * 1000),
        "amount_due": v.amount_due,
        "amount_paid": v.amount_paid,
    }


class OutcomeStore(Protocol):
    async def init(self) -> None: ...
    async def write_outcome(self, row: dict[str, Any]) -> None: ...
    async def write_vetoes(self, rows: list[dict[str, Any]]) -> None: ...


class NullOutcomeStore:
    """Keeps rows in memory. What the demo runs on with no database up."""

    backend = "memory"

    def __init__(self) -> None:
        self.outcomes: list[dict] = []
        self.vetoes: list[dict] = []

    async def init(self) -> None:
        return None

    async def write_outcome(self, row: dict[str, Any]) -> None:
        self.outcomes.append(row)

    async def write_vetoes(self, rows: list[dict[str, Any]]) -> None:
        self.vetoes.extend(rows)


class ClickHouseOutcomeStore:
    backend = "clickhouse"

    def __init__(self, client: Any) -> None:
        self.client = client

    async def init(self) -> None:
        for ddl in CLICKHOUSE_DDL:
            self.client.command(ddl)

    async def write_outcome(self, row: dict[str, Any]) -> None:
        self.client.insert("outcomes", [[row[c] for c in COLUMNS]], column_names=COLUMNS)

    async def write_vetoes(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self.client.insert(
            "vetoes",
            [[r.get(c, "") for c in VETO_COLUMNS] for r in rows],
            column_names=VETO_COLUMNS,
        )


class PostgresOutcomeStore:
    """The documented ClickHouse fallback (slip protocol, 1 day behind)."""

    backend = "postgres"

    def __init__(self, pool: Any) -> None:
        self.pool = pool

    async def init(self) -> None:
        async with self.pool.acquire() as c:
            for ddl in POSTGRES_DDL:
                await c.execute(ddl)

    async def write_outcome(self, row: dict[str, Any]) -> None:
        from datetime import datetime, timezone

        vals = [
            datetime.fromtimestamp(row["ts"], tz=timezone.utc)
            if c == "ts" else (bool(row[c]) if c == "gate_allowed" else row[c])
            for c in COLUMNS
        ]
        ph = ",".join(f"${i+1}" for i in range(len(COLUMNS)))
        async with self.pool.acquire() as conn:
            await conn.execute(
                f"INSERT INTO outcomes ({','.join(COLUMNS)}) VALUES ({ph})", *vals
            )

    async def write_vetoes(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        from datetime import datetime, timezone

        ph = ",".join(f"${i+1}" for i in range(len(VETO_COLUMNS)))
        async with self.pool.acquire() as conn:
            for r in rows:
                vals = [
                    datetime.fromtimestamp(r.get("ts", 0), tz=timezone.utc)
                    if c == "ts" else r.get(c, "")
                    for c in VETO_COLUMNS
                ]
                await conn.execute(
                    f"INSERT INTO vetoes ({','.join(VETO_COLUMNS)}) VALUES ({ph})", *vals
                )


def _port_open(host: str, port: int, timeout: float = 0.4) -> bool:
    """Cheap liveness probe before building a client.

    The ClickHouse driver retries through urllib3 on a refused
    connection, which turns "not running" into seconds of wall time on
    every startup. A refused TCP connect answers the same question in
    microseconds.
    """
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


async def build_outcome_store(cfg: Any, pool: Any = None) -> OutcomeStore:
    """ClickHouse, else Postgres, else memory. Never raises."""
    try:
        import clickhouse_connect

        url = cfg.clickhouse_url.replace("http://", "").split(":")
        host = url[0]
        port = int(url[1]) if len(url) > 1 else 8123
        if not _port_open(host, port):
            raise ConnectionError(f"clickhouse not listening on {host}:{port}")
        client = clickhouse_connect.get_client(
            host=host, port=port,
            username="nishchay", password="nishchay", database="default",
            connect_timeout=3,
        )
        store = ClickHouseOutcomeStore(client)
        await store.init()
        return store
    except Exception:                              # noqa: BLE001
        pass

    if pool is not None:
        try:
            store = PostgresOutcomeStore(pool)
            await store.init()
            return store
        except Exception:                          # noqa: BLE001
            pass

    return NullOutcomeStore()
