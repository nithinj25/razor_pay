-- Confusion matrix and headline metrics.
-- A query, not a script: the numbers in the pitch must be re-derivable
-- by a reviewer typing this at a prompt.
--
-- Runs on ClickHouse; the CREATEs are ANSI enough for Postgres if the
-- slip protocol drops ClickHouse (MergeTree -> nothing, DateTime -> timestamptz).

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
) ENGINE = MergeTree ORDER BY (ts, order_id);

CREATE TABLE IF NOT EXISTS vetoes (
    ts          DateTime,
    trace_id    String,
    order_id    String,
    action      String,
    rule        String,
    reason      String,
    confidence  Float32,
    evidence    String
) ENGINE = MergeTree ORDER BY (ts, order_id);


-- 1. Confusion matrix. The diagonal is correct; anything off it is an
--    exception we must be able to explain by name.
SELECT ground_truth, verdict, count() AS n
FROM outcomes
GROUP BY ground_truth, verdict
ORDER BY ground_truth, verdict;


-- 2. The two invariants. Both must be zero. These are not percentiles -
--    one row here is a bug, not a regression in a metric.
SELECT
    countIf(action = 'SEND_RECOVERY_LINK' AND ground_truth = 'ORDER_SETTLED')
        AS false_positive_links,
    countIf(action IN ('SEND_RECOVERY_LINK','REFUND','CAPTURE')
            AND status IN ('EXECUTED','STUBBED')
            AND ground_truth = 'ORDER_SETTLED')
        AS money_actions_on_settled_orders
FROM outcomes;


-- 3. Where the LLM actually gets used. The claim is that five of seven
--    services never call it and that A/B/C resolve at zero calls; this
--    is what makes that checkable rather than rhetorical.
SELECT
    scenario,
    any(ground_truth)      AS truth,
    sum(llm_calls)         AS llm_calls,
    sum(tokens_in + tokens_out) AS tokens,
    round(avg(latency_ms))      AS avg_ms
FROM outcomes
GROUP BY scenario
ORDER BY scenario;


-- 4. Veto log, by rule. This is the audit trail the track brief asks for:
--    every refusal, with the invariant that produced it.
SELECT rule, action, count() AS n, any(reason) AS example
FROM vetoes
GROUP BY rule, action
ORDER BY n DESC;


-- 5. Latency distribution against the SLO (verdict p95 < 15s).
SELECT
    quantile(0.50)(latency_ms) AS p50_ms,
    quantile(0.95)(latency_ms) AS p95_ms,
    max(latency_ms)            AS max_ms,
    countIf(latency_ms > 15000) AS slo_breaches
FROM outcomes;


-- 6. Before/after, if the baseline's rows are loaded with scenario
--    prefixed 'baseline:'. Kept as one query so the pitch number and the
--    demo number cannot drift apart.
SELECT
    if(startsWith(scenario, 'baseline:'), 'baseline', 'nishchay') AS system,
    countIf(action = 'SEND_RECOVERY_LINK' AND ground_truth = 'ORDER_SETTLED') AS duplicates,
    sum(if(action = 'SEND_RECOVERY_LINK' AND ground_truth = 'ORDER_SETTLED', amount_due, 0)) / 100
        AS duplicated_rupees
FROM outcomes
GROUP BY system;
