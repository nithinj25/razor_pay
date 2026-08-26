# REJECTED.md — architectures we discarded, and why

Razorpay's own engineering writeups document their dead ends: Bumblebee
walks through an n8n prototype that hit branch explosion near 40 nodes,
then a single ReAct agent that died on token bloat, before arriving at
the Planner/Fetchers/Analyzer split. This file is the same discipline.

Each entry states what we built, what broke it, and what replaced it.
Entries 4 and 5 were found *during* implementation, by a test — those
are the honest ones.

---

## 1. Mutable status field, last write wins

**Built:** `orders.status` as a column, updated on each webhook.
`UPDATE orders SET status = ? WHERE id = ?`. It is what almost every
recovery agent does, and it is what our baseline (`harness/baseline.py`)
still does so we have a "before" number.

**What broke it:** Razorpay webhooks are at-least-once **and unordered**.
Scenario A is the counter-example, and it is not exotic — it is the most
common failure mode on the platform:

```
event_time +18  payment.failed     pay_1     delivered at +55
event_time +47  payment.captured   pay_2     delivered at +52
```

Arrival order writes CAPTURED then FAILED. The column ends on FAILED, the
agent fires a recovery link, and because that link creates a **new
order**, Razorpay's per-order clubbing cannot catch the duplicate. The
customer is charged twice.

**Replaced with:** `resolve(observations, now)` — a pure fold over an
append-only log, ordered by `event_time`. Order-independence is asserted
by a Hypothesis property test over every permutation
(`tests/test_fold.py::test_fold_is_order_independent`), and the failure
mode is kept executable in `test_lww_would_fail` so the justification
cannot rot.

**Cost:** every read is a fold rather than a lookup. Measured p95 to
verdict is 16 ms across the six scenarios, so this bought correctness for
nothing that matters at our volume.

---

## 2. `now + 86400` for the RBI T+1 window

**Built:** `deadline = failure_ts + 86400`. One line, obviously right.

**What broke it:** RBI's harmonised TAT is T+1 **banking days**, and
Indian banking days are not weekdays. Sundays are closed, the **2nd and
4th Saturday** of each month are closed (the 1st, 3rd and 5th are working
days), and gazetted holidays stack on top.

Scenario D is dated Friday 23 Jan 2026, 19:40 IST. The 24th is a 4th
Saturday, the 25th a Sunday, the 26th is Republic Day. The naive deadline
lands **Saturday 19:40**; the real one lands **Tuesday 27th, 23:59** —
4.18x further out. An agent using the naive form acts on Saturday, while
the bank is still going to reverse the debit unaided on Tuesday. It
produces a duplicate charge that looks like a logic bug but is a calendar
bug.

**Replaced with:** `core/banking.py`, modelling the actual RBI rule
including the 2nd/4th-Saturday distinction. `naive_deadline()` is kept
deliberately so the console can show both numbers side by side rather
than asserting the gap.

**Rejected sub-option:** treating *all* Saturdays as closed. Simpler, and
safe in the sense that it only ever delays action — but wrong, and a
reviewer who knows Indian banking would spot it immediately.

---

## 3. Kafka keyed on `payment_id`

**Built:** `publish(topic="raw", key=payment_id)`. It is the natural key:
one message, one payment.

**What broke it:** sibling attempts on the same order land on different
partitions and therefore different consumers. The single question this
system exists to answer — *did any other attempt on this order already
succeed?* — becomes a cross-partition race. I3 is unenforceable.

**Replaced with:** `key=order_id`. The order, not the payment, is the
unit of resolution. `tests/test_ingress.py::test_partition_key_is_order_id_never_payment_id`
asserts it.

**Cost:** hot-order skew. Irrelevant at one merchant; would need a
composite key at platform scale.

---

## 4. Literal terminality: "failed is non-terminal until the settlement horizon"

**Built:** exactly as ARCHITECTURE §4 reads.
`NON_TERMINAL = created | authorized | (failed ∧ age ≤ SETTLE_HORIZON)`,
and I3 vetoes a recovery link while *any* sibling is non-terminal.

**What broke it:** the gate's own test suite, on the first run. Scenario
B is a customer who explicitly cancelled — no debit ever occurred — and
the rule vetoed the recovery link for three banking days. Scenario E, a
pre-debit failure during a confirmed outage, was vetoed too. Read
literally, the invariant blocks *every* recovery for three days. That is
not caution; it is a system that cannot do its job, and it would have
made scenario B — the one that proves we are not merely cautious — fail.

**Replaced with:** `is_flippable(payment, now, cfg, evidence)`. The
question I3 actually asks is not "is this attempt terminal?" but **"could
this attempt still become captured or authorized?"** An attempt is in
play while it could still flip:

- customer-attributable cancellation → never reached the bank → not in play
- pre-debit failure during a known, method-scoped outage → not in play
- everything else → in play until the settlement horizon

**Why this is the honest fix rather than a loophole:** the danger I3
guards against is a *second order created while another attempt may still
succeed*. A payment that provably cannot succeed poses no such danger.
The rule got narrower and strictly more correct, not weaker.

---

## 5. Penalising every unavailable probe

**Built:** `_degrade()` subtracted 0.12 of confidence per unavailable
evidence source, uncapped — a direct reading of I9, "missing evidence
lowers confidence, which fails the gate floor, which yields NOOP".

**What broke it:** the first end-to-end replay. Scenario E's verdict came
out at **0.56**, far below the 0.90 floor for `SEND_RECOVERY_LINK`, so
the link was vetoed and the strategist's whole demo evaporated. The cause
was not an outage: three HTTP fetchers had returned "unavailable" because
no API client was configured in the offline demo path. The system was
taxing itself for probes it had never attempted.

**Replaced with:** two changes.

1. **A cap.** Absent evidence can subtract at most 0.30. Without a
   ceiling, a handful of unconfigured stubs sinks every verdict below
   every floor — which is not caution, it is a different failure.
2. **Skip vs fail.** A probe that is *not configured* is skipped and
   reported on the decision (`degraded: skipped (no API client): ...`).
   Only a probe that was **attempted and failed** produces unavailable
   evidence and costs confidence. An unconfigured probe is a known
   unknown; a failed one is a new one.

Chaos 3 still passes: killing a *configured* downtime fetcher drops
scenario E from `CONFIRMED_FAILED@0.92` to `PENDING_TAT@0.63` and flags
the gap. Degradation still biases toward inaction — it just no longer
fires on a configuration state.

---

## 6. Letting the model choose the action

**Built:** the strategist emitted a full `RecoveryIntent`, `action` field
included, and the gate checked it.

**What broke it:** nothing yet — which is the point. It was rejected on
inspection rather than on failure. With `action` in the model's output
schema, a successful injection has a *syntactically valid* path to
`REFUND` or `SEND_RECOVERY_LINK`. The gate would still catch it, but the
defence would rest on one layer.

**Replaced with:** the model emits `Composition`, which has **no `action`
field at all** — only a template choice, its variables, and a channel.
The action is set by `services/strategist/graph.py::_to_intent`, which is
reached only on a `CONFIRMED_FAILED` verdict. The model chooses *how* to
intervene; it cannot choose *whether*. That is I7 expressed in a type
rather than in a check.

`tests/test_strategist.py::test_model_cannot_choose_the_action` asserts
the field's absence, so it cannot be reintroduced quietly.

---

## 7. Considered and not built

| Option | Why not |
|---|---|
| Single ReAct agent over all tools | Bumblebee documents this failing on token bloat and sequential tool calls. Our fetchers are parallel and our routing is deterministic; a ReAct loop would add tokens and remove guarantees. |
| n8n / visual workflow | Branch explosion, per Bumblebee's own writeup. The strategist's conditional turn 3 is exactly the shape that explodes. |
| LLM-generated message text | Illegal under DLT before it is unsafe: templates must be pre-registered. Template selection is both the compliant and the injection-proof design. |
| Polling Razorpay instead of webhooks | Cost, and it does not solve ordering — a poller sees the same inverted reality, just later. |
| ClickHouse from day 1 | Kept, but the slip protocol drops it for Postgres. Outcome analytics at six scenarios do not need a columnar store. |
