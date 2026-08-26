# BUILD.md — Step-by-step build guide

Every day has an **acceptance test**. If it doesn't pass, you are not
done with that day — do not move on. If a day slips, take the cut listed
for that day, not from a later one.

---

## Day 0 — Setup (do this today, before any code)

### 0.1 Accounts and keys

1. **Razorpay test account** — sign up, no KYC needed for test mode.
2. **API keys** — Dashboard → Account & Settings → API Keys → Generate
   Test Key. You get `rzp_test_xxx` (key_id) and a secret shown **once**.
3. **Webhook** — Dashboard → Account & Settings → Webhooks → Add New.
   URL is your ngrok URL. **Set your own webhook secret** — this is
   *not* the API secret. Confusing the two is the single most common
   integration bug.
4. Subscribe to: `payment.authorized`, `payment.failed`,
   `payment.captured`, `order.paid`, `refund.created`,
   `refund.processed`, `refund.failed`, `settlement.processed`,
   `payment.downtime.started`, `payment.downtime.updated`,
   `payment.downtime.resolved`.
5. **Anthropic API key.**
6. **Raise the Payment Downtime API support request now.** It is not
   enabled by default and the lead time is unknown. This is the only
   dependency you cannot code around.

### 0.2 Environment

```bash
mkdir nishchay && cd nishchay
git init
python -m venv .venv && source .venv/bin/activate
pip install fastapi uvicorn httpx pydantic redis asyncpg \
            clickhouse-connect aiokafka langgraph anthropic \
            rich pytest hypothesis pybreaker python-dotenv
pip freeze > requirements.txt
```

`.env` — and add it to `.gitignore` **now**, before the first commit:

```
RZP_KEY_ID=rzp_test_xxx
RZP_KEY_SECRET=xxx
RZP_WEBHOOK_SECRET=xxx          # NOT the API secret
RZP_WEBHOOK_SECRET_PREV=        # dual-secret window, E8
ANTHROPIC_API_KEY=sk-ant-xxx
POSTGRES_DSN=postgresql://nishchay:nishchay@localhost:5432/nishchay
REDIS_URL=redis://localhost:6379
KAFKA_BOOTSTRAP=localhost:19092
CLICKHOUSE_URL=http://localhost:8123
TAT_WINDOW_BANKING_DAYS=1
SETTLE_HORIZON_DAYS=3
MAX_TURNS=6
MAX_TOKENS_PER_RESOLUTION=8000
```

### 0.3 docker-compose.yml

```yaml
services:
  postgres:
    image: postgres:16
    environment:
      POSTGRES_USER: nishchay
      POSTGRES_PASSWORD: nishchay
      POSTGRES_DB: nishchay
    ports: ["5432:5432"]
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
  redpanda:
    image: redpandadata/redpanda:latest
    command:
      - redpanda start --overprovisioned --smp 1 --memory 1G
      - --kafka-addr PLAINTEXT://0.0.0.0:29092,OUTSIDE://0.0.0.0:19092
      - --advertise-kafka-addr PLAINTEXT://redpanda:29092,OUTSIDE://localhost:19092
    ports: ["19092:19092"]
  clickhouse:
    image: clickhouse/clickhouse-server:latest
    ports: ["8123:8123"]
    ulimits:
      nofile: {soft: 262144, hard: 262144}
```

### 0.4 ngrok

```bash
ngrok http 8000        # paste the https URL into the Razorpay webhook
```

**Acceptance:** `docker compose up -d` → all four healthy. `.env`
populated. Downtime request raised.

---

## Day 1 — Ingress

**Build:** `services/ingress/main.py`

```python
@app.post("/webhook/razorpay")
async def webhook(request: Request):
    raw = await request.body()                # RAW bytes, never re-serialise
    sig = request.headers.get("X-Razorpay-Signature", "")
    if not verify_any_secret(raw, sig):       # current + previous (E8)
        return Response(status_code=400)
    eid = request.headers["X-Razorpay-Event-Id"]
    if not await redis.set(f"evt:{eid}", 1, nx=True, ex=7*86400):
        return Response(status_code=200)      # duplicate, drop
    await append_observation(eid, raw)
    return Response(status_code=200)
```

**Pitfall:** if you `json.loads()` then re-serialise before verifying,
the signature will never match. Verify the raw bytes.

**Acceptance**
```bash
curl -X POST localhost:8000/webhook/razorpay -d '{}' -H 'X-Razorpay-Signature: bad'
# → 400
python -m harness.sign_and_post fixtures/clean_failure.json
# → 200, one row in observations; run twice → still one row
```

**Cut if slipping:** none. Day 1 is not optional.

---

## Day 2 — Event store + fold ← **CHECKPOINT**

This is the day the project succeeds or fails.

**Build:** `core/events.py`, `core/fold.py`, `core/verdicts.py`,
`core/banking.py`

```python
def resolve(order_id: str, now: int) -> Verdict:
    obs = load(order_id)                       # ALL siblings
    obs.sort(key=lambda o: (o.event_time, o.event_id))
    return fold(obs, now)                      # pure — no I/O
```

`fold` must be a pure function. No database calls, no clock reads, no
network. `now` is a parameter.

**Acceptance — the property test is the whole point**
```python
@given(st.permutations(SCENARIO_A_OBSERVATIONS))
def test_order_independent(perm):
    assert resolve_from(perm, NOW) == resolve_from(CANONICAL, NOW)

def test_lww_would_fail():
    """Documents why we don't use a mutable status field."""
    assert naive_lww(SCENARIO_A_RECEIPT_ORDER) == "FAILED"   # wrong
    assert resolve_from(SCENARIO_A_RECEIPT_ORDER, NOW).verdict == "ORDER_SETTLED"
```

Also: `banking.py` must give **Tuesday** for a Friday 19:40 IST failure
with a Monday holiday (E14).

**Cut if slipping:** drop ClickHouse, use Postgres for outcomes. Buy the
day back here, never from days 6–9.

---

## Day 3 — Fixtures + replay harness

**Build:** `harness/fixtures/*.json`, `harness/replay.py`,
`harness/fixtures/labels.json`

Adapt payloads from Razorpay's webhook docs. Six scenarios (A–F, see
GUARDRAILS §4). Each fixture is a timeline:

```json
{ "scenario": "A", "ground_truth": "ORDER_SETTLED",
  "timeline": [
    {"at": 52, "event": "payment.captured", "event_time": 47, "file": "..."},
    {"at": 55, "event": "payment.failed",   "event_time": 18, "file": "..."}
  ]}
```

`replay.py` signs each with the real webhook secret and POSTs on a
synthetic clock.

**Acceptance:** `python -m harness.replay --scenario A --speed 4x`
delivers events in the inverted order, all signatures verify, six
scenarios all replay.

**Cut:** F (prompt injection) can move to day 7.

---

## Day 4 — Baseline agent + crude console

**Build:** `harness/baseline.py`, `web/console.html` (rough)

The baseline is the naive agent: mutable status, LWW, link on every
`payment.failed`. **You need it for the delta, and its failure becomes
`docs/REJECTED.md` entry v1.**

Build Screen 1 rough now — split-pane, SSE, no styling. Working blind
through logs for six days is how projects die.

**Acceptance**
```bash
python -m harness.baseline --all
# → prints duplicate orders created: N   (N > 0, this is your before-number)
```
Console shows both panes updating live on scenario A.

**Cut:** styling. Not the split-pane.

---

## Day 5 — Triage + Kafka

**Build:** `services/triage/`, Kafka producer/consumer, DLQ

Pure classifier over `source` × `step` × `reason`. **No LLM. No I/O.**
Kafka key is `order_id` — verify siblings land on one partition.

**Acceptance:** unit tests cover every triage branch; scenarios A and B
classify correctly; a poison message lands in DLQ after N retries.

**Cut:** Kafka → in-process asyncio queue. Note the swap in REJECTED.md.

---

## Day 6 — Resolver graph

**Build:** `services/resolver/graph.py`, `planner.py`, `fetchers/`,
`analyzer.py`

Get the graph executing **with one fetcher and a hardcoded route first**.
Wiring is where time goes, not prompts. Then add parallelism, circuit
breakers, pruning.

Deterministic rules run before any LLM call. Scenarios A, B, C must
resolve with **zero** LLM calls.

**Acceptance**
```bash
pytest tests/test_resolver.py       # ScriptedLLM, not the real API
python -m harness.replay --scenario C
# → PENDING_TAT, then UNCAPTURED_AUTH after the +5m event
assert llm_calls == 0 for scenarios A, B, C
```

**Cut:** `settlement` and `bank_prior` fetchers → stubs.

---

## Day 7 — Strategist graph

**Build:** `services/strategist/`, `RecoveryIntent`, template registry

Forced `tool_choice` so the model cannot return prose. `MAX_TURNS=6`
enforced in the router, not the prompt. Untrusted-data block for `notes`.

**Acceptance:** scenario E produces `RCV_UPI_ALT` + WhatsApp +
`SERVICE_IMPLICIT`; scenario F's injection fails the schema or is vetoed
downstream; a forced 7-turn loop terminates at 6.

**Cut:** `probe_history` → stub returning empty. E degrades but still
demonstrates the loop.

---

## Day 8 — Gate + executor + scheduler

**Build:** `services/gate/`, `services/executor/`, `services/scheduler/`,
LangGraph `PostgresSaver`

Gate re-derives everything itself (I8). Every veto persisted. Executor
idempotency keyed on `(payment_id, action, evidence_version)`. Scheduler
is a Redis ZSET with banking-day due times.

**Acceptance:** F is vetoed with a logged reason; the same intent
submitted twice executes once; killing a worker mid-graph and restarting
resumes from checkpoint to the same verdict.

**Cut:** none — this is the "bounded and gated" requirement.

---

## Day 9 — Chaos + metrics

**Build:** `harness/chaos.py`, `analytics/matrix.sql`, outcome writes

All nine faults from GUARDRAILS §5. Chaos 5 (kill resolver mid-flight) is
the one you demo.

**Acceptance**
```bash
python -m harness.chaos --all        # 9/9 pass
clickhouse-client --query "$(cat analytics/matrix.sql)"
# → confusion matrix over six scenarios
```
Headline table populated: duplicates **0**, false positives **0**.

**Cut:** chaos 6–9. Keep 1–5.

---

## Day 10 — Frontend + docs

**Build:** Screens 2–4, README, `docs/REJECTED.md` (3 entries),
one live test-mode loop

The live loop: create a real order, pay it in test mode, watch a real
signed webhook hit ngrok. Fifteen seconds of footage that proves the
integration is real.

README first screen must carry: the problem in three lines, the results
table, and **one command to run it**.

**Acceptance:** fresh clone → `docker compose up` →
`python -m harness.demo --all` works with no manual steps. Ask someone
else to try it.

**Cut:** Screens 2–4 → static screenshots. Screen 1 survives.

---

## Day 11 — Record

Shot list in GUARDRAILS §8 and the plan below.

| Time | Shot |
|---|---|
| 0:00–0:20 | **Cold open** — baseline double-charges. No title card. |
| 0:20–0:45 | Razorpay's own docs: `failed → captured` via in-app retry |
| 0:45–1:05 | The gap: clubbing is per-order; a recovery link makes a new order |
| 1:05–2:15 | Scenarios A → B → C split-screen. **B proves it isn't just cautious.** |
| 2:15–2:50 | Scenario E — strategist picks UPI over downed netbanking |
| 2:50–3:20 | `docker kill resolver` → recovery → same verdict. Kill LLM → safe UNRESOLVED. |
| 3:20–4:00 | Architecture. "Five of seven services have zero LLM calls." |
| 4:00–4:35 | Live SQL: confusion matrix, then the veto log |
| 4:35–5:00 | Scenario D → UNRESOLVED, evidence packet with RRN |

Terminal at **18pt minimum**. Record voiceover separately. Show the speed
multiplier on screen so compression is never hidden. 4:30 beats 5:00.

---

## Commands cheat sheet

```bash
docker compose up -d
uvicorn services.ingress.main:app --reload --port 8000
ngrok http 8000

python -m harness.replay --scenario A --speed 4x
python -m harness.baseline --all
python -m harness.demo --all --headless
python -m harness.chaos --all

pytest -q                                   # ScriptedLLM only
pytest tests/test_fold.py -q                # the property tests
clickhouse-client --query "$(cat analytics/matrix.sql)"
```

---

## Pitfalls, ranked by how much time they cost

1. **Webhook secret ≠ API secret.** Different values, different places.
2. **Re-serialising the body before HMAC.** Verify raw bytes.
3. **`localhost` as a webhook URL.** Must be public — use ngrok.
4. **Floats for money.** Everything is int paise.
5. **`received_at` used for ordering.** Always `event_time`.
6. **Kafka keyed on `payment_id`.** Must be `order_id`.
7. **Real LLM calls in tests.** Slow, flaky, expensive. Use ScriptedLLM.
8. **`now + 86400` for T+1.** Banking days (E14).
9. **Committing `.env`.** Add to `.gitignore` before the first commit.
10. **Building the frontend on day 10.** Rough version day 4.

---

## Slip protocol

| If you are behind by | Do this |
|---|---|
| ½ day | take that day's listed cut |
| 1 day | drop ClickHouse → Postgres; drop chaos 6–9 |
| 2 days | drop Kafka → asyncio queue; Screens 2–4 → screenshots |
| 3 days | drop the strategist; submit resolver + gate only, and say so |

**Never cut:** the fold and its property tests, the baseline comparison,
the confusion matrix, chaos 1–5, Screen 1, the video.
