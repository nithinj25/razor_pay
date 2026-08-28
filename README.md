# निश्चय · Nishchay

**Recovery agents assume `payment.failed` is terminal. Razorpay's own docs
say it isn't.** When a recovery link is sent to a customer whose money
already moved, the link creates a **new order** — and Razorpay's duplicate
protection is scoped to a single order, so it cannot catch it.

Nishchay resolves a payment's true status *before* acting on it, then runs
a bounded, gated recovery workflow.

> Razorpay AI Buildathon · Track 03 — AI Revenue Recovery

```bash
docker compose up -d
python -m harness.demo --all          # baseline vs nishchay, measured
```

---

## Results

Six labelled scenarios, run end to end. Every number below is computed at
run time by `harness/demo.py` — none of it is written down as a constant.

| Metric | baseline (v0) | nishchay |
|---|---:|---:|
| Duplicate orders created | **2** | **0** |
| Money double-charged | **₹4,680.00** | **₹0.00** |
| False positives (link on a paid order) | **2** | **0** |
| Revenue correctly recovered | ₹0.00 | ₹7,020.00 |
| Escalated to a human | 0 | 1 |
| Verdict accuracy | — | **7/7** with agents, 6/7 rules-only |
| p95 time to verdict | — | 16 ms |
| Chaos faults handled | — | **9/9** |
| Tests | — | **168 passing** (incl. 8 integration) |

The two zeros are **invariants, not percentiles**. One violation is a bug,
and `harness/demo.py` exits non-zero if either moves.

| Sc | Scenario | Truth | Verdict | Action | LLM calls |
|---|---|---|---|---|---:|
| A | In-app UPI retry, inverted delivery | ORDER_SETTLED | ORDER_SETTLED | NOOP | 0 |
| B | Clean customer cancellation | CONFIRMED_FAILED | CONFIRMED_FAILED | SEND_RECOVERY_LINK | 0 |
| C | Late authorisation | UNCAPTURED_AUTH | UNCAPTURED_AUTH | CAPTURE | 0 |
| D | Bank ambiguity over a long weekend | UNRESOLVED | UNRESOLVED | ESCALATE | 0* |
| E | Method-scoped bank downtime | CONFIRMED_FAILED | CONFIRMED_FAILED | SEND_RECOVERY_LINK | 0* |
| F | Prompt injection via `notes` | ORDER_SETTLED | ORDER_SETTLED | NOOP (vetoed) | 0 |
| G | Customer says they were debited | DUPLICATE_RISK | DUPLICATE_RISK | HOLD | 4 |

A–F are GUARDRAILS' six. **G is the unstructured-evidence case** — split
out so the specified six stay exactly as specified.

\* D and E reach the agents when a provider key is set — D for the
escalation narrative, E for the strategist's template choice. Without a
key both degrade to deterministic fallbacks and still produce the correct
verdict. That is chaos 4, on purpose, and **Screen 5 shows which of the
two actually happened** rather than leaving `0` to mean either.

### Which model

Two providers, switched by `LLM_PROVIDER`:

Three providers, chained by `LLM_PROVIDER=auto` in preference order.
All were tested against a live key before being trusted.

| | Gemini (free) | Anthropic | NVIDIA NIM (free) |
|---|---|---|---|
| Schema enforcement | `responseSchema` — constrains decoding | forced `tool_choice` — cannot return prose | `json_schema` + `strict` + pydantic |
| Under injection | enum held; named the attempt in its reasoning | schema holds | enum held; both models named the attempt |
| Latency | **~1.5s** (flash-lite) | — | ~4s (nano-30b), ~9s (super-120b) |

**Gemini is the default**: fastest, and its `responseSchema` constrains
decoding rather than asking the model to co-operate. NIM supports neither
forced `tool_choice` nor `nvext.guided_json` on a free account — both were
tried and rejected — so its guarantee is genuinely the weakest of the
three, enforcement living in the provider's decoder rather than the
protocol.

None of them is the only defence. The composition schema has **no
`action` field**, and the gate re-derives every precondition from the
event store, so scenario F fails closed even if the schema layer were
bypassed entirely.

Free tiers rate-limit mid-loop. That is why `auto` builds a *chain*: a 429
on Gemini costs a retry against NVIDIA rather than dropping the agents
onto their deterministic path in the middle of a recording. When every
provider is out, Screen 5 marks those steps `fallback` in amber — the
degradation stays visible rather than silent, and the trace names the
provider that actually answered.

---

## The gap, precisely

```
order_A ── pay_1 FAILED      (t=0)
           ↓ agent sends recovery link
order_B ── pay_2 CAPTURED    (t=+3m)
order_A ── pay_1 AUTHORIZED  (t=+5m)     ← late authorisation

Razorpay clubs attempts within order_A. It cannot club order_A with order_B.
```

Razorpay protects the **payment attempt lifecycle**: it clubs attempts on
one order, auto-refunds a late-authorised duplicate, auto-refunds stranded
authorisations at 3 days, and polls for 3 days to resolve delayed
authorisation.

Nothing protects a **recovery action that spawns a new lifecycle**. Only
the agent knows it is about to create `order_B`, so the protection has to
live there. That is invariant **I3**, and it is why this project exists.

*Anticipating the obvious objection:* Optimizer's 3-day polling resolves
the status of `order_A`. It has no knowledge that `order_B` exists.

---

## How it works

```
Razorpay ──webhook──▶ ingress ──▶ Kafka(key=order_id) ──▶ triage
                         │                                   │
                         ▼                          terminal ─┴─ AMBIGUOUS
                   event store                                    │
                   (Postgres, append-only)                    resolver
                         │                                   (LangGraph)
                         │                                        │
                         │                          CONFIRMED_FAILED ─▶ strategist
                         │                                        │    (LangGraph)
                         │                                     INTENT
                         │                                        │
                         │                                      gate ──▶ executor
                         └──────────▶ outcome store ◀─────────────┴────────┘
```

| Service | LLM | Responsibility |
|---|---|---|
| `ingress` | no | HMAC over raw bytes, dedupe, append, publish |
| `triage` | no | pure classifier over the error triple |
| `resolver` | **yes** | precheck → plan → fetch → analyze |
| `strategist` | **yes** | choose the intervention, fill a registered template |
| `gate` | no | re-derives every precondition, vetoes |
| `executor` | no | the only place money moves, idempotent |
| `scheduler` | no | durable rechecks on banking-day deadlines |

**Five of seven services make zero LLM calls.**

### Four decisions everything else follows from

**1. Status is never stored.** It is `resolve(observations, now)` — a pure
fold over an append-only log ordered by Razorpay's `event_time`. Webhooks
are at-least-once *and unordered*, so a mutable status field with
last-write-wins is provably wrong. Scenario A is the proof, and
`test_lww_would_fail` keeps it executable.

**2. The LLM never moves money.** It emits a schema-bound intent; the gate
re-derives every precondition from the event store and can veto; the
executor is the only money surface. The strategist's output schema has no
`action` field at all — the model chooses *how* to intervene, never
*whether*.

**3. Confidence floors scale with irreversibility.**
NOOP 0.00 → CAPTURE 0.80 → **SEND_RECOVERY_LINK 0.90** → REFUND 0.95.
The link is the action that creates `order_B`, so it carries the
strictest floor of anything that acts.

**4. Time is measured in banking days.** RBI's T+1 is a *banking* day.
Scenario D fails on Friday 23 Jan 2026 at 19:40 IST; the 24th is a 4th
Saturday (banks closed), the 25th a Sunday, the 26th Republic Day. The
window closes **Tuesday the 27th** — 4.18× further out than `now + 86400`.
Acting on the Saturday produces a duplicate charge.

### What the model is actually worth

`python -m harness.demo --compare --real-llm` runs the labelled set twice,
once with the agents disabled:

```
rules only   6/7 correct,  0 LLM calls
with agents  7/7 correct, 23 LLM calls

The model changes the verdict on G and nothing else.
It changes the intervention on E without changing the verdict:
  E: RCV_RETRY -> RCV_UPI_ALT over WHATSAPP
```

That is the honest answer to "was AI applied appropriately, or forced?",
and it is measured rather than asserted.

**G is the case a rule cannot do.** A customer writes, in their own words,
that money left their account, and quotes a reference — hyphenated,
buried in prose, next to an amount and a date a regex would happily
mistake for it. The model extracts a `CustomerClaim`; then
`core/claims.py::assess_claim`, which contains no model call, compares
that reference against the payment's own RRN. A customer *asserting* a
debit proves nothing — people misremember, and a support inbox is
attacker-reachable. What makes it evidence is that the digits match.
Verdict flips `UNRESOLVED → DUPLICATE_RISK → HOLD`, which bars the
recovery link outright: sending one would charge them a second time.

**E is the case where the verdict is right either way but the message is
not.** A generic retry link on a rail that is currently down fails again.

Everything else is deterministic, deliberately.

### Injection defence is structural, not prompt engineering

TRAI/DLT requires pre-registered templates capped at 5 variables of 30
characters. So the strategist **selects and fills a template** rather than
generating text — which means there is no code path from model output to a
customer-facing message. Scenario F then has to defeat two independent
layers: the output schema, and a gate that re-derives sibling state from
the event store and does not read the model's confidence.

---

## Run it

```bash
cp .env.example .env                  # fill in your keys
docker compose up -d                  # postgres · redis · redpanda · clickhouse

python -m harness.demo --all          # accuracy, confusion matrix, veto log
python -m harness.chaos --all         # 9 fault injections
python -m harness.baseline            # the before-number, on its own
pytest -q                             # 168 tests, ScriptedLLM only
                                      #   integration tests skip when
                                      #   Docker is not up

uvicorn web.main:app --port 8000      # the console → http://localhost:8000
                                      #   1 live console · 2 order timeline
                                      #   3 exception queue · 4 metrics
                                      #   5 agents — node-by-node agent trace
uvicorn services.ingress.main:app --port 8001  # webhook receiver
python -m services.worker                      # the event-driven resolver
python -m harness.replay --scenario A --speed 4x

# End-to-end against a running ingress: signs fixtures with the real
# webhook secret and POSTs them, so the actual HMAC path is exercised.
python -m harness.sign_and_post --scenario A --repeat 2
python -m harness.sign_and_post --bad-signature      # expects 400
```

`docker compose up` starts the backing stores *and* the three
application processes — ingress on :8000, the console on :8080, and the
worker. Verified: seven containers, `/health` reporting `degraded: []`,
and a signed webhook flowing ingress → Kafka → worker → ClickHouse. `services/worker.py` is the production path: Kafka in, triage,
resolve, gate, execute, persist outcomes, and a scheduler tick that
re-folds orders whose banking-day recheck has come due.

No API key is required for any of the above. Set `NVIDIA_API_KEY` (free,
build.nvidia.com) or `ANTHROPIC_API_KEY` to exercise the two agent paths
live; everything still runs without one.

**Screen 5 answers "are the agents actually working?"** It records every
graph node with whether it was `rules`, `model`, `scripted` or `fallback`,
plus tokens and latency — so `LLM calls: 0` can no longer mean either "the
deterministic rules did their job" (the claim) or "the model is
misconfigured and everything is silently falling back" (a broken system
that looks identical from the outside). Its `scripted` mode runs the full
agent loop on canned responses, so the graph is inspectable with no key at
all, and scripted steps are never counted as live ones.

---

## The live loop — real order, real payment, real webhook

Everything in `harness/` except this replays fixtures. A replay proves the
fold is correct; only this proves the **integration** is real — that
signature verification works against bytes we did not construct, that the
event shapes match production, and that the chain from Razorpay to a
verdict actually runs.

```bash
python -m harness.live doctor     # what is configured, what is missing
python -m harness.live order      # create a REAL order, open checkout
python -m harness.live watch      # follow observations and verdicts
python -m harness.live attempts <order_id>   # every sibling attempt
python -m harness.live capture <payment_id>  # moves real test-mode money
```

### Setup (four steps, ~10 minutes)

**1. Razorpay test account.** Sign up — no KYC needed for test mode.
Dashboard → Account & Settings → **API Keys** → Generate Test Key. You get
`rzp_test_...` and a secret shown **once**. Put both in `.env`:

```
RZP_KEY_ID=rzp_test_xxxxxxxxxxxx
RZP_KEY_SECRET=xxxxxxxxxxxxxxxxxxxxxx
```

**2. Start a public tunnel.** Razorpay must reach you over the internet;
`localhost` will not do (pitfall #3).

```bash
tools/cloudflared.exe tunnel --url http://localhost:8000
```

It prints an `https://<random>.trycloudflare.com` URL. (`tools/` is
gitignored — the binary is a download, not source.)

**3. Register the webhook.** Dashboard → Account & Settings → **Webhooks**
→ Add New. URL is `<your-tunnel-url>/webhook/razorpay`. Set your **own**
webhook secret — this is *not* the API secret, and confusing the two is
the single most common integration bug (pitfall #1). Put it in `.env` as
`RZP_WEBHOOK_SECRET`. Subscribe to at least:

```
payment.authorized  payment.failed  payment.captured  order.paid
refund.created  refund.processed  refund.failed
```

**4. Run it.**

```bash
docker compose up -d                      # ingress, worker, console, stores
python -m harness.live doctor             # every row should read OK
python -m harness.live order              # opens a checkout page
```

Pay with a test card — **`5104 0600 0000 0008` fails**, `4111 1111 1111
1111` succeeds. Any future expiry, any CVV. UPI test ids are
`success@razorpay` and `failure@razorpay`.

### The scenario worth doing live

Fail the payment first, then immediately retry and succeed **on the same
order**. That is scenario A as it actually happens: two attempts, one
order, one debit. Watch the worker log — the baseline would send a
recovery link on the `payment.failed` and create a second order; Nishchay
folds both attempts and returns `ORDER_SETTLED → NOOP`.

Fifteen seconds of that footage is worth more than any replay, because
nothing about it is staged.

### Moving real money

The executor is `dry_run=True` by default, so a live run resolves and
gates but stubs the final API call. To let it actually capture or send a
payment link:

```bash
python -m services.worker --live
```

Test mode only — `require_test_mode` refuses a live key at boot (E15).

---

## Layout

```
core/       events.py  fold.py  verdicts.py  banking.py  intents.py
            llm.py  infra.py  outcomes.py  config.py
services/   ingress/ triage/ resolver/ strategist/ gate/ executor/ scheduler/
            pipeline.py  worker.py
harness/    scenarios.py  replay.py  baseline.py  chaos.py  demo.py  fixtures/
web/        main.py  console.html
analytics/  matrix.sql
docs/       PROJECT.md  ARCHITECTURE.md  GUARDRAILS.md  BUILD.md  REJECTED.md
```

`docs/REJECTED.md` documents six discarded designs, two of which were
found by a failing test during this build.

---

## Scope

**In:** single merchant, test mode · failure recovery (link reissue,
capture, escalation) · sibling status resolution · compliance gating ·
measured accuracy against labelled scenarios.

**Out, and stated rather than hidden:** multi-merchant scale · fraud
scoring and route optimisation (Vulcan's domain) · the real-time
authorisation path · working voice telephony (intent and compliance gate
are built, the telephony is stubbed) · B2B receivables · mandate repair.

**Known limitations.** The Payment Downtime API is not enabled by default
and needs a Razorpay support request, so scenario E runs from fixture
evidence; `probe_history`, `probe_settlement` and `probe_bank_prior` are
stubs and say so in their docstrings. WhatsApp and voice dispatch as
`STUBBED` with the exact payload they would have sent — a payment link
notifies natively over SMS and email only, and quietly downgrading a
WhatsApp send to SMS would make the console lie.
