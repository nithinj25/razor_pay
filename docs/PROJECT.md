# PROJECT.md — What we are building and why

---

## 1. One paragraph

Recovery agents listen for `payment.failed` and text the customer a
fresh payment link. That assumes `payment.failed` is terminal. Razorpay
documents two ways it isn't. When a recovery link is sent to a customer
whose money already moved, the link creates a **new order**, and
Razorpay's duplicate protection is scoped to a single order — so it
cannot catch it. **Nishchay resolves a payment's true status before
acting, then runs a bounded recovery workflow.**

---

## 2. The submission context

| | |
|---|---|
| Event | Razorpay AI Buildathon |
| Track | **03 — AI Revenue Recovery** |
| Brief | *Build an agent that detects revenue at risk, determines the right intervention, and executes a bounded recovery workflow: from payment failures and checkout abandonment to overdue receivables.* |
| The bar | Every money action explainable, bounded and gated. Show the audit trail and one failure handled gracefully. |
| Deliverables | public GitHub repo · 5-min pitch video · architecture · panel review |
| Deadline | applications close **5 September 2026** |

Judged on: whether AI was applied appropriately rather than forced;
how runtime failures were identified and handled with graceful
fallbacks; measured accuracy with an honest exception list.

---

## 3. The domain facts this is built on

Every claim below is from Razorpay's own documentation or RBI/TRAI
regulation. Sources in §7.

### 3.1 `payment.failed` is not terminal

**(a) In-app UPI retry — common.** A customer pays via a TPAP (PhonePe,
GPay). The attempt fails on a wrong PIN or low balance. Razorpay emits
`payment.failed`. The TPAP offers an immediate in-app retry; the
customer succeeds; `payment.captured` follows. Razorpay documents this
as a distinct cause of `failed → captured`, separate from late
authorisation. **Rate: unpublished.**

**(b) Late authorisation — rare.** Bank↔Razorpay communication is
interrupted; payment marked Failed, moves to Authorized later.
**Razorpay states this is under 0.5% of total payments.** Uncaptured
late-authorised payments are auto-refunded at 3 days.

### 3.2 What Razorpay already protects

Do not claim these as gaps.

| Protection | Mechanism |
|---|---|
| Duplicate attempts on one order | Orders API **clubs attempts within an `order_id`**. If one succeeds and another is late-authorised, the late one is **refunded immediately**. |
| Stranded authorisations | uncaptured payments **auto-refunded at 3 days** |
| Status uncertainty | Optimizer **polls for 3 days** to resolve delayed authorisation |

### 3.3 The gap

Clubbing is scoped to `order_id`. **A recovery link creates a new
order.**

```
order_A ── pay_1 FAILED (t=0)
           ↓ agent sends recovery link
order_B ── pay_2 CAPTURED (t=+3m)
order_A ── pay_1 AUTHORIZED (t=+5m)

Razorpay clubs within order_A. It cannot club order_A with order_B.
```

**Razorpay protects the payment attempt lifecycle. Nothing protects a
recovery action that spawns a new lifecycle.** Only the agent knows it
is about to create order_B, so the protection must live there.

### 3.4 Regulatory clocks

**RBI harmonised TAT.** Failed online transactions (UPI, IMPS, cards,
NACH, PPI) must be auto-reversed within **T+1**, with compensation
credited automatically. A "failed transaction" is defined as one not
completed **for reasons not attributable to the customer**.

Two consequences:
- The grey zone largely self-resolves inside T+1 → the correct early
  action is usually to wait.
- RBI's attributable-to-customer test maps directly onto Razorpay's
  `error.source` field. A customer-cancelled payment was never a debit,
  so the wait does not apply.

**T+1 is a banking day, not 24 hours.** A Friday-evening failure over a
long weekend has a window roughly 3.5× longer than naive
`now + 86400`.

### 3.5 Outbound communication (TRAI / TCCCPR)

Not covered by Razorpay's docs. Governs every recovery message.

- Messages are classified **Promotional / Transactional / Service
  Implicit / Service Explicit**. Fully-blocked DND subscribers still
  receive Transactional, Service Implicit and Government messages.
- **Service Implicit** = triggered by a customer action. A failure
  notification qualifies. A later nudge is contested. Anything with an
  incentive is **Promotional** and blocked on DND.
- Voice outbound numbers must use the correct series — **140 for
  promotional, 160 for transactional/service** (one source says 1600;
  verify before quoting).
- **DLT templates must be pre-registered**, limited to **5 variables of
  30 chars each**. WhatsApp needs Business API via an approved BSP.
- 2025 amendment: complaint threshold for action dropped from ten to
  five; **explicit consent validity capped at 7 days**; TRAI can act
  directly against a sender.
- Penalty tail: repeat violations can mean **suspension from the DLT
  platform, which blocks transactional traffic too** — a bad recovery
  campaign can break your OTP delivery.

**Design consequence:** the strategist **selects and fills a registered
template**, it does not generate free text.

---

## 4. Scope

### In scope
- Single merchant, test mode
- Payment-failure recovery: link reissue, capture, escalation
- Status resolution across sibling attempts on an order
- Compliance gating of outbound channel and category
- Measured accuracy against labelled scenarios

### Out of scope — say so, don't apologise
- Multi-merchant platform scale
- Fraud scoring / route optimisation (Vulcan's domain)
- Real-time authorisation path
- Working voice telephony (intent + gate built; executor stubbed — see
  GUARDRAILS §6)
- B2B receivables, mandate repair (designed-adjacent, not built)

---

## 5. Standards we are matching

Razorpay's engineering team publishes their agent architectures
(Bumblebee for risk, Viveka for RCA). Four practices repeat:

| Practice | What we do |
|---|---|
| **Document rejected architectures** — Bumblebee's writeup walks through an n8n prototype that hit branch explosion near 40 nodes, then a single ReAct agent that died on token bloat and sequential tool calls, before the Planner/Fetchers/Analyzer split | `docs/REJECTED.md`, minimum three entries |
| **Report before/after metrics** — Bumblebee: tokens −60%, latency 35s→8–12s, success 88%→99%+. Viveka: MTTI −80%, MTTR −50–60% | every claim gets a v0/v2 pair |
| **Prune at the edge** — fetchers return compact JSON with confidence and provenance; deterministic thresholds before any LLM call; per-agent temperature | log `tokens_in` per node |
| **Degrade gracefully and flag** — proceed on partial data, flag the gap for manual review rather than blocking | missing evidence → lower confidence → NOOP + escalate |

Viveka is built on **LangGraph**; we match the framework deliberately.

---

## 6. Positioning against Razorpay's shipped agents

| Their agent | What it does | Why we are not it |
|---|---|---|
| Vulcan | scores routes, detects fraud, real time | scoring, hot path — not our problem |
| Agent Studio — Subscription Recovery | retries failed subscription payments, nudges | acts on failure; does not verify it |
| Bumblebee | merchant website risk review | different domain |
| Viveka | infra RCA — pods, deploys, logs | our failures are counterparty-side, not infra |

**One-line answer:** *Vulcan scores and routes. Their recovery agent
retries and nudges. Neither resolves whether the payment actually failed
before acting. That's a correctness gap, not a feature gap.*

---

## 7. References

**Razorpay docs**
- Webhooks overview — https://razorpay.com/docs/webhooks/
- Payments webhook events (in-app retry note) — https://razorpay.com/docs/webhooks/payments/
- Webhooks FAQ (retries, secret rotation, replay) — https://razorpay.com/docs/webhooks/faqs/
- Handle late authorisation — https://razorpay.com/docs/payments/payments/late-authorisation/handle/
- Late authorisation overview — https://razorpay.com/docs/payment-gateway/payments/late-authorization/
- Payment error codes — https://razorpay.com/docs/errors/payments/list/
- Fetch payments (`acquirer_data`, `rrn`) — https://razorpay.com/docs/api/payments/fetch-all-payments/
- Capture & refund settings (3-day rule) — https://razorpay.com/docs/payments/optimizer/capture-refund-settings/

**Razorpay engineering**
- Bumblebee — https://dev.to/razorpaytech/meet-bumblebee-agentic-ai-flagging-risky-merchants-in-under-90-seconds-2nlf
- Project Viveka — https://dev.to/razorpaytech/project-viveka-a-multi-agent-ai-that-does-root-cause-analysis-in-under-90-seconds-4g44
- Agent Studio — https://razorpay.com/agent-studio/
- Buildathon — https://razorpay.com/buildathon/

**Regulatory** — verify against primary text before quoting figures
- RBI harmonised TAT circular (failed transactions, T+1, compensation)
- TRAI TCCCPR 2018 + 2025 amendment
- NPCI UPI Ecosystem Statistics — BD/TD, Debit Reversal Success %,
  Deemed Approved (Pending Credit Confirmation)

**Confidence flags**
- `[VERIFY]` voice number series (140/160 vs 1600)
- `[VERIFY]` settlement recon endpoint path
- `[VERIFY]` NPCI per-bank granularity — if usable, the bank-prior
  fetcher becomes real data instead of a stub
- `[BLOCKED]` Payment Downtime API is **not enabled by default** —
  support request required, unknown lead time
