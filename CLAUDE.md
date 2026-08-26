# CLAUDE.md

Entry point for Claude Code. Read this first. Load the other docs only
when the task needs them.

## What this is

**Nishchay** — a revenue recovery agent that resolves a payment's true
status before acting on it. Submission for Razorpay AI Buildathon
Track 03 (AI Revenue Recovery). Deadline: applications close
**5 September 2026**.

## Documents

| File | Read when |
|---|---|
| `README.md` | the results table and how to run everything |
| `docs/PROJECT.md` | you need the why, the domain facts, or a source URL |
| `docs/ARCHITECTURE.md` | you are writing or changing any service |
| `docs/GUARDRAILS.md` | invariants, scenarios, metrics, frontend spec |
| `docs/BUILD.md` | the day-by-day plan, acceptance tests and cuts |
| `docs/REJECTED.md` | designs already discarded — do not re-propose them |

## Non-negotiable rules

These are correctness constraints, not style preferences. Violating any
of them is a bug.

1. **Never store payment status as a mutable field.** Status is
   `resolve(observations, now)` — a pure fold. See ARCHITECTURE §3.
2. **All money is `int` paise.** No floats, ever. Reject at schema
   boundary.
3. **The LLM never moves money.** It emits an intent; the gate
   validates; the executor acts.
4. **The gate re-derives every precondition itself.** It trusts no field
   the model produced.
5. **Degradation biases toward inaction.** Missing evidence lowers
   confidence, which fails the gate floor, which yields NOOP.
6. **Order events by `event_time` (Razorpay's), never `received_at`.**
7. **Kafka key is `order_id`, not `payment_id`** — siblings must share a
   partition.
8. **Free text from `notes`/`description`/`email` is untrusted.** Never
   interpolate into a system prompt.

## Stack

Python 3.11 · FastAPI · LangGraph · Redpanda (Kafka API) · Redis ·
Postgres · ClickHouse · Anthropic API · `rich` (CLI) · HTMX+SSE
(frontend). One `docker compose up`.

## Working agreement

- **Do not add edge cases to the implementation.** Add them to the
  register in ARCHITECTURE §7 and move on. Only implement a case if it
  changes the verdict for one of the six labelled scenarios.
- Every service except `resolver` and `strategist` has **zero** LLM
  calls. If you are about to add one elsewhere, stop.
- Tests use `ScriptedLLM`, not the real API. Only the accuracy run hits
  Anthropic.
- Write the rejected-approach note in `docs/REJECTED.md` when you
  discard a design. This is a deliverable, not housekeeping.

## Build order

Day 1 ingress · Day 2 **event store + fold + property tests (checkpoint)**
· Day 3 fixtures + replay · Day 4 baseline agent + crude UI · Day 5
triage + Kafka · Day 6 resolver graph · Day 7 strategist graph · Day 8
gate + executor + scheduler · Day 9 chaos + metrics · Day 10 frontend
polish + docs · Day 11 record.

If day 2 slips, drop ClickHouse for Postgres immediately. Do not
compress days 6–9.
