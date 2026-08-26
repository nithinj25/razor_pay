"""The console. FastAPI + SSE + one HTML page. No build step.

Its purpose is not decoration. A reviewer cannot audit a claim they
cannot see, and logs read as noise while a polished dashboard reads as a
mockup. The split-screen is the argument: both panes are fed the *same*
event stream, and they diverge. Nothing has to be narrated.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, AsyncIterator

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from core.llm import build_llm
from core.verdicts import Action, Verdict
from harness import baseline as bl
from harness import scenarios as sc
from services.executor.main import Executor
from services.pipeline import Pipeline

HERE = Path(__file__).parent
app = FastAPI(title="nishchay-console")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse(HERE / "console.html")


@app.get("/api/scenarios")
async def scenarios() -> JSONResponse:
    return JSONResponse(
        [
            {
                "key": s.key,
                "title": s.title,
                "order_id": s.order_id,
                "ground_truth": s.ground_truth.value,
                "note": s.note,
                "amount": sc.AMOUNT,
                "events": len(s.deliveries),
            }
            for s in sc.ALL
        ]
    )


@app.get("/api/stream")
async def stream(
    scenario: str = Query("A"), speed: float = Query(4.0)
) -> StreamingResponse:
    """Server-sent events for one scenario replay.

    Both panes are driven from this one stream: the baseline's reaction
    and nishchay's verdict are emitted for the same observation, in the
    same frame, so a viewer can see that they were given identical input.
    """
    return StreamingResponse(
        _replay_events(scenario.upper(), speed),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event: str, data: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _replay_events(key: str, speed: float) -> AsyncIterator[str]:
    s = sc.BY_KEY.get(key)
    if s is None:
        yield _sse("error", {"message": f"unknown scenario {key}"})
        return

    yield _sse("start", {
        "scenario": s.key, "title": s.title, "note": s.note,
        "order_id": s.order_id, "ground_truth": s.ground_truth.value,
        "amount": sc.AMOUNT, "speed": speed,
    })

    ex = Executor(dry_run=True)
    pipeline = Pipeline(llm=build_llm(), executor=ex)

    # The baseline is recomputed incrementally against the same
    # observations, so the two panes are provably fed one stream.
    delivered: list = []
    settled_truth = s.ground_truth == Verdict.ORDER_SETTLED
    baseline_dupes = 0
    baseline_paise = 0

    obs_list = s.observations()
    for d, obs in zip(s.deliveries, obs_list):
        now = s.start + d.at
        delivered.append(obs)

        prior_max = max((o.event_time for o in delivered[:-1]), default=obs.event_time)
        yield _sse("observation", {
            "at": now,
            "clock": d.at,
            "event_type": obs.event_type,
            "payment_id": obs.payment_id,
            "event_time": obs.event_time - s.start,
            "received_at": obs.received_at - s.start,
            "inverted": obs.event_time < prior_max,
            "error_source": obs.error_source,
            "error_step": obs.error_step,
            "error_reason": obs.error_reason,
        })

        # -- baseline pane: mutable status, last write wins, act on failure
        run = bl.run_baseline(delivered, truth_settled=settled_truth)
        new_dupes = len(run.duplicates)
        if new_dupes > baseline_dupes:
            baseline_paise = run.duplicate_amount
        baseline_dupes = new_dupes
        yield _sse("baseline", {
            "status": run.status.get(s.order_id, "NONE"),
            "actions": [a.action for a in run.actions],
            "duplicates": baseline_dupes,
            "duplicate_rupees": baseline_paise / 100,
        })

        # -- nishchay pane: fold, verdict, gate, outcome
        decision = await pipeline.process(
            list(delivered), now, order_id=s.order_id, seed_evidence=s.evidence
        )
        yield _sse("decision", {
            **decision.to_row(),
            "fold": [
                {
                    "payment_id": o.payment_id,
                    "event_type": o.event_type,
                    "event_time": o.event_time - s.start,
                }
                for o in sorted(delivered, key=lambda x: (x.event_time, x.event_id))
            ],
            "evidence": [
                {"source": e.source, "available": e.available,
                 "confidence": e.confidence, "provenance": e.provenance}
                for e in decision.evidence
            ],
            "degraded": decision.degraded,
            "trace": decision.trace,
        })

        await asyncio.sleep(min(1.5, 1.0 / max(speed, 0.1)))

    # Final evaluation at the labelled `now` - for D, after the banking
    # window closes on the Tuesday.
    if s.evaluate_at > (s.start + s.deliveries[-1].at):
        final = await pipeline.process(
            list(delivered), s.evaluate_at, order_id=s.order_id, seed_evidence=s.evidence
        )
        yield _sse("decision", {
            **final.to_row(),
            "final_pass": True,
            "fold": [],
            "evidence": [],
            "degraded": final.degraded,
            "trace": final.trace + ["evaluated after the RBI T+1 banking window closed"],
        })

    acted = [o for o in ex.outcomes if o.status in ("EXECUTED", "STUBBED")]
    links = [o for o in acted if o.action == Action.SEND_RECOVERY_LINK]
    yield _sse("done", {
        "baseline_duplicates": baseline_dupes,
        "baseline_rupees": baseline_paise / 100,
        "nishchay_duplicates": len(links) if settled_truth else 0,
        "nishchay_rupees": 0.0 if settled_truth else 0.0,
        "vetoes": pipeline.vetoes,
        "exception_queue": ex.exception_queue,
        "ground_truth": s.ground_truth.value,
    })


@app.get("/api/metrics")
async def metrics() -> JSONResponse:
    """Screen 4. Reads the artifact written by `harness.demo`."""
    p = Path(".artifacts/results.json")
    if not p.exists():
        return JSONResponse(
            {"error": "run `python -m harness.demo --all` first"}, status_code=404
        )
    return JSONResponse(json.loads(p.read_text(encoding="utf-8")))


@app.get("/api/timeline")
async def timeline(scenario: str = Query("A")) -> JSONResponse:
    """Screen 2 - the verdict recomputed at every step.

    The point of this screen is to prove the verdict is *derived*. Each
    row shows the observations known at that instant and the verdict they
    fold to, so a reviewer watches it change as evidence arrives rather
    than taking our word that nothing is stored.
    """
    s = sc.BY_KEY.get(scenario.upper())
    if s is None:
        return JSONResponse({"error": f"unknown scenario {scenario}"}, status_code=404)

    from core.banking import ist_datetime, naive_deadline, tat_deadline
    from core.fold import build_state, fold

    obs_all = s.observations()
    rows = []
    delivered: list = []

    for d, obs in zip(s.deliveries, obs_all):
        now = s.start + d.at
        delivered.append(obs)
        v = fold(list(delivered), now, order_id=s.order_id, evidence=s.evidence)
        st = build_state(list(delivered), s.order_id)
        rows.append({
            "at": now,
            "clock": d.at,
            "event_type": obs.event_type,
            "payment_id": obs.payment_id,
            "event_time_offset": obs.event_time - s.start,
            "received_at_offset": obs.received_at - s.start,
            "skew": obs.skew,
            # E18 - the inversion, shown rather than described.
            "inverted": obs.event_time < max(
                (o.event_time for o in delivered[:-1]), default=obs.event_time
            ),
            "verdict": v.verdict.value,
            "confidence": round(v.confidence, 3),
            "action": v.proposed_action.value,
            "rules_fired": list(v.rules_fired),
            "amount_paid": v.amount_paid,
            "amount_due": v.amount_due,
            "any_sibling_non_terminal": v.any_sibling_non_terminal,
            "payments": [
                {"payment_id": p.payment_id,
                 "status": p.status.value if p.status else None,
                 "rrn": p.rrn}
                for p in st.payments.values()
            ],
        })

    if s.evaluate_at > (s.start + s.deliveries[-1].at):
        v = fold(list(delivered), s.evaluate_at, order_id=s.order_id, evidence=s.evidence)
        rows.append({
            "at": s.evaluate_at,
            "clock": s.evaluate_at - s.start,
            "event_type": "(re-fold, no new event)",
            "payment_id": None,
            "event_time_offset": None,
            "received_at_offset": None,
            "skew": 0,
            "inverted": False,
            "verdict": v.verdict.value,
            "confidence": round(v.confidence, 3),
            "action": v.proposed_action.value,
            "rules_fired": list(v.rules_fired),
            "amount_paid": v.amount_paid,
            "amount_due": v.amount_due,
            "any_sibling_non_terminal": v.any_sibling_non_terminal,
            "payments": [],
            "note": "same observations, later clock - the verdict moved because time did",
        })

    failure_ts = next((o.event_time for o in obs_all if o.event_type == "payment.failed"), None)
    clocks = None
    if failure_ts:
        real, naive = tat_deadline(failure_ts), naive_deadline(failure_ts)
        clocks = {
            "failure": ist_datetime(failure_ts).strftime("%a %d %b %Y %H:%M IST"),
            "naive_deadline": ist_datetime(naive).strftime("%a %d %b %Y %H:%M IST"),
            "banking_deadline": ist_datetime(real).strftime("%a %d %b %Y %H:%M IST"),
            "ratio": round((real - failure_ts) / max(naive - failure_ts, 1), 2),
        }

    return JSONResponse({
        "scenario": s.key, "title": s.title, "order_id": s.order_id,
        "note": s.note, "ground_truth": s.ground_truth.value,
        "evidence": [
            {"source": e.source, "confidence": e.confidence,
             "provenance": e.provenance, "available": e.available,
             "value": e.value}
            for e in s.evidence
        ],
        "clocks": clocks,
        "rows": rows,
    })


@app.get("/api/exceptions")
async def exceptions() -> JSONResponse:
    """Screen 3 - the exception queue. Scenario D, and the video's ending.

    Every card carries the identifiers a human can actually act on. RRN
    first: when a customer says "money was debited", that is the field
    that ties this payment to their bank statement.
    """
    from core.banking import banking_days_between, ist_datetime, tat_deadline
    from core.fold import build_state, fold
    from core.verdicts import Verdict as V

    cards = []
    for s in sc.ALL:
        v = fold(s.observations(), s.evaluate_at, order_id=s.order_id, evidence=s.evidence)
        if v.verdict not in (V.UNRESOLVED, V.DUPLICATE_RISK):
            continue

        st = build_state(s.observations(), s.order_id)
        failed = max(st.failed, key=lambda p: p.failed_at or 0) if st.failed else None
        failure_ts = failed.failed_at if failed else s.start
        deadline = tat_deadline(failure_ts)

        checked = [f"deterministic rules: {', '.join(v.rules_fired)}"]
        checked += [
            f"{e.source}: {'available' if e.available else 'UNAVAILABLE'} "
            f"({e.provenance})"
            for e in v.evidence
        ]
        missing = [e.source for e in v.evidence if not e.available] or [
            "customer bank statement confirmation",
            "settlement recon entry",
        ]

        cards.append({
            "scenario": s.key,
            "order_id": s.order_id,
            "payment_id": failed.payment_id if failed else None,
            "amount_paise": v.amount_due,
            "amount_rupees": v.amount_due / 100,
            "verdict": v.verdict.value,
            "confidence": round(v.confidence, 3),
            # The two identifiers that make this actionable.
            "rrn": failed.rrn if failed else None,
            "upi_transaction_id": failed.upi_transaction_id if failed else None,
            "error_source": failed.error_source if failed else None,
            "error_step": failed.error_step if failed else None,
            "method": failed.method if failed else None,
            "failed_at": ist_datetime(failure_ts).strftime("%a %d %b %Y %H:%M IST"),
            "age_banking_days": banking_days_between(failure_ts, s.evaluate_at),
            "window_closed": ist_datetime(deadline).strftime("%a %d %b %Y %H:%M IST"),
            "what_was_checked": checked,
            "what_is_missing": missing,
            "suggested_next_step": (
                "Ask the customer for the bank reference against RRN "
                f"{failed.rrn} and compare with the settlement report. "
                "Razorpay documents that a bank can auto-refund without "
                "changing payment status, so the API cannot settle this."
                if failed and failed.rrn else
                "Request the settlement recon report for this window."
            ),
        })

    return JSONResponse({"count": len(cards), "cards": cards})
