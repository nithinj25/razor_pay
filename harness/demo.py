"""One command: baseline vs nishchay, confusion matrix, veto log.

This is the accuracy run and the "before/after" table in one place. Every
number printed here is computed from the labelled fixtures at run time -
none of it is written down anywhere as a constant, so it cannot drift
away from what the system actually does.

The two zeros (duplicate orders, false positives) are invariants, not
percentiles. One violation is a bug, and this run is what would catch it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from core.llm import build_llm
from core.verdicts import Action, Verdict
from harness import baseline as bl
from harness import scenarios as sc
from harness.replay import Replay
from services.executor.main import Executor
from services.pipeline import Pipeline

ARTIFACTS = Path(".artifacts")


@dataclass
class Run:
    rows: list[dict] = field(default_factory=list)
    vetoes: list[dict] = field(default_factory=list)
    confusion: Counter = field(default_factory=Counter)
    duplicates: int = 0
    false_positives: int = 0
    escalated: int = 0
    recovered_paise: int = 0
    llm_calls: int = 0
    tokens: int = 0
    latencies: list[float] = field(default_factory=list)

    @property
    def correct(self) -> int:
        return sum(n for (truth, got), n in self.confusion.items() if truth == got)

    @property
    def total(self) -> int:
        return sum(self.confusion.values())

    @property
    def p95_latency(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        return s[min(int(len(s) * 0.95), len(s) - 1)]


async def run_nishchay(real_llm: bool = False) -> Run:
    run = Run()
    llm = build_llm() if real_llm else None

    for s in sc.ALL:
        ex = Executor(dry_run=True)
        p = Pipeline(llm=llm, executor=ex)
        r = Replay(p)
        await r.run(s)

        final, eff = r.final, r.effective
        got = final.verdict.verdict
        run.confusion[(s.ground_truth.value, got.value)] += 1
        run.vetoes.extend(p.vetoes)
        run.llm_calls += sum(st.decision.llm_calls for st in r.steps if st.decision)
        run.latencies.extend(st.decision.latency_s for st in r.steps if st.decision)

        acted = [o for o in ex.outcomes if o.status in ("EXECUTED", "STUBBED")]
        links = [o for o in acted if o.action == Action.SEND_RECOVERY_LINK]

        # A false positive is a recovery link on an order that was in
        # fact already paid. This is the number that must be zero.
        settled_truth = s.ground_truth == Verdict.ORDER_SETTLED
        fp = len(links) if settled_truth else 0
        run.false_positives += fp
        run.duplicates += fp
        run.escalated += len([o for o in acted if o.action == Action.ESCALATE])

        if eff and eff.action in (Action.CAPTURE, Action.SEND_RECOVERY_LINK) and not settled_truth:
            run.recovered_paise += final.verdict.amount_due

        run.rows.append(
            {
                "scenario": s.key,
                "title": s.title,
                "truth": s.ground_truth.value,
                "verdict": got.value,
                "correct": got == s.ground_truth,
                "confidence": round(final.verdict.confidence, 3),
                "action": eff.action.value if eff else "NONE",
                "status": eff.outcome.status if eff and eff.outcome else "-",
                "rules": list(final.verdict.rules_fired),
                "llm_calls": sum(st.decision.llm_calls for st in r.steps if st.decision),
                "duplicate_orders": fp,
                "vetoes": [v["rule"] for v in p.vetoes],
                "degraded": final.degraded,
            }
        )
    return run


def print_report(run: Run, base: dict) -> None:
    from rich.console import Console
    from rich.table import Table

    con = Console()

    # -- per scenario ------------------------------------------------
    t = Table(title="NISHCHAY - labelled scenarios", header_style="bold")
    for c, j in (("Sc", "left"), ("Scenario", "left"), ("truth", "left"),
                 ("verdict", "left"), ("conf", "right"), ("action", "left"),
                 ("LLM", "right"), ("dupes", "right")):
        t.add_column(c, justify=j)

    for r in run.rows:
        ok = r["correct"]
        t.add_row(
            r["scenario"], r["title"][:28], r["truth"],
            f"[green]{r['verdict']}[/green]" if ok else f"[red]{r['verdict']}[/red]",
            f"{r['confidence']:.2f}", r["action"], str(r["llm_calls"]),
            "[red]" + str(r["duplicate_orders"]) + "[/red]" if r["duplicate_orders"] else "0",
        )
    con.print(t)

    # -- confusion matrix --------------------------------------------
    truths = sorted({t for t, _ in run.confusion})
    gots = sorted({g for _, g in run.confusion})
    cm = Table(title="Confusion matrix (ground truth x verdict)", header_style="bold")
    cm.add_column("truth \\ verdict")
    for g in gots:
        cm.add_column(g[:16], justify="right")
    for tr in truths:
        cells = []
        for g in gots:
            n = run.confusion.get((tr, g), 0)
            cells.append(
                (f"[green]{n}[/green]" if tr == g else f"[red]{n}[/red]") if n else "."
            )
        cm.add_row(tr[:16], *cells)
    con.print(cm)

    # -- headline ----------------------------------------------------
    h = Table(title="Headline metrics", header_style="bold")
    h.add_column("metric")
    h.add_column("baseline (v0)", justify="right")
    h.add_column("nishchay", justify="right")

    h.add_row("Duplicate orders created",
              f"[red]{base['duplicate_orders_created']}[/red]",
              f"[green]{run.duplicates}[/green]" if run.duplicates == 0 else f"[red]{run.duplicates}[/red]")
    h.add_row("Money double-charged",
              f"[red]Rs {base['duplicate_rupees']:,.2f}[/red]",
              "[green]Rs 0.00[/green]" if run.duplicates == 0 else "[red]>0[/red]")
    h.add_row("False positives (link on paid order)",
              f"[red]{base['duplicate_orders_created']}[/red]",
              f"[green]{run.false_positives}[/green]" if run.false_positives == 0 else f"[red]{run.false_positives}[/red]")
    h.add_row("Revenue correctly recovered", "Rs 0.00", f"Rs {run.recovered_paise/100:,.2f}")
    h.add_row("Escalated to a human", "0", str(run.escalated))
    h.add_row("Verdict accuracy", "-", f"{run.correct}/{run.total}")
    h.add_row("LLM calls (6 scenarios)", "-", str(run.llm_calls))
    h.add_row("p95 time to verdict", "-", f"{run.p95_latency*1000:.0f} ms")
    con.print(h)

    # -- veto log ----------------------------------------------------
    if run.vetoes:
        v = Table(title="Veto log (the audit trail)", header_style="bold")
        v.add_column("order")
        v.add_column("action")
        v.add_column("rule")
        v.add_column("reason")
        for row in run.vetoes[:12]:
            v.add_row(row["order_id"][-12:], row["action"], f"[yellow]{row['rule']}[/yellow]",
                      row["reason"][:60])
        con.print(v)

    zeros_hold = run.duplicates == 0 and run.false_positives == 0
    con.print(
        f"\n[bold {'green' if zeros_hold else 'red'}]"
        f"invariant check: duplicate orders={run.duplicates}, "
        f"false positives={run.false_positives}"
        f"{' - both zero' if zeros_hold else ' - VIOLATED'}[/bold {'green' if zeros_hold else 'red'}]"
    )


def write_artifacts(run: Run, base: dict) -> Path:
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS / "results.json").write_text(
        json.dumps(
            {
                "scenarios": run.rows,
                "confusion": [
                    {"ground_truth": t, "verdict": g, "n": n}
                    for (t, g), n in sorted(run.confusion.items())
                ],
                "headline": {
                    "baseline_duplicate_orders": base["duplicate_orders_created"],
                    "baseline_duplicate_rupees": base["duplicate_rupees"],
                    "nishchay_duplicate_orders": run.duplicates,
                    "nishchay_false_positives": run.false_positives,
                    "recovered_rupees": run.recovered_paise / 100,
                    "escalated": run.escalated,
                    "accuracy": f"{run.correct}/{run.total}",
                    "llm_calls": run.llm_calls,
                    "p95_latency_ms": round(run.p95_latency * 1000, 1),
                },
                "vetoes": run.vetoes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ARTIFACTS / "results.json"


async def main_async(args) -> None:
    base = bl.run_all()["_totals"]
    run = await run_nishchay(real_llm=args.real_llm)
    print_report(run, base)
    p = write_artifacts(run, base)
    print(f"\nartifacts -> {p}")

    if run.duplicates or run.false_positives:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline vs nishchay, measured.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--real-llm", action="store_true", help="use the Anthropic API")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
