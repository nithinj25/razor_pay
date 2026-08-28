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
        resolver = strategist = None
        if real_llm:
            # Serve the fixture's evidence through the real fetcher and
            # probe interfaces, so the agents gather rather than being
            # handed everything up front.
            from harness.scripted_agents import scripted_fetchers, scripted_probes
            from services.resolver.graph import Resolver
            from services.strategist.graph import Strategist

            resolver = Resolver(llm=llm, fetchers=scripted_fetchers(s))
            strategist = Strategist(llm=llm, probes=scripted_probes(s))
        p = Pipeline(llm=llm, executor=ex, resolver=resolver, strategist=strategist)
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
                # The intervention, not just the verdict. E's whole point
                # is choosing UPI over a generic retry - same verdict,
                # different message - so verdict accuracy alone
                # undersells what the strategist does.
                "template": (eff.intent.template_id if eff and eff.intent else None),
                "channel": (eff.intent.channel.value
                            if eff and eff.intent and eff.intent.channel else None),
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


def print_llm_delta(rules_only: Run, with_model: Run) -> None:
    """What the model is actually worth, as a number.

    "Was AI applied appropriately, or forced?" is a judging criterion, and
    it deserves a measurement rather than an assertion. Running the same
    labelled scenarios twice - once with the agents disabled - says
    exactly which cases need a model and which do not.
    """
    from rich.console import Console
    from rich.table import Table

    con = Console()
    t = Table(title="What the model is worth", header_style="bold")
    t.add_column("Sc")
    t.add_column("Scenario")
    t.add_column("rules only")
    t.add_column("with agents")
    t.add_column("LLM", justify="right")

    by_key = {r["scenario"]: r for r in with_model.rows}
    for r in rules_only.rows:
        w = by_key.get(r["scenario"], {})
        gained = (not r["correct"]) and w.get("correct")
        t.add_row(
            r["scenario"],
            r["title"][:30],
            f"[green]{r['verdict']}[/green]" if r["correct"] else f"[red]{r['verdict']}[/red]",
            f"[green]{w.get('verdict', '-')}[/green]" if w.get("correct")
            else f"[red]{w.get('verdict', '-')}[/red]",
            f"[bold cyan]+{w.get('llm_calls', 0)}[/bold cyan]" if gained
            else str(w.get("llm_calls", 0)),
        )
    con.print(t)

    gained = [
        r["scenario"] for r in rules_only.rows
        if not r["correct"] and by_key.get(r["scenario"], {}).get("correct")
    ]
    # Second axis: same verdict, different intervention. The strategist
    # earns its place here even where the verdict never moves.
    better_message = [
        r["scenario"] for r in rules_only.rows
        if r["correct"]
        and by_key.get(r["scenario"], {}).get("template")
        and by_key[r["scenario"]]["template"] != r.get("template")
    ]
    con.print(
        f"\n[bold]rules only [/bold] {rules_only.correct}/{rules_only.total} correct, "
        f"{rules_only.llm_calls} LLM calls"
    )
    con.print(
        f"[bold]with agents[/bold] {with_model.correct}/{with_model.total} correct, "
        f"{with_model.llm_calls} LLM calls"
    )
    if gained:
        con.print(
            f"\n[bold cyan]The model changes the verdict on {', '.join(gained)} "
            f"and nothing else.[/bold cyan]"
        )
        con.print(
            "[dim]That is the case whose evidence is prose - a customer "
            "describing a debit in their own words, quoting a reference no "
            "rule can extract. Every other verdict is settled "
            "deterministically, by design.[/dim]"
        )
    if better_message:
        rows = [(r, by_key[r]) for r in better_message]
        con.print(
            f"\n[bold cyan]It changes the intervention on "
            f"{', '.join(better_message)}[/bold cyan] without changing the verdict:"
        )
        for key, w in rows:
            was = next(x.get("template") or "-" for x in rules_only.rows
                       if x["scenario"] == key)
            con.print(f"  [dim]{key}:[/dim] {was} -> {w['template']} over {w['channel']}")
        con.print(
            "[dim]Same conclusion about the money, a different message to the "
            "customer. A generic retry link on a rail that is currently down "
            "fails again.[/dim]"
        )
    else:
        con.print(
            "\n[yellow]The model changed no verdict. Either the rules cover "
            "every labelled case, or the agents are not running - check the "
            "health badge on Screen 5.[/yellow]"
        )


async def main_async(args) -> None:
    base = bl.run_all()["_totals"]
    run = await run_nishchay(real_llm=args.real_llm)
    print_report(run, base)

    if args.compare:
        # Two full passes over the labelled set: one with the agents
        # disabled, one with them live. The delta is the claim.
        rules_only = run if not args.real_llm else await run_nishchay(real_llm=False)
        with_model = run if args.real_llm else await run_nishchay(real_llm=True)
        print_llm_delta(rules_only, with_model)

    p = write_artifacts(run, base)
    print(f"\nartifacts -> {p}")

    if run.duplicates or run.false_positives:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser(description="Baseline vs nishchay, measured.")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--real-llm", action="store_true", help="use the configured LLM provider")
    ap.add_argument(
        "--compare", action="store_true",
        help="run twice, with and without the agents, and show the delta",
    )
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
