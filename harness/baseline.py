"""The naive recovery agent - v0. This is the number we have to beat.

It is written the way most recovery agents are actually written, and not
strawmanned: a mutable status field updated in arrival order, and a
payment link on every `payment.failed`. Nothing here is incompetent. It
is simply built on the assumption that `payment.failed` is terminal.

Its output is the "before" column of every metric, and its failure mode
is REJECTED.md entry v1.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from core.events import Observation
from harness import scenarios as sc


@dataclass
class BaselineAction:
    order_id: str
    action: str
    reason: str
    amount: int = 0
    #: True when this action created a second order for money that had
    #: already moved. Razorpay's per-order clubbing cannot catch it,
    #: because the link is a *new* order.
    duplicate: bool = False


@dataclass
class BaselineRun:
    actions: list[BaselineAction] = field(default_factory=list)
    #: order_id -> status, mutated in arrival order. The bug, in one field.
    status: dict[str, str] = field(default_factory=dict)

    @property
    def duplicates(self) -> list[BaselineAction]:
        return [a for a in self.actions if a.duplicate]

    @property
    def duplicate_amount(self) -> int:
        return sum(a.amount for a in self.duplicates)


def run_baseline(obs: list[Observation], truth_settled: bool) -> BaselineRun:
    """Process observations in ARRIVAL order, acting on each failure.

    `truth_settled` is only used to label the outcome afterwards - the
    baseline itself has no way to know it, which is precisely the point.
    """
    run = BaselineRun()
    for o in sorted(obs, key=lambda x: x.received_at):
        if o.status:
            run.status[o.order_id] = o.status.upper()   # last write wins

        if o.event_type != "payment.failed":
            continue

        # No sibling check, no wait, no evidence. Fire the link.
        run.actions.append(
            BaselineAction(
                order_id=o.order_id,
                action="SEND_RECOVERY_LINK",
                reason="payment.failed received",
                amount=o.amount,
                duplicate=truth_settled,
            )
        )
    return run


def run_all() -> dict:
    """Every scenario through the naive agent. Prints the before-number."""
    summary: dict[str, dict] = {}
    total_dupes = 0
    total_amount = 0

    for s in sc.ALL:
        obs = s.observations()
        settled = s.ground_truth.value == "ORDER_SETTLED"
        run = run_baseline(obs, truth_settled=settled)
        n = len(run.duplicates)
        total_dupes += n
        total_amount += run.duplicate_amount
        summary[s.key] = {
            "title": s.title,
            "final_status_field": run.status.get(s.order_id, "NONE"),
            "truth": s.ground_truth.value,
            "actions": [a.action for a in run.actions],
            "duplicate_orders": n,
            "duplicate_paise": run.duplicate_amount,
        }

    summary["_totals"] = {
        "duplicate_orders_created": total_dupes,
        "duplicate_paise": total_amount,
        "duplicate_rupees": total_amount / 100,
    }
    return summary


def main() -> None:
    from rich.console import Console
    from rich.table import Table

    con = Console()
    s = run_all()

    t = Table(title="BASELINE (v0) - naive recovery agent", header_style="bold")
    t.add_column("Sc")
    t.add_column("Scenario")
    t.add_column("status field")
    t.add_column("ground truth")
    t.add_column("actions")
    t.add_column("dupes", justify="right")

    for key in ("A", "B", "C", "D", "E", "F"):
        r = s[key]
        bad = r["duplicate_orders"] > 0
        t.add_row(
            key,
            r["title"],
            f"[red]{r['final_status_field']}[/red]" if bad else r["final_status_field"],
            r["truth"],
            ", ".join(r["actions"]) or "-",
            f"[red]{r['duplicate_orders']}[/red]" if bad else "0",
        )
    con.print(t)

    tot = s["_totals"]
    con.print(
        f"\n[bold red]duplicate orders created: {tot['duplicate_orders_created']}[/bold red]"
        f"   [red]Rs {tot['duplicate_rupees']:,.2f} charged twice[/red]"
    )
    con.print("[dim]This is the before-number. Nishchay's target for both is 0.[/dim]")
    print("\n" + json.dumps(s["_totals"], indent=2))


if __name__ == "__main__":
    main()
