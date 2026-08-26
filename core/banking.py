"""Banking-day arithmetic in IST. Pure — no clock reads, no I/O.

E14 / I10. RBI's harmonised TAT is "T+1 **day**", and a day here is a
*banking* day, not 86400 seconds. A Friday-evening failure ahead of a
long weekend has a window roughly 3.5x longer than `now + 86400`, and an
agent using the naive form acts while the bank is still going to reverse
the debit on its own. That is the single most expensive off-by-one in
this system: it produces a duplicate charge that looks like a bug in our
logic rather than in our calendar.

Indian bank working days follow RBI's actual rule, which is not
"weekdays":

  * every Sunday is closed;
  * the **2nd and 4th Saturday** of each month are closed, while the
    1st, 3rd and 5th Saturdays are ordinary working days;
  * plus the gazetted holiday list.

Modelling the Saturday rule properly matters — treating all Saturdays as
closed would push deadlines late (safe but wrong), and treating none as
closed would push them early (a duplicate charge).
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30), "IST")

#: RBI-observed national bank holidays. State-specific holidays are
#: deliberately excluded: this is a single-merchant demo, and an
#: over-broad calendar only ever delays action (safe), never advances it.
#: Extend per-merchant in production.
NATIONAL_HOLIDAYS_2026: frozenset[date] = frozenset(
    {
        date(2026, 1, 1),    # New Year / annual closing (most states)
        date(2026, 1, 26),   # Republic Day (Monday)
        date(2026, 3, 4),    # Holi
        date(2026, 4, 1),    # Bank annual closing of accounts
        date(2026, 4, 3),    # Good Friday
        date(2026, 5, 1),    # May Day
        date(2026, 8, 15),   # Independence Day
        date(2026, 10, 2),   # Gandhi Jayanti
        date(2026, 11, 8),   # Diwali / Laxmi Pujan
        date(2026, 12, 25),  # Christmas
    }
)

DEFAULT_HOLIDAYS = NATIONAL_HOLIDAYS_2026


def ist_datetime(ts: int) -> datetime:
    """Epoch seconds → IST-aware datetime."""
    return datetime.fromtimestamp(ts, tz=IST)


def ist_date(ts: int) -> date:
    return ist_datetime(ts).date()


def nth_weekday_of_month(d: date) -> int:
    """Which occurrence of its weekday `d` is within its month (1-based).

    The 24th of a month that falls on a Saturday is the 4th Saturday iff
    this returns 4.
    """
    return (d.day - 1) // 7 + 1


def is_banking_day(d: date, holidays: frozenset[date] = DEFAULT_HOLIDAYS) -> bool:
    """True if Indian banks settle on this calendar date."""
    if d.weekday() == 6:                       # Sunday
        return False
    if d.weekday() == 5 and nth_weekday_of_month(d) in (2, 4):
        return False                           # 2nd / 4th Saturday
    return d not in holidays


def next_banking_day(d: date, holidays: frozenset[date] = DEFAULT_HOLIDAYS) -> date:
    """The first banking day strictly after `d`."""
    probe = d + timedelta(days=1)
    for _ in range(60):                        # guard against a pathological calendar
        if is_banking_day(probe, holidays):
            return probe
        probe += timedelta(days=1)
    raise RuntimeError(f"no banking day within 60 days of {d} — check the calendar")


def add_banking_days(
    ts: int, n: int, holidays: frozenset[date] = DEFAULT_HOLIDAYS
) -> date:
    """Advance `n` banking days from the IST date of `ts`.

    `n=1` on a Friday before a closed Saturday, a Sunday and a Monday
    holiday returns the Tuesday.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    d = ist_date(ts)
    for _ in range(n):
        d = next_banking_day(d, holidays)
    return d


def end_of_day_ist(d: date) -> int:
    """Epoch seconds at 23:59:59 IST on `d`.

    RBI requires reversal *by* end of T+1, so the window closes at the
    end of that banking day, not at the same clock time.
    """
    return int(datetime.combine(d, time(23, 59, 59), tzinfo=IST).timestamp())


def tat_deadline(
    failure_ts: int, banking_days: int = 1, holidays: frozenset[date] = DEFAULT_HOLIDAYS
) -> int:
    """Epoch second at which the RBI auto-reversal window closes.

    Until this instant a debit may still reverse itself, so acting is
    premature — the correct move is NOOP plus a scheduled re-fold.
    """
    return end_of_day_ist(add_banking_days(failure_ts, banking_days, holidays))


def banking_days_between(
    start_ts: int, end_ts: int, holidays: frozenset[date] = DEFAULT_HOLIDAYS
) -> int:
    """Count banking days elapsed in (start, end]. Negative spans give 0."""
    if end_ts <= start_ts:
        return 0
    d, end_d, n = ist_date(start_ts), ist_date(end_ts), 0
    while d < end_d:
        d += timedelta(days=1)
        if is_banking_day(d, holidays):
            n += 1
    return n


def within_tat_window(
    failure_ts: int,
    now: int,
    banking_days: int = 1,
    holidays: frozenset[date] = DEFAULT_HOLIDAYS,
) -> bool:
    """True while the bank may still reverse the debit unaided."""
    return now <= tat_deadline(failure_ts, banking_days, holidays)


def naive_deadline(failure_ts: int, days: int = 1) -> int:
    """The wrong answer, kept for the demo.

    Pitfall #8. Screen 2 shows this beside `tat_deadline` so the gap is
    visible rather than asserted.
    """
    return failure_ts + days * 86_400
