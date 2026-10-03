"""Unit tests for M3 schedule math — no network, no API keys.

Run: python tests/test_scheduler.py
"""

import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_browser.workflows import (
    Workflow,
    describe_schedule,
    next_run,
    parse_duration,
)

LA = ZoneInfo("America/Los_Angeles")


def dt(s: str) -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=LA)


def test_every():
    assert next_run({"every": "6h"}, dt("2026-10-03T09:00:00"), LA) == dt("2026-10-03T15:00:00")
    assert next_run({"every": "30m"}, dt("2026-10-03T09:00:00"), LA) == dt("2026-10-03T09:30:00")
    assert parse_duration("1d").days == 1
    print("ok: every")


def test_daily():
    # after 07:00 -> tomorrow 07:00
    assert next_run({"daily": "07:00"}, dt("2026-10-03T09:00:00"), LA) == dt("2026-10-04T07:00:00")
    # before 07:00 -> today 07:00
    assert next_run({"daily": "07:00"}, dt("2026-10-03T06:00:00"), LA) == dt("2026-10-03T07:00:00")
    print("ok: daily")


def test_weekly():
    # 2026-10-02 is a Friday; next Mon 09:00 is 2026-10-05
    nxt = next_run({"weekly": {"days": ["mon"], "at": "09:00"}}, dt("2026-10-02T10:00:00"), LA)
    assert (nxt.year, nxt.month, nxt.day, nxt.hour) == (2026, 10, 5, 9), nxt
    # Monday 08:00 -> same day 09:00
    nxt = next_run({"weekly": {"days": ["mon", "wed"], "at": "09:00"}}, dt("2026-10-05T08:00:00"), LA)
    assert (nxt.month, nxt.day, nxt.hour) == (10, 5, 9), nxt
    # Monday 10:00 -> Wednesday 09:00
    nxt = next_run({"weekly": {"days": ["mon", "wed"], "at": "09:00"}}, dt("2026-10-05T10:00:00"), LA)
    assert (nxt.month, nxt.day, nxt.hour) == (10, 7, 9), nxt
    print("ok: weekly")


def test_cron():
    # Mondays 09:00; from Friday -> Monday
    nxt = next_run({"cron": "0 9 * * 1"}, dt("2026-10-02T10:00:00"), LA)
    assert (nxt.month, nxt.day, nxt.hour, nxt.weekday()) == (10, 5, 9, 0), nxt
    print("ok: cron")


def test_never_run_is_due_now():
    before = datetime.now(LA)
    nxt = next_run({"daily": "07:00"}, None, LA)
    after = datetime.now(LA)
    assert before <= nxt <= after
    print("ok: never-run is due immediately")


def test_describe_and_load():
    assert describe_schedule({"every": "6h"}) == "every 6h"
    assert describe_schedule({"daily": "07:00"}) == "daily at 07:00"
    tdir = Path(__file__).resolve().parent.parent / "templates"
    wf = Workflow.from_yaml(tdir / "tech-briefing.yaml")
    assert wf.name == "tech-briefing"
    assert wf.use_orchestrator is True
    assert wf.workers == 2
    print("ok: describe + template load")


if __name__ == "__main__":
    test_every()
    test_daily()
    test_weekly()
    test_cron()
    test_never_run_is_due_now()
    test_describe_and_load()
    print("\nAll scheduler tests passed.")
