"""M3: scheduled recurring workflows.

Polar's headline feature: "Schedule workflows… hourly, daily, weekly, or
custom schedules" that "carry the same context into every run." Workflows are
YAML files in workflows/. The scheduler (``agentic_browser schedule``) fires
due workflows, runs them through the orchestrator/agent, and persists results
plus cross-run memory in the shared brain so each run builds on the last.

Schedule shapes (one per workflow):
    schedule: {every: "30m" | "6h" | "1d"}     # fixed interval
    schedule: {daily: "07:00"}                  # every day at HH:MM
    schedule: {weekly: {days: [mon, wed], at: "09:00"}}
    schedule: {cron: "0 9 * * 1-5"}            # needs the `croniter` package

Times are interpreted in the workflow's timezone (``timezone:`` field,
AGENTIC_TZ env, or the system local zone).

Safety: scheduled runs are unattended, so they ALWAYS run with auto_approve.
Keep scheduled workflows to low-stakes, read-only tasks.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from .agent import BrowserAgent
from .browser import BrowserSession
from .memory import SecondBrain
from .models import Router
from .orchestrator import Orchestrator

DURATION_RE = re.compile(r"^\s*(\d+)\s*([mhd])\s*$")
DAY_INDEX = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}


def resolve_tz(name: str | None):
    """Resolve a timezone name to a tzinfo. Falls back to AGENTIC_TZ, then the
    system local zone (used as-is — abbreviations like 'PDT' are not valid
    IANA keys, so we never round-trip through a string)."""
    from datetime import tzinfo

    if name:
        return ZoneInfo(name)
    env = os.environ.get("AGENTIC_TZ")
    if env:
        return ZoneInfo(env)
    local = datetime.now().astimezone().tzinfo
    return local if isinstance(local, tzinfo) else ZoneInfo("UTC")


def parse_duration(s: str) -> timedelta:
    m = DURATION_RE.match(s)
    if not m:
        raise ValueError(f"Bad duration {s!r}: expected like '30m', '6h', '1d'.")
    n, unit = int(m.group(1)), m.group(2)
    return {"m": timedelta(minutes=n), "h": timedelta(hours=n), "d": timedelta(days=n)}[unit]


def _parse_hhmm(s: str) -> tuple[int, int]:
    h, m = s.split(":")
    return int(h), int(m)


def next_run(schedule: dict, after: datetime | None, tz: ZoneInfo) -> datetime:
    """Next fire time strictly after `after` (naive `after` assumed in tz).

    `after=None` means "never ran" -> due immediately (returns now).
    """
    now = datetime.now(tz)
    if after is None:
        return now
    if after.tzinfo is None:
        after = after.replace(tzinfo=tz)

    if "every" in schedule:
        return after + parse_duration(str(schedule["every"]))

    if "daily" in schedule:
        h, m = _parse_hhmm(str(schedule["daily"]))
        cand = after.replace(hour=h, minute=m, second=0, microsecond=0)
        if cand <= after:
            cand += timedelta(days=1)
        return cand

    if "weekly" in schedule:
        spec = schedule["weekly"]
        days = [DAY_INDEX[d.lower()[:3]] for d in spec["days"]]
        h, m = _parse_hhmm(str(spec["at"]))
        best = None
        for d in days:
            delta = (d - after.weekday()) % 7
            cand = (after + timedelta(days=delta)).replace(hour=h, minute=m, second=0, microsecond=0)
            if cand <= after:
                cand += timedelta(days=7)
            best = cand if best is None or cand < best else best
        return best  # type: ignore[return-value]

    if "cron" in schedule:
        try:
            from croniter import croniter
        except ImportError:
            raise RuntimeError("A workflow uses cron: but `croniter` is not installed (pip install croniter).")
        return croniter(str(schedule["cron"]), after).get_next(datetime)

    raise ValueError(f"Unknown schedule shape: {schedule!r}")


def describe_schedule(schedule: dict) -> str:
    if "every" in schedule:
        return f"every {schedule['every']}"
    if "daily" in schedule:
        return f"daily at {schedule['daily']}"
    if "weekly" in schedule:
        spec = schedule["weekly"]
        return f"weekly on {', '.join(spec['days'])} at {spec['at']}"
    if "cron" in schedule:
        return f"cron {schedule['cron']}"
    return str(schedule)


@dataclass
class Workflow:
    name: str
    task: str
    schedule: dict
    description: str = ""
    timezone: str | None = None
    use_orchestrator: bool = False
    workers: int = 2
    max_steps: int = 30
    vars: dict = field(default_factory=dict)
    profile: str | None = None

    @classmethod
    def from_yaml(cls, path: Path) -> "Workflow":
        data = yaml.safe_load(path.read_text())
        data["name"] = data.get("name") or path.stem
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

    def render_task(self) -> str:
        try:
            return self.task.format(**self.vars)
        except KeyError as e:
            raise ValueError(f"Workflow {self.name!r}: task references undefined var {e}")


def load_workflows(workflows_dir: Path) -> list[Workflow]:
    workflows_dir = Path(workflows_dir)
    out = []
    for p in sorted(workflows_dir.glob("*.yaml")) + sorted(workflows_dir.glob("*.yml")):
        if p.name.startswith("."):
            continue
        out.append(Workflow.from_yaml(p))
    return out


class Scheduler:
    """Finds due workflows, runs them unattended, records history in the brain."""

    def __init__(
        self,
        workflows_dir: str | Path,
        brain: SecondBrain,
        router: Router,
        headless: bool = True,
        state_path: str | Path | None = None,
    ) -> None:
        self.workflows_dir = Path(workflows_dir)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self.brain = brain
        self.router = router
        self.headless = headless
        self.state_path = Path(state_path or self.workflows_dir / ".scheduler-state.json")
        self.state: dict[str, str] = {}
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text())
            except json.JSONDecodeError:
                self.state = {}

    def _save_state(self) -> None:
        self.state_path.write_text(json.dumps(self.state, indent=2))

    def _last_run(self, wf: Workflow) -> datetime | None:
        iso = self.state.get(wf.name)
        if not iso:
            return None
        dt = datetime.fromisoformat(iso)
        return dt

    def due_workflows(self, now: datetime | None = None) -> list[tuple[Workflow, datetime]]:
        due = []
        for wf in load_workflows(self.workflows_dir):
            tz = resolve_tz(wf.timezone)
            now_tz = (now or datetime.now(tz)).astimezone(tz)
            last = self._last_run(wf)
            if last is None:
                due.append((wf, now_tz))  # never ran -> due immediately
                continue
            if now_tz >= next_run(wf.schedule, last, tz):
                due.append((wf, next_run(wf.schedule, last, tz)))
        return due

    def run_workflow(self, wf: Workflow, headless: bool | None = None) -> str:
        """Execute one workflow now (auto-approved: no human is watching)."""
        tz = resolve_tz(wf.timezone)
        now = datetime.now(tz)
        run_id = now.strftime("%Y%m%d-%H%M%S")
        task_text = (
            f"[Scheduled run of workflow \"{wf.name}\" — {describe_schedule(wf.schedule)}. "
            f"Persistent memory for this workflow is at brain path "
            f"\"workflows/{wf.name}/memory/\". Start by reading "
            f"\"workflows/{wf.name}/memory/last-run.md\" for context from the previous run.]\n\n"
            + wf.render_task()
        )
        print(f"[scheduler] running workflow '{wf.name}' (run {run_id})")
        headless = self.headless if headless is None else headless
        t0 = time.time()
        try:
            if wf.use_orchestrator:
                orch = Orchestrator(
                    brain=self.brain,
                    router=self.router,
                    max_workers=wf.workers,
                    headless=headless,
                    trajectory_root=Path("trajectories") / "workflows" / wf.name,
                    auto_approve=True,
                )
                answer = orch.run(task_text)
                steps_info = "orchestrated"
            else:
                session = BrowserSession(headless=headless, profile_dir=wf.profile).start()
                try:
                    agent = BrowserAgent(
                        session=session,
                        brain=self.brain,
                        router=self.router,
                        model_role="worker",
                        max_steps=wf.max_steps,
                        name=f"workflow-{wf.name}",
                        trajectory_root=Path("trajectories") / "workflows" / wf.name,
                        auto_approve=True,
                    )
                    result = agent.run(task_text)
                    answer = result.answer or "(no answer — agent did not finish)"
                    steps_info = f"{result.steps} steps"
                finally:
                    session.close()
        except Exception as e:
            answer = f"WORKFLOW FAILED: {type(e).__name__}: {e}"
            steps_info = "failed"
        elapsed = time.time() - t0

        base = f"workflows/{wf.name}/runs/{run_id}"
        self.brain.write(
            f"{base}/result.md",
            f"# {wf.name} — run {run_id}\n\n"
            f"Schedule: {describe_schedule(wf.schedule)}\n"
            f"Finished: {now.isoformat()} ({elapsed:.0f}s, {steps_info})\n\n{answer}\n",
        )
        self.brain.write(
            f"workflows/{wf.name}/memory/last-run.md",
            f"# Last run: {run_id} ({now.isoformat()})\n\n{answer[:3000]}\n",
        )
        self.state[wf.name] = now.isoformat()
        self._save_state()
        print(f"[scheduler] workflow '{wf.name}' done in {elapsed:.0f}s")
        return answer

    def run_once(self) -> list[str]:
        """Fire all due workflows right now. Returns their answers."""
        answers = []
        for wf, _ in self.due_workflows():
            answers.append(self.run_workflow(wf))
        if not answers:
            print("[scheduler] nothing due")
        return answers

    def loop(self, poll_seconds: int = 60) -> None:
        print(f"[scheduler] watching {self.workflows_dir} (poll every {poll_seconds}s) — Ctrl+C to stop")
        try:
            while True:
                try:
                    self.run_once()
                except Exception as e:
                    print(f"[scheduler] error: {type(e).__name__}: {e}")
                time.sleep(poll_seconds)
        except KeyboardInterrupt:
            print("\n[scheduler] stopped")
