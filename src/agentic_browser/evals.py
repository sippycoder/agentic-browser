"""Eval harness: Odysseys-style rubric judging on live-web tasks.

A task = start URL + goal + rubrics. The agent runs, the full trajectory
(screenshots + actions) is kept, and a judge model grades EACH rubric
independently from the trajectory — exactly like the Odysseys per-rubric
trajectory judge. A task is "perfect" only if every rubric passes.

This is the measurement loop Polar's team treats as a first-class R&D surface
("browser-agent evals" is one of their listed frontier problems). v0 ships with
three small live-web tasks in evals/tasks/.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .agent import BrowserAgent
from .browser import BrowserSession
from .memory import SecondBrain
from .models import ModelMessage, Router

JUDGE_PROMPT = """You are grading a browser agent's work against ONE rubric item.

Rubric requirement: {requirement}
Verification notes: {notes}

The agent was given this goal: {goal}

Trajectory (each step: the action taken and the page URL; screenshots follow):
{trajectory}

Examine the trajectory and the screenshots. Reply with ONLY a JSON object:
{{"pass": true/false, "reasoning": "one or two sentences citing what you saw"}}
A rubric passes only if the trajectory shows the requirement was actually met —
not merely attempted.
"""


@dataclass
class Rubric:
    id: str
    requirement: str
    notes: str = ""


@dataclass
class EvalTask:
    id: str
    start_url: str
    goal: str
    rubrics: list[Rubric] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, path: Path) -> "EvalTask":
        data = yaml.safe_load(path.read_text())
        start_url = data["start_url"]
        # M5: fixture:// URLs resolve to the repo's hermetic eval fixtures,
        # so evals never depend on the live web changing under them.
        if start_url.startswith("fixture://"):
            fixture = Path(__file__).resolve().parents[2] / "evals" / "fixtures" / start_url[len("fixture://"):]
            start_url = fixture.as_uri()
        return cls(
            id=data["id"],
            start_url=start_url,
            goal=data["goal"],
            rubrics=[Rubric(**r) for r in data.get("rubrics", [])],
        )


@dataclass
class TaskGrade:
    task_id: str
    answer: str
    steps: int
    finished: bool
    rubric_results: list[dict]
    perfect: bool
    elapsed_s: float


def _trajectory_text(traj_dir: Path, router: Router) -> tuple[str, list[str]]:
    """Compact step list + the last few screenshots for the judge."""
    lines: list[str] = []
    images: list[str] = []
    log = traj_dir / "trajectory.jsonl"
    steps: list[dict] = []
    if log.exists():
        for line in log.read_text().splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("type"):  # M5: skip meta/tool_result/result records
                continue
            steps.append(e)
    for s in steps:
        thought = (s.get("thought") or "")[:200].replace("\n", " ")
        lines.append(
            f"step {s['step']}: {s['tool']} {json.dumps(s.get('args', {}))[:160]} "
            f"[{s.get('url', '')[:70]}] thought: {thought}"
        )
    pngs = sorted(traj_dir.glob("step-*.png"))
    for p in pngs[-4:]:  # last 4 screenshots, like a trajectory judge would see
        import base64 as _b64

        images.append(_b64.b64encode(p.read_bytes()).decode())
    return "\n".join(lines), images


def grade_task(
    task: EvalTask, result_answer: str, traj_dir: Path, router: Router
) -> TaskGrade:
    t0 = time.time()
    traj_text, images = _trajectory_text(traj_dir, router)
    rubric_results = []
    for rubric in task.rubrics:
        resp = router.generate(
            "judge",
            [
                ModelMessage(
                    role="system",
                    text=JUDGE_PROMPT.format(
                        requirement=rubric.requirement,
                        notes=rubric.notes or "(none)",
                        goal=task.goal,
                        trajectory=traj_text or "(no steps recorded)",
                    ),
                ),
                ModelMessage(
                    role="user",
                    text=f"Agent's final answer was: {result_answer[:1000]}\n\nGrade rubric '{rubric.id}' now.",
                    images=images,
                ),
            ],
            max_tokens=1024,
            tag=f"eval:{task.id}:judge",
        )
        text = resp.text.strip()
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[1] if "\n" in text else text
            text = text.lstrip()
            if text.startswith("json"):
                text = text[4:]
        try:
            verdict = json.loads(text)
            passed = bool(verdict.get("pass"))
            reasoning = str(verdict.get("reasoning", ""))[:300]
        except Exception:
            passed, reasoning = False, f"unparseable judge output: {text[:200]}"
        rubric_results.append(
            {"rubric": rubric.id, "pass": passed, "reasoning": reasoning}
        )
    return TaskGrade(
        task_id=task.id,
        answer=result_answer,
        steps=0,
        finished=True,
        rubric_results=rubric_results,
        perfect=all(r["pass"] for r in rubric_results),
        elapsed_s=time.time() - t0,
    )


def run_eval(
    tasks_dir: Path,
    task_ids: list[str] | None = None,
    headless: bool = True,
    max_steps: int = 25,
    prompt_pack: str | None = None,
) -> list[TaskGrade]:
    router = Router()
    brain = SecondBrain(Path("brain-eval"))
    grades: list[TaskGrade] = []
    tasks = [EvalTask.from_yaml(p) for p in sorted(tasks_dir.glob("*.yaml"))]
    if task_ids:
        tasks = [t for t in tasks if t.id in task_ids]
    for task in tasks:
        session = BrowserSession(headless=headless).start()
        try:
            agent = BrowserAgent(
                session=session,
                brain=brain,
                router=router,
                model_role="worker",
                max_steps=max_steps,
                name=f"eval-{task.id}",
                trajectory_root=Path("trajectories") / "evals",
                auto_approve=True,  # evals run unattended; gates stay in the trajectory log
                prompt_pack=prompt_pack,
            )
            result = agent.run(task.goal, start_url=task.start_url)
            grade = grade_task(task, result.answer, result.trajectory_dir, router)
            grade.steps = result.steps
            grade.finished = result.finished
            grades.append(grade)
        finally:
            session.close()
    return grades


def report(grades: list[TaskGrade]) -> str:
    lines = ["# Eval report", ""]
    for g in grades:
        status = "PERFECT" if g.perfect else "FAIL"
        lines.append(f"## {g.task_id}: {status} ({g.steps} steps, {g.elapsed_s:.0f}s)")
        for r in g.rubric_results:
            mark = "✓" if r["pass"] else "✗"
            lines.append(f"  {mark} {r['rubric']}: {r['reasoning']}")
        lines.append(f"  answer: {g.answer[:300]}")
        lines.append("")
    perfect = sum(1 for g in grades if g.perfect)
    lines.append(f"**Perfect rate: {perfect}/{len(grades)}**")
    return "\n".join(lines)
