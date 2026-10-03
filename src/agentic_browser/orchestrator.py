"""Orchestrator: hierarchical planning + massively parallel subagents.

Mirrors Polar's disclosed harness: an orchestrator model decomposes the goal,
spawns worker subagents (each with its own BrowserSession — i.e. its own
"browser window"), coordinates through the shared SecondBrain, then merges the
workers' findings into a final answer. Synchronous UI micro-tasks can be routed
to the cheaper `micro` model role via the worker's model_role parameter.

M4: workers run behind a WorkerBackend interface. LocalBackend runs them in
threads against local Chromium (v0 behavior). HttpBackend POSTs subtasks to a
worker service (`agentic_browser serve`) — the honest foundation of a cloud
worker pool: same interface, remote execution. The production evolution is many
such services behind a queue + autoscaler.
"""

from __future__ import annotations

import abc
import json
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from .agent import BrowserAgent
from .browser import BrowserSession
from .memory import SecondBrain
from .models import ModelMessage, Router

PLAN_PROMPT = """You are the orchestrator of a browser-agent fleet. Decompose the user's
goal into independent subtasks that can be executed in parallel by worker
agents, each driving its own browser. Workers share a filesystem ("brain") for
coordination — tell them which brain paths to write findings to.

Reply with ONLY a JSON object:
{
  "subtasks": [
    {"id": "s1", "goal": "...", "start_url": "https://...", "brain_path": "tasks/<TASK>/notes/s1.md"},
    ...
  ],
  "merge_instructions": "How to combine the subtask findings into the final answer."
}
Keep it to at most 4 subtasks. If the goal is simple, use exactly 1 subtask.
"""

MERGE_PROMPT = """You are the orchestrator merging parallel browser-agent workers' findings.

Original goal:
{goal}

Merge instructions:
{merge_instructions}

Worker findings:
{findings}

Produce the final answer to the original goal. Be concrete and cite the key facts."""


@dataclass
class WorkerContext:
    """Everything a worker needs; carried to local threads or HTTP services."""

    brain: SecondBrain
    router: Router
    headless: bool = True
    trajectory_root: Path = Path("trajectories")
    auto_approve: bool = False
    task_id: str = ""
    max_steps: int = 30


class WorkerBackend(abc.ABC):
    """Runs one worker subtask, returns {id, answer, finished, steps}."""

    @abc.abstractmethod
    def run_worker(self, subtask: dict, ctx: WorkerContext) -> dict:
        ...


class LocalBackend(WorkerBackend):
    """In-process workers: one thread + one Chromium session each."""

    def run_worker(self, subtask: dict, ctx: WorkerContext) -> dict:
        session = BrowserSession(headless=ctx.headless).start()
        try:
            agent = BrowserAgent(
                session=session,
                brain=ctx.brain,
                router=ctx.router,
                model_role="worker",
                max_steps=ctx.max_steps,
                name=subtask["id"],
                trajectory_root=Path(ctx.trajectory_root) / ctx.task_id,
                auto_approve=ctx.auto_approve,
            )
            result = agent.run(subtask["goal"], start_url=subtask.get("start_url"))
            if result.answer and subtask.get("brain_path"):
                ctx.brain.write(
                    subtask["brain_path"],
                    f"# {subtask['id']}: {subtask['goal']}\n\n{result.answer}\n",
                )
            return {
                "id": subtask["id"],
                "answer": result.answer,
                "finished": result.finished,
                "steps": result.steps,
            }
        finally:
            session.close()


class HttpBackend(WorkerBackend):
    """Remote workers via a worker service (`agentic_browser serve`).

    The service runs the browser + agent and returns the result. Brain
    coordination works when the service shares the brain filesystem (localhost
    demo); across machines it needs shared storage — the known M4-hard part.
    No auth in v0: bind to localhost or put it behind your own auth.
    """

    def __init__(self, base_url: str, timeout: int = 1800) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def run_worker(self, subtask: dict, ctx: WorkerContext) -> dict:
        import urllib.request

        payload = json.dumps(
            {
                "subtask": subtask,
                "task_id": ctx.task_id,
                "headless": ctx.headless,
                "max_steps": ctx.max_steps,
                "auto_approve": ctx.auto_approve,
            }
        ).encode()
        req = urllib.request.Request(
            self.base_url + "/run-worker",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {
                "id": subtask.get("id", "?"),
                "answer": f"(remote worker failed: {type(e).__name__}: {e})",
                "finished": False,
                "steps": 0,
            }


@dataclass
class Orchestrator:
    brain: SecondBrain
    router: Router
    max_workers: int = 4
    headless: bool = True
    trajectory_root: Path = Path("trajectories")
    auto_approve: bool = False  # background runs can't prompt; approve high-risk automatically
    backend: WorkerBackend | None = None  # default: LocalBackend

    def _plan(self, goal: str, task_id: str) -> dict:
        resp = self.router.generate(
            "orchestrator",
            [
                ModelMessage(role="system", text=PLAN_PROMPT),
                ModelMessage(
                    role="user",
                    text=f"TASK_ID: {task_id}\nGOAL: {goal}",
                ),
            ],
            max_tokens=2048,
            tag="orchestrator:plan",
        )
        text = resp.text.strip()
        # tolerate code fences
        if text.startswith("```"):
            text = text.strip("`").split("\n", 1)[1] if "\n" in text else text
            if text.lstrip().startswith("json"):
                text = text.lstrip()[4:]
        plan = json.loads(text)
        assert isinstance(plan.get("subtasks"), list) and plan["subtasks"], "empty plan"
        return plan

    def run(self, goal: str) -> str:
        task_id = uuid.uuid4().hex[:8]
        self.brain.write(f"tasks/{task_id}/goal.md", goal)
        plan = self._plan(goal, task_id)
        self.brain.write(f"tasks/{task_id}/plan.md", json.dumps(plan, indent=2))

        backend = self.backend or LocalBackend()
        ctx = WorkerContext(
            brain=self.brain,
            router=self.router,
            headless=self.headless,
            trajectory_root=self.trajectory_root,
            auto_approve=self.auto_approve,
            task_id=task_id,
        )
        subtasks = plan["subtasks"][: self.max_workers]
        findings: list[dict] = []
        with ThreadPoolExecutor(max_workers=len(subtasks)) as pool:
            futures = {
                pool.submit(backend.run_worker, st, ctx): st for st in subtasks
            }
            for fut in as_completed(futures):
                try:
                    findings.append(fut.result())
                except Exception as e:
                    st = futures[fut]
                    findings.append(
                        {"id": st["id"], "answer": f"(worker failed: {e})", "finished": False, "steps": 0}
                    )

        findings_text = "\n\n".join(
            f"--- {f['id']} (finished={f['finished']}, steps={f['steps']}) ---\n{f['answer']}"
            for f in sorted(findings, key=lambda x: x["id"])
        )
        merged = self.router.generate(
            "orchestrator",
            [
                ModelMessage(
                    role="system",
                    text=MERGE_PROMPT.format(
                        goal=goal,
                        merge_instructions=plan.get("merge_instructions", ""),
                        findings=findings_text,
                    ),
                ),
                ModelMessage(role="user", text="Merge the findings now."),
            ],
            max_tokens=4096,
            tag="orchestrator:merge",
        )
        self.brain.write(f"tasks/{task_id}/result.md", merged.text)
        return merged.text
