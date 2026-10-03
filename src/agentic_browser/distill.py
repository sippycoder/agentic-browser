"""M5: the flywheel — distill trajectories into training data and prompt packs.

Polar's disclosed flywheel: trajectories from agent runs become training data
for smaller specialist models. v0 implements the full loop up to the training
step:

    trajectories/**/trajectory.jsonl
        -> quality filter (finished runs, no injection flags, sane length)
        -> fine-tune JSONL export (OpenAI chat format with tool calls —
           accepted by OpenAI, Together, Fireworks, and other fine-tune APIs)
        -> few-shot prompt pack (works TODAY: prepended to the agent's system
           prompt via --prompt-pack; re-run evals to measure the gain)

Moonshot's API does not offer fine-tuning, so the export targets providers
that do; the prompt pack is the closed loop that runs against any provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

FINETUNE_SYSTEM = """You operate a web browser via tools. Observe the page, reason briefly, \
then call exactly one tool per step. Use browser_navigate to go to URLs, browser_click \
with the element's ref to click, browser_fill to type into fields, browser_get_text to \
read page content, and finish with the final answer when the task is complete."""


@dataclass
class DistilledTrajectory:
    path: Path
    task: str = ""
    start_url: str = ""
    model_role: str = ""
    agent: str = ""
    finished: bool = False
    answer: str = ""
    steps: int = 0
    actions: list[dict] = field(default_factory=list)  # step/thought/tool/args/page/url/risk
    results: dict = field(default_factory=dict)  # step -> tool_result text
    has_injection_flag: bool = False
    legacy: bool = False  # pre-M5 format (no meta/result records)

    @property
    def action_count(self) -> int:
        return len([a for a in self.actions if a.get("tool") not in ("finish", "force_finish")])


def _parse_one(path: Path) -> DistilledTrajectory | None:
    try:
        lines = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except (json.JSONDecodeError, OSError):
        return None
    if not lines:
        return None
    t = DistilledTrajectory(path=path)
    saw_meta = False
    for e in lines:
        etype = e.get("type")
        if etype == "meta":
            saw_meta = True
            t.task = e.get("task", "")
            t.start_url = e.get("start_url", "")
            t.model_role = e.get("model_role", "")
            t.agent = e.get("agent", "")
        elif etype == "result":
            t.finished = bool(e.get("finished"))
            t.answer = e.get("answer", "")
            t.steps = int(e.get("steps", 0))
        elif etype == "tool_result":
            t.results[int(e.get("step", -1))] = e.get("result", "")
        elif e.get("tool") == "injection_flag":
            t.has_injection_flag = True
        elif "tool" in e:
            t.actions.append(e)
    t.legacy = not saw_meta
    if t.legacy:
        # Heuristic for pre-M5 logs: a logged `finish` action means it finished.
        t.finished = any(a.get("tool") == "finish" for a in t.actions)
        t.steps = max([a.get("step", 0) for a in t.actions] or [0])
    return t


def iter_trajectories(root: str | Path = "trajectories") -> list[DistilledTrajectory]:
    out = []
    for p in sorted(Path(root).rglob("trajectory.jsonl")):
        t = _parse_one(p)
        if t is not None:
            out.append(t)
    return out


def quality_filter(
    trajs: list[DistilledTrajectory],
    require_finished: bool = True,
    min_actions: int = 1,
    max_steps: int = 80,
    exclude_injection: bool = True,
) -> list[DistilledTrajectory]:
    out = []
    for t in trajs:
        if require_finished and not t.finished:
            continue
        if exclude_injection and t.has_injection_flag:
            continue
        if not (min_actions <= t.action_count <= max_steps):
            continue
        if not t.actions:
            continue
        out.append(t)
    return out


def _tool_call_msg(step_idx: int, action: dict) -> dict:
    return {
        "id": f"call_{step_idx}",
        "type": "function",
        "function": {
            "name": action.get("tool", ""),
            "arguments": json.dumps(action.get("args", {})),
        },
    }


def to_finetune_messages(traj: DistilledTrajectory, system: str = FINETUNE_SYSTEM) -> list[dict]:
    """One trajectory -> OpenAI chat fine-tuning messages with tool calls."""
    messages: list[dict] = [{"role": "system", "content": system}]
    user_text = f"TASK: {traj.task or '(browser task)'}"
    if traj.start_url:
        user_text += f"\nStart by navigating to: {traj.start_url}"
    messages.append({"role": "user", "content": user_text})

    call_idx = 0
    for a in traj.actions:
        tool = a.get("tool", "")
        if tool in ("injection_flag",):
            continue
        thought = (a.get("thought") or "").strip()
        call_idx += 1
        call_id = f"call_{call_idx}"
        messages.append(
            {
                "role": "assistant",
                "content": thought or None,
                "tool_calls": [
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {
                            "name": tool,
                            "arguments": json.dumps(a.get("args", {})),
                        },
                    }
                ],
            }
        )
        outcome = traj.results.get(a.get("step", -1), "")
        result_text = f"[{tool}] {outcome}" if outcome else f"[{tool}] done (url: {a.get('url', '')})"
        messages.append(
            {"role": "tool", "tool_call_id": call_id, "content": result_text[:1000]}
        )
        if tool == "finish":
            break
    if traj.answer and (not messages or messages[-1].get("role") != "assistant" or messages[-1].get("content") != traj.answer):
        messages.append({"role": "assistant", "content": f"Final answer: {traj.answer[:1500]}"})
    return messages


def export_jsonl(trajs: list[DistilledTrajectory], out_path: str | Path, system: str = FINETUNE_SYSTEM) -> int:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w") as f:
        for t in trajs:
            f.write(json.dumps({"messages": to_finetune_messages(t, system)}) + "\n")
            n += 1
    return n


def mine_fewshots(
    trajs: list[DistilledTrajectory],
    max_examples: int = 8,
    max_thought_chars: int = 500,
) -> list[dict]:
    """Mine (situation -> thought -> action) exemplars from successful runs.

    Prefers steps with substantive reasoning and a spread of tools. Skips
    overlong, rambling thoughts — confused reasoning distills badly.
    """
    candidates = []
    for t in trajs:
        for a in t.actions:
            tool = a.get("tool", "")
            thought = (a.get("thought") or "").strip()
            if tool in ("finish", "force_finish", "injection_flag"):
                continue
            if not (20 <= len(thought) <= max_thought_chars):
                continue
            candidates.append(
                {
                    "tool": tool,
                    "thought": thought,
                    "args": a.get("args", {}),
                    "url": a.get("url", ""),
                    "page": (a.get("page") or "")[:400],
                    "task": t.task,
                }
            )
    # round-robin over tools for diversity, longest thoughts first
    by_tool: dict[str, list[dict]] = {}
    for c in sorted(candidates, key=lambda c: -len(c["thought"])):
        by_tool.setdefault(c["tool"], []).append(c)
    examples: list[dict] = []
    while len(examples) < max_examples and any(by_tool.values()):
        for tool in sorted(by_tool):
            if by_tool[tool] and len(examples) < max_examples:
                examples.append(by_tool[tool].pop(0))
    return examples


def write_pack(examples: list[dict], out_path: str | Path, n_runs: int = 0) -> str:
    """Write a few-shot prompt pack (markdown). Returns the pack text."""
    lines = [
        "# Distilled behavior examples",
        "",
        f"Learned from {n_runs} successful agent run(s). Imitate this reasoning style:",
        "",
    ]
    for i, e in enumerate(examples, 1):
        lines += [
            f"## Example {i} — `{e['tool']}`",
            "",
            f"Situation: on {e['url'] or '(page)'}, page shows: {e['page'] or '(no snapshot)'}",
            "",
            f"Thought: {e['thought']}",
            "",
            f"Action: `{e['tool']}({json.dumps(e['args'])})`",
            "",
        ]
    text = "\n".join(lines)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    return text


def report(trajs: list[DistilledTrajectory]) -> str:
    total = len(trajs)
    finished = sum(1 for t in trajs if t.finished)
    flagged = sum(1 for t in trajs if t.has_injection_flag)
    legacy = sum(1 for t in trajs if t.legacy)
    actions = sum(t.action_count for t in trajs)
    eligible = len(quality_filter(trajs))
    lines = [
        "Distillation report",
        f"  trajectories: {total} ({legacy} legacy format)",
        f"  finished: {finished}/{total}",
        f"  injection-flagged: {flagged}",
        f"  total actions: {actions}",
        f"  eligible for training: {eligible}",
        f"  generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
    ]
    return "\n".join(lines)
