"""Unit tests for M5 distillation — synthetic trajectories, no network."""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_browser.distill import (
    export_jsonl,
    iter_trajectories,
    mine_fewshots,
    quality_filter,
    report,
    to_finetune_messages,
    write_pack,
)


def _write_traj(d: Path, name: str, lines: list[dict]) -> Path:
    p = d / name / "trajectory.jsonl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(json.dumps(l) for l in lines))
    return p


def _new_format(finished: bool = True, flagged: bool = False) -> list[dict]:
    lines = [
        {"type": "meta", "task": "get the headline", "start_url": "https://example.com",
         "model_role": "worker", "agent": "w1", "ts": 1.0},
        {"step": 1, "thought": "The page loaded. I need the main heading, so let me read the page text.",
         "tool": "browser_get_text", "args": {}, "screenshot": "step-01.png",
         "url": "https://example.com/", "risk": None, "page": "Example Domain ..."},
        {"type": "tool_result", "step": 1, "tool": "browser_get_text", "result": "Example Domain\n..."},
        {"step": 2, "thought": "I have the headline. Done.",
         "tool": "finish", "args": {"answer": "Example Domain"}, "screenshot": "step-02.png",
         "url": "https://example.com/", "risk": None, "page": ""},
    ]
    if flagged:
        lines.insert(2, {"step": 1, "thought": "", "tool": "injection_flag",
                          "args": {"markers": ["persona-override"]}, "screenshot": "s.png",
                          "url": "https://example.com/", "risk": "security", "page": ""})
    lines.append({"type": "result", "finished": finished, "answer": "Example Domain", "steps": 2})
    return lines


def _legacy_format() -> list[dict]:
    return [
        {"step": 1, "thought": "Navigate first.", "tool": "browser_navigate",
         "args": {"url": "https://example.com"}, "screenshot": "step-01.png", "url": "about:blank"},
        {"step": 2, "thought": "Done.", "tool": "finish",
         "args": {"answer": "ok"}, "screenshot": "step-02.png", "url": "https://example.com/"},
    ]


def test_parse_and_filter():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_traj(d, "good", _new_format(finished=True))
        _write_traj(d, "bad", _new_format(finished=False))
        _write_traj(d, "flagged", _new_format(finished=True, flagged=True))
        _write_traj(d, "old", _legacy_format())
        trajs = iter_trajectories(d)
        assert len(trajs) == 4, len(trajs)
        good = next(t for t in trajs if t.path.parent.name == "good")
        assert good.finished and good.task == "get the headline"
        assert good.results[1].startswith("Example Domain")
        assert not good.legacy
        old = next(t for t in trajs if t.path.parent.name == "old")
        assert old.legacy and old.finished  # finish-tool heuristic
        flagged = next(t for t in trajs if t.path.parent.name == "flagged")
        assert flagged.has_injection_flag

        eligible = quality_filter(trajs)
        names = {t.path.parent.name for t in eligible}
        assert names == {"good", "old"}, names
    print("ok: parse + quality filter")


def test_finetune_messages():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_traj(d, "good", _new_format(finished=True))
        (traj,) = iter_trajectories(d)
        msgs = to_finetune_messages(traj)
        roles = [m["role"] for m in msgs]
        assert roles[0] == "system" and roles[1] == "user"
        assert "get the headline" in msgs[1]["content"]
        asst = [m for m in msgs if m["role"] == "assistant" and m.get("tool_calls")]
        assert asst and asst[0]["tool_calls"][0]["function"]["name"] == "browser_get_text"
        tools = [m for m in msgs if m["role"] == "tool"]
        assert tools and "Example Domain" in tools[0]["content"]
        assert msgs[-1]["role"] == "assistant" and "Example Domain" in msgs[-1]["content"]
        n = export_jsonl([traj], d / "ft.jsonl")
        assert n == 1
        row = json.loads((d / "ft.jsonl").read_text().splitlines()[0])
        assert "messages" in row and row["messages"][0]["role"] == "system"
    print("ok: fine-tune export")


def test_fewshots_and_pack():
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        _write_traj(d, "good", _new_format(finished=True))
        (traj,) = iter_trajectories(d)
        examples = mine_fewshots([traj], max_examples=4)
        assert examples, "expected mined examples"
        assert all(len(e["thought"]) >= 20 for e in examples)
        assert "finish" not in {e["tool"] for e in examples}
        text = write_pack(examples, d / "pack.md", n_runs=1)
        assert "browser_get_text" in text and (d / "pack.md").exists()
        rep = report([traj])
        assert "eligible for training: 1" in rep
    print("ok: few-shot mining + pack + report")


if __name__ == "__main__":
    test_parse_and_filter()
    test_finetune_messages()
    test_fewshots_and_pack()
    print("\nAll M5 tests passed.")
