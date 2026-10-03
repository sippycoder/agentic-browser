"""Smoke tests with no network and no API keys.

Run: python tests/test_smoke.py
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_browser.agent import TOOLS, BrowserAgent
from agentic_browser.memory import SecondBrain
from agentic_browser.models import MockProvider, ModelMessage, Router, ToolCall


class FakeSession:
    """Pretends to be a BrowserSession; records actions."""

    def __init__(self):
        self.actions = []
        self._n = 0

    def observe(self, with_screenshot=True):
        from agentic_browser.browser import Observation

        self._n += 1
        return Observation(
            url="https://example.com",
            title="Example",
            elements=[{"ref": "e1", "role": "link", "name": "More information"}],
            screenshot_b64=None,
        )

    def navigate(self, url):
        self.actions.append(("navigate", url))
        return f"Navigated to {url}"

    def click(self, ref):
        self.actions.append(("click", ref))
        return f"Clicked [{ref}]"

    def fill(self, ref, text, submit=False):
        self.actions.append(("fill", ref, text))
        return "Filled."

    def press(self, key):
        self.actions.append(("press", key))
        return "Pressed."

    def scroll(self, direction="down", pixels=600):
        self.actions.append(("scroll", direction))
        return "Scrolled."

    def go_back(self):
        return "Went back."

    def page_text(self, max_chars=8000):
        return "Example Domain"


def test_tool_schemas_valid():
    for t in TOOLS:
        assert t["name"] and t.get("parameters", {}).get("type") == "object", t
    names = [t["name"] for t in TOOLS]
    assert "finish" in names and "browser_click" in names
    print("ok: tool schemas valid (%d tools)" % len(TOOLS))


def test_brain_roundtrip():
    with tempfile.TemporaryDirectory() as d:
        brain = SecondBrain(d)
        brain.write("tasks/t1/notes/a.md", "hello")
        assert "hello" in brain.read("tasks/t1/notes/a.md")
        assert "tasks/t1/notes/a.md" in brain.list("tasks")
        try:
            brain.read("../../etc/passwd")
            raise AssertionError("path traversal not blocked")
        except ValueError:
            pass
    print("ok: brain roundtrip + traversal guard")


def test_agent_loop_finishes():
    script = [
        ModelMessage(
            role="assistant",
            text="I see a link, I'll click it.",
            tool_calls=[ToolCall(id="c1", name="browser_click", arguments={"ref": "e1"})],
        ),
        ModelMessage(
            role="assistant",
            text="Done.",
            tool_calls=[ToolCall(id="c2", name="finish", arguments={"answer": "Example Domain"})],
        ),
    ]
    router = Router(provider=MockProvider(script))
    with tempfile.TemporaryDirectory() as d:
        session = FakeSession()
        brain = SecondBrain(Path(d) / "brain")
        agent = BrowserAgent(
            session=session,
            brain=brain,
            router=router,
            max_steps=5,
            trajectory_root=Path(d) / "traj",
            use_micro_gate=False,  # keep this test focused on loop mechanics
            auto_approve=True,
        )
        result = agent.run("Report the heading.")
        assert result.finished, "agent did not finish"
        assert result.answer == "Example Domain"
        assert ("click", "e1") in session.actions
        log = result.trajectory_dir / "trajectory.jsonl"
        assert log.exists()
        lines = [json.loads(l) for l in log.read_text().splitlines()]
        actions = [l for l in lines if not l.get("type")]  # M5: skip meta/tool_result/result records
        assert any(l["tool"] == "browser_click" for l in actions)
        assert any(l["tool"] == "finish" for l in actions)
    print("ok: agent loop observes, acts, finishes, logs trajectory")


def test_router_requires_config():
    import os

    os.environ["AGENTIC_NO_DOTENV"] = "1"
    for var in ("AGENTIC_WORKER_MODEL", "AGENTIC_ORCHESTRATOR_MODEL",
                "AGENTIC_MICRO_MODEL", "AGENTIC_JUDGE_MODEL"):
        os.environ.pop(var, None)
    try:
        Router().generate("worker", [ModelMessage(role="user", text="hi")])
        raise AssertionError("expected RuntimeError")
    except RuntimeError as e:
        assert "AGENTIC_WORKER_MODEL" in str(e)
    print("ok: router fails loudly without model config")


def test_risk_gates():
    # Heuristic path: no model calls needed.
    agent = BrowserAgent(
        session=FakeSession(),
        brain=SecondBrain(tempfile.mkdtemp()),
        router=Router(provider=MockProvider([])),
        use_micro_gate=False,
    )
    agent._last_elements = [
        {"ref": "e9", "role": "button", "name": "Buy now"},
        {"ref": "e3", "role": "link", "name": "Learn more"},
    ]
    assert agent.assess_risk(
        "browser_fill", {"ref": "e1", "text": "x", "submit": True}, "https://x", "t"
    ), "submit=True should be high risk"
    assert agent.assess_risk(
        "browser_click", {"ref": "e9"}, "https://x", "t"
    ), "'Buy now' click should be high risk"
    assert agent.assess_risk(
        "browser_click", {"ref": "e3"}, "https://x", "t"
    ) is None, "plain link click should pass heuristics"
    assert agent.assess_risk(
        "browser_scroll", {"direction": "down"}, "https://x", "t"
    ) is None
    assert agent.assess_risk(
        "browser_navigate", {"url": "https://x"}, "https://x", "t"
    ) is None
    # Micro-model path: scripted HIGH verdict.
    script = [ModelMessage(role="assistant", text="HIGH: this submits payment")]
    agent2 = BrowserAgent(
        session=FakeSession(),
        brain=SecondBrain(tempfile.mkdtemp()),
        router=Router(provider=MockProvider(script)),
        use_micro_gate=True,
    )
    agent2._last_elements = [{"ref": "e2", "role": "link", "name": "Details"}]
    reason = agent2.assess_risk("browser_click", {"ref": "e2"}, "https://x", "t")
    assert reason, "micro HIGH verdict should flag the action"
    assert "payment" in reason
    print("ok: risk gates (heuristics + micro-model)")


if __name__ == "__main__":
    import os

    os.environ["AGENTIC_NO_DOTENV"] = "1"  # hermetic: never read the real .env
    test_tool_schemas_valid()
    test_brain_roundtrip()
    test_agent_loop_finishes()
    test_router_requires_config()
    test_risk_gates()
    print("\nAll smoke tests passed.")
