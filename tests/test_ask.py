"""Tests for the ask-back channel and the ask_user tool.

Run: python tests/test_ask.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_browser.agent import TOOLS, BrowserAgent
from agentic_browser.ask import (
    AskChannel,
    CallbackAskChannel,
    QueueAskChannel,
    TerminalAskChannel,
    channel_from_auto,
)
from agentic_browser.memory import SecondBrain
from agentic_browser.models import MockProvider, ModelMessage, Router, ToolCall

import sys as _sys

_sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_smoke import FakeSession  # noqa: E402


def test_terminal_channels():
    assert TerminalAskChannel(auto_approve=True).ask("Approve?") == "yes"
    assert TerminalAskChannel(auto_deny=True).ask("Approve?") == "no"
    assert channel_from_auto(True).ask("Approve?") == "yes"
    print("ok: terminal channels (auto approve/deny)")


def test_callback_channel():
    seen = []

    def fn(prompt, options):
        seen.append((prompt, options))
        return "42"

    ch = CallbackAskChannel(fn)
    assert ch.ask("What is the code?") == "42"
    assert seen == [("What is the code?", None)]
    print("ok: callback channel")


def test_ask_user_in_tools():
    assert "ask_user" in [t["name"] for t in TOOLS]
    print("ok: ask_user registered in TOOLS")


def test_approval_routes_through_channel():
    with tempfile.TemporaryDirectory() as d:
        brain = SecondBrain(Path(d) / "brain")
        router = Router(provider=MockProvider([]))
        agent = BrowserAgent(
            session=FakeSession(),
            brain=brain,
            router=router,
            trajectory_root=Path(d) / "traj",
            ask_channel=TerminalAskChannel(auto_approve=True),
        )
        assert agent._request_approval("browser_click", {"ref": "e1"}, "test", "https://x") is True
        agent2 = BrowserAgent(
            session=FakeSession(),
            brain=brain,
            router=router,
            trajectory_root=Path(d) / "traj2",
            ask_channel=TerminalAskChannel(auto_deny=True),
        )
        assert agent2._request_approval("browser_click", {"ref": "e1"}, "test", "https://x") is False
    print("ok: approval gate routes through the ask channel")


def test_ask_user_flow():
    """The agent asks a question mid-task; the channel's answer resumes it."""
    questions = []

    def fn(prompt, options):
        questions.append(prompt)
        return "123456"

    script = [
        ModelMessage(
            role="assistant",
            text="I need the 2FA code.",
            tool_calls=[ToolCall(id="c1", name="ask_user", arguments={"question": "What is the 2FA code?"})],
        ),
        ModelMessage(
            role="assistant",
            text="Got it.",
            tool_calls=[ToolCall(id="c2", name="finish", arguments={"answer": "Code was 123456"})],
        ),
    ]
    router = Router(provider=MockProvider(script))
    with tempfile.TemporaryDirectory() as d:
        agent = BrowserAgent(
            session=FakeSession(),
            brain=SecondBrain(Path(d) / "brain"),
            router=router,
            max_steps=5,
            trajectory_root=Path(d) / "traj",
            use_micro_gate=False,
            ask_channel=CallbackAskChannel(fn),
        )
        result = agent.run("log in")
        assert questions == ["What is the 2FA code?"], questions
        assert result.finished and "123456" in result.answer, result.answer
    print("ok: ask_user question -> channel answer -> task resumes")


def test_queue_channel():
    import queue as _queue
    import threading as _threading

    ask_q, ans_q = _queue.Queue(), _queue.Queue()
    ch = QueueAskChannel(ask_q, ans_q)
    box = {}

    def worker():
        box["answer"] = ch.ask("What is the code?", options=["a", "b"])

    t = _threading.Thread(target=worker, daemon=True)
    t.start()
    q = ask_q.get(timeout=5)
    assert q == {"prompt": "What is the code?", "options": ["a", "b"]}, q
    ans_q.put("777")
    t.join(timeout=5)
    assert box["answer"] == "777"
    print("ok: queue channel (cross-thread ask/answer)")


if __name__ == "__main__":
    test_terminal_channels()
    test_callback_channel()
    test_queue_channel()
    test_ask_user_in_tools()
    test_approval_routes_through_channel()
    test_ask_user_flow()
    print("ALL ASK TESTS PASSED")
