"""Tests for the MCP elicitation ask-back bridge.

Deterministic: calls the (decorated) browser_run_task tool function directly
with a fake ctx and a stub BrowserAgent, verifying that ask-backs travel
from the agent loop out through MCP elicitation and answers flow back.

Run: python tests/test_mcp_ask.py
"""

import asyncio
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mcp.types import ElicitResult

from agentic_browser import mcp_server
from agentic_browser.agent import AgentResult
from agentic_browser.mcp_server import (
    ElicitationAskChannel,
    _ApprovalSchema,
    _QuestionSchema,
)


class FakeCtx:
    """Stands in for the MCP Context: records elicitations, auto-answers."""

    def __init__(self, approve: bool = True, answer: str = "123456"):
        self.asked = []
        self._approve = approve
        self._answer = answer

    async def elicit(self, message, schema):
        self.asked.append((message, schema))
        if schema is _ApprovalSchema:
            return ElicitResult(action="accept", content={"approved": self._approve})
        return ElicitResult(action="accept", content={"answer": self._answer})


class StubAgent:
    """Captures the ask channel, then asks one approval + one question."""

    captured = None

    def __init__(self, **kwargs):
        StubAgent.captured = kwargs.get("ask_channel")
        self._kwargs = kwargs

    def run(self, task: str):
        ch = self._kwargs["ask_channel"]
        assert isinstance(ch, ElicitationAskChannel)
        ok = ch.ask("HIGH-RISK action proposed: browser_click — Approve?", options=["yes", "no"])
        ans = ch.ask("What is the 2FA code?")
        return AgentResult(answer=f"approved={ok} code={ans}", steps=2, finished=True,
                           trajectory_dir=Path("/tmp/x"), elapsed_s=0.1)


class FakeSession:
    def close(self):
        pass


def test_elicitation_bridge():
    sid = "testsession1"
    mcp_server._sessions[sid] = FakeSession()
    mcp_server._locks[sid] = threading.Lock()
    real_agent = mcp_server.BrowserAgent
    mcp_server.BrowserAgent = StubAgent
    try:
        ctx = FakeCtx(approve=True, answer="123456")
        out = asyncio.run(mcp_server.browser_run_task(sid, "do the thing", ctx))
        assert "approved=yes" in out, out
        assert "code=123456" in out, out
        # the server asked twice: once with the approval schema, once with the question schema
        assert len(ctx.asked) == 2, ctx.asked
        assert ctx.asked[0][1] is _ApprovalSchema, "approval must use the boolean schema"
        assert ctx.asked[1][1] is _QuestionSchema, "question must use the text schema"
        assert "Approve?" in ctx.asked[0][0]
        assert "2FA" in ctx.asked[1][0]
    finally:
        mcp_server.BrowserAgent = real_agent
        del mcp_server._sessions[sid]
        del mcp_server._locks[sid]
    print("ok: elicitation bridge (approval + question round-trip)")


def test_elicitation_decline_denies():
    sid = "testsession2"
    mcp_server._sessions[sid] = FakeSession()
    mcp_server._locks[sid] = threading.Lock()
    real_agent = mcp_server.BrowserAgent
    mcp_server.BrowserAgent = StubAgent
    try:
        ctx = FakeCtx(approve=False, answer="123456")
        out = asyncio.run(mcp_server.browser_run_task(sid, "do the thing", ctx))
        assert "approved=no" in out, out
    finally:
        mcp_server.BrowserAgent = real_agent
        del mcp_server._sessions[sid]
        del mcp_server._locks[sid]
    print("ok: declined elicitation -> denied approval")


if __name__ == "__main__":
    test_elicitation_bridge()
    test_elicitation_decline_denies()
    print("ALL MCP ASK TESTS PASSED")
