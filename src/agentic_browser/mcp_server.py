"""MCP server: let any main agent drive the browser.

Any MCP-capable main agent (Claude, Cursor, …) can:

  - browser_start_session -> get a session_id (one Chromium page)
  - drive it step by step: navigate / snapshot / click / fill / press /
    scroll / back / get_text / screenshot
  - browser_run_task -> hand a goal to the autonomous BrowserAgent loop,
    which runs inside this session and asks back when it needs something
  - browser_close_session when done

Ask-back uses MCP elicitation (server -> client): approvals for high-risk
actions and the agent's ask_user questions arrive as elicitation requests;
the main agent's response resumes the loop.

Run:  python -m agentic_browser.cli mcp        (stdio)
Wire into a client with: {"command": "<venv>/bin/python",
                          "args": ["-m", "agentic_browser.cli", "mcp"]}
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from pathlib import Path

from mcp.server.mcpserver import Context, MCPServer
from mcp.types import ImageContent
from pydantic import BaseModel

from .agent import BrowserAgent
from .ask import AskChannel
from .browser import BrowserSession
from .memory import SecondBrain
from .models import Router

mcp = MCPServer("agentic-browser")

_sessions: dict[str, BrowserSession] = {}
_locks: dict[str, threading.Lock] = {}
_registry_lock = threading.Lock()

_router: Router | None = None
_brain: SecondBrain | None = None


def _shared() -> tuple[Router, SecondBrain]:
    global _router, _brain
    if _router is None:
        _router = Router()
        _brain = SecondBrain(Path("brain"))
    return _router, _brain


def _get(session_id: str) -> BrowserSession:
    s = _sessions.get(session_id)
    if s is None:
        raise ValueError(f"unknown session: {session_id}")
    return s


async def _in_session(session_id: str, fn, *args):
    """Run sync browser work in a worker thread, one session at a time."""
    lock = _locks.get(session_id)
    if lock is None:
        raise ValueError(f"unknown session: {session_id}")
    s = _get(session_id)
    return await asyncio.to_thread(_with_lock, lock, fn, s, *args)


def _with_lock(lock: threading.Lock, fn, s: BrowserSession, *args):
    with lock:
        return fn(s, *args)


# -- elicitation-backed ask channel -------------------------------------

class _ApprovalSchema(BaseModel):
    approved: bool


class _QuestionSchema(BaseModel):
    answer: str


class ElicitationAskChannel(AskChannel):
    """Ask the main agent via MCP elicitation. The agent loop runs in a
    worker thread; questions are marshalled back onto the server's event
    loop, which delivers them to the client."""

    def __init__(self, ctx: Context, loop: asyncio.AbstractEventLoop) -> None:
        self._ctx = ctx
        self._loop = loop

    def _elicit(self, message: str, schema):
        fut = asyncio.run_coroutine_threadsafe(self._ctx.elicit(message, schema), self._loop)
        try:
            return fut.result(timeout=600)
        except Exception:
            return None

    @staticmethod
    def _value(res, key: str):
        c = res.content
        if isinstance(c, dict):
            return c.get(key)
        return getattr(c, key, None)

    def ask(self, prompt: str, options: list[str] | None = None) -> str:
        if options == ["yes", "no"]:
            res = self._elicit(prompt, _ApprovalSchema)
            if res and res.action == "accept" and res.content:
                return "yes" if self._value(res, "approved") else "no"
            return "no"
        res = self._elicit(prompt, _QuestionSchema)
        if res and res.action == "accept" and res.content:
            return self._value(res, "answer") or ""
        return ""


# -- tools ---------------------------------------------------------------

@mcp.tool()
async def browser_start_session(headless: bool = True) -> str:
    """Start a browser session. Returns a session_id for the other tools."""
    def _start() -> str:
        sid = uuid.uuid4().hex[:12]
        with _registry_lock:
            _sessions[sid] = BrowserSession(headless=headless).start()
            _locks[sid] = threading.Lock()
        return sid
    return await asyncio.to_thread(_start)


@mcp.tool()
async def browser_navigate(session_id: str, url: str) -> str:
    """Navigate the session's page to a URL."""
    return await _in_session(session_id, lambda s: s.navigate(url))


@mcp.tool()
async def browser_snapshot(session_id: str) -> str:
    """Ref-tagged interactive element tree of the current page."""
    def _snap(s: BrowserSession) -> str:
        obs = s.observe(with_screenshot=False)
        lines = [f"url: {obs.url}", f"title: {obs.title}"]
        for e in obs.elements:
            lines.append(f"[{e.get('ref')}] {e.get('role')} {e.get('name', '')[:60]}")
        return "\n".join(lines)
    return await _in_session(session_id, _snap)


@mcp.tool()
async def browser_click(session_id: str, ref: str) -> str:
    """Click the element with the given ref, e.g. 'e12'."""
    return await _in_session(session_id, lambda s: s.click(ref))


@mcp.tool()
async def browser_fill(session_id: str, ref: str, text: str) -> str:
    """Fill the element with the given ref with text."""
    return await _in_session(session_id, lambda s: s.fill(ref, text))


@mcp.tool()
async def browser_press(session_id: str, key: str) -> str:
    """Press a key, e.g. 'Enter', 'Escape', 'Tab'."""
    return await _in_session(session_id, lambda s: s.press(key))


@mcp.tool()
async def browser_scroll(session_id: str, direction: str = "down", pixels: int = 600) -> str:
    """Scroll the page."""
    return await _in_session(session_id, lambda s: s.scroll(direction, pixels))


@mcp.tool()
async def browser_back(session_id: str) -> str:
    """Go back one page."""
    return await _in_session(session_id, lambda s: s.back())


@mcp.tool()
async def browser_get_text(session_id: str, max_chars: int = 8000) -> str:
    """Visible text of the current page."""
    return await _in_session(session_id, lambda s: s.get_text()[:max_chars])


@mcp.tool()
async def browser_screenshot(session_id: str) -> ImageContent:
    """Screenshot of the current page as an image."""
    def _shot(s: BrowserSession) -> ImageContent:
        obs = s.observe(with_screenshot=True)
        return ImageContent(data=obs.screenshot_b64 or "", mimeType="image/png")
    return await _in_session(session_id, _shot)


@mcp.tool()
async def browser_run_task(
    session_id: str, task: str, ctx: Context, max_steps: int = 30
) -> str:
    """Run an autonomous browser task in this session. The agent works the
    task step by step and asks back (via elicitation) when it needs an
    approval or information. Returns the final answer."""
    loop = asyncio.get_running_loop()
    channel = ElicitationAskChannel(ctx, loop)

    def _run() -> str:
        lock = _locks[session_id]
        with lock:
            s = _get(session_id)
            router, brain = _shared()
            agent = BrowserAgent(
                session=s,
                brain=brain,
                router=router,
                model_role="worker",
                max_steps=max_steps,
                name="mcp-worker",
                trajectory_root=Path("trajectories") / "mcp" / session_id,
                ask_channel=channel,
            )
            result = agent.run(task)
            status = "done" if result.finished else "unfinished"
            return f"[{status} in {result.steps} steps]\n{result.answer}"

    return await asyncio.to_thread(_run)


@mcp.tool()
async def browser_close_session(session_id: str) -> str:
    """Close a browser session and free its resources."""
    def _close() -> str:
        with _registry_lock:
            s = _sessions.pop(session_id, None)
            _locks.pop(session_id, None)
        if s is None:
            return f"unknown session: {session_id}"
        try:
            s.close()
        except Exception:
            pass
        return f"closed {session_id}"
    return await asyncio.to_thread(_close)


def main() -> None:
    import asyncio as _asyncio

    _asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
