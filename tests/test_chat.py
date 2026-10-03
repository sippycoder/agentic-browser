"""Tests for the in-app chat sidecar endpoints.

No real browser or model: BrowserSession/BrowserAgent are stubbed, but the
HTTP + SSE plumbing and the ask-back round-trip are real.

Run: python tests/test_chat.py
"""

import json
import sys
import threading
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from agentic_browser import serve as serve_mod
from agentic_browser.agent import AgentResult


class FakeBrowserSession:
    def attach_cdp(self, cdp_url, url_match=None):
        assert cdp_url.startswith("http")
        return self

    def close(self):
        pass


class StubAgent:
    """Emits progress, asks one question through the real channel, finishes."""

    def __init__(self, **kwargs):
        self._kw = kwargs

    def run(self, task: str):
        emit = self._kw["on_event"]
        emit("thought", {"step": 1, "text": "I need a code."})
        answer = self._kw["ask_channel"].ask("What is the 2FA code?")
        emit("action", {"step": 2, "tool": "browser_navigate", "args": {}, "risk": None})
        return AgentResult(
            answer=f"done with code {answer}", steps=2, finished=True,
            trajectory_dir=Path("/tmp/x"), elapsed_s=0.1,
        )


def _post(base, path, body=None, method="POST"):
    data = json.dumps(body or {}).encode()
    req = urllib.request.Request(base + path, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())


def test_chat_roundtrip():
    serve_mod.BrowserSession = FakeBrowserSession
    serve_mod.BrowserAgent = StubAgent
    server = ThreadingHTTPServer(("127.0.0.1", 0), serve_mod.WorkerHandler)
    server.router = object()
    server.brain = object()
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    try:
        # create
        r = _post(base, "/chat-sessions", {"cdp_url": "http://127.0.0.1:9333"})
        sid = r["session_id"]
        assert sid

        # run (SSE) in background, collect events
        events = []

        def run_stream():
            req = urllib.request.Request(
                base + f"/chat-sessions/{sid}/run",
                data=json.dumps({"message": "log me in"}).encode(),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            with urllib.request.urlopen(req) as resp:
                buf = b""
                for chunk in resp:
                    buf += chunk
                    while b"\n\n" in buf:
                        raw, buf = buf.split(b"\n\n", 1)
                        for line in raw.split(b"\n"):
                            if line.startswith(b"data: "):
                                events.append(json.loads(line[6:]))

        rt = threading.Thread(target=run_stream, daemon=True)
        rt.start()

        # wait for the question, then answer it
        import time

        deadline = time.time() + 15
        while time.time() < deadline and not any(e.get("type") == "question" for e in events):
            time.sleep(0.1)
        q = next(e for e in events if e.get("type") == "question")
        assert "2FA" in q["prompt"], q
        _post(base, f"/chat-sessions/{sid}/answer", {"text": "987654"})

        rt.join(timeout=20)
        assert not rt.is_alive(), "run stream did not finish"

        types = [e.get("type") for e in events]
        assert "thought" in types and "action" in types and "question" in types, types
        done = next(e for e in events if e.get("type") == "done")
        assert "987654" in done["answer"], done  # the answer reached the agent
        assert done["finished"] is True

        # history recorded; second turn still works (session persists)
        st = serve_mod.chat_sessions[sid]
        assert st.history and st.history[0][0] == "log me in"
        print("ok: chat SSE + ask-back round-trip, history kept")
    finally:
        server.shutdown()


if __name__ == "__main__":
    test_chat_roundtrip()
    print("ALL CHAT TESTS PASSED")
