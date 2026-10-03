"""M4: HTTP worker service — the cloud-pool foundation.

POST /run-worker   {subtask, task_id, headless, max_steps, auto_approve}
                   -> {id, answer, finished, steps}
POST /run-tab-task {task, cdp_url, url_match, max_steps, auto_approve}
                   -> {answer, finished, steps}   (drives a tab in the
                      Frontier browser app over CDP — the user watches)
GET  /health       -> {"ok": true}

Run: agentic_browser serve --port 8000
The orchestrator talks to it via HttpBackend (--backend http://127.0.0.1:8000).

Each request launches its own Chromium + agent (heavy but isolated).
The service loads its own .env for model config. No auth in v0: bind to
localhost or put it behind your own auth/reverse proxy. For multi-machine
pools the services must share the brain filesystem (NFS/object store) —
that shared state is the known hard part of the cloud evolution.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agent import BrowserAgent
from .browser import BrowserSession
from .memory import SecondBrain
from .models import Router


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "AgenticBrowserWorker/0.4"

    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self) -> None:
        if self.path == "/run-tab-task":
            self._handle_tab_task()
            return
        if self.path != "/run-worker":
            self._json({"error": "not found"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode() or "{}")
            subtask = req["subtask"]
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            self._json({"error": f"bad request: {e}"}, 400)
            return

        router: Router = self.server.router  # type: ignore[attr-defined]
        brain: SecondBrain = self.server.brain  # type: ignore[attr-defined]
        session = BrowserSession(headless=req.get("headless", True)).start()
        try:
            agent = BrowserAgent(
                session=session,
                brain=brain,
                router=router,
                model_role="worker",
                max_steps=int(req.get("max_steps", 30)),
                name=subtask.get("id", "remote-worker"),
                trajectory_root=Path("trajectories") / "remote" / req.get("task_id", "x"),
                auto_approve=bool(req.get("auto_approve", True)),
            )
            result = agent.run(subtask["goal"], start_url=subtask.get("start_url"))
            if result.answer and subtask.get("brain_path"):
                brain.write(
                    subtask["brain_path"],
                    f"# {subtask['id']}: {subtask['goal']}\n\n{result.answer}\n",
                )
            self._json(
                {
                    "id": subtask.get("id", "?"),
                    "answer": result.answer,
                    "finished": result.finished,
                    "steps": result.steps,
                }
            )
        except Exception as e:
            self._json(
                {
                    "id": subtask.get("id", "?"),
                    "answer": f"(remote worker failed: {type(e).__name__}: {e})",
                    "finished": False,
                    "steps": 0,
                }
            )
        finally:
            session.close()

    def _handle_tab_task(self) -> None:
        """Drive a tab in the Frontier browser app over CDP.

        The user watches the agent work in their own visible tab. Approvals
        can't prompt (no terminal), so tab tasks run auto-approved — the
        human supervision here is watching the tab itself.
        """
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode() or "{}")
            task, cdp_url = req["task"], req["cdp_url"]
        except (ValueError, KeyError, json.JSONDecodeError) as e:
            self._json({"error": f"bad request: {e}"}, 400)
            return

        router: Router = self.server.router  # type: ignore[attr-defined]
        brain: SecondBrain = self.server.brain  # type: ignore[attr-defined]
        session = BrowserSession().attach_cdp(cdp_url, req.get("url_match"))
        try:
            agent = BrowserAgent(
                session=session,
                brain=brain,
                router=router,
                model_role=req.get("model_role", "worker"),
                max_steps=int(req.get("max_steps", 40)),
                name="frontier-tab",
                trajectory_root=Path("trajectories") / "frontier",
                auto_approve=bool(req.get("auto_approve", True)),
            )
            result = agent.run(task)
            self._json(
                {
                    "answer": result.answer,
                    "finished": result.finished,
                    "steps": result.steps,
                    "trajectory": str(result.trajectory_dir),
                }
            )
        except Exception as e:
            self._json(
                {
                    "answer": f"(tab task failed: {type(e).__name__}: {e})",
                    "finished": False,
                    "steps": 0,
                }
            )
        finally:
            session.close()

    def log_message(self, fmt: str, *args) -> None:
        print(f"[serve] {self.address_string()} {fmt % args}")


def serve(port: int = 8000, brain_root: str = "brain", host: str = "127.0.0.1") -> None:
    router = Router()  # loads .env; raises clearly if unconfigured
    brain = SecondBrain(brain_root)
    server = ThreadingHTTPServer((host, port), WorkerHandler)
    server.router = router  # type: ignore[attr-defined]
    server.brain = brain  # type: ignore[attr-defined]
    print(f"[serve] worker service on http://{host}:{port} (brain: {brain_root}) — Ctrl+C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[serve] stopped")
