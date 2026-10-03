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
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .agent import BrowserAgent
from .ask import QueueAskChannel
from .browser import BrowserSession
from .memory import SecondBrain
from .models import Router


class ChatSessionState:
    """One in-app chat: conversation history plus the queues that carry
    ask-back questions to the UI and answers back.

    Threading: Playwright's sync API is bound to its creating thread, so
    each chat owns ONE worker thread that does every browser call for the
    session's lifetime — attach, agent runs, and close all happen there.
    The HTTP handlers only enqueue jobs and stream events."""

    def __init__(self, sid: str, cdp_url: str, url_match: str | None,
                 router: Router, brain: SecondBrain) -> None:
        self.id = sid
        self.cdp_url = cdp_url
        self.url_match = url_match
        self.router = router
        self.brain = brain
        self.history: list[tuple[str, str]] = []  # (user message, agent answer)
        self.run_lock = threading.Lock()
        self.ask_queue: queue.Queue = queue.Queue()
        self.answer_queue: queue.Queue = queue.Queue()
        self.event_queue: queue.Queue = queue.Queue()  # (run_id, etype, data)
        self.job_queue: queue.Queue = queue.Queue()  # ("run", run_id, msg, task, steps) | ("stop",)
        self.session: BrowserSession | None = None
        self.thread = threading.Thread(target=self._worker, daemon=True,
                                       name=f"chat-{sid}")
        self.thread.start()

    def _worker(self) -> None:
        while True:
            job = self.job_queue.get()
            if job[0] == "stop":
                break
            _, run_id, user_message, task, max_steps = job
            try:
                if self.session is None:
                    self.session = BrowserSession().attach_cdp(self.cdp_url, self.url_match)
                channel = QueueAskChannel(self.ask_queue, self.answer_queue,
                                           meta={"run_id": run_id})
                agent = BrowserAgent(
                    session=self.session,
                    brain=self.brain,
                    router=self.router,
                    model_role="worker",
                    max_steps=max_steps,
                    name="frontier-chat",
                    trajectory_root=Path("trajectories") / "frontier-chat" / self.id,
                    ask_channel=channel,
                    on_event=lambda et, d, rid=run_id: self.event_queue.put((rid, et, d)),
                )
                result = agent.run(task)
                self.history.append((user_message, result.answer))
                self.event_queue.put((run_id, "done", {
                    "answer": result.answer,
                    "finished": result.finished,
                    "steps": result.steps,
                }))
            except Exception as e:
                self.event_queue.put((run_id, "done", {
                    "answer": f"(chat turn failed: {type(e).__name__}: {e})",
                    "finished": False,
                    "steps": 0,
                }))
        try:
            if self.session is not None:
                self.session.close()
        except Exception:
            pass

    def stop(self) -> None:
        self.job_queue.put(("stop",))


chat_sessions: dict[str, ChatSessionState] = {}
chat_lock = threading.Lock()


class WorkerHandler(BaseHTTPRequestHandler):
    server_version = "AgenticBrowserWorker/0.4"

    def _json(self, obj: dict, status: int = 200) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")  # localhost UI
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
        if self.path == "/chat-sessions":
            self._handle_chat_create()
            return
        if self.path.startswith("/chat-sessions/"):
            parts = self.path.split("/")
            # /chat-sessions/{id}/run | /chat-sessions/{id}/answer
            if len(parts) == 4 and parts[3] == "run":
                self._handle_chat_run(parts[2])
                return
            if len(parts) == 4 and parts[3] == "answer":
                self._handle_chat_answer(parts[2])
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

    # -- in-app chat ----------------------------------------------------
    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", 0))
            return json.loads(self.rfile.read(length).decode() or "{}")
        except (ValueError, json.JSONDecodeError):
            return {}

    def _cors(self) -> None:
        # The Electron renderer fetches these directly (file:// origin).
        self.send_header("Access-Control-Allow-Origin", "*")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_DELETE(self) -> None:
        parts = self.path.split("/")
        if len(parts) == 3 and parts[1] == "chat-sessions":
            with chat_lock:
                st = chat_sessions.pop(parts[2], None)
            if st:
                st.stop()  # worker closes the session on its own thread
            self._json({"ok": True})
            return
        self._json({"error": "not found"}, 404)

    def _handle_chat_create(self) -> None:
        """Start a chat. The tab attach happens lazily on the session's
        worker thread (Playwright is thread-bound); here we just check the
        CDP endpoint is reachable."""
        req = self._read_json()
        cdp_url = req.get("cdp_url")
        if not cdp_url:
            self._json({"error": "cdp_url required"}, 400)
            return
        try:
            import urllib.request as _url

            with _url.urlopen(cdp_url.rstrip("/") + "/json/version", timeout=5):
                pass
        except Exception as e:
            self._json({"error": f"CDP unreachable: {e}"}, 502)
            return
        router: Router = self.server.router  # type: ignore[attr-defined]
        brain: SecondBrain = self.server.brain  # type: ignore[attr-defined]
        sid = uuid.uuid4().hex[:12]
        with chat_lock:
            chat_sessions[sid] = ChatSessionState(
                sid, cdp_url, req.get("url_match"), router, brain)
        self._json({"session_id": sid})

    def _handle_chat_run(self, sid: str) -> None:
        """Run one chat turn as SSE. Streams status/thought/action/question,
        then done. Events carry the run id so a stale turn's leftovers
        can't leak into a newer stream."""
        with chat_lock:
            st = chat_sessions.get(sid)
        if st is None:
            self._json({"error": "unknown chat session"}, 404)
            return
        if not st.run_lock.acquire(blocking=False):
            self._json({"error": "a turn is already running"}, 409)
            return
        try:
            req = self._read_json()
            message = (req.get("message") or "").strip()
            if not message:
                self._json({"error": "message required"}, 400)
                return

            # Conversation context: last few exchanges keep the agent grounded
            # in what was already discussed and done in this tab.
            ctx_lines = []
            for um, aa in st.history[-6:]:
                ctx_lines.append(f"User: {um[:400]}")
                ctx_lines.append(f"Agent: {aa[:600]}")
            ctx = "\n".join(ctx_lines)
            task = (
                (f"Conversation so far in this tab:\n{ctx}\n\n" if ctx else "")
                + f"User: {message}\n\nRespond to the latest user message. "
                "You are driving the user's visible browser tab — they watch "
                "everything. If you need information (a code, a choice, a "
                "clarification), ask with the ask_user tool."
            )

            run_id = uuid.uuid4().hex[:8]
            st.job_queue.put(("run", run_id, message, task, int(req.get("max_steps", 40))))

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self._cors()
            self.end_headers()

            def send(evt: dict) -> None:
                self.wfile.write(f"data: {json.dumps(evt)}\n\n".encode())
                self.wfile.flush()

            send({"type": "status", "text": "working…"})
            while True:
                try:
                    q = st.ask_queue.get_nowait()
                    if q.get("run_id") == run_id:
                        send({"type": "question", "prompt": q["prompt"],
                              "options": q.get("options")})
                except queue.Empty:
                    pass
                try:
                    rid, etype, data = st.event_queue.get(timeout=1.0)
                    if rid != run_id:
                        continue  # stale turn's leftovers
                    send({"type": etype, **data})
                    if etype == "done":
                        break
                except queue.Empty:
                    self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # UI went away mid-stream; the worker thread keeps going
        finally:
            st.run_lock.release()

    def _handle_chat_answer(self, sid: str) -> None:
        with chat_lock:
            st = chat_sessions.get(sid)
        if st is None:
            self._json({"error": "unknown chat session"}, 404)
            return
        req = self._read_json()
        st.answer_queue.put(req.get("text", ""))
        self._json({"ok": True})

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
