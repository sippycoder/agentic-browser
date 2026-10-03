"""Ask channels: how the browser agent asks back.

The agent is conversational — when it needs something it can't decide itself
(an approval, a login code, a clarification), it asks through a channel
instead of blocking on a terminal or guessing. The channel decides who
answers: the human at a terminal, a callback (an embedding program), or a
main agent over MCP elicitation.

Approvals and questions are the same mechanism: _request_approval is just
ask("Approve X?") with a yes/no shape.
"""

from __future__ import annotations

import abc
from typing import Callable


class AskChannel(abc.ABC):
    """Ask the operator a question; return their response as a string."""

    @abc.abstractmethod
    def ask(self, prompt: str, options: list[str] | None = None) -> str:
        ...


class TerminalAskChannel(AskChannel):
    """Ask the human at the terminal. auto_approve answers 'yes' to everything
    (evals, background runs); auto_deny answers 'no'."""

    def __init__(self, auto_approve: bool = False, auto_deny: bool = False) -> None:
        self.auto_approve = auto_approve
        self.auto_deny = auto_deny

    def ask(self, prompt: str, options: list[str] | None = None) -> str:
        if self.auto_approve:
            print(f"  [ask] {prompt} -> auto-approved")
            return "yes"
        if self.auto_deny:
            print(f"  [ask] {prompt} -> auto-denied")
            return "no"
        print(f"\n  [ask] {prompt}")
        if options:
            print(f"  [ask] options: {', '.join(options)}")
        try:
            return input("  [ask] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("  [ask] no input available")
            return ""


class CallbackAskChannel(AskChannel):
    """Delegate the question to a callable: fn(prompt, options) -> str."""

    def __init__(self, fn: Callable[[str, list[str] | None], str]) -> None:
        self.fn = fn

    def ask(self, prompt: str, options: list[str] | None = None) -> str:
        return self.fn(prompt, options) or ""


class QueueAskChannel(AskChannel):
    """Ask across a process boundary (the Frontier chat sidecar).

    Puts the question on ask_queue and blocks until the answer arrives on
    answer_queue. The serving layer forwards questions to the UI over SSE
    and posts answers back. A timeout avoids leaking the agent thread if
    nobody answers."""

    def __init__(self, ask_queue, answer_queue, timeout: float = 600,
                 meta: dict | None = None) -> None:
        self.ask_queue = ask_queue
        self.answer_queue = answer_queue
        self.timeout = timeout
        self.meta = meta or {}

    def ask(self, prompt: str, options: list[str] | None = None) -> str:
        self.ask_queue.put({"prompt": prompt, "options": options, **self.meta})
        try:
            return self.answer_queue.get(timeout=self.timeout) or ""
        except Exception:
            return ""


def channel_from_auto(auto_approve: bool) -> AskChannel:
    return TerminalAskChannel(auto_approve=auto_approve)
