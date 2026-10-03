"""SecondBrain: the shared virtual filesystem.

Polar's disclosed "second brain" — a cloud virtual filesystem shared across all
subagents that doubles as task memory *and* the inter-agent coordination layer.
In v0 it is a local directory; the interface is what matters (write/read/list/
append), so it can be backed by cloud storage later without changing agents.

Layout convention:
    tasks/<task_id>/plan.md        — orchestrator's plan
    tasks/<task_id>/notes/<worker>.md — worker findings
    tasks/<task_id>/result.md      — merged result
    memory/*.md                    — long-lived learnings ("compounding memory")
"""

from __future__ import annotations

from pathlib import Path


class SecondBrain:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _resolve(self, path: str) -> Path:
        p = (self.root / path).resolve()
        if self.root.resolve() not in p.parents and p != self.root.resolve():
            raise ValueError(f"Path escapes brain root: {path!r}")
        return p

    def write(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"Wrote {len(content)} chars to {path}"

    def append(self, path: str, content: str) -> str:
        p = self._resolve(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a") as f:
            f.write(content if content.endswith("\n") else content + "\n")
        return f"Appended to {path}"

    def read(self, path: str) -> str:
        p = self._resolve(path)
        if not p.exists():
            return f"(no file at {path})"
        return p.read_text()[:20000]

    def list(self, prefix: str = "") -> str:
        base = self._resolve(prefix) if prefix else self.root
        if not base.exists():
            return "(empty)"
        files = sorted(
            str(p.relative_to(self.root)) for p in base.rglob("*") if p.is_file()
        )
        return "\n".join(files[:200]) if files else "(empty)"
