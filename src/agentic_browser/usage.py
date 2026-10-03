"""M4: credit metering.

Polar meters everything in credits; here we meter in dollars against the
provider's published price card. Every model call's token usage is captured,
priced, and appended to a JSONL ledger. A budget guard refuses new work once
rolling spend exceeds AGENTIC_BUDGET_USD.

Prices are per 1M tokens: (input_cache_miss, input_cache_hit, output).
Verified 2026-10-03 against Moonshot's published rates; override with
AGENTIC_PRICES='{"moonshot:kimi-k3": [3.0, 0.3, 15.0]}'.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

DEFAULT_PRICES: dict[str, tuple[float, float, float]] = {
    "moonshot:kimi-k3": (3.00, 0.30, 15.00),
    "moonshot:kimi-k2.6": (0.95, 0.16, 4.00),
    "moonshot:kimi-k2.7-code": (0.95, 0.19, 4.00),
    "moonshot:kimi-k2.7-code-highspeed": (1.90, 0.38, 8.00),
}


def price_table() -> dict[str, tuple[float, float, float]]:
    table = dict(DEFAULT_PRICES)
    raw = os.environ.get("AGENTIC_PRICES", "").strip()
    if raw:
        try:
            for k, v in json.loads(raw).items():
                table[k] = (float(v[0]), float(v[1]), float(v[2]))
        except (json.JSONDecodeError, ValueError, IndexError, TypeError):
            pass
    return table


def cost_of(model: str, prompt_tokens: int, completion_tokens: int, cached_tokens: int = 0) -> float:
    miss, hit, out = price_table().get(model, (0.0, 0.0, 0.0))
    fresh = max(0, prompt_tokens - cached_tokens)
    return (fresh * miss + cached_tokens * hit + completion_tokens * out) / 1_000_000


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class UsageLedger:
    """Thread-safe append-only ledger of model spend."""

    path: Path = Path("usage/ledger.jsonl")

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def record(
        self,
        role: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        tag: str = "",
    ) -> float:
        cost = cost_of(model, prompt_tokens, completion_tokens, cached_tokens)
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "role": role,
            "model": model,
            "tag": tag,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": cached_tokens,
            "cost_usd": round(cost, 6),
        }
        with self._lock:
            with self.path.open("a") as f:
                f.write(json.dumps(entry) + "\n")
        return cost

    def _entries(self, days: float) -> list[dict]:
        if not self.path.exists():
            return []
        cutoff = time.time() - days * 86400
        out = []
        with self._lock:
            for line in self.path.read_text().splitlines():
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = datetime.fromisoformat(e["ts"]).timestamp()
                if ts >= cutoff:
                    out.append(e)
        return out

    def spend_since(self, days: float = 30) -> float:
        return round(sum(e["cost_usd"] for e in self._entries(days)), 4)

    def summary(self, days: float = 1) -> dict:
        entries = self._entries(days)
        by_model: dict[str, dict] = {}
        for e in entries:
            m = by_model.setdefault(
                e["model"], {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "cost_usd": 0.0}
            )
            m["calls"] += 1
            m["prompt_tokens"] += e["prompt_tokens"]
            m["completion_tokens"] += e["completion_tokens"]
            m["cost_usd"] = round(m["cost_usd"] + e["cost_usd"], 4)
        return {
            "days": days,
            "calls": len(entries),
            "total_usd": round(sum(e["cost_usd"] for e in entries), 4),
            "by_model": by_model,
        }

    def check_budget(self, days: float = 30) -> None:
        """Raise BudgetExceeded if the rolling spend hit AGENTIC_BUDGET_USD."""
        budget = os.environ.get("AGENTIC_BUDGET_USD", "").strip()
        if not budget:
            return
        try:
            limit = float(budget)
        except ValueError:
            return
        spent = self.spend_since(days)
        if spent >= limit:
            raise BudgetExceeded(
                f"Spend ${spent:.4f} hit the budget ${limit:.2f} "
                f"(AGENTIC_BUDGET_USD, {days:g}-day window). Raise it or unset to continue."
            )


def format_report(summary: dict) -> str:
    lines = [
        f"Usage — last {summary['days']:g} day(s): "
        f"{summary['calls']} calls, ${summary['total_usd']:.4f}"
    ]
    for model, m in sorted(summary["by_model"].items(), key=lambda kv: -kv[1]["cost_usd"]):
        lines.append(
            f"  {model}: {m['calls']} calls, "
            f"{m['prompt_tokens'] + m['completion_tokens']:,} tokens, ${m['cost_usd']:.4f}"
        )
    return "\n".join(lines)
