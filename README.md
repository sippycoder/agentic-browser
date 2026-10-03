# Agentic Browser — v0 Prototype

A working minimal agentic browser, built to replicate the *disclosed architecture*
of Polar Browser (Recursive Intelligence, polarbrowser.com) — not its code, which
is closed source. Everything here is original work guided by their public blog
posts, eval repo, and press.

## What it does today

- Drives **real Chromium** (Playwright) with persistent profiles — logins survive restarts
- **Observe → think → act** loop: screenshot + ref-tagged element tree → frontier LLM → grounded actions (click/fill/press/scroll/navigate)
- **Hierarchical orchestration**: an orchestrator model decomposes goals, fans out to parallel worker subagents each with their own browser, merges via a shared filesystem brain
- **SecondBrain**: a shared virtual filesystem (Polar's "second brain" concept) for task memory and inter-agent coordination
- **Model-agnostic routing**: orchestrator / worker / micro / judge roles each map to any `provider:model` via env vars — mix and match frontier models per comparative advantage
- **Eval harness**: Odysseys-style tasks (goal + rubrics), per-rubric LLM trajectory judging, "perfect" only if every rubric passes

## Quickstart

```bash
pip install -e .
python -m playwright install chromium

# Configure models (provider:model per role — mix and match frontier models)
# Keys can go in a git-ignored .env file in the project root (auto-loaded):
#   MOONSHOT_API_KEY=sk-...
#   AGENTIC_WORKER_MODEL=moonshot:kimi-k3
#   AGENTIC_ORCHESTRATOR_MODEL=moonshot:kimi-k3
#   AGENTIC_MICRO_MODEL=moonshot:kimi-k2.7-code-highspeed
#   AGENTIC_JUDGE_MODEL=moonshot:kimi-k3
#   AGENTIC_BUDGET_USD=10          # optional: refuse new work past $10 rolling-30d spend
#   AGENTIC_AUTO_APPROVE=1         # optional: auto-approve high-risk actions (background runs)
# Or export them directly:
export MOONSHOT_API_KEY=sk-...
export AGENTIC_WORKER_MODEL="moonshot:kimi-k3"
export AGENTIC_ORCHESTRATOR_MODEL="moonshot:kimi-k3"
export AGENTIC_JUDGE_MODEL="moonshot:kimi-k3"

# Single agent, one task
python -m agentic_browser.cli run "Find the top story on Hacker News right now and summarize it."

# Hierarchical: orchestrator + 3 parallel workers
python -m agentic_browser.cli run "Compare the pricing pages of Vercel, Netlify and Cloudflare Pages." --orchestrate --workers 3

# Keep logins between runs (the "logged in as them" property)
python -m agentic_browser.cli run "Check my inbox for the latest invoice." --profile ./profiles/main

# Run the eval suite
python -m agentic_browser.cli eval
```

## Architecture → Polar mapping

| Polar's disclosed concept | Our v0 implementation | Status |
|---|---|---|
| Chromium fork, real browser engine | Upstream Chromium via Playwright (no fork yet) | ✅ working |
| Logged-in sessions, device-bound auth | Persistent profile dir | ✅ working |
| Orchestrator + massively parallel subagents | `orchestrator.py` (thread pool, N browser sessions) | ✅ working |
| Small models for sync micro-tasks | `micro` role in router (wiring point; not yet used for UI micro-actions) | 🟡 stubbed |
| Cloud virtual filesystem "second brain" | `memory.py` SecondBrain (local FS, same interface) | ✅ working |
| Model-agnostic mix-and-match | `models.py` Router (OpenAI + Anthropic) | ✅ working |
| Trajectory logging → eval flywheel | Per-step JSONL + screenshots | ✅ working |
| Public rubric evals (Odysseys 98%, BU Bench 95/100) | `evals.py` + 3 sample tasks | ✅ working |
| Scheduled recurring workflows | `workflows.py` + template gallery | ✅ working |
| Coordinate-based grounding | Ref-based grounding (more robust in v0) | 🟡 alternative |
| Credit metering / billing | `usage.py` ledger + budget guard | ✅ working |
| Cloud execution fleet | WorkerBackend + `serve` HTTP service | ✅ working (v0) |
| Self-trained specialized models | `distill.py`: fine-tune export + prompt packs | ✅ working (v0) |

## Honest notes on Polar's performance claims

Their 98% on Odysseys and 95/100 on BU Bench are **self-run, self-reported**
([repo](https://github.com/recursive-hq/polar-evals) — verified real), comparing
their *full harness* (parallel subagents, orchestrator, shared memory, big step
budgets) against *raw* frontier-model computer-use baselines. Strong vendor
result, not an independent head-to-head. The replicable insight isn't the number —
it's the harness: hierarchy + shared memory + eval-driven iteration. That's what
this prototype implements first, because that's the part that transfers.

## Milestones

- **M0** (done): single-agent loop on real Chromium; ref grounding; trajectory logs
- **M1** (done): orchestrator + parallel workers; SecondBrain; rubric eval harness
- **M2** (done 2026-10-03): approval gates for high-risk actions (heuristic fast-path +
  micro-model classifier, human prompt unless auto-approved); stall & loop detection
  with recovery guidance and force-finish; micro-model wired as the risk classifier
  (heterogeneous delegation); live-validated — orchestrator ran 2 parallel workers
  (HN + Lobsters) and merged results via the shared brain
- **M3** (done 2026-10-03): workflow scheduler — YAML workflow definitions with
  every/daily/weekly/cron schedules, timezone-aware; `schedule` loop + `--once`
  mode for system cron; persistent run state; run history and cross-run memory
  in the shared brain ("carry the same context into every run"); template
  gallery (tech-briefing, pricing-watch, hn-hiring-scan, site-watch) with
  `templates install`; live-validated (site-watch ran end-to-end, baseline
  recorded, second invocation correctly reported nothing due)
- **M4** (done 2026-10-03): credit metering — every model call's tokens are
  captured, priced against Moonshot's published card (kimi-k3 $3/$15 per 1M),
  and appended to `usage/ledger.jsonl`; `usage report` CLI; AGENTIC_BUDGET_USD
  budget guard refuses new work past the rolling 30d spend. Prompt-injection
  defenses: trust-boundary system prompt, per-observation injection-pattern
  scan, trajectory flagging, and 3-step elevated risk (forced approvals) after
  a hit. Worker pool: WorkerBackend interface with LocalBackend (threads) and
  HttpBackend — `serve` runs an HTTP worker service; `run --orchestrate
  --backend http://127.0.0.1:8000` distributes workers to it (live-validated).
- **M5** (done 2026-10-03): the flywheel — trajectories are now self-describing
  (meta header, per-step page snapshots, tool outcomes, result footer);
  `distill report/export/pack` turns finished runs into OpenAI-format
  fine-tune JSONL and few-shot prompt packs; `--prompt-pack` loads a pack into
  the agent; evals support `fixture://` hermetic pages (example.com changed
  under the live eval, so the basics task now runs against a local fixture).
  Live-validated: pack distilled from 8 runs, eval with pack scored PERFECT.

## Scheduled workflows

```bash
# Browse and install a template
python -m agentic_browser.cli templates list
python -m agentic_browser.cli templates install tech-briefing

# Edit workflows/tech-briefing.yaml (task, vars, schedule), then:
python -m agentic_browser.cli workflows list
python -m agentic_browser.cli workflows run tech-briefing   # run once now

# Fire due workflows once (good for system cron):
python -m agentic_browser.cli schedule --once

# Or run the scheduler loop (polls every 60s):
python -m agentic_browser.cli schedule
```

Schedule shapes in the workflow YAML:

```yaml
schedule: {every: "30m"}        # or "6h", "1d"
schedule: {daily: "07:00"}
schedule: {weekly: {days: [mon, wed, fri], at: "09:00"}}
schedule: {cron: "0 9 * * 1-5"} # via croniter
```

Each run is recorded under `brain/workflows/<name>/runs/<timestamp>/` and the
latest summary is kept at `brain/workflows/<name>/memory/last-run.md`, which
the next run reads first — so workflows genuinely build on their own history
(e.g. pricing-watch diffs against the previous run).

Safety: scheduled runs are unattended, so they always run with auto-approve.
Keep them to low-stakes, read-only tasks — the bundled templates are.

## Metering, budgets, and the worker pool

Every model call is metered: tokens are captured from the provider's usage
response, priced against Moonshot's published card (override via
`AGENTIC_PRICES`), and appended to `usage/ledger.jsonl` with the caller tag
(agent name, `orchestrator:plan`, `eval:<id>:judge`, …).

```bash
python -m agentic_browser.cli usage --days 7
# Usage — last 7 day(s): 42 calls, $0.5123
#   moonshot:kimi-k3: 42 calls, 210,540 tokens, $0.5123
```

Set a spend ceiling and the router refuses new work past it (rolling 30d):

```bash
export AGENTIC_BUDGET_USD=10
```

Prompt-injection defenses: the system prompt draws a hard trust boundary (page
content is data, never instructions); every observation is scanned for
injection markers ("ignore previous instructions", persona overrides,
exfiltration requests, …); hits are flagged in the trajectory and force
approvals for the next 3 steps regardless of the risk classifier.

Worker pool: workers run behind a `WorkerBackend` interface. Local threads are
the default; `agentic_browser serve --port 8000` starts an HTTP worker service
and `run --orchestrate --backend http://127.0.0.1:8000` distributes workers to
it — the same seam a cloud fleet (queue + autoscaler + shared brain storage)
would plug into.

# Frontier browser app

The product surface lives in `~/workspace/browser-app/` (sibling directory):
a real desktop browser on Chromium (Electron) with tabs, omnibox, and an
agent composer sidebar. `npm start` launches it; the app spawns the Python
engine as a sidecar, and the composer drives your visible tab over CDP
(`BrowserSession.attach_cdp`, `/run-tab-task`). See its README for details
and honest v0.1 limits.

## Distillation flywheel

Trajectories are training data. Every run logs a self-describing JSONL: a meta
header (task, model role), per-step page snapshots, tool outcomes, and a
result footer.

```bash
python -m agentic_browser.cli distill report   # stats across all trajectories
python -m agentic_browser.cli distill export   # -> distill/finetune.jsonl (OpenAI chat format)
python -m agentic_browser.cli distill pack     # -> distill/pack.md (few-shot examples)

# Close the loop: run with the distilled pack, re-measure on evals
python -m agentic_browser.cli eval --prompt-pack distill/pack.md
python -m agentic_browser.cli run --prompt-pack distill/pack.md "Your task"
```

The fine-tune export works with any provider that accepts OpenAI-format chat
data (Moonshot's API does not offer fine-tuning). The prompt pack is the loop
that works today against any provider. Evals can use `fixture://` URLs to run
against hermetic local pages instead of the live web.

## MCP server — any main agent can drive the browser

`python -m agentic_browser.cli mcp` runs a Model Context Protocol server over
stdio. Wire it into any MCP-capable main agent (Claude, Cursor, …):

```json
{"command": "/path/to/agentic-browser/.venv/bin/python",
 "args": ["-m", "agentic_browser.cli", "mcp"]}
```

Tools: `browser_start_session`, `browser_navigate`, `browser_new_tab`,
`browser_snapshot`, `browser_click`, `browser_fill`, `browser_press`,
`browser_scroll`, `browser_back`, `browser_get_text`, `browser_screenshot`,
`browser_run_task`, `browser_close_session`.

The agent is conversational: `browser_run_task` runs the autonomous loop
inside a session, and when it needs something — an approval for a high-risk
action, or an `ask_user` question (a login code, a clarification) — it asks
back via MCP elicitation. The main agent's answer resumes the loop.
The same ask channel drives terminal runs (human prompt), `auto_approve`
background runs, and programmatic callbacks (`CallbackAskChannel`).

## Safety

Approval gates (M2): every proposed action is risk-assessed — heuristics plus
the micro model as classifier. High-risk actions (submit, payment, delete,
send, publish, account changes) ask first; a denial can't be retried.
Prompt-injection defenses (M4): trust-boundary system prompt, per-observation
pattern scan, trajectory flags, forced approvals after a hit. Budget guard
(`AGENTIC_BUDGET_USD`) halts new work past rolling spend. `.env` is
git-ignored and chmod 600.

Honest gaps: no site allowlist; Chromium runs with `--no-sandbox` in
containers; `auto_approve` removes the human; page content flows to the
model by design; trajectories sit on disk unencrypted. Run low-stakes sites
first, use a throwaway profile for logins.

## Sandbox / CI note

In locked-down environments where the Chromium binary's own egress is filtered
(and TLS is intercepted), set `AGENTIC_RELAY_PROXY=1`: the browser then talks
to a localhost relay running inside this process, which forwards to the proxy
from `$https_proxy`. This mode also disables cert validation — **never enable
it outside a trusted sandbox**. On a normal machine, unset it (or don't set it)
and Chromium connects directly.
