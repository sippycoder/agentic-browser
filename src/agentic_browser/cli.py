"""CLI: run tasks, run evals.

    python -m agentic_browser.cli run "Research ..." [--headed] [--profile DIR] [--orchestrate]
    python -m agentic_browser.cli eval [--task wikipedia-python] [--headed]

Requires model config in env (see models.py) and an API key for the provider.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from agentic_browser.agent import BrowserAgent
from agentic_browser.browser import BrowserSession
from agentic_browser.evals import report, run_eval
from agentic_browser.memory import SecondBrain
from agentic_browser.models import Router
from agentic_browser.orchestrator import HttpBackend, Orchestrator
from agentic_browser.serve import serve
from agentic_browser.usage import UsageLedger, format_report
from agentic_browser.workflows import Scheduler, describe_schedule, load_workflows


def cmd_run(args: argparse.Namespace) -> int:
    router = Router()
    brain = SecondBrain(Path(args.brain))
    auto_approve = args.auto_approve or os.environ.get("AGENTIC_AUTO_APPROVE") == "1"
    session = BrowserSession(
        headless=not args.headed, profile_dir=args.profile
    ).start()
    try:
        if args.orchestrate:
            backend = HttpBackend(args.backend) if args.backend else None
            orch = Orchestrator(
                brain=brain,
                router=router,
                max_workers=args.workers,
                headless=not args.headed,
                auto_approve=auto_approve,
                backend=backend,
            )
            answer = orch.run(args.task)
            print("\n=== FINAL ANSWER ===\n")
            print(answer)
        else:
            agent = BrowserAgent(
                session=session,
                brain=brain,
                router=router,
                model_role="worker",
                max_steps=args.max_steps,
                trajectory_root=Path(args.trajectories),
                auto_approve=auto_approve,
                prompt_pack=args.prompt_pack,
            )
            result = agent.run(args.task, start_url=args.start_url)
            print(f"\n=== DONE (finished={result.finished}, steps={result.steps}, "
                  f"{result.elapsed_s:.0f}s) ===")
            print(f"trajectory: {result.trajectory_dir}\n")
            print(result.answer or "(no answer — agent did not call finish)")
    finally:
        session.close()
    return 0


def cmd_schedule(args: argparse.Namespace) -> int:
    router = Router()
    brain = SecondBrain(Path(args.brain))
    sched = Scheduler(args.workflows_dir, brain, router, headless=not args.headed)
    if args.once:
        sched.run_once()
    else:
        sched.loop(poll_seconds=args.poll)
    return 0


def cmd_workflows(args: argparse.Namespace) -> int:
    wdir = Path(args.workflows_dir)
    if args.action == "list":
        wfs = load_workflows(wdir)
        if not wfs:
            print(f"No workflows in {wdir} — try: agentic_browser templates install tech-briefing")
            return 0
        for wf in wfs:
            print(f"- {wf.name}: {wf.description or '(no description)'} [{describe_schedule(wf.schedule)}]")
        return 0
    if args.action == "run":
        wfs = {wf.name: wf for wf in load_workflows(wdir)}
        if args.name not in wfs:
            print(f"Unknown workflow {args.name!r} in {wdir}")
            return 1
        router = Router()
        brain = SecondBrain(Path(args.brain))
        sched = Scheduler(wdir, brain, router, headless=not args.headed)
        answer = sched.run_workflow(wfs[args.name], headless=not args.headed)
        print("\n=== RESULT ===\n")
        print(answer)
        return 0
    return 1


def cmd_templates(args: argparse.Namespace) -> int:
    tdir = Path(__file__).resolve().parents[2] / "templates"
    if args.action == "list":
        for p in sorted(tdir.glob("*.yaml")):
            data = __import__("yaml").safe_load(p.read_text())
            print(f"- {p.stem}: {data.get('description', '')} [{describe_schedule(data['schedule'])}]")
        return 0
    if args.action == "install":
        src = tdir / f"{args.name}.yaml"
        if not src.exists():
            print(f"Unknown template {args.name!r}")
            return 1
        wdir = Path(args.workflows_dir)
        wdir.mkdir(parents=True, exist_ok=True)
        dest = wdir / f"{args.name}.yaml"
        if dest.exists():
            print(f"{dest} already exists — not overwriting")
            return 1
        dest.write_text(src.read_text())
        print(f"Installed {args.name} -> {dest}")
        print("Edit the vars/task, then run: agentic_browser schedule --once")
        return 0
    return 1


def cmd_serve(args: argparse.Namespace) -> int:
    serve(port=args.port, brain_root=args.brain, host=args.host)
    return 0


def cmd_mcp(args: argparse.Namespace) -> int:
    from agentic_browser.mcp_server import main as mcp_main

    mcp_main(cdp_url=args.cdp_url)
    return 0


def cmd_distill(args: argparse.Namespace) -> int:
    from agentic_browser.distill import (
        export_jsonl,
        iter_trajectories,
        mine_fewshots,
        quality_filter,
        report,
        write_pack,
    )

    trajs = iter_trajectories(args.trajectories_dir)
    if args.action == "report":
        print(report(trajs))
        return 0
    eligible = quality_filter(trajs)
    if args.action == "export":
        out = args.out or "distill/finetune.jsonl"
        n = export_jsonl(eligible, out)
        print(f"Exported {n} trajectories -> {out} (OpenAI chat fine-tune format)")
        return 0
    if args.action == "pack":
        out = args.out or "distill/pack.md"
        examples = mine_fewshots(eligible, max_examples=args.max_examples)
        write_pack(examples, out, n_runs=len(eligible))
        print(f"Wrote {len(examples)} few-shot examples from {len(eligible)} runs -> {out}")
        print("Use with: run --prompt-pack", out, " / eval --prompt-pack", out)
        return 0
    return 1
    serve(port=args.port, brain_root=args.brain, host=args.host)
    return 0


def cmd_usage(args: argparse.Namespace) -> int:
    ledger = UsageLedger(Path(args.ledger))
    print(format_report(ledger.summary(days=args.days)))
    budget = os.environ.get("AGENTIC_BUDGET_USD", "").strip()
    if budget:
        print(f"Budget: ${float(budget):.2f} / 30d — spent ${ledger.spend_since(30):.4f}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    grades = run_eval(
        Path(args.tasks_dir),
        task_ids=args.task,
        headless=not args.headed,
        max_steps=args.max_steps,
        prompt_pack=args.prompt_pack,
    )
    text = report(grades)
    print(text)
    Path("eval-report.md").write_text(text)
    print("\nWrote eval-report.md")
    return 0 if all(g.perfect for g in grades) else 1


def main() -> int:
    ap = argparse.ArgumentParser(prog="agentic-browser")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="Run a single task with one agent.")
    r.add_argument("task")
    r.add_argument("--start-url")
    r.add_argument("--headed", action="store_true")
    r.add_argument("--profile", default=None, help="Persistent Chromium profile dir (keeps logins).")
    r.add_argument("--max-steps", type=int, default=40)
    r.add_argument("--brain", default="brain")
    r.add_argument("--trajectories", default="trajectories")
    r.add_argument("--orchestrate", action="store_true", help="Use the hierarchical orchestrator.")
    r.add_argument("--workers", type=int, default=3)
    r.add_argument("--auto-approve", action="store_true",
                   help="Auto-approve high-risk actions (also via AGENTIC_AUTO_APPROVE=1). Needed for background runs.")
    r.add_argument("--backend", default=None,
                   help="Worker backend URL for orchestrated runs, e.g. http://127.0.0.1:8000 (see 'serve'). Default: local threads.")
    r.add_argument("--prompt-pack", default=None,
                   help="Path to a distilled few-shot prompt pack (see 'distill pack').")
    r.set_defaults(fn=cmd_run)

    e = sub.add_parser("eval", help="Run the eval suite.")
    e.add_argument("--task", action="append", default=None)
    e.add_argument("--headed", action="store_true")
    e.add_argument("--max-steps", type=int, default=25)
    e.add_argument("--tasks-dir", default=str(Path(__file__).resolve().parents[2] / "evals" / "tasks"))
    e.add_argument("--prompt-pack", default=None,
                   help="Path to a distilled few-shot prompt pack (see 'distill pack').")
    e.set_defaults(fn=cmd_eval)

    s = sub.add_parser("schedule", help="Run the workflow scheduler (recurring tasks).")
    s.add_argument("--once", action="store_true", help="Fire due workflows once and exit (for system cron).")
    s.add_argument("--poll", type=int, default=60, help="Poll interval in seconds for the loop.")
    s.add_argument("--headed", action="store_true")
    s.add_argument("--brain", default="brain")
    s.add_argument("--workflows-dir", default="workflows")
    s.set_defaults(fn=cmd_schedule)

    w = sub.add_parser("workflows", help="List or manually run workflows.")
    w.add_argument("action", choices=["list", "run"])
    w.add_argument("name", nargs="?", default=None, help="Workflow name (for run).")
    w.add_argument("--headed", action="store_true")
    w.add_argument("--brain", default="brain")
    w.add_argument("--workflows-dir", default="workflows")
    w.set_defaults(fn=cmd_workflows)

    t = sub.add_parser("templates", help="Browse and install workflow templates.")
    t.add_argument("action", choices=["list", "install"])
    t.add_argument("name", nargs="?", default=None, help="Template name (for install).")
    t.add_argument("--workflows-dir", default="workflows")
    t.set_defaults(fn=cmd_templates)

    v = sub.add_parser("serve", help="Run an HTTP worker service for remote orchestration.")
    v.add_argument("--port", type=int, default=8000)
    v.add_argument("--host", default="127.0.0.1")
    v.add_argument("--brain", default="brain")
    v.set_defaults(fn=cmd_serve)

    m = sub.add_parser("mcp", help="Run the MCP server (stdio) so any main agent can drive the browser.")
    m.add_argument("--cdp-url", default=None,
                   help="Attach to a running browser over CDP (e.g. the Frontier app) "
                        "instead of launching headless Chromium.")
    m.set_defaults(fn=cmd_mcp)

    u = sub.add_parser("usage", help="Show metered model spend.")
    u.add_argument("action", nargs="?", choices=["report"], default="report")
    u.add_argument("--days", type=float, default=1.0)
    u.add_argument("--ledger", default="usage/ledger.jsonl")
    u.set_defaults(fn=cmd_usage)

    d = sub.add_parser("distill", help="Distill trajectories into training data / prompt packs (M5 flywheel).")
    d.add_argument("action", choices=["report", "export", "pack"])
    d.add_argument("--trajectories-dir", default="trajectories")
    d.add_argument("--out", default=None, help="Output path (default: distill/finetune.jsonl or distill/pack.md).")
    d.add_argument("--max-examples", type=int, default=8)
    d.set_defaults(fn=cmd_distill)

    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
