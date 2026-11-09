"""Command line for the test bench: python -m bench --agent agents/baseline.py"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="python -m bench", description="Run an agent architecture over tasks.")
    parser.add_argument("--agent", type=Path, default=ROOT / "agents" / "baseline.py", help="file with async solve(env)")
    parser.add_argument("--tasks", type=Path, action="append",
                        help="task directory, repeatable (default: task-dir and my-tasks)")
    parser.add_argument("--only", help="comma-separated task ids")
    parser.add_argument("--type", dest="task_type", help="only tasks of this type")
    parser.add_argument("--stub", action="store_true", help="use the scripted stub instead of a real model")
    parser.add_argument("--budget", type=float, help="token budget for the whole run")
    parser.add_argument("--task-cap", type=float, default=6000.0, help="cost limit per task")
    parser.add_argument("--task-timeout", type=float, default=600.0, help="seconds per task")
    parser.add_argument("--parallel", type=int, default=5, help="sessions at once; 1 makes logs easier to read")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs", help="where to keep run results")
    return parser.parse_args()


def select_and_copy(args: argparse.Namespace, dest: Path) -> None:
    """Copy the chosen tasks, so that a run never changes the originals."""
    only = set(args.only.split(",")) if args.only else None
    dest.mkdir(parents=True)
    for source in args.tasks:
        for task_json in sorted(source.glob("*/task.json")):
            meta = json.loads(task_json.read_text(encoding="utf-8"))
            if only and meta.get("task_id") not in only:
                continue
            if args.task_type and meta.get("task_type") != args.task_type:
                continue
            target = dest / task_json.parent.name
            if target.exists():
                raise SystemExit(f"duplicate task folder name: {task_json.parent.name}")
            shutil.copytree(task_json.parent, target,
                            ignore=shutil.ignore_patterns("__pycache__", ".DS_Store", "answer.txt"))


def load_agent(path: Path):
    spec = importlib.util.spec_from_file_location(f"agent_{path.stem}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "solve"):
        raise SystemExit(f"{path} has no async solve(env) function")
    return module


def print_report(report) -> None:
    print(f"{report.task_id:<24} {report.task_type:<14} {report.status:<11} points {report.points:>4.0f}/{report.reward:<3.0f}"
          f" cost {report.cost:>8.1f}  calls {report.calls:>3}  {report.seconds:>6.1f}s  {report.answer[:24]}"
          + (f"  ! {report.error[:120]}" if report.error else ""), flush=True)


def main() -> int:
    args = parse_args()
    args.agent = args.agent.resolve()
    args.tasks = [path.resolve() for path in (args.tasks or [ROOT / "task-dir", ROOT / "my-tasks"])]
    run_dir = args.runs_dir.resolve() / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.agent.stem}"
    if not args.agent.is_file():
        print(f"agent file not found: {args.agent}", file=sys.stderr)
        return 2
    base_url, api_key = os.getenv("LLM_BASE_URL"), os.getenv("LLM_API_KEY")
    if not args.stub and not (base_url and api_key):
        print("Set LLM_BASE_URL and LLM_API_KEY, or add --stub to check the wiring without a model.", file=sys.stderr)
        return 2

    tasks_dir = run_dir / "tasks"
    select_and_copy(args, tasks_dir)
    # openJiuwen creates ./logs when imported, so leave the project root first.
    os.chdir(run_dir)
    logging.basicConfig(level=logging.ERROR, format="%(levelname)s %(message)s", stream=sys.stderr)
    if os.name == "nt":
        sys.stdout.reconfigure(encoding="utf-8")

    from harness.framework_logs import quiet_framework_logs

    quiet_framework_logs(run_dir)
    from bench.core import run_suite, summarize
    from harness.settings import DEFAULT_MODELS, Settings, normalize_base_url
    from harness.tasks import load_tasks

    stub = None
    if args.stub:
        sys.path.insert(0, str(ROOT / "tests"))
        from stub_llm import StubLLM

        stub = StubLLM(tasks_dir)
        base_url, api_key = stub.start(), "stub"
    settings = Settings(
        tasks_dir=tasks_dir, out=run_dir / "solutions.json", base_url=normalize_base_url(base_url), api_key=api_key,
        models=DEFAULT_MODELS, budget=args.budget, work_root=run_dir, trajectory_path=run_dir / "trajectories.jsonl",
        workers=args.parallel, task_cost_cap=args.task_cap,
    )
    tasks = load_tasks(tasks_dir)
    if not tasks:
        print("no tasks matched", file=sys.stderr)
        return 2

    module = load_agent(args.agent)
    print(f"agent {args.agent.name}, {len(tasks)} tasks, {'STUB, no real model' if stub else settings.base_url}", flush=True)
    reports = asyncio.run(run_suite(module, tasks, settings, run_dir, args.task_timeout, print_report))

    summary = summarize(reports)
    (run_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    max_points = sum(t.reward for t in tasks if t.check != "impossible")
    print(f"\npoints {summary['points']:.0f}/{max_points:.0f}   cost {summary['cost']:.1f}   "
          f"points per 1k tokens {summary['points_per_1k_tokens']}   {summary['statuses']}")
    for task_type, entry in sorted(summary["by_type"].items()):
        print(f"  {task_type:<14} tasks {entry['tasks']:>3.0f}  cost {entry['cost']:>9.1f}  points {entry['points']:>4.0f}")
    print(f"results: {run_dir}")
    sys.stdout.flush()
    logging.shutdown()
    os._exit(0)  # skip teardown: pooled HTTP clients must not delay exit


if __name__ == "__main__":
    sys.exit(main())
