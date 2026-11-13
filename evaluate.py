"""Scores a finished run the way the final check does, on tasks that carry reference answers (`expected`).

Usage: python evaluate.py [tasks dir] [--report trajectories.jsonl]   (default: ./task-dir)
With --report it also compares the planned cost estimates with what solving took, per group of tasks, from the
file a run writes when TRAJECTORIES is set (see estimate_table).
main.py also calls score_tasks() at the end of a run. Solving never reads these scoring fields.

Rules, by the task's `check` field:
- exact_int: answer.txt parsed as an integer equals `expected`
- exact_str: answer.txt equals `expected` exactly, after trimming surrounding whitespace
- unittest: test_*.py in workspace/ pass, and so do the `hidden_tests` from task.json when present
A task without answer.txt scores nothing.
A task with "solvable": false (see make_testset.py) is right when left unanswered; it is worth no points, as in the
final run, and is left out of the maximum.
A task without `expected` (other than unittest) cannot be scored here and is left out of the maximum.
"""
import json
import statistics
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

CORRECT, WRONG, NO_ANSWER, NO_REFERENCE = "correct", "wrong", "no answer", "no reference"
SKIPPED_OK, ANSWERED_UNSOLVABLE = "skipped ok", "answered unsolvable"


def _tests_pass(workspace, hidden_tests):
    """Names of failing test files; the workspace itself is never changed."""
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp) / "ws"
        shutil.copytree(workspace, work)
        if hidden_tests:
            (work / "test_hidden_check.py").write_text(hidden_tests, encoding="utf-8")
        return [test.name for test in sorted(work.glob("test_*.py"))
                if subprocess.run([sys.executable, test.name], cwd=work, capture_output=True,
                                  timeout=60).returncode != 0]


def check_task(task_dir):
    """Return (status, detail, meta) for one task folder."""
    meta = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    check = meta.get("check")
    answer_file = task_dir / "answer.txt"
    has_answer = answer_file.exists()
    if meta.get("solvable") is False:
        if has_answer:
            return ANSWERED_UNSOLVABLE, "answered an unsolvable task: " + repr(
                answer_file.read_text(encoding="utf-8").strip()[:40]), meta
        return SKIPPED_OK, "unsolvable, correctly left unanswered", meta
    if check != "unittest" and "expected" not in meta:
        return NO_REFERENCE, "no expected answer in task.json", meta
    if not has_answer:
        return NO_ANSWER, "no answer.txt", meta
    answer = answer_file.read_text(encoding="utf-8").strip()

    if check == "unittest":
        failed = _tests_pass(task_dir / "workspace", meta.get("hidden_tests"))
        if failed:
            return WRONG, "failing: " + ", ".join(name.replace("test_hidden_check.py", "hidden tests")
                                                  for name in failed), meta
        return CORRECT, "tests pass" + (" incl. hidden" if meta.get("hidden_tests") else ""), meta
    expected = str(meta["expected"]).strip()
    if check == "exact_int":
        try:
            correct = int(answer) == int(expected)
        except ValueError:
            correct = False
    else:
        correct = answer == expected
    return (CORRECT if correct else WRONG), f"answer {answer[:40]!r}, expected {expected!r}", meta


def score_tasks(tasks_dir):
    """Score every task folder in tasks_dir. Returns points, the reachable maximum and per-task results."""
    results = []
    for task_json in sorted(Path(tasks_dir).glob("*/task.json")):
        try:
            status, detail, meta = check_task(task_json.parent)
        except Exception as err:  # one broken task must not hide the score of the others
            status, detail, meta = NO_REFERENCE, f"cannot check: {err}", {"task_id": task_json.parent.name}
        reward = meta.get("reward", 0)
        results.append({"task_id": meta.get("task_id", task_json.parent.name), "status": status,
                        "reward": reward, "points": reward if status == CORRECT else 0, "detail": detail})
    scorable = [r for r in results if r["status"] in (CORRECT, WRONG, NO_ANSWER)]
    return {
        "points": sum(r["points"] for r in results),
        "max_points": sum(r["reward"] for r in scorable),
        "correct": sum(r["status"] == CORRECT for r in results),
        "scorable": len(scorable),
        "unsolvable": sum(r["status"] in (SKIPPED_OK, ANSWERED_UNSOLVABLE) for r in results),
        "unsolvable_skipped": sum(r["status"] == SKIPPED_OK for r in results),
        "unscorable": sum(r["status"] == NO_REFERENCE for r in results),
        "tasks": results,
    }


def summary_line(report):
    line = f"points {report['points']}/{report['max_points']}, {report['correct']}/{report['scorable']} tasks correct"
    if report["unsolvable"]:
        line += f"; unsolvable left unanswered {report['unsolvable_skipped']}/{report['unsolvable']}"
    if report["unscorable"]:
        line += f"; {report['unscorable']} without a reference answer"
    return line


def _group(meta):
    """Group label: task type plus the suite labels from make_suites.py, or solvable/unsolvable."""
    kind = meta.get("trap") or meta.get("difficulty") or ("unsolvable" if meta.get("solvable") is False else "solvable")
    return f"{meta.get('task_type', '?')}/{kind}"


def read_trajectories(path):
    """Rows of a TRAJECTORIES file merged per task: a task tried again in the second pass has two rows."""
    tasks = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        task = tasks.setdefault(row["task_id"], {"estimated_cost": row["estimated_cost"], "actual_cost": 0.0,
                                                  "answered": False, "rows": 0})
        task["actual_cost"] += row["actual_cost"]
        task["answered"] = task["answered"] or row["answered"]
        task["rows"] += 1
    return tasks


def estimate_table(tasks_dir, trajectories):
    """Lines comparing the planned cost estimate of each task with what solving it cost, per group of tasks.

    trajectories: the result of read_trajectories. Groups are the task type plus the labels make_suites.py
    puts in task.json (trap or difficulty), or solvable/unsolvable.
    n: tasks in the group; tried: tasks the run worked on (planned, or picked up in the second pass).
    est / actual: median cost per task in budget units (output tokens + input tokens / 10).
    ratio: estimated / actual cost per task, median over the group; above 1 the estimate is too high.
    in 2x: share of tried tasks whose estimate is between half and double the actual cost.
    right: tasks answered correctly, or left unanswered when unsolvable.
    """
    groups = {}
    for task_json in sorted(Path(tasks_dir).glob("*/task.json")):
        meta = json.loads(task_json.read_text(encoding="utf-8"))
        g = groups.setdefault(_group(meta), {"n": 0, "tried": 0, "answered": 0, "est": [], "actual": [],
                                             "ratio": [], "right": 0})
        g["n"] += 1
        g["right"] += check_task(task_json.parent)[0] in (CORRECT, SKIPPED_OK)
        row = trajectories.get(meta["task_id"])
        if row is None:
            continue
        g["tried"] += 1
        g["answered"] += row["answered"]
        g["est"].append(row["estimated_cost"])
        g["actual"].append(row["actual_cost"])
        if row["actual_cost"] > 0:
            g["ratio"].append(row["estimated_cost"] / row["actual_cost"])

    def med(values, fmt="{:.0f}"):
        return fmt.format(statistics.median(values)) if values else "-"

    lines = [f"{'group':<30} {'n':>3} {'tried':>5} {'est':>6} {'actual':>6} {'ratio':>6} {'in 2x':>6} "
             f"{'answered':>8} {'right':>6}"]
    for name, g in sorted(groups.items()):
        within = f"{sum(0.5 <= r <= 2 for r in g['ratio']) / len(g['ratio']):.0%}" if g["ratio"] else "-"
        lines.append(f"{name:<30} {g['n']:>3} {g['tried']:>5} {med(g['est']):>6} {med(g['actual']):>6} "
                     f"{med(g['ratio'], '{:.2f}'):>6} {within:>6} {g['answered']:>8} "
                     f"{str(g['right']) + '/' + str(g['n']):>6}")
    return lines


def main():
    args = sys.argv[1:]
    report_path = None
    if "--report" in args:
        at = args.index("--report")
        report_path = args[at + 1]
        del args[at:at + 2]
    report = score_tasks(args[0] if args else "task-dir")
    for r in report["tasks"]:
        print(f"{r['task_id']:<22} {r['status']:<20} {r['points']}/{r['reward']}  {r['detail']}")
    print("\n" + summary_line(report))
    if report_path:
        print()
        for line in estimate_table(args[0] if args else "task-dir", read_trajectories(report_path)):
            print(line)


if __name__ == "__main__":
    main()
