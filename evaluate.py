"""Scores a finished run the way the final check does, on tasks that carry reference answers (`expected`).

Usage: python evaluate.py [tasks dir] [--report run_report.json]   (default: ./task-dir)
With --report it also compares the judge's estimates with what solving took, per group of tasks (see estimate_table).
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


def _judge_view(judge):
    """(raw token estimate per model, cost estimate per model, step estimate, judged infeasible) from a report entry.

    Reads both report formats: the old one with the judge's own token count per model ({model: {tokens, seconds}},
    -1 meaning infeasible) and the current one ({"verdict": ..., "estimates": {model: {steps, tokens, cost}}}).
    """
    if "estimates" in judge:
        estimates = judge["estimates"]
        return ({m: e["tokens"] for m, e in estimates.items()}, {m: e["cost"] for m, e in estimates.items()},
                next(iter(estimates.values()))["steps"] if estimates else None,
                judge.get("verdict") is not None and not estimates)
    tokens = {m: e["tokens"] for m, e in judge.items() if e["tokens"] != -1 and e["seconds"] != -1}
    return tokens, {}, None, bool(judge) and not tokens


def estimate_table(tasks_dir, run_report):
    """Lines comparing the judge's estimates with what solving took, per group of tasks.

    est tok / actual: median tokens (input + output) estimated for the planned model / used by solving. When a task
    was not planned, the median estimate over models is taken.
    ratio: estimated / actual tokens per task, median over the group; above 1 the judge overestimates.
    in 2x: share of solved tasks whose estimate is between half and double the actual tokens.
    est cost / cost: median cost in competition units, estimated (current report format only) / actual.
    steps: median steps estimated / model calls made.
    right: tasks answered correctly, or left unanswered when unsolvable.
    """
    groups = {}
    for task_json in sorted(Path(tasks_dir).glob("*/task.json")):
        meta = json.loads(task_json.read_text(encoding="utf-8"))
        entry = run_report["tasks"].get(meta["task_id"])
        if entry is None:
            continue
        g = groups.setdefault(_group(meta), {"n": 0, "infeasible": 0, "planned": 0, "outcomes": {}, "est": [],
                                             "actual": [], "ratio": [], "cost": [], "est_cost": [], "steps": [],
                                             "calls": [], "right": 0})
        g["n"] += 1
        g["right"] += check_task(task_json.parent)[0] in (CORRECT, SKIPPED_OK)
        tokens, costs, steps, infeasible = _judge_view(entry["judge"])
        g["infeasible"] += infeasible
        planned = entry.get("planned_model")
        estimate = tokens.get(planned) if planned else (statistics.median(tokens.values()) if tokens else None)
        if estimate is not None:
            g["est"].append(estimate)
        if planned in costs:
            g["est_cost"].append(costs[planned])
        if steps is not None:
            g["steps"].append(steps)
        solve = entry.get("solve")
        if not planned or not solve:
            continue
        g["planned"] += 1
        g["outcomes"][solve["outcome"]] = g["outcomes"].get(solve["outcome"], 0) + 1
        actual = sum(a["input_tokens"] + a["output_tokens"] for a in solve["attempts"])
        g["cost"].append(sum(a["cost"] for a in solve["attempts"]))
        g["calls"].append(sum(a["calls"] for a in solve["attempts"]))
        if actual and estimate is not None:  # an estimate of 0 tokens counts, as ratio 0
            g["actual"].append(actual)
            g["ratio"].append(estimate / actual)

    def med(values, fmt="{:.0f}"):
        return fmt.format(statistics.median(values)) if values else "-"

    lines = [f"{'group':<30} {'n':>3} {'infeas':>6} {'plan':>4} {'est tok':>8} {'actual':>8} {'ratio':>6} "
             f"{'in 2x':>6} {'est cost':>8} {'cost':>6} {'steps':>7} {'right':>6}  outcomes"]
    for name, g in sorted(groups.items()):
        within = f"{sum(0.5 <= r <= 2 for r in g['ratio']) / len(g['ratio']):.0%}" if g["ratio"] else "-"
        outcomes = ", ".join(f"{k} {v}" for k, v in sorted(g["outcomes"].items()))
        steps = f"{med(g['steps'])}/{med(g['calls'])}"
        lines.append(f"{name:<30} {g['n']:>3} {g['infeasible']:>6} {g['planned']:>4} {med(g['est']):>8} "
                     f"{med(g['actual']):>8} {med(g['ratio'], '{:.2f}'):>6} {within:>6} {med(g['est_cost']):>8} "
                     f"{med(g['cost']):>6} {steps:>7} {str(g['right']) + '/' + str(g['n']):>6}  {outcomes}")
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
        for line in estimate_table(args[0] if args else "task-dir",
                                   json.loads(Path(report_path).read_text(encoding="utf-8"))):
            print(line)


if __name__ == "__main__":
    main()
