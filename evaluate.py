"""Scores a finished run the way the final check does, on tasks that carry reference answers (`expected`).

Usage: python evaluate.py [tasks dir]   (default: ./task-dir)
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


def main():
    report = score_tasks(sys.argv[1] if len(sys.argv) > 1 else "task-dir")
    for r in report["tasks"]:
        print(f"{r['task_id']:<22} {r['status']:<20} {r['points']}/{r['reward']}  {r['detail']}")
    print("\n" + summary_line(report))


if __name__ == "__main__":
    main()
