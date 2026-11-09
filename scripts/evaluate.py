"""Development only: score a finished run against task.json. Reads `expected`, so it is excluded from the image."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def check_task(task_dir: Path) -> tuple[bool, float, str]:
    """Return (correct, reward, task id) for one task directory."""
    meta = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    answer_file = task_dir / "answer.txt"
    answer = answer_file.read_text(encoding="utf-8").strip() if answer_file.exists() else None
    check = meta.get("check")
    if check == "impossible":  # own test tasks: the right move is to spend nothing and answer nothing
        return answer is None, 0.0, meta["task_id"]
    if answer is None:
        correct = False
    elif check == "unittest":
        workspace = task_dir / "workspace"
        correct = all(
            subprocess.run([sys.executable, test.name], cwd=workspace, capture_output=True,
                           timeout=60).returncode == 0
            for test in workspace.glob("test_*.py")
        )
    elif check == "exact_int":
        try:
            correct = int(answer) == int(meta["expected"])
        except ValueError:
            correct = False
    else:
        correct = answer == str(meta["expected"]).strip()
    return correct, float(meta.get("reward", 0)), meta["task_id"]


def evaluate(tasks_dir: Path) -> dict:
    results = [check_task(path.parent) for path in sorted(tasks_dir.glob("*/task.json"))]
    return {
        "score": sum(reward for correct, reward, _ in results if correct),
        "max_score": sum(reward for _, reward, _ in results),
        "solved": sorted(task_id for correct, _, task_id in results if correct),
        "failed": sorted(task_id for correct, _, task_id in results if not correct),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tasks_dir", type=Path)
    print(json.dumps(evaluate(parser.parse_args().tasks_dir), indent=1))
