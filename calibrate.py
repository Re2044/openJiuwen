"""Rebuilds estimates.json from a finished run: python calibrate.py [tasks dir] [trajectories.jsonl] [output]

Uses the trajectories the executor writes (model, cost, time of every attempt) and the reference answers
of the task set (via evaluate.py). Only solvable tasks with a reference answer are counted.
An attempt is a success when it produced the accepted answer and that answer is correct.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import config
from evaluate import CORRECT, NO_ANSWER, WRONG, check_task


def main():
    tasks_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "task-dir-final-run")
    trajectories = Path(sys.argv[2] if len(sys.argv) > 2 else config.TRAJECTORIES_FILE)
    samples = defaultdict(list)  # (task type, model) -> [(cost, seconds, success)]
    for line in trajectories.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        task_dir = tasks_dir / record["task_id"]
        if not (task_dir / "task.json").exists():
            continue
        status, _, _ = check_task(task_dir)
        if status not in (CORRECT, WRONG, NO_ANSWER):  # unsolvable or without a reference answer
            continue
        for attempt in record["attempts"]:
            if attempt["calls"] == 0:
                continue  # the model never answered, e.g. it is not available to the key
            success = attempt["submitted"] and status == CORRECT
            samples[(record["task_type"], attempt["model"])].append((attempt["cost"], attempt["seconds"], success))

    types = defaultdict(dict)
    for (task_type, model), rows in sorted(samples.items()):
        types[task_type][model] = {
            "cost": round(statistics.median(row[0] for row in rows)),
            "seconds": round(statistics.median(row[1] for row in rows)),
            "success": round(sum(row[2] for row in rows) / len(rows), 2),
            "n": len(rows),
        }
        print(f"{task_type:<14} {model:<26} n {len(rows):>3}  cost {types[task_type][model]['cost']:>6}  "
              f"seconds {types[task_type][model]['seconds']:>4}  success {types[task_type][model]['success']:.2f}")
    if not types:
        sys.exit("no usable attempts found")
    table = {"source": f"calibrate.py on {tasks_dir.name} and {trajectories.name}",
             "note": "cost and seconds are medians per attempt; n is the number of attempts", "types": types}
    output = Path(sys.argv[3]) if len(sys.argv) > 3 else Path(__file__).with_name("estimates.json")
    output.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    print(f"{output} written")


if __name__ == "__main__":
    main()
