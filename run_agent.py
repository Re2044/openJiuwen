"""Entry point of the competition container.

    python /app/run_agent.py --tasks-dir /data/tasks [--out solutions.json]

--tasks-dir and --out override the TASKS and OUT environment variables; everything else is read from the environment
(LLM_BASE_URL, LLM_API_KEY, BUDGET, ...) as described in README.md. The run itself is main.py.
"""
import argparse
import os
import runpy
from pathlib import Path

parser = argparse.ArgumentParser(description="Plan and solve the tasks in a task folder within the token budget.")
parser.add_argument("--tasks-dir", help="folder with one subfolder per task (default: TASKS)")
parser.add_argument("--out", help="file or folder for the solutions file (default: OUT)")
args, _ = parser.parse_known_args()
if args.tasks_dir:
    os.environ["TASKS"] = args.tasks_dir
if args.out:
    os.environ["OUT"] = args.out

runpy.run_path(str(Path(__file__).with_name("main.py")), run_name="__main__")
