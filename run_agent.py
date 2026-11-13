"""Entry point of the competition run: python run_agent.py --tasks-dir DIR [--out FILE] [--budget N]

The arguments become the TASKS, OUT and BUDGET variables that main.py reads; a variable already set in the
environment is kept when its argument is not given.
"""
import argparse
import os
import runpy
from pathlib import Path

parser = argparse.ArgumentParser(description="Solve the tasks of a folder within a token budget.")
parser.add_argument("--tasks-dir", "--tasks", dest="tasks", help="folder with the task folders")
parser.add_argument("--out", "--out-file", "--output", dest="out", help="file for the saved solutions")
parser.add_argument("--budget", type=int, help="token budget of the run, in output tokens")
args, _ = parser.parse_known_args()  # unknown arguments must not stop the run

if args.tasks:
    os.environ["TASKS"] = args.tasks
if args.out:
    os.environ["OUT"] = args.out
if args.budget is not None:
    os.environ["BUDGET"] = str(args.budget)

runpy.run_path(str(Path(__file__).with_name("main.py")), run_name="__main__")
