"""Competition entry point: python /app/run_agent.py --tasks-dir /data/tasks"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Runs as a script from any cwd: the harness package sits next to this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.settings import Settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Budget-aware task harness on openJiuwen")
    parser.add_argument("--tasks-dir", default=os.getenv("TASKS"), help="directory with task folders")
    parser.add_argument("--out", default=os.getenv("OUT"), help="optional solutions file or directory")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stderr)
    log = logging.getLogger("harness")
    log.setLevel(logging.INFO)
    try:
        settings = Settings.from_env(args.tasks_dir, args.out)
    except ValueError as err:
        log.error("%s", err)
        return 2

    # openJiuwen writes ./logs relative to the cwd, and /app is read-only in the container.
    settings.work_root.mkdir(parents=True, exist_ok=True)
    os.chdir(settings.work_root)

    # Imported after chdir: these pull in openJiuwen.
    from harness.framework_logs import quiet_framework_logs

    quiet_framework_logs(settings.work_root)
    from harness.runner import run

    summary = asyncio.run(run(settings))
    log.info("finished: %s", summary)
    logging.shutdown()
    sys.stdout.flush()
    sys.stderr.flush()
    # Skip interpreter teardown: pooled HTTP clients to a dead proxy must not delay exit.
    os._exit(0)


if __name__ == "__main__":
    sys.exit(main())
