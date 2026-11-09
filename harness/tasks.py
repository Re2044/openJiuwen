"""Task loading and answer storage.

Only whitelisted fields of task.json are read. The `expected` field holds the
reference answer in the example tasks, so runtime code must never touch it;
scripts/evaluate.py is the only place that may.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("harness")


@dataclass(frozen=True)
class Task:
    task_id: str
    task_type: str
    instruction: str
    reward: float
    check: str
    root: Path

    @property
    def workspace(self) -> Path | None:
        path = self.root / "workspace"
        return path if path.is_dir() else None

    def workdir(self, work_root: Path) -> Path:
        """Where tools run. Tasks without a workspace get a scratch directory, keeping task.json out of reach."""
        if self.workspace:
            return self.workspace
        path = work_root / "work" / self.task_id
        path.mkdir(parents=True, exist_ok=True)
        return path


def load_tasks(tasks_dir: Path) -> list[Task]:
    tasks = []
    for path in sorted(tasks_dir.glob("*/task.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            tasks.append(Task(
                task_id=str(raw.get("task_id") or path.parent.name),
                task_type=str(raw.get("task_type", "unknown")),
                instruction=str(raw["instruction"]),
                reward=float(raw.get("reward", 0)),
                check=str(raw.get("check", "")),
                root=path.parent,
            ))
        except (OSError, ValueError, KeyError, TypeError) as err:
            log.warning("skipping malformed task %s: %s", path, err)
    return tasks


def _write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class AnswerStore:
    """Persists every answer at once: a run can be killed at any moment."""

    def __init__(self, out: Path | None) -> None:
        self._answers: dict[str, str] = {}
        self._solutions_file = self._resolve_solutions_file(out)

    @staticmethod
    def _resolve_solutions_file(out: Path | None) -> Path | None:
        if out is None:
            return None
        # The rules call OUT a file but create it with `mkdir -p`, so accept both.
        return out / "solutions.json" if out.is_dir() or not out.suffix else out

    def save(self, task: Task, answer: str) -> bool:
        """Write answer.txt to the task root and its workspace; the rules and instructions name different places."""
        targets = [task.root / "answer.txt"]
        if task.workspace:
            targets.append(task.workspace / "answer.txt")
        written = 0
        for target in targets:
            try:
                _write_atomic(target, answer + "\n")
                written += 1
            except OSError as err:
                log.error("cannot write %s: %s", target, err)
        self._answers[task.task_id] = answer
        self._write_solutions()
        return written > 0

    def _write_solutions(self) -> None:
        if self._solutions_file is None:
            return
        try:
            self._solutions_file.parent.mkdir(parents=True, exist_ok=True)
            _write_atomic(self._solutions_file, json.dumps(self._answers, ensure_ascii=False, indent=1))
        except OSError as err:
            log.error("cannot write %s: %s", self._solutions_file, err)
