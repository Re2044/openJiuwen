"""Expected cost, time and success of a model on a task type, measured on earlier runs (estimates.json).

This replaces asking a judge model about every task: the task type is already in task.json, so the lookup
costs no tokens and no time, and the numbers come from real runs. calibrate.py rebuilds the table.
"""
import json
import logging
from dataclasses import dataclass
from pathlib import Path

import config

log = logging.getLogger(__name__)
TABLE_FILE = Path(__file__).with_name("estimates.json")


@dataclass
class Estimate:
    cost: float  # competition units: output tokens + input tokens / 10
    seconds: float
    success: float  # share of solvable tasks answered correctly
    measured: bool  # False when borrowed from other models or task types


def _average(entries):
    """Combine table entries, weighted by how many tasks each was measured on."""
    weight = sum(entry["n"] for entry in entries)
    return {key: sum(entry[key] * entry["n"] for entry in entries) / weight for key in ("cost", "seconds", "success")}


def _reasons(model):
    """Whether the model reasons before answering; estimates are borrowed between models of the same kind."""
    return not any(marker in model for marker in config.NO_REASONING_MARKERS)


class Estimator:
    def __init__(self, path=TABLE_FILE):
        self.types = json.loads(Path(path).read_text(encoding="utf-8"))["types"]
        self._warned = set()

    def is_measured(self, task_type, model):
        return model in self.types.get(task_type, {})

    def estimate(self, task_type, model):
        by_model = self.types.get(task_type)
        if by_model and model in by_model:
            return Estimate(**{k: by_model[model][k] for k in ("cost", "seconds", "success")}, measured=True)
        # No measurement: borrow from models with the same reasoning mode, in this task type if it is known.
        candidates = by_model.items() if by_model else [
            item for models in self.types.values() for item in models.items()]  # unknown task type: use all types
        candidates = list(candidates)
        pool = [e for m, e in candidates if _reasons(m) == _reasons(model)] or [e for _, e in candidates]
        if (task_type, model) not in self._warned:
            self._warned.add((task_type, model))
            log.warning("no measurements for %s on %s tasks, borrowing from similar models", model, task_type)
        return Estimate(**_average(pool), measured=False)
