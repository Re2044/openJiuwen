"""Token accounting shared by all sessions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("harness")

INPUT_TOKENS_PER_OUTPUT = 10


def call_cost(usage: Any) -> float:
    """Competition cost of one call: output tokens + input tokens / 10.

    Reasoning counts as output whichever field carries it. OpenAI-style APIs
    include it in output_tokens; reasoning larger than output can only mean a
    separate field, so it is added then. Confirm this against the real proxy.
    """
    output = usage.output_tokens or 0
    reasoning = usage.reasoning_tokens or 0
    if reasoning > output:
        output += reasoning
    return (usage.input_tokens or 0) / INPUT_TOKENS_PER_OUTPUT + output


class Ledger:
    """Spending and proxy health for the whole run."""

    def __init__(self, limit: float | None, max_llm_failures: int) -> None:
        self.limit = limit
        self.spent = 0.0
        self.calls = 0
        self._failures = 0
        self._max_failures = max_llm_failures
        self.proxy_down = False

    def charge(self, cost: float) -> None:
        self.spent += cost
        self.calls += 1
        self._failures = 0

    def note_llm_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._max_failures and not self.proxy_down:
            self.proxy_down = True
            log.warning("%d consecutive LLM failures: treating the budget as exhausted", self._failures)

    @property
    def remaining(self) -> float | None:
        return None if self.limit is None else self.limit - self.spent

    @property
    def halted(self) -> bool:
        return self.proxy_down or (self.limit is not None and self.spent >= self.limit)


class Trajectory:
    """Append-only JSONL log of every call and task outcome, the training data for the estimator."""

    def __init__(self, path: Path) -> None:
        self._path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, **record: Any) -> None:
        try:
            with self._path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except OSError as err:
            log.warning("cannot write trajectory: %s", err)
