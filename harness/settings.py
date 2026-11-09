"""Run configuration, read from the environment."""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

# Cheapest first; the next model is tried only after the previous one gave no accepted answer.
DEFAULT_MODELS = ("qwen3.8-27b-noreason", "qwen3.8-27b")


def normalize_base_url(url: str) -> str:
    """The OpenAI client expects the /v1 prefix; add it when the proxy address is bare."""
    url = url.rstrip("/")
    return url + "/v1" if urlparse(url).path == "" else url


@dataclass(frozen=True)
class Settings:
    tasks_dir: Path
    out: Path | None
    base_url: str
    api_key: str
    models: tuple[str, ...]
    budget: float | None
    work_root: Path
    trajectory_path: Path
    workers: int = 5
    deadline_sec: float = 3300.0  # the container is killed at one hour
    llm_timeout_sec: float = 240.0
    max_iterations: int = 8
    max_tokens: int = 2000
    task_cost_cap: float = 6000.0  # stop-loss per task, in output-token equivalents
    bash_timeout_sec: float = 60.0
    llm_retries: int = 2
    max_llm_failures: int = 6  # consecutive failures that mean the proxy cut us off

    @classmethod
    def from_env(cls, tasks_dir: str | None, out: str | None) -> "Settings":
        base_url = os.getenv("LLM_BASE_URL")
        api_key = os.getenv("LLM_API_KEY")
        missing = [name for name, value in (
            ("--tasks-dir or TASKS", tasks_dir), ("LLM_BASE_URL", base_url), ("LLM_API_KEY", api_key),
        ) if not value]
        if missing:
            raise ValueError("missing required settings: " + ", ".join(missing))
        tasks_path = Path(tasks_dir).resolve()
        if not tasks_path.is_dir():
            raise ValueError(f"tasks directory does not exist: {tasks_path}")

        work_root = Path(os.getenv("HARNESS_WORK_DIR") or Path(tempfile.gettempdir()) / "harness").resolve()
        models = tuple(m.strip() for m in os.getenv("HARNESS_MODELS", "").split(",") if m.strip())
        budget = os.getenv("TOKEN_BUDGET")
        return cls(
            tasks_dir=tasks_path,
            out=Path(out).resolve() if out else None,
            base_url=normalize_base_url(base_url),
            api_key=api_key,
            models=models or DEFAULT_MODELS,
            budget=float(budget) if budget else None,
            work_root=work_root,
            trajectory_path=Path(os.getenv("HARNESS_TRAJECTORY") or work_root / "trajectories.jsonl").resolve(),
            deadline_sec=float(os.getenv("HARNESS_DEADLINE_SEC", cls.deadline_sec)),
            llm_timeout_sec=float(os.getenv("HARNESS_LLM_TIMEOUT_SEC", cls.llm_timeout_sec)),
        )
