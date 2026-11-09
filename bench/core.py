"""Test bench: runs an agent architecture over tasks, in isolation, and scores it.

An architecture is a Python file with one required function, `async def solve(env)`,
and an optional `def plan(tasks)` that picks and orders tasks. See agents/ for examples.
"""
from __future__ import annotations

import asyncio
import time
import traceback
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

from pydantic import BaseModel

from harness.agent import (
    AttemptResult,
    GuardedTests,
    MeterRail,
    SYSTEM_PROMPT,
    accept_answer,
    build_agent,
    build_model,
    build_tools,
    run_bash,
)
from harness.budget import Ledger, Trajectory
from harness.runner import adopt_written_answer, plan as default_plan
from harness.settings import Settings
from harness.tasks import AnswerStore, Task
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, ModelCallInputs
from scripts.evaluate import check_task

DEFAULT_MODEL = "qwen3.8-27b-noreason"
MESSAGE_FIELDS_TO_DROP = {"usage_metadata", "metadata", "parser_content", "prompt_token_ids",
                          "completion_token_ids", "logprobs"}


class BudgetExhausted(Exception):
    """Raised by env.ask when the run budget or the per-task cap is used up."""


def _plain(message: Any) -> Any:
    if isinstance(message, BaseModel):
        message = message.model_dump(exclude_none=True)
    if isinstance(message, dict):
        return {key: value for key, value in message.items() if key not in MESSAGE_FIELDS_TO_DROP}
    return message


class RecordingRail(MeterRail):
    """MeterRail that also saves what the model saw and answered, for reading after the run."""

    def __init__(self, *args: Any, transcript: Trajectory) -> None:
        super().__init__(*args)
        self._transcript = transcript
        self._messages_seen = 0

    async def after_model_call(self, ctx: AgentCallbackContext) -> None:
        await super().after_model_call(ctx)
        if not isinstance(ctx.inputs, ModelCallInputs):
            return
        messages = list(ctx.inputs.messages or [])
        new = messages[self._messages_seen:] if len(messages) >= self._messages_seen else messages
        self._messages_seen = len(messages)
        self._transcript.write(event="model_call", new_messages=[_plain(m) for m in new],
                               response=_plain(ctx.inputs.response))


class TaskEnv:
    """Everything an architecture needs for one task. Reference answers are not reachable from here."""

    def __init__(self, task: Task, settings: Settings, ledger: Ledger, trajectory: Trajectory, store: AnswerStore,
                 run_dir: Path) -> None:
        self.task = task
        self.workdir = task.workdir(settings.work_root)
        self._settings = settings
        self._ledger = ledger
        self._store = store
        self._state = AttemptResult(model="")
        self._guard = GuardedTests(self.workdir)
        self._transcript = Trajectory(run_dir / "transcripts" / f"{task.task_id}.jsonl")
        self._agent_home = run_dir / "agent_home" / task.task_id
        self._agent_home.mkdir(parents=True, exist_ok=True)
        self.meter = RecordingRail(task, self._state, ledger, trajectory, settings.task_cost_cap,
                                   transcript=self._transcript)

    task_id = property(lambda self: self.task.task_id)
    task_type = property(lambda self: self.task.task_type)
    instruction = property(lambda self: self.task.instruction)
    reward = property(lambda self: self.task.reward)
    spent = property(lambda self: self._state.cost, doc="Cost of this task so far, in output-token equivalents.")
    calls = property(lambda self: self._state.calls)
    submitted = property(lambda self: self._state.submitted)
    budget_left = property(lambda self: self._ledger.remaining, doc="None when the run has no budget set.")

    def model(self, name: str = DEFAULT_MODEL, max_tokens: int | None = None) -> Any:
        """An openJiuwen Model. Attach env.meter wherever it is used, or calls go uncounted."""
        return build_model(self._settings, name, max_tokens)

    def tools(self) -> list:
        """The `bash` and `submit` tools as openJiuwen tools, bound to this task."""
        return build_tools(self.task, self.workdir, self._guard, self._store, self._settings, self._state)

    def make_agent(self, model: str = DEFAULT_MODEL, system_prompt: str = SYSTEM_PROMPT, max_iterations: int = 8,
                   max_tokens: int | None = None, tools: list | None = None, rails: list | None = None,
                   **overrides: Any) -> Any:
        """A metered DeepAgent with the costly defaults off; `overrides` go to create_deep_agent."""
        return build_agent(
            self.task, self.model(model, max_tokens), self._agent_home,
            tools=self.tools() if tools is None else tools,
            rails=[self.meter, *(rails or [])],
            system_prompt=system_prompt, max_iterations=max_iterations, **overrides,
        )

    async def run_agent(self, agent: Any, query: str | None = None) -> Any:
        """Invoke an agent on the task instruction, or on your own query."""
        return await agent.invoke({"query": query or self.instruction,
                                   "conversation_id": f"{self.task_id}-{uuid.uuid4().hex[:8]}"})

    async def ask(self, prompt: str, model: str = DEFAULT_MODEL, system: str | None = None,
                  max_tokens: int | None = None) -> str:
        """One metered model call without tools, for cheap steps such as classifying or estimating a task."""
        if self._ledger.halted or self._state.cost >= self._settings.task_cost_cap:
            raise BudgetExhausted(f"spent {self._state.cost:.0f} on this task, {self._ledger.spent:.0f} in the run")
        messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        response = await self.model(model, max_tokens).invoke(messages)
        self.meter.record(response.usage_metadata)
        self._transcript.write(event="ask", model=model, new_messages=messages, response=_plain(response))
        return str(response.content or "").strip()

    async def bash(self, command: str) -> str:
        """Run a command in the task's working directory. Costs no tokens."""
        return await run_bash(command, self.workdir, self._settings.bash_timeout_sec)

    async def submit(self, answer: str) -> str:
        """Save the final answer. Returns "Accepted." or the reason, such as failing tests."""
        return await accept_answer(self.task, answer, self._guard, self._store, self._settings, self._state)

    async def finish(self) -> None:
        """Called by the bench after solve: accepts an answer.txt written directly, restores edited tests."""
        if not self.submitted:
            self._state.submitted = await adopt_written_answer(self.task, self.workdir, self._guard, self._store,
                                                               self._settings)
        self._guard.restore()

    def log_error(self, text: str) -> None:
        self._transcript.write(event="error", traceback=text)


@dataclass
class TaskReport:
    task_id: str
    task_type: str
    reward: float
    status: str  # solved, wrong, no_answer, skipped_ok (correctly left an impossible task), not_run, error
    points: float = 0.0
    cost: float = 0.0
    calls: int = 0
    seconds: float = 0.0
    answer: str = ""
    error: str = ""


def _status(task: Task, correct: bool, submitted: bool, error: str) -> str:
    if error:
        return "error"
    if task.check == "impossible":
        return "skipped_ok" if correct else "wrong"
    if correct:
        return "solved"
    return "wrong" if submitted else "no_answer"


async def run_task(solve: Callable[[TaskEnv], Awaitable[Any]], task: Task, settings: Settings, ledger: Ledger,
                   trajectory: Trajectory, store: AnswerStore, run_dir: Path, task_timeout: float) -> TaskReport:
    env = TaskEnv(task, settings, ledger, trajectory, store, run_dir)
    started = time.monotonic()
    error = ""
    try:
        await asyncio.wait_for(solve(env), task_timeout)
    except asyncio.TimeoutError:
        error = f"timed out after {task_timeout:.0f}s"
    except Exception as err:  # a bug in one architecture run must not stop the other tasks
        error = f"{type(err).__name__}: {err}"
        env.log_error(traceback.format_exc())
    await env.finish()

    correct, points, _ = check_task(task.root)
    answer_file = task.root / "answer.txt"
    report = TaskReport(
        task_id=task.task_id, task_type=task.task_type, reward=task.reward,
        status=_status(task, correct, env.submitted, error),
        points=points if correct else 0.0,
        cost=round(env.spent, 1), calls=env.calls, seconds=round(time.monotonic() - started, 1),
        answer=answer_file.read_text(encoding="utf-8").strip()[:60] if answer_file.exists() else "",
        error=error[:300],
    )
    trajectory.write(event="task", task_id=task.task_id, task_type=task.task_type, reward=task.reward,
                     status=report.status, cost=env.spent, calls=env.calls, seconds=report.seconds)
    return report


async def run_suite(module: Any, tasks: list[Task], settings: Settings, run_dir: Path, task_timeout: float,
                    on_report: Callable[[TaskReport], None]) -> list[TaskReport]:
    ledger = Ledger(settings.budget, settings.max_llm_failures)
    trajectory = Trajectory(settings.trajectory_path)
    store = AnswerStore(settings.out)
    chosen = list(getattr(module, "plan", default_plan)(tasks))
    chosen_ids = {task.task_id for task in chosen}
    reports = [TaskReport(t.task_id, t.task_type, t.reward, "not_run") for t in tasks if t.task_id not in chosen_ids]
    for report, task in ((r, next(t for t in tasks if t.task_id == r.task_id)) for r in reports):
        # plan() leaving out an impossible task is exactly the wanted behaviour.
        if task.check == "impossible":
            report.status = "skipped_ok"
    gate = asyncio.Semaphore(settings.workers)

    async def guarded(task: Task) -> None:
        async with gate:
            if ledger.halted:
                reports.append(TaskReport(task.task_id, task.task_type, task.reward, "not_run",
                                          error="budget exhausted"))
                return
            report = await run_task(module.solve, task, settings, ledger, trajectory, store, run_dir, task_timeout)
            reports.append(report)
            on_report(report)

    await asyncio.gather(*(guarded(task) for task in chosen))
    return reports


def summarize(reports: list[TaskReport]) -> dict:
    cost = sum(r.cost for r in reports)
    points = sum(r.points for r in reports)
    statuses: dict[str, int] = {}
    for report in reports:
        statuses[report.status] = statuses.get(report.status, 0) + 1
    by_type: dict[str, dict[str, float]] = {}
    for report in reports:
        entry = by_type.setdefault(report.task_type, {"tasks": 0, "cost": 0.0, "points": 0.0})
        entry["tasks"] += 1
        entry["cost"] = round(entry["cost"] + report.cost, 1)
        entry["points"] += report.points
    return {
        "points": points,
        "cost": round(cost, 1),
        "points_per_1k_tokens": round(points / cost * 1000, 2) if cost else None,
        "statuses": statuses,
        "by_type": by_type,
        "tasks": [asdict(r) for r in sorted(reports, key=lambda r: r.task_id)],
    }
