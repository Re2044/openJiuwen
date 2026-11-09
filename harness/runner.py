"""Picks the task order, runs up to five sessions in parallel, saves answers as they arrive."""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from pathlib import Path
from typing import Any

from harness.agent import (
    UNITTEST_CHECK,
    AttemptResult,
    GuardedTests,
    accept_answer,
    run_attempt,
)
from harness.budget import Ledger, Trajectory
from harness.settings import Settings
from harness.tasks import AnswerStore, Task, load_tasks

log = logging.getLogger("harness")

# Placeholder estimates in output-token equivalents, until measured on the real proxy.
PRIOR_COST = {"math": 150.0, "data_analysis": 200.0, "search": 300.0, "coding": 800.0}
DEFAULT_PRIOR_COST = 800.0
MAX_GUESS_CHARS = 100


def estimate_cost(task: Task) -> float:
    return PRIOR_COST.get(task.task_type, DEFAULT_PRIOR_COST)


def plan(tasks: list[Task]) -> list[Task]:
    """Best points per token first. No partial credit means this order also works without knowing the budget."""
    return sorted(tasks, key=lambda t: (-t.reward / estimate_cost(t), estimate_cost(t), t.task_id))


def next_affordable(queue: deque[Task], ledger: Ledger) -> Task | None:
    """Pop the best remaining task the budget can still pay for."""
    remaining = ledger.remaining
    if remaining is None:
        return queue.popleft() if queue else None
    for task in list(queue):
        if estimate_cost(task) <= remaining:
            queue.remove(task)
            return task
    return None


async def solve_task(task: Task, settings: Settings, ledger: Ledger, trajectory: Trajectory,
                     store: AnswerStore) -> bool:
    workdir = task.workdir(settings.work_root)
    guard = GuardedTests(workdir)
    spent = 0.0
    guess = ""
    submitted = False
    used_model = ""

    for model in settings.models:
        for retry in range(settings.llm_retries + 1):
            if ledger.halted or spent >= settings.task_cost_cap:
                break
            result = await run_attempt(task, model, settings, ledger, trajectory, store, guard, workdir,
                                       settings.task_cost_cap - spent)
            spent += result.cost
            used_model = model
            submitted = submitted or result.submitted
            if submitted:
                break
            if not result.llm_error:
                guess = result.final_text or guess
                break
            ledger.note_llm_failure()
            await asyncio.sleep(2.0 * (retry + 1))
        if submitted:
            break
        submitted = await adopt_written_answer(task, workdir, guard, store, settings)
        if submitted:
            break

    guard.restore()
    if not submitted and guess and len(guess) <= MAX_GUESS_CHARS and task.check != UNITTEST_CHECK:
        # A wrong answer costs nothing, an unsubmitted one earns nothing.
        store.save(task, guess)
        log.info("%s: saved unverified final text as a best guess", task.task_id)

    trajectory.write(event="task", task_id=task.task_id, task_type=task.task_type, reward=task.reward,
                     model=used_model, submitted=submitted, cost=spent)
    log.info("%s: %s, cost %.0f, run total %.0f", task.task_id, "submitted" if submitted else "no answer",
             spent, ledger.spent)
    return submitted


async def adopt_written_answer(task: Task, workdir: Path, guard: GuardedTests, store: AnswerStore,
                                settings: Settings) -> bool:
    """The instructions ask for answer.txt, so a model may write it directly instead of calling submit."""
    path = workdir / "answer.txt"
    if not path.is_file():
        return False
    result = AttemptResult(model="")
    await accept_answer(task, path.read_text(encoding="utf-8", errors="replace"), guard, store, settings, result)
    return result.submitted


async def run(settings: Settings) -> dict[str, Any]:
    tasks = load_tasks(settings.tasks_dir)
    log.info("loaded %d tasks from %s, models %s, budget %s", len(tasks), settings.tasks_dir, settings.models,
             settings.budget)

    ledger = Ledger(settings.budget, settings.max_llm_failures)
    trajectory = Trajectory(settings.trajectory_path)
    store = AnswerStore(settings.out)
    queue = deque(plan(tasks))
    solved: list[str] = []

    async def worker() -> None:
        while not ledger.halted:
            task = next_affordable(queue, ledger)
            if task is None:
                return
            try:
                if await solve_task(task, settings, ledger, trajectory, store):
                    solved.append(task.task_id)
            except Exception:
                log.exception("%s: unexpected failure", task.task_id)

    workers = [asyncio.create_task(worker()) for _ in range(settings.workers)]
    _, pending = await asyncio.wait(workers, timeout=settings.deadline_sec)
    if pending:
        log.warning("deadline reached, stopping %d running sessions", len(pending))
        for job in pending:
            job.cancel()
        await asyncio.wait(pending, timeout=10)

    return {"tasks": len(tasks), "submitted": len(solved), "cost": round(ledger.spent, 1), "calls": ledger.calls,
            "unstarted": len(queue), "proxy_down": ledger.proxy_down}
