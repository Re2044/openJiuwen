"""Runs the plan: solves the chosen tasks in parallel and saves every answer as soon as it is accepted."""
import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import config
from solver import Ledger, TestGuard, solve_task, task_workdir

log = logging.getLogger(__name__)


def _write_atomic(path, text):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


class AnswerStore:
    """Writes answer.txt into the task folder (and its workspace/) and keeps the solutions file current."""

    def __init__(self, out):
        out = Path(out)
        # The rules call OUT a file but create it with `mkdir -p`, so accept a directory too.
        self.solutions_file = out / "solutions.json" if out.is_dir() else out
        self.answers = {}

    def save(self, task, answer):
        targets = [Path(task.path) / "answer.txt"]
        workspace = Path(task.path) / "workspace"
        if workspace.is_dir():  # the rules and the task instructions name different folders
            targets.append(workspace / "answer.txt")
        for target in targets:
            try:
                _write_atomic(target, answer + "\n")
            except OSError as err:
                log.error("cannot write %s: %s", target, err)
        self.answers[task.task_id] = answer
        try:
            _write_atomic(self.solutions_file, json.dumps(self.answers, ensure_ascii=False, indent=1))
        except OSError as err:
            log.error("cannot write %s: %s", self.solutions_file, err)


class Executor:
    def __init__(self, base_url, api_key, budget, out, unavailable_models=()):
        self.unavailable = set(unavailable_models)  # models this key cannot use; more are found while solving
        self.base_url = base_url
        self.api_key = api_key
        self.ledger = Ledger(budget)
        self.store = AnswerStore(out)
        self.scratch = Path(tempfile.mkdtemp(prefix="solver-"))
        self.records = {}  # task id -> what solving it took, for the run report

    async def run(self, plan):
        """plan: list of (task, estimate) pairs. Returns a summary dict."""
        # Most reward per estimated token first, so that a budget cut-off loses the least.
        plan = sorted(plan, key=lambda item: -item[0].reward / max(item[1].tokens, 1))
        log.info("solving %d tasks, %d at a time, budget left %.0f; answers go to answer.txt and %s", len(plan),
                 config.PARALLEL_SESSIONS, self.ledger.budget, os.path.abspath(self.store.solutions_file))
        gate = asyncio.Semaphore(config.PARALLEL_SESSIONS)

        async def guarded(task, estimate):
            async with gate:
                if self.ledger.exhausted:
                    self.records[task.task_id] = {"outcome": "not started", "attempts": []}
                    log.warning("%s: not started, %s", task.task_id,
                                "proxy stopped answering" if self.ledger.proxy_down else "budget exhausted")
                    return task, False
                return task, await self.solve(task, estimate)

        results = await asyncio.gather(*(guarded(task, estimate) for task, estimate in plan))
        solved = [task for task, ok in results if ok]
        summary = {
            "planned": len(plan),
            "answered": len(solved),
            "reward_if_all_correct": sum(task.reward for task in solved),
            "solving_cost": round(self.ledger.spent, 1),
            "model_calls": self.ledger.calls,
        }
        log.info("solving done: %d/%d answered, reward %d if all answers are correct, cost %.0f in %d calls",
                 summary["answered"], summary["planned"], summary["reward_if_all_correct"], self.ledger.spent,
                 self.ledger.calls)
        return summary

    async def solve(self, task, estimate):
        """Planned model first, then fallback models, within the per-task cost cap.

        A model that this key cannot use is skipped and does not count as an attempt.
        """
        workdir = task_workdir(task, self.scratch)
        guard = TestGuard(workdir)
        models = [estimate.model_name] + [m for m in config.FALLBACK_MODELS if m != estimate.model_name]
        spent = 0.0
        attempts_made = 0
        record = self.records[task.task_id] = {"outcome": "no answer", "attempts": []}
        try:
            for model in models:
                if model in self.unavailable:
                    continue
                if attempts_made >= config.MAX_MODELS_PER_TASK:
                    break
                attempts_made += 1
                for retry in range(config.LLM_ERROR_RETRIES + 1):
                    if self.ledger.exhausted or spent >= config.TASK_COST_CAP:
                        return False
                    log.info("%s: start with %s (estimated cost %d, reward %d)", task.task_id, model,
                             estimate.tokens, task.reward)
                    started = time.monotonic()
                    attempt = await solve_task(task, model, workdir, guard, self.store, self.ledger, self.base_url,
                                               self.api_key, config.TASK_COST_CAP - spent)
                    spent += attempt.cost
                    record["attempts"].append({
                        "model": model, "calls": attempt.calls, "cost": round(attempt.cost, 1),
                        "input_tokens": attempt.input_tokens, "output_tokens": round(attempt.output_tokens),
                        "seconds": round(time.monotonic() - started, 1), "submitted": attempt.submitted,
                        "gave_up": attempt.gave_up, "llm_error": attempt.llm_error})
                    log.info("%s: %s with %s, %d calls, cost %.0f", task.task_id,
                             "answered" if attempt.submitted else "no answer", model, attempt.calls, attempt.cost)
                    if attempt.submitted:
                        record["outcome"] = "answered"
                        return True
                    if attempt.gave_up:
                        record["outcome"] = "gave up"
                        return False  # judged impossible: another model would pay to reach the same verdict
                    if attempt.model_unavailable:
                        self.unavailable.add(model)
                        attempts_made -= 1
                        log.warning("%s: %s is not available to this key, trying another model", task.task_id, model)
                        break
                    if not attempt.llm_error:
                        # The model just did not solve it; a retry would cost the same for nothing. In test runs
                        # a second model never rescued a solvable task this way, so by default stop here.
                        if not config.FALLBACK_ON_NO_ANSWER:
                            return False
                        break
                    pause = 3 * (retry + 1)
                    log.warning("%s: model call failed, retrying in %ds", task.task_id, pause)
                    await asyncio.sleep(pause)
            return False
        finally:
            guard.restore()
