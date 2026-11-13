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
    def __init__(self, base_url, api_key, budget, out, unavailable_models=(), deadline=None):
        self.unavailable = set(unavailable_models)  # models this key cannot use; more are found while solving
        self.deadline = deadline  # time.monotonic() value after which no new attempt starts
        self.outcomes = {}  # task id -> {"answered", "gave_up", "tried": models already used on it}
        self.base_url = base_url
        self.api_key = api_key
        self.ledger = Ledger(budget)
        self.store = AnswerStore(out)
        self.scratch = Path(tempfile.mkdtemp(prefix="solver-"))
        # TRAJECTORIES in .env: a JSON-lines file with the estimate and the real cost of every task, the data
        # for calibrating the estimates. Off by default.
        self.trajectories = os.getenv("TRAJECTORIES") or None

    def record(self, task, estimate, answered, spent, seconds, models_tried):
        log.info("%s: estimated %d units / %d s, actual %.0f units / %.0f s, %s", task.task_id, estimate.tokens,
                 estimate.seconds, spent, seconds, "answered" if answered else "no answer")
        if not self.trajectories:
            return
        row = {"task_id": task.task_id, "task_type": task.task_type, "reward": task.reward,
               "planned_model": estimate.model_name, "models_tried": models_tried,
               "estimated_cost": estimate.tokens, "estimated_seconds": estimate.seconds,
               "actual_cost": round(spent, 1), "actual_seconds": round(seconds, 1), "answered": answered}
        try:
            with open(self.trajectories, "a", encoding="utf-8") as file:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        except OSError as err:
            log.error("cannot write %s: %s", self.trajectories, err)

    async def run(self, plan, spare=()):
        """plan: list of (task, estimate) pairs; spare: tasks the plan left out. Returns a summary dict.

        After the plan, whatever budget is left goes to a second pass: tasks that ended without an answer get
        one more attempt with another model, and the left-out tasks get their first. A task the agent gave up
        on as unsolvable is not repeated.
        """
        # Most reward per estimated token first, so that a budget cut-off loses the least.
        by_value = lambda item: -item[0].reward / max(item[1].tokens, 1)
        plan = sorted(plan, key=by_value)
        log.info("solving %d tasks, %d at a time, budget left %.0f; answers go to answer.txt and %s", len(plan),
                 config.PARALLEL_SESSIONS, self.ledger.budget, os.path.abspath(self.store.solutions_file))
        gate = asyncio.Semaphore(config.PARALLEL_SESSIONS)

        async def guarded(task, estimate, second_pass=False):
            async with gate:
                # A task needs at least one full call; without room for it the task would only burn its start.
                if self.ledger.proxy_down or not self.ledger.fits(config.SOLVER_MAX_TOKENS + config.CALL_RESERVE_MARGIN):
                    log.warning("%s: not started, %s", task.task_id,
                                "proxy stopped answering" if self.ledger.proxy_down else "no budget left for a call")
                    return task, False
                if second_pass and self.deadline is not None \
                        and time.monotonic() + config.TASK_TIMEOUT_SECONDS > self.deadline:
                    log.warning("%s: not repeated, too little time left", task.task_id)
                    return task, False
                return task, await self.solve(task, estimate, second_pass)

        results = await asyncio.gather(*(guarded(task, estimate) for task, estimate in plan))
        solved = [task for task, ok in results if ok]
        log.info("first pass done: %d/%d answered, cost %.0f in %d calls", len(solved), len(plan), self.ledger.spent,
                 self.ledger.calls)

        again = []
        if config.SECOND_PASS:
            unanswered = [(task, estimate) for task, estimate in plan
                          if not self.outcomes.get(task.task_id, {}).get("answered")
                          and not self.outcomes.get(task.task_id, {}).get("gave_up")]
            again = sorted(unanswered + list(spare), key=by_value)
        if again:
            log.info("second pass: %d tasks without an answer and %d left out of the plan, budget left %.0f",
                     len(again) - len(spare), len(spare), self.ledger.budget - self.ledger.spent)
            results = await asyncio.gather(*(guarded(task, estimate, True) for task, estimate in again))
            solved += [task for task, ok in results if ok]

        summary = {
            "planned": len(plan),
            "answered": len(solved),
            "second_pass": len(again),
            "reward_if_all_correct": sum(task.reward for task in solved),
            "solving_cost": round(self.ledger.spent, 1),
            "model_calls": self.ledger.calls,
        }
        log.info("solving done: %d/%d answered, reward %d if all answers are correct, cost %.0f in %d calls",
                 summary["answered"], len(plan) + len(spare), summary["reward_if_all_correct"], self.ledger.spent,
                 self.ledger.calls)
        return summary

    async def solve(self, task, estimate, second_pass=False):
        """Planned model first, then fallback models, within the per-task cost cap.

        A model that this key cannot use is skipped and does not count as an attempt. In the second pass the
        task gets one attempt, with a model it has not been tried on when there is one.
        """
        workdir = task_workdir(task, self.scratch)
        guard = TestGuard(workdir)
        models = [estimate.model_name] + [m for m in config.FALLBACK_MODELS if m != estimate.model_name]
        outcome = self.outcomes.setdefault(task.task_id, {"answered": False, "gave_up": False, "tried": []})
        max_models = config.MAX_MODELS_PER_TASK
        if second_pass:
            models = [m for m in models if m not in outcome["tried"]] + [m for m in models if m in outcome["tried"]]
            max_models = 1
        spent = 0.0
        cost_cap = config.TASK_COST_CAP_BY_TYPE.get(task.task_type, config.TASK_COST_CAP)
        attempts_made = 0
        answered = False
        models_tried = []
        started = time.monotonic()
        try:
            for model in models:
                if model in self.unavailable:
                    continue
                if attempts_made >= max_models:
                    break
                if attempts_made and spent > config.FALLBACK_MAX_SPENT:
                    log.info("%s: no second model, the first attempt already cost %.0f", task.task_id, spent)
                    break
                attempts_made += 1
                models_tried.append(model)
                for retry in range(config.LLM_ERROR_RETRIES + 1):
                    if (self.ledger.proxy_down or spent >= cost_cap
                            or not self.ledger.fits(config.SOLVER_MAX_TOKENS + config.CALL_RESERVE_MARGIN)):
                        return False
                    log.info("%s: start with %s (estimated %d tokens, reward %d)", task.task_id, model,
                             estimate.tokens, task.reward)
                    attempt = await solve_task(task, model, workdir, guard, self.store, self.ledger, self.base_url,
                                               self.api_key, cost_cap - spent)
                    spent += attempt.cost
                    log.info("%s: %s with %s, %d calls, cost %.0f", task.task_id,
                             "answered" if attempt.submitted else "gave up" if attempt.gave_up else "no answer", model, attempt.calls, attempt.cost)
                    if attempt.submitted:
                        answered = True
                        return True
                    if attempt.gave_up:
                        outcome["gave_up"] = True
                        return False  # unsolvable: no answer, no second model, no second pass
                    if attempt.model_unavailable:
                        self.unavailable.add(model)
                        attempts_made -= 1
                        log.warning("%s: %s is not available to this key, trying another model", task.task_id, model)
                        break
                    if not attempt.llm_error:
                        break  # the model just did not solve it; a retry would cost the same for nothing
                    pause = 3 * (retry + 1)
                    log.warning("%s: model call failed, retrying in %ds", task.task_id, pause)
                    await asyncio.sleep(pause)
            return False
        finally:
            guard.restore()
            outcome["answered"] = answered
            outcome["tried"] += [m for m in models_tried if m not in outcome["tried"]]
            self.record(task, estimate, answered, spent, time.monotonic() - started, models_tried)
