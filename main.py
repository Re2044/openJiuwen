from openai import OpenAI
from data_access import TasksRepository, TaskData, TaskEstimate, APIService
from dp import DP
import os
import sys
import time
import math
import logging
from types import SimpleNamespace
import config
from dotenv import load_dotenv

load_dotenv()

# LOG_LEVEL=DEBUG also shows every task loaded and the token usage of each judge call.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("main.log", encoding="utf-8")],
)
for noisy in ("httpx", "httpx2", "httpcore", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("main")
log.info("started, log file %s", os.path.abspath("main.log"))

data_accessor = TasksRepository()

tasks = data_accessor.task_data_array

api_service = APIService(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

budget_in_tokens = int(os.getenv("BUDGET"))
log.info("LLM endpoint %s, budget %d tokens, time limit %ds", os.getenv("LLM_BASE_URL"), budget_in_tokens,
         config.TIME_LIMIT_IN_SECONDS)

task_estimates = []
# PARALLEL_SESSIONS in .env: how many tasks are solved at once (a free plan may allow fewer requests at a time).
if os.getenv("PARALLEL_SESSIONS"):
    config.PARALLEL_SESSIONS = max(1, int(os.getenv("PARALLEL_SESSIONS")))
    log.info("parallel sessions set by PARALLEL_SESSIONS: %d", config.PARALLEL_SESSIONS)

# MODELS in .env limits the run to these models, e.g. the ones a free plan includes.
if os.getenv("MODELS"):
    config.MODEL_NAME_LIST = [model.strip() for model in os.getenv("MODELS").split(",") if model.strip()]
    config.FALLBACK_MODELS = [model for model in config.FALLBACK_MODELS if model in config.MODEL_NAME_LIST]
    log.info("models limited by MODELS: %s", ", ".join(config.MODEL_NAME_LIST))

# USE_JUDGE=1 in .env plans with the judge model instead of the estimates table, for comparison.
if os.getenv("USE_JUDGE"):
    config.USE_JUDGE = os.getenv("USE_JUDGE") == "1"
log.info("planning with %s", "the judge model" if config.USE_JUDGE else "the estimates table (estimates.json)")

judge_model = config.JUDGE_MODEL_NAME
estimate_labels = []  # task id of each entry in task_estimates, for the logs
estimate_success = []  # expected share of correct answers for each entry in task_estimates

# The agent's commands need a working bash and python3; check before spending anything.
import shell

try:
    shell.check()
except RuntimeError as err:
    log.error("%s", err)
    sys.exit(2)

# A free plan may not include every model; asking the judge about those would waste the budget.
all_models = list(dict.fromkeys(config.MODEL_NAME_LIST + ([judge_model] if config.USE_JUDGE else [])
                                + config.FALLBACK_MODELS))
available_models = [model for model in all_models if api_service.is_model_available(model)]
unavailable_models = [model for model in all_models if model not in available_models]
log.info("models available to this key: %s%s", ", ".join(available_models) or "none",
         f"; not available: {', '.join(unavailable_models)}" if unavailable_models else "")
model_names = [model for model in config.MODEL_NAME_LIST if model in available_models]
if not model_names:
    log.error("none of the models in MODEL_NAME_LIST is available to this key")
    sys.exit(2)

if not config.USE_JUDGE:
    # Estimates come from the table of earlier measurements: no model calls, no waiting.
    from estimator import Estimator

    estimator = Estimator()
    for task in tasks:
        # Measured models first: when two estimates are equal the plan keeps the earlier one.
        for model_name in sorted(model_names, key=lambda model: not estimator.is_measured(task.task_type, model)):
            entry = estimator.estimate(task.task_type, model_name)
            task_estimates.append(TaskEstimate(model_name=model_name, tokens=round(entry.cost),
                                               seconds=round(entry.seconds), reward=task.reward))
            estimate_labels.append(task.task_id)
            estimate_success.append(entry.success)
    for task_type in sorted({task.task_type for task in tasks}):
        log.info("estimates for %s: %s", task_type, "; ".join(
            f"{model} cost {entry.cost:.0f}, {entry.seconds:.0f} s, success {entry.success:.2f}"
            + ("" if entry.measured else " (borrowed)")
            for model in model_names for entry in [estimator.estimate(task_type, model)]))
elif judge_model not in available_models:
    log.error("the judge model %s is not available to this key; change JUDGE_MODEL_NAME in config.py", judge_model)
    sys.exit(2)

total_calls = len(tasks) * len(model_names) if config.USE_JUDGE else 0
if config.USE_JUDGE:
    log.info("estimating %d tasks x %d models = %d calls to judge %s", len(tasks), len(model_names),
             total_calls, judge_model)
estimation_started = time.monotonic()
call_number = 0
for task in tasks if config.USE_JUDGE else []:
    for model_name in model_names:
        call_number += 1
        log.info("[%d/%d] %s on %s: asking judge", call_number, total_calls, task.task_id, model_name)
        task_estimate = api_service.EstimateTask(judge_model, task.instruction, model_name, budget_in_tokens)

        if task_estimate.tokens == -1 or task_estimate.seconds == -1:
            log.info("[%d/%d] judged infeasible, skipped (%.1fs)", call_number, total_calls,
                     api_service.last_call_seconds)
            continue
        log.info("[%d/%d] estimate %d tokens, %d s (%.1fs, judge spent %.0f so far)", call_number, total_calls,
                 task_estimate.tokens, task_estimate.seconds, api_service.last_call_seconds, api_service.cost)
        
        task_estimate.model_name = model_name
        task_estimate.reward = task.reward
        task_estimates.append(task_estimate)
        estimate_labels.append(task.task_id)
        estimate_success.append(1.0)  # the judge gives no success estimate

log.info("estimation done in %.0fs: %d estimates; model calls so far cost %.0f in competition units (%d input + "
         "%d output tokens)", time.monotonic() - estimation_started, len(task_estimates), api_service.cost,
         api_service.input_tokens, api_service.output_tokens)

# Tasks are solved PARALLEL_SESSIONS at a time, so the time available to the plan is counted in session-seconds.
# Time is planned in coarse steps and reward as expected reward (reward x success) to keep the DP table small.
session_seconds = int(config.TIME_LIMIT_IN_SECONDS * config.PARALLEL_SESSIONS * config.TIME_SAFETY_SHARE)
time_step = max(1, math.ceil(session_seconds / config.DP_TIME_BUCKETS))
budget_left = int(budget_in_tokens - api_service.cost)
dp_items = [SimpleNamespace(tokens=estimate.tokens,
                            seconds=math.ceil(estimate.seconds / time_step),
                            reward=round(estimate.reward * success * config.DP_REWARD_SCALE))
            for estimate, success in zip(task_estimates, estimate_success)]

dp = DP()
res = dp.calc(budget_left, session_seconds // time_step, dp_items, groups=estimate_labels)

scaled_reward, chosen = res
log.info("plan: %d of %d tasks, expected reward %.1f, estimated cost %d of %d, estimated time %d of %d "
         "session-seconds", len(chosen), len(tasks), scaled_reward / config.DP_REWARD_SCALE,
         sum(task_estimates[index - 1].tokens for index in chosen), budget_left,
         sum(task_estimates[index - 1].seconds for index in chosen), session_seconds)
for index in chosen:
    estimate = task_estimates[index - 1]
    log.info("  %s on %s: %d tokens, %d s, reward %d", estimate_labels[index - 1], estimate.model_name,
             estimate.tokens, estimate.seconds, estimate.reward)

# Solve the planned tasks. Imported here because loading openJiuwen takes a few seconds.
import asyncio
from executor import Executor

tasks_by_id = {task.task_id: task for task in tasks}
plan = [(tasks_by_id[estimate_labels[index - 1]], task_estimates[index - 1]) for index in chosen]
executor = Executor(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    budget=budget_in_tokens - api_service.cost,  # what the judge spent is gone already
    out=os.getenv("OUT") or "solutions.json",
    unavailable_models=unavailable_models,
)
summary = asyncio.run(executor.run(plan))
log.info("finished: %s; total spent %.0f of %d (planning %.0f, solving %.0f)", summary,
         api_service.cost + summary["solving_cost"], budget_in_tokens, api_service.cost, summary["solving_cost"])

# Score the saved answers the way the final check does. Only possible where task.json has `expected`,
# as in the example tasks; answers are already saved, so this cannot change them. SCORE_RUN=0 turns it off.
if os.getenv("SCORE_RUN", "1") != "0":
    from evaluate import score_tasks, summary_line

    report = score_tasks(data_accessor.path)
    if report["scorable"] or report["unsolvable"]:
        for result in report["tasks"]:
            log.info("  %-22s %-20s %d/%d  %s", result["task_id"], result["status"], result["points"],
                     result["reward"], result["detail"])
        log.info("score: %s", summary_line(report))
    else:
        log.info("score: no task has a reference answer, nothing to score")