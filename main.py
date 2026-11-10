from openai import OpenAI
from data_access import TasksRepository, TaskData, TaskEstimate, APIService
from dp import DP
import os
import sys
import time
import logging
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

judge_model = config.JUDGE_MODEL_NAME
estimate_labels = []  # task id of each entry in task_estimates, for the logs

# The agent's commands need a working bash and python3; check before spending anything.
import shell

try:
    shell.check()
except RuntimeError as err:
    log.error("%s", err)
    sys.exit(2)

# A free plan may not include every model; asking the judge about those would waste the budget.
all_models = list(dict.fromkeys(config.MODEL_NAME_LIST + [judge_model] + config.FALLBACK_MODELS))
available_models = [model for model in all_models if api_service.is_model_available(model)]
unavailable_models = [model for model in all_models if model not in available_models]
log.info("models available to this key: %s%s", ", ".join(available_models) or "none",
         f"; not available: {', '.join(unavailable_models)}" if unavailable_models else "")
if judge_model not in available_models:
    log.error("the judge model %s is not available to this key; change JUDGE_MODEL_NAME in config.py", judge_model)
    sys.exit(2)
model_names = [model for model in config.MODEL_NAME_LIST if model in available_models]

total_calls = len(tasks) * len(model_names)
log.info("estimating %d tasks x %d models = %d calls to judge %s", len(tasks), len(model_names),
         total_calls, judge_model)
estimation_started = time.monotonic()
call_number = 0
for task in tasks:
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

log.info("estimation done in %.0fs: %d usable estimates; judge spent %d input + %d output tokens = %.0f in "
         "competition units", time.monotonic() - estimation_started, len(task_estimates), api_service.input_tokens,
         api_service.output_tokens, api_service.cost)

dp = DP()
res = dp.calc(budget_in_tokens, config.TIME_LIMIT_IN_SECONDS, task_estimates, groups=estimate_labels)

best_reward, chosen = res
log.info("plan: total reward %s from %d estimates", best_reward, len(chosen))
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
log.info("finished: %s; total spent %.0f of %d (judge %.0f, solving %.0f)", summary,
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