from openai import OpenAI
from data_access import TasksRepository, TaskData, TaskEstimate, APIService, estimates_from_steps, estimates_from_table
from dp import DP
import os
import sys
import tempfile
import time
import logging
import math
from concurrent.futures import ThreadPoolExecutor
import config
from dotenv import load_dotenv

load_dotenv()
run_started = time.monotonic()  # the hour of the run counts from here



def open_log_file():
    """main.log next to the code, or in the temp folder when that is read-only, as in the competition container."""
    for folder in (os.getenv("LOG_DIR") or os.getcwd(), tempfile.gettempdir()):
        try:
            return logging.FileHandler(os.path.join(folder, "main.log"), encoding="utf-8")
        except OSError:
            continue
    return None


# LOG_LEVEL=DEBUG also shows every task loaded and the token usage of each judge call.
log_file = open_log_file()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)] + ([log_file] if log_file else []),
)
for noisy in ("httpx", "httpx2", "httpcore", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("main")
log.info("started, log file %s", log_file.baseFilename if log_file else "none (no writable folder)")

data_accessor = TasksRepository()

tasks = data_accessor.task_data_array

# Unsolvable tasks that show without a model call: contradictory tests, input files that are not there.
import precheck

checked = []
for task in tasks:
    reason = precheck.unsolvable_reason(task)
    if reason:
        log.info("%s: unsolvable, not planned: %s", task.task_id, reason)
    else:
        checked.append(task)
log.info("precheck: %d of %d tasks found unsolvable without model calls", len(tasks) - len(checked), len(tasks))
tasks = checked

api_service = APIService(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

# The competition's launch command passes no budget, so there is a default; the proxy enforces the real one.
if os.getenv("BUDGET"):
    budget_in_tokens = int(os.getenv("BUDGET"))
else:
    budget_in_tokens = config.DEFAULT_BUDGET
    log.warning("BUDGET is not set, assuming %d; the run goes on until the proxy stops answering", budget_in_tokens)
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
    # Names from another provider (e.g. OpenRouter) match nothing in config.py; then MODELS is the fallback order too.
    config.FALLBACK_MODELS = ([model for model in config.FALLBACK_MODELS if model in config.MODEL_NAME_LIST]
                              or list(config.MODEL_NAME_LIST))
    log.info("models limited by MODELS: %s", ", ".join(config.MODEL_NAME_LIST))

# JUDGE_MODEL in .env replaces the judge from config.py, for a provider that names the models differently.
if os.getenv("JUDGE_MODEL"):
    config.JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL").strip()
    log.info("judge model set by JUDGE_MODEL: %s", config.JUDGE_MODEL_NAME)
judge_model = config.JUDGE_MODEL_NAME
# USE_JUDGE=1 turns the judge calls back on; by default the estimates cost no tokens (see config.USE_JUDGE).
use_judge = os.getenv("USE_JUDGE", "1" if config.USE_JUDGE else "0") not in ("", "0")
estimate_labels = []  # task id of each entry in task_estimates, for the logs

# The agent's commands need a working bash and python3; check before spending anything.
import shell

try:
    shell.check()
except RuntimeError as err:
    log.error("%s", err)
    sys.exit(2)

# A free plan may not include every model; asking the judge about those would waste the budget.
all_models = list(dict.fromkeys(config.MODEL_NAME_LIST + ([judge_model] if use_judge else [])
                                + config.FALLBACK_MODELS))
available_models = [model for model in all_models if api_service.is_model_available(model)]
unavailable_models = [model for model in all_models if model not in available_models]
log.info("models available to this key: %s%s", ", ".join(available_models) or "none",
         f"; not available: {', '.join(unavailable_models)}" if unavailable_models else "")
if use_judge and judge_model not in available_models:
    log.error("the judge model %s is not available to this key; change JUDGE_MODEL_NAME in config.py or set JUDGE_MODEL", judge_model)
    sys.exit(2)
model_names = [model for model in config.MODEL_NAME_LIST if model in available_models]

estimation_started = time.monotonic()


def judge(task):
    """Runs in a worker thread. A judge that fails must not stop the run or cost the task its chance."""
    try:
        return api_service.EstimateTask(judge_model, task, model_names)
    except Exception as err:
        return err, 0.0


def from_table(task):
    """No model call: the measured average cost of the task's type. Unsolvable tasks are left to the agent,
    which sees the files and gives up in its first call."""
    return estimates_from_table(task, model_names), 0.0


if use_judge:
    log.info("estimating %d tasks for %d models: one call per task to judge %s, %d at a time", len(tasks),
             len(model_names), judge_model, config.PARALLEL_SESSIONS)
else:
    log.info("estimating %d tasks for %d models from the cost table, without model calls", len(tasks),
             len(model_names))

with ThreadPoolExecutor(max_workers=config.PARALLEL_SESSIONS) as pool:
    for number, (task, (estimates, seconds)) in enumerate(zip(tasks, pool.map(judge if use_judge else from_table, tasks)), start=1):
        label = f"[{number}/{len(tasks)}] {task.task_id}"
        if estimates is None:
            log.info("%s: judged infeasible, skipped (%.1fs)", label, seconds)
            continue
        if isinstance(estimates, Exception):
            log.warning("%s: no estimate from the judge (%s: %s), using the default %d steps", label,
                        type(estimates).__name__, str(estimates)[:120], config.DEFAULT_STEPS)
            estimates = estimates_from_steps(config.DEFAULT_STEPS, task, model_names)
        log.log(logging.INFO if use_judge else logging.DEBUG, "%s: %s (%.1fs)", label,
                "; ".join(f"{e.model_name} {e.tokens} units, {e.seconds} s" for e in estimates), seconds)
        for estimate in estimates:
            task_estimates.append(estimate)
            estimate_labels.append(task.task_id)

log.info("estimation done in %.0fs: %d usable estimates; model checks and judge spent %d input + %d output tokens = %.0f in "
         "competition units", time.monotonic() - estimation_started, len(task_estimates), api_service.input_tokens,
         api_service.output_tokens, api_service.cost)

# The plan gets what is left of the hour and of the budget. Sessions run in parallel, so a task takes
# 1/PARALLEL_SESSIONS of its seconds from the wall clock.
seconds_left = max(1, int(config.TIME_LIMIT_IN_SECONDS - config.RUN_SAFETY_MARGIN_SECONDS
                          - (time.monotonic() - run_started)))
budget_left = int(budget_in_tokens - api_service.cost)
plan_estimates = [estimate.model_copy(update={"seconds": math.ceil(estimate.seconds / config.PARALLEL_SESSIONS)})
                  for estimate in task_estimates]
log.info("planning for %d units and %d s of wall clock", budget_left, seconds_left)
dp = DP()
res = dp.calc(budget_left, seconds_left, plan_estimates, groups=estimate_labels)

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
# Tasks the plan left out, each with its cheapest estimate: they are tried if budget remains after the plan.
planned_ids = {task.task_id for task, _ in plan}
spare_estimates = {}
for label, estimate in zip(estimate_labels, task_estimates):
    if label not in planned_ids and (label not in spare_estimates or estimate.tokens < spare_estimates[label].tokens):
        spare_estimates[label] = estimate
spare = [(tasks_by_id[label], estimate) for label, estimate in spare_estimates.items()]


def solutions_path():
    """OUT when given; otherwise solutions.json next to the code, or in the tasks folder when that is read-only."""
    if os.getenv("OUT"):
        return os.getenv("OUT")
    return "solutions.json" if os.access(os.getcwd(), os.W_OK) else os.path.join(data_accessor.path, "solutions.json")


executor = Executor(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
    budget=budget_in_tokens - api_service.cost,  # what the judge spent is gone already
    out=solutions_path(),
    unavailable_models=unavailable_models,
    deadline=run_started + config.TIME_LIMIT_IN_SECONDS - config.RUN_SAFETY_MARGIN_SECONDS,
)
summary = asyncio.run(executor.run(plan, spare))
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