from openai import OpenAI
from data_access import TasksRepository, TaskData, TaskEstimate, APIService, estimate_from_verdict, is_reasoning_model
from dp import DP
import os
import sys
import time
import logging
import tempfile
import config
from workspace_summary import summarize
from dotenv import load_dotenv

load_dotenv()

# Files the run writes (log, default solutions file) go to the working folder, or to the temp folder when the working
# folder is read-only, as /app is in the competition container.
RUN_DIR = os.getcwd() if os.access(os.getcwd(), os.W_OK) else tempfile.gettempdir()
LOG_FILE = os.path.join(RUN_DIR, "main.log")

# LOG_LEVEL=DEBUG also shows every task loaded and the token usage of each judge call.
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")],
)
for noisy in ("httpx", "httpx2", "httpcore", "openai"):
    logging.getLogger(noisy).setLevel(logging.WARNING)
log = logging.getLogger("main")
log.info("started, log file %s", LOG_FILE)

data_accessor = TasksRepository()

tasks = data_accessor.task_data_array

api_service = APIService(
    base_url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)

budget_in_tokens = int(os.getenv("BUDGET"))
log.info("LLM endpoint %s, budget %d, time limit %ds", os.getenv("LLM_BASE_URL"), budget_in_tokens,
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
if os.getenv("JUDGE_WORKSPACE_SUMMARY") == "0":  # e.g. to measure what the summary is worth
    config.JUDGE_WORKSPACE_SUMMARY = False
if os.getenv("SOLVER_WARNINGS") == "0":
    config.SOLVER_WARNINGS = False
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

log.info("estimating %d tasks with judge %s, one call per task, for models %s", len(tasks), judge_model,
         ", ".join(model_names))
estimation_started = time.monotonic()
judge_estimates = {}  # task id -> the judge's verdict and the estimate per model, for the run report
for number, task in enumerate(tasks, start=1):
    workspace = summarize(task.path, task.instruction) if config.JUDGE_WORKSPACE_SUMMARY else None
    verdict = api_service.JudgeTask(judge_model, task, workspace)
    judged = judge_estimates[task.task_id] = {"verdict": verdict.model_dump() if verdict else None, "estimates": {},
                                              "workspace": workspace}
    if verdict is not None and verdict.solvable_probability < config.MIN_SOLVABLE_PROBABILITY:
        log.info("[%d/%d] %s: judged unsolvable (p=%.2f), left out (%.1fs)", number, len(tasks), task.task_id,
                 verdict.solvable_probability, api_service.last_call_seconds)
        continue
    models = model_names
    if verdict is not None and verdict.needs_reasoning_model:
        models = [m for m in model_names if is_reasoning_model(m)] or model_names
    for model_name in models:
        estimate = estimate_from_verdict(verdict, task.task_type, model_name)
        judged["estimates"][model_name] = estimate
        task_estimates.append(TaskEstimate(model_name=model_name, tokens=estimate["cost"],
                                           seconds=estimate["seconds"], reward=task.reward))
        estimate_labels.append(task.task_id)
    log.info("[%d/%d] %s: %s, %d steps%s, cost %s (%.1fs, judge spent %.0f so far)", number, len(tasks),
             task.task_id, verdict.difficulty if verdict else "no verdict", estimate["steps"],
             ", needs reasoning" if verdict is not None and verdict.needs_reasoning_model else "",
             "/".join(str(e["cost"]) for e in judged["estimates"].values()), api_service.last_call_seconds,
             api_service.cost)

log.info("estimation done in %.0fs: %d usable estimates; judge spent %d input + %d output tokens = %.0f in "
         "competition units", time.monotonic() - estimation_started, len(task_estimates), api_service.input_tokens,
         api_service.output_tokens, api_service.cost)

dp = DP()
res = dp.calc(budget_in_tokens, config.TIME_LIMIT_IN_SECONDS, task_estimates, groups=estimate_labels)

best_reward, chosen = res
log.info("plan: total reward %s from %d estimates", best_reward, len(chosen))
for index in chosen:
    estimate = task_estimates[index - 1]
    log.info("  %s on %s: cost %d, %d s, reward %d", estimate_labels[index - 1], estimate.model_name,
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
    out=os.getenv("OUT") or os.path.join(RUN_DIR, "solutions.json"),
    unavailable_models=unavailable_models,
)
summary = asyncio.run(executor.run(plan))
log.info("finished: %s; total spent %.0f of %d (judge %.0f, solving %.0f)", summary,
         api_service.cost + summary["solving_cost"], budget_in_tokens, api_service.cost, summary["solving_cost"])

# What the judge predicted next to what solving took, per task; evaluate.py compares them.
import json
from pathlib import Path

planned = {estimate_labels[index - 1]: task_estimates[index - 1].model_name for index in chosen}
run_report = {
    "tasks_dir": data_accessor.path,
    "budget": budget_in_tokens,
    "judge": {"model": judge_model, "calls": api_service.calls, "input_tokens": api_service.input_tokens,
              "output_tokens": api_service.output_tokens, "cost": round(api_service.cost, 1)},
    "summary": summary,
    "tasks": {task.task_id: {"reward": task.reward, "judge": judge_estimates[task.task_id],
                             "planned_model": planned.get(task.task_id),
                             "solve": executor.records.get(task.task_id)} for task in tasks},
}
report_path = executor.store.solutions_file.with_name("run_report.json")
report_path.write_text(json.dumps(run_report, ensure_ascii=False, indent=1), encoding="utf-8")
log.info("run report with the judge's estimates and actual costs: %s", report_path.resolve())

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
    from evaluate import estimate_table

    for line in estimate_table(data_accessor.path, run_report):
        log.info("%s", line)