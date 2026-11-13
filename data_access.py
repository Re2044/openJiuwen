from dataclasses import dataclass
import os
import json
import logging
import threading
import time
from pydantic import BaseModel, Field
import config
import models
from openai import OpenAI, LengthFinishReasonError, NotFoundError, PermissionDeniedError, RateLimitError
import ratelimit

log = logging.getLogger(__name__)


class TasksRepository:
    def __init__(self, path_to_task_dir: str = "./task-dir"):
        if os.getenv("TASKS") is not None and os.getenv("TASKS") != "":
            path_to_task_dir = os.getenv("TASKS")

        self.path = os.path.abspath(path_to_task_dir)
        self.task_data_array = self.get_task_data_array(path_to_task_dir)
        log.info("loaded %d tasks from %s", len(self.task_data_array), os.path.abspath(path_to_task_dir))

    def get_task_data_array(self, path_to_task_dir: str):
        task_data_array = []
        folders = [os.path.join(path_to_task_dir, f) for f in os.listdir(path_to_task_dir) if os.path.isdir(os.path.join(path_to_task_dir, f))]
        for f in folders:
            task_file_path = os.path.join(f, "task.json")

            try:
                with open(task_file_path, 'r', encoding="utf-8") as file:
                    json_data = json.load(file)
                task_data = TaskData(
                    task_id = json_data["task_id"],
                    task_type = json_data.get("task_type", ""),
                    instruction = json_data["instruction"],
                    reward = json_data["reward"],
                    path = os.path.abspath(f),
                    check = json_data.get("check", ""),
                )
            except (OSError, ValueError, KeyError) as err:  # one broken folder must not stop the run
                log.warning("skipping %s: %s: %s", f, type(err).__name__, err)
                continue
            task_data_array.append(task_data)
            log.debug("task %s: type %s, reward %s", task_data.task_id, task_data.task_type, task_data.reward)

        return task_data_array

class APIService:
    def __init__(self, base_url, api_key):
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        # Judge spending so far, for the logs.
        self.calls = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self._lock = threading.Lock()  # judge calls run in several threads

    @property
    def cost(self):
        """Spending in competition units: output tokens + input tokens / 10."""
        return self.output_tokens + self.input_tokens / 10
    

    def _record_usage(self, resp):
        with self._lock:
            self.calls += 1
            if resp.usage is not None:
                self.input_tokens += resp.usage.prompt_tokens
                self.output_tokens += resp.usage.completion_tokens

    def is_model_available(self, model_name):
        """One tiny call to see whether this key may use the model (a free plan may not include all of them)."""
        api_name, extra_body = models.request_model(model_name)
        try:
            resp = self.client.chat.completions.create(
                model=api_name, messages=[{"role": "user", "content": "Reply with OK."}], max_tokens=1,
                extra_body=extra_body)
        except (PermissionDeniedError, NotFoundError) as err:
            log.warning("model %s is not available to this key: %s", model_name, getattr(err, "message", err))
            return False
        except Exception as err:  # not an access problem, so do not rule the model out
            log.warning("model %s check failed, assuming it is available: %s", model_name, err)
            return True
        self._record_usage(resp)
        return True

    def EstimateTask(self, judge_model, task, model_names):
        """One judge call for a task: (an estimate per model, seconds the call took).

        The estimates are None when the judge finds the task infeasible. Raises when the judge gives no
        usable answer; the tokens of such a call are still counted.
        """
        started = time.monotonic()
        # JUDGE_NO_REASONING=1 in .env: ask the provider (OpenRouter) to turn the judge's reasoning off, which is
        # what the -noreason models of the competition proxy do by themselves.
        extra_body = {"reasoning": {"enabled": False}} if os.getenv("JUDGE_NO_REASONING", "0") not in ("", "0") else None
        question = (f"Task type: {task.task_type}\nTask: {task.instruction}\n"
                    f"Workspace: {workspace_summary(task.path)}")
        for wait_number in range(config.RATE_LIMIT_MAX_WAITS + 1):
            try:
                resp = self.client.chat.completions.parse(
                    model=judge_model,
                    messages=[
                        {"role": "system", "content": config.SYSTEM},
                        {"role": "user", "content": question},
                    ],
                    response_format=TaskJudgement,
                    temperature=0,
                    extra_body=extra_body,
                )
                break
            except LengthFinishReasonError as err:
                # The judge used up its output limit (usually on reasoning): the tokens are spent, so count them.
                self._record_usage(err.completion)
                raise
            except RateLimitError as err:
                if wait_number == config.RATE_LIMIT_MAX_WAITS:
                    raise
                wait = ratelimit.wait_seconds(err)
                log.info("judge rate limited, waiting %.0fs: %s", wait, getattr(err, "message", err))
                time.sleep(wait)

        seconds = time.monotonic() - started
        self._record_usage(resp)
        if resp.usage is not None:
            log.debug("judge call for %s took %.1fs: %d input, %d output tokens", task.task_id, seconds,
                      resp.usage.prompt_tokens, resp.usage.completion_tokens)
        else:
            log.warning("judge response has no usage data, spending is undercounted")

        judgement = resp.choices[0].message.parsed
        if judgement is None:
            raise ValueError("the judge returned no parsed answer")
        if not judgement.feasible:
            return None, seconds
        return estimates_from_steps(judgement.steps, task, model_names), seconds


def cost_per_step(model_name):
    return config.COST_PER_STEP.get(model_name.split("/")[-1], config.DEFAULT_COST_PER_STEP)


def estimates_from_table(task, model_names):
    """An estimate per model without any model call: the measured average of the task's type."""
    cost = config.COST_BY_TASK_TYPE.get(task.task_type, config.DEFAULT_TASK_COST)
    seconds = config.SECONDS_BY_TASK_TYPE.get(task.task_type, config.DEFAULT_TASK_SECONDS)
    return [TaskEstimate(model_name=name, tokens=max(1, round(cost * config.MODEL_COST_FACTOR.get(name.split("/")[-1], 1.0))),
                         seconds=seconds, reward=task.reward) for name in model_names]


def estimates_from_steps(steps, task, model_names):
    """An estimate per model from the number of agent steps and the measured cost of a step."""
    steps = min(max(int(steps), 1), config.SOLVER_MAX_ITERATIONS)
    return [TaskEstimate(model_name=name, tokens=steps * cost_per_step(name), seconds=steps * config.SECONDS_PER_STEP,
                         reward=task.reward) for name in model_names]


def workspace_summary(task_path):
    """The task's files with their sizes, for the judge. Costs no model calls."""
    workspace = os.path.join(task_path, "workspace")
    if not os.path.isdir(workspace):
        return "no files"
    files = []
    for root, _, names in os.walk(workspace):
        for name in names:
            if name == "answer.txt":  # left by an earlier run
                continue
            full = os.path.join(root, name)
            files.append((os.path.relpath(full, workspace).replace(os.sep, "/"), os.path.getsize(full)))
    if not files:
        return "no files"
    files.sort()
    shown = ", ".join(f"{name} ({size} bytes)" for name, size in files[:config.WORKSPACE_SUMMARY_MAX_FILES])
    if len(files) > config.WORKSPACE_SUMMARY_MAX_FILES:
        rest = files[config.WORKSPACE_SUMMARY_MAX_FILES:]
        shown += f", and {len(rest)} more files ({sum(size for _, size in rest)} bytes)"
    return shown


@dataclass
class TaskData:
    task_id: str
    task_type: str
    instruction: str
    reward: int
    path: str = ""  # absolute path of the task folder
    check: str = ""  # how the answer is checked: exact_int, exact_str, unittest

class TaskJudgement(BaseModel):
    """What the judge returns for one task."""
    feasible: bool = Field(description="False when the task cannot be completed with the given files and tools")
    steps: int = Field(description="Agent steps needed, counting the final submit")


class TaskEstimate(BaseModel):
    model_name: str
    tokens: int = Field(description="Expected cost in budget units: output tokens + input tokens / 10")
    seconds: int = Field(description="Wall-clock time expected, in seconds")
    reward: int