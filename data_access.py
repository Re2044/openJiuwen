from dataclasses import dataclass
import os
import json
import logging
import time
from pydantic import BaseModel, Field
import config
from openai import OpenAI, NotFoundError, PermissionDeniedError, RateLimitError
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

            with open(task_file_path, 'r') as file:
                json_data = json.load(file)
                task_data = TaskData(
                    task_id = json_data["task_id"],
                    task_type = json_data["task_type"],
                    instruction = json_data["instruction"],
                    reward = json_data["reward"],
                    path = os.path.abspath(f),
                    check = json_data.get("check", ""),
                )
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
        self.last_call_seconds = 0.0

    @property
    def cost(self):
        """Spending in competition units: output tokens + input tokens / 10."""
        return self.output_tokens + self.input_tokens / 10
    

    def _record_usage(self, resp):
        self.calls += 1
        if resp.usage is not None:
            self.input_tokens += resp.usage.prompt_tokens
            self.output_tokens += resp.usage.completion_tokens

    def is_model_available(self, model_name):
        """One tiny call to see whether this key may use the model (a free plan may not include all of them)."""
        try:
            resp = self.client.chat.completions.create(
                model=model_name, messages=[{"role": "user", "content": "Reply with OK."}], max_tokens=1)
        except (PermissionDeniedError, NotFoundError) as err:
            log.warning("model %s is not available to this key: %s", model_name, getattr(err, "message", err))
            return False
        except Exception as err:  # not an access problem, so do not rule the model out
            log.warning("model %s check failed, assuming it is available: %s", model_name, err)
            return True
        self._record_usage(resp)
        return True

    def JudgeTask(self, judge_model, task, workspace=None):
        """One judge call per task. Returns a JudgeVerdict, or None when the call or its answer failed.

        workspace: an optional summary of the task's files (workspace_summary.summarize).
        """
        content = f"Task type: {task.task_type}\nInstruction: {task.instruction}"
        if workspace:
            content += f"\nWorkspace files (summary made by a script):\n{workspace}"
        started = time.monotonic()
        resp = None
        try:
            for wait_number in range(config.RATE_LIMIT_MAX_WAITS + 1):
                try:
                    resp = self.client.chat.completions.parse(
                        model=judge_model,
                        messages=[
                            {"role": "system", "content": config.JUDGE_SYSTEM},
                            {"role": "user", "content": content},
                        ],
                        response_format=JudgeVerdict,
                        temperature=0,
                    )
                    break
                except RateLimitError as err:
                    if wait_number == config.RATE_LIMIT_MAX_WAITS:
                        raise
                    wait = ratelimit.wait_seconds(err)
                    log.info("judge rate limited, waiting %.0fs: %s", wait, getattr(err, "message", err))
                    time.sleep(wait)
        except Exception as err:  # a failed judge call falls back to the task type's default, see estimate_from_verdict
            log.warning("judge call for %s failed, using the default for %s: %s", task.task_id, task.task_type,
                        str(err)[:200])
            return None
        finally:
            self.last_call_seconds = time.monotonic() - started
            if resp is not None:
                self._record_usage(resp)
                if resp.usage is None:
                    log.warning("judge response has no usage data, spending is undercounted")
        return resp.choices[0].message.parsed

@dataclass
class TaskData:
    task_id: str
    task_type: str
    instruction: str
    reward: int
    path: str = ""  # absolute path of the task folder
    check: str = ""  # how the answer is checked: exact_int, exact_str, unittest

class JudgeVerdict(BaseModel):
    """What the judge returns. Ranges are enforced by estimate_from_verdict, not here, so that an out-of-range
    answer is clamped instead of failing to parse."""
    difficulty: str = Field(description="trivial, easy, medium or hard")
    steps: int = Field(description="model calls the agent needs, from 1 to 8")
    solvable_probability: float = Field(description="chance from 0 to 1 that the task can be done at all")
    needs_reasoning_model: bool = Field(description="true when a model without reasoning would likely fail")


def is_reasoning_model(model_name):
    return not model_name.endswith("-noreason")


def estimate_from_verdict(verdict, task_type, model_name):
    """Steps, tokens, cost (competition units) and seconds for one model, from the judge's step count.

    The step count is clamped to the range seen for the task type; without a verdict the type's default is used.
    Call n sends first + (n - 1) * growth input tokens, since every call resends the history.
    """
    low, high, default = config.STEP_LIMITS.get(task_type, config.DEFAULT_STEP_LIMITS)
    steps = default if verdict is None else min(max(int(verdict.steps), low), high)
    first, growth = config.STEP_INPUT_TOKENS.get(task_type, config.DEFAULT_STEP_INPUT_TOKENS)
    input_tokens = steps * first + growth * steps * (steps - 1) // 2
    output_tokens = steps * config.OUTPUT_TOKENS_PER_CALL.get(model_name, config.DEFAULT_OUTPUT_TOKENS_PER_CALL)
    return {"steps": steps, "tokens": input_tokens + output_tokens,
            "cost": round(output_tokens + input_tokens / 10), "seconds": steps * config.SECONDS_PER_CALL}


class TaskEstimate(BaseModel):
    """One planning option: a task on a model. `tokens` is the expected cost in competition units, which is
    what the budget is counted in."""
    model_name: str
    tokens: int = Field(description="Expected cost in competition units: output tokens + input tokens / 10")
    seconds: int = Field(description="Wall-clock time expected, in seconds")
    reward: int