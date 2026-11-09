"""One attempt at one task: an openJiuwen agent with two small tools and a metering rail.

The tool set is deliberately tiny. Tool schemas are resent on every call, and
openJiuwen's stock tools cost about 2800 input tokens per call against about 100 here.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig
from openjiuwen.core.foundation.tool import tool
from openjiuwen.core.single_agent import AgentCard
from openjiuwen.core.single_agent.rail.base import (
    AgentCallbackContext,
    AgentRail,
    ModelCallInputs,
    ToolCallInputs,
)
from openjiuwen.harness.factory import create_deep_agent

from harness.budget import Ledger, Trajectory, call_cost
from harness.settings import Settings
from harness.tasks import AnswerStore, Task

log = logging.getLogger("harness")

SYSTEM_PROMPT = (
    "You solve one small task with tools. Be brief and do not explain.\n"
    "bash runs a command in the task's working directory; prefer one command that computes the result.\n"
    "submit records the final answer and ends the task. Pass only the bare value, for example 42 or Osaka, "
    "or DONE when the task asks for that. Use submit instead of writing answer.txt yourself."
)

MAX_TOOL_OUTPUT_CHARS = 2000
UNITTEST_CHECK = "unittest"
# Task instructions call python3, which on a Windows dev machine is only a Microsoft Store alias.
BASH_PREAMBLE = f'python3() {{ "{Path(sys.executable).as_posix()}" "$@"; }}; ' if os.name == "nt" else ""
FINISH_SUBMITTED = "submitted"
FINISH_STOPPED = "stopped"


def truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    head = limit // 4
    return f"{text[:head]}\n...[{len(text) - limit} chars cut]...\n{text[-(limit - head):]}"


def child_env() -> dict[str, str]:
    """Commands written by the model never see the proxy credentials."""
    env = {k: v for k, v in os.environ.items() if k not in ("LLM_API_KEY", "LLM_BASE_URL")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


async def run_process(argv: list[str], cwd: Path, timeout: float) -> tuple[int | None, str]:
    """Run a command; the return code is None when it timed out."""
    proc = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, env=child_env(), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None, f"timed out after {timeout:.0f}s"
    except asyncio.CancelledError:
        proc.kill()
        raise
    return proc.returncode, stdout.decode("utf-8", errors="replace")


class GuardedTests:
    """Keeps the model from passing a coding task by editing its tests."""

    def __init__(self, workdir: Path) -> None:
        self._workdir = workdir
        self._originals = {p: p.read_bytes() for p in workdir.glob("test_*.py")}

    def restore(self) -> None:
        for path, content in self._originals.items():
            if not path.exists() or path.read_bytes() != content:
                log.info("restoring edited test file %s", path)
                path.write_bytes(content)

    async def failures(self, timeout: float) -> str | None:
        """Return failure output, or None when every test file passes."""
        self.restore()
        for path in self._originals:
            code, output = await run_process([sys.executable, path.name], self._workdir, timeout)
            if code != 0:
                return truncate(output)
        return None


@dataclass
class AttemptResult:
    model: str
    submitted: bool = False
    llm_error: bool = False
    cost: float = 0.0
    calls: int = 0
    final_text: str = ""


class MeterRail(AgentRail):
    """Charges every call to the shared ledger and ends the attempt as early as possible."""

    priority = 950  # ahead of other rails: a call we cannot pay for is refused first

    def __init__(self, task: Task, result: AttemptResult, ledger: Ledger, trajectory: Trajectory,
                 cost_allowance: float) -> None:
        super().__init__()
        self._task = task
        self._result = result
        self._ledger = ledger
        self._trajectory = trajectory
        self._allowance = cost_allowance

    async def before_model_call(self, ctx: AgentCallbackContext) -> None:
        if self._ledger.halted or self._result.cost >= self._allowance:
            ctx.request_force_finish({"output": "", "result_type": FINISH_STOPPED})

    async def after_model_call(self, ctx: AgentCallbackContext) -> None:
        if not isinstance(ctx.inputs, ModelCallInputs):
            return
        self.record(getattr(ctx.inputs.response, "usage_metadata", None))

    def record(self, usage: Any) -> None:
        """Charge one model call to the ledger and the trajectory."""
        if usage is None:
            log.warning("%s: response without usage data", self._task.task_id)
            return
        cost = call_cost(usage)
        self._ledger.charge(cost)
        self._result.cost += cost
        self._result.calls += 1
        self._trajectory.write(
            event="call", task_id=self._task.task_id, task_type=self._task.task_type,
            model=usage.model_name or self._result.model, call=self._result.calls,
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens, cost=cost,
        )

    async def on_model_exception(self, ctx: AgentCallbackContext) -> None:
        self._result.llm_error = True

    async def after_tool_call(self, ctx: AgentCallbackContext) -> None:
        # Stopping here skips the closing model call, which would resend the whole context for nothing.
        if isinstance(ctx.inputs, ToolCallInputs) and self._result.submitted:
            ctx.request_force_finish({"output": "", "result_type": FINISH_SUBMITTED})


async def run_bash(command: str, workdir: Path, timeout: float) -> str:
    if "task.json" in command:
        return "Refused: task.json is not part of the task data."
    bash_path = shutil.which("bash") or "bash"
    code, output = await run_process([bash_path, "-c", BASH_PREAMBLE + command], workdir, timeout)
    return truncate(output) if code == 0 else f"exit code {code}\n{truncate(output)}"


def build_tools(task: Task, workdir: Path, guard: GuardedTests, store: AnswerStore, settings: Settings,
                result: AttemptResult) -> list:
    @tool(name="bash", description="Run a bash command in the working directory and return its output.")
    async def bash(command: str) -> str:
        return await run_bash(command, workdir, settings.bash_timeout_sec)

    @tool(name="submit", description="Record the final answer and end the task.")
    async def submit(answer: str) -> str:
        return await accept_answer(task, answer, guard, store, settings, result)

    return [bash, submit]


async def accept_answer(task: Task, answer: str, guard: GuardedTests, store: AnswerStore, settings: Settings,
                        result: AttemptResult) -> str:
    answer = str(answer).strip()
    if not answer:
        return "Not accepted: the answer is empty."
    if task.check == UNITTEST_CHECK:
        failures = await guard.failures(settings.bash_timeout_sec)
        if failures is not None:
            return f"Not accepted, tests still fail:\n{failures}"
    if not store.save(task, answer):
        return "Not accepted: the answer could not be saved."
    result.submitted = True
    return "Accepted."


def build_model(settings: Settings, model_name: str, max_tokens: int | None = None) -> Model:
    return Model(
        model_client_config=ModelClientConfig(
            client_provider="OpenAI",
            api_key=settings.api_key,
            api_base=settings.base_url,
            timeout=settings.llm_timeout_sec,
            stream_first_chunk_timeout=settings.llm_timeout_sec,
        ),
        model_config=ModelRequestConfig(model=model_name, max_tokens=max_tokens or settings.max_tokens),
    )


def build_agent(task: Task, model: Model, agent_home: str | Path, tools: list, rails: list, system_prompt: str,
                max_iterations: int, **overrides: Any) -> Any:
    """A DeepAgent with the costly defaults turned off; `overrides` can turn any of them back on."""
    options = {
        "parallel_tool_calls": False,
        "prompt_mode": "none",
        # These defaults add a paid image-support probe and prompt text, and their retries are not metered here.
        "enable_read_image_multimodal": False,
        "enable_security_rail": False,
        "enable_model_anomaly_detection_rail": False,
        **overrides,
    }
    return create_deep_agent(
        model,
        card=AgentCard(id=f"{task.task_id}-{uuid.uuid4().hex[:8]}", name="task_solver"),
        workspace=str(agent_home),  # DeepAgent keeps its own artifacts here, away from the task directory
        tools=tools,
        rails=rails,
        system_prompt=system_prompt,
        max_iterations=max_iterations,
        **options,
    )


async def run_attempt(task: Task, model_name: str, settings: Settings, ledger: Ledger, trajectory: Trajectory,
                      store: AnswerStore, guard: GuardedTests, workdir: Path, cost_allowance: float) -> AttemptResult:
    result = AttemptResult(model=model_name)
    rail = MeterRail(task, result, ledger, trajectory, cost_allowance)
    with tempfile.TemporaryDirectory(dir=settings.work_root, ignore_cleanup_errors=True) as agent_home:
        agent = build_agent(
            task, build_model(settings, model_name), agent_home,
            tools=build_tools(task, workdir, guard, store, settings, result),
            rails=[rail], system_prompt=SYSTEM_PROMPT, max_iterations=settings.max_iterations,
        )
        inputs = {"query": task.instruction, "conversation_id": f"{task.task_id}-{uuid.uuid4().hex[:8]}"}
        try:
            # A generous outer limit in case the client timeout does not fire on a stalled proxy.
            outcome = await asyncio.wait_for(
                agent.invoke(inputs), settings.llm_timeout_sec * (settings.max_iterations + 1),
            )
        except asyncio.TimeoutError:
            log.warning("%s: attempt with %s timed out", task.task_id, model_name)
            result.llm_error = True
            return result
        except Exception as err:  # for example max_iterations; provider errors also set llm_error via the rail
            log.warning("%s: attempt with %s failed: %s", task.task_id, model_name, str(err)[:300])
            return result

    if isinstance(outcome, dict):
        result.final_text = str(outcome.get("output") or "").strip()
    return result
