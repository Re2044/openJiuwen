"""Solves one task with an openJiuwen agent that has two tools: bash and submit.

The tool set is kept tiny on purpose: tool schemas are resent with every model
call, and openJiuwen's stock tools cost about 2800 input tokens per call.
"""
import asyncio
import logging
import os
import sys
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path

# openJiuwen logs every request and writes ./logs while it loads, so configure it before importing the rest.
from openjiuwen.core.common.logging.log_config import configure_log_config

configure_log_config({
    "backend": "default",
    "level": "CRITICAL",  # failed calls are reported by our own warnings, in one line
    "propagate": False,
    "log_path": os.path.join(tempfile.gettempdir(), "openjiuwen-logs"),
    "output": ["console"],
    "interface_output": ["console"],
    "performance_output": ["console"],
})
# Its internal loggers report every callback at INFO, "llm" repeats each failed call as a long JSON line,
# and alembic announces its plugins while openJiuwen loads.
logging.getLogger("openjiuwen").setLevel(logging.WARNING)
logging.getLogger("llm").setLevel(logging.CRITICAL)
logging.getLogger("alembic").setLevel(logging.WARNING)

from openjiuwen.core.foundation.llm import Model, ModelClientConfig, ModelRequestConfig  # noqa: E402
from openjiuwen.core.foundation.tool import tool  # noqa: E402
from openjiuwen.core.single_agent import AgentCard  # noqa: E402
from openjiuwen.core.single_agent.rail.base import AgentCallbackContext, AgentRail, ModelCallInputs, ToolCallInputs  # noqa: E402
from openjiuwen.harness.factory import create_deep_agent  # noqa: E402

import config  # noqa: E402
import ratelimit  # noqa: E402
import shell  # noqa: E402

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You solve one small task with tools. Be brief and do not explain.\n"
    "bash runs a command in the task's working directory; prefer one command that computes the result.\n"
    "submit records the final answer and ends the task. Pass only the bare value, for example 42 or Osaka, "
    "or DONE when the task asks for that. Use submit instead of writing answer.txt yourself."
)
MAX_TOOL_OUTPUT_CHARS = 2000
# Signs in a failed call that this model is not available to the key, as opposed to the proxy being down.
MODEL_UNAVAILABLE_MARKERS = ("Error code: 403", "Error code: 404", "PermissionDenied", "NotFoundError")


def call_cost(usage):
    """Competition cost of one call: output tokens + input tokens / 10. Reasoning counts as output."""
    output = usage.output_tokens or 0
    reasoning = usage.reasoning_tokens or 0
    if reasoning > output:  # reported separately rather than inside output_tokens
        output += reasoning
    return (usage.input_tokens or 0) / 10 + output


class Ledger:
    """Spending shared by all sessions of a run."""

    def __init__(self, budget):
        self.budget = budget
        self.spent = 0.0
        self.calls = 0
        self.consecutive_llm_errors = 0

    def charge(self, cost):
        self.spent += cost
        self.calls += 1
        self.consecutive_llm_errors = 0

    def forgive_llm_error(self):
        """Undo the count of a failed call that was not the proxy's fault, such as a model outside the plan."""
        self.consecutive_llm_errors = max(0, self.consecutive_llm_errors - 1)

    @property
    def proxy_down(self):
        return self.consecutive_llm_errors >= config.MAX_CONSECUTIVE_LLM_ERRORS

    @property
    def exhausted(self):
        return self.spent >= self.budget or self.proxy_down


@dataclass
class Attempt:
    task_id: str
    model: str
    submitted: bool = False
    answer: str = ""
    cost: float = 0.0
    calls: int = 0
    llm_error: bool = False
    model_unavailable: bool = False


def truncate(text, limit=MAX_TOOL_OUTPUT_CHARS):
    if len(text) <= limit:
        return text
    head = limit // 4
    return f"{text[:head]}\n...[{len(text) - limit} chars cut]...\n{text[-(limit - head):]}"


async def run_command(argv, cwd, timeout):
    """Run a command; the return code is None on timeout. The proxy key is not passed to it."""
    env = {k: v for k, v in os.environ.items() if k not in ("LLM_API_KEY", "LLM_BASE_URL")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = await asyncio.create_subprocess_exec(*argv, cwd=cwd, env=env, stdout=asyncio.subprocess.PIPE,
                                                stderr=asyncio.subprocess.STDOUT)
    try:
        stdout, _ = await asyncio.wait_for(proc.communicate(), timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return None, f"timed out after {timeout}s"
    except asyncio.CancelledError:
        proc.kill()
        raise
    return proc.returncode, stdout.decode("utf-8", errors="replace")


def task_workdir(task, scratch_root):
    """Where the agent works. Tasks without workspace/ get a scratch folder, keeping task.json out of reach."""
    workspace = Path(task.path) / "workspace"
    if workspace.is_dir():
        return workspace
    scratch = Path(scratch_root) / task.task_id
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch


class TestGuard:
    """Checks coding tasks before accepting them and undoes any edits to the test files."""

    def __init__(self, workdir):
        self.workdir = Path(workdir)
        self.originals = {path: path.read_bytes() for path in self.workdir.glob("test_*.py")}

    def restore(self):
        for path, content in self.originals.items():
            if not path.exists() or path.read_bytes() != content:
                log.warning("restoring edited test file %s", path)
                path.write_bytes(content)

    async def failures(self):
        """Failure output, or None when all tests pass."""
        self.restore()
        for path in self.originals:
            code, output = await run_command([sys.executable, path.name], self.workdir, config.BASH_TIMEOUT_SECONDS)
            if code != 0:
                return truncate(output)
        return None


class MeterRail(AgentRail):
    """Bills every model call to the ledger, logs it, and ends the attempt as early as possible."""

    priority = 950  # before other rails: a call that cannot be paid for is refused first

    def __init__(self, attempt, ledger, cost_allowance):
        super().__init__()
        self.attempt = attempt
        self.ledger = ledger
        self.allowance = cost_allowance

    async def before_model_call(self, ctx: AgentCallbackContext):
        if self.ledger.exhausted or self.attempt.cost >= self.allowance:
            log.warning("%s: stopping, budget or task cap reached (task %.0f, run %.0f/%.0f)", self.attempt.task_id,
                        self.attempt.cost, self.ledger.spent, self.ledger.budget)
            ctx.request_force_finish({"output": "", "result_type": "stopped"})

    async def after_model_call(self, ctx: AgentCallbackContext):
        if not isinstance(ctx.inputs, ModelCallInputs):
            return
        if ctx.inputs.response is None:  # the call failed; on_model_exception records that
            return
        usage = getattr(ctx.inputs.response, "usage_metadata", None)
        if usage is None:
            log.warning("%s: response without usage data, spending is undercounted", self.attempt.task_id)
            return
        cost = call_cost(usage)
        self.ledger.charge(cost)
        self.attempt.cost += cost
        self.attempt.calls += 1
        log.info("%s [%s] call %d: %d in, %d out -> %.0f (task %.0f, run %.0f/%.0f)", self.attempt.task_id,
                 self.attempt.model, self.attempt.calls, usage.input_tokens, usage.output_tokens, cost,
                 self.attempt.cost, self.ledger.spent, self.ledger.budget)

    async def on_model_exception(self, ctx: AgentCallbackContext):
        self.attempt.llm_error = True
        self.ledger.consecutive_llm_errors += 1

    async def after_tool_call(self, ctx: AgentCallbackContext):
        # Ending right after an accepted answer skips a closing model call that would resend the whole context.
        if isinstance(ctx.inputs, ToolCallInputs) and self.attempt.submitted:
            ctx.request_force_finish({"output": "", "result_type": "submitted"})


class PatientModel(Model):
    """A Model that waits out rate limits (HTTP 429) instead of failing, so the agent keeps its progress.

    openJiuwen creates its HTTP clients without retries, and its own retrying rail is turned off here
    because its retries would not be metered.
    """

    async def invoke(self, *args, **kwargs):
        waits = 0
        while True:
            try:
                return await super().invoke(*args, **kwargs)
            except Exception as err:
                wait = ratelimit.wait_seconds(err)
                if wait is None or waits >= config.RATE_LIMIT_MAX_WAITS:
                    raise
                waits += 1
                log.info("%s: rate limited, waiting %.0fs (%d/%d): %s", self.model_config.model_name, wait, waits,
                         config.RATE_LIMIT_MAX_WAITS, str(err).split("Error code:")[-1][:120].strip())
                await asyncio.sleep(wait)


def build_model(base_url, api_key, model_name):
    return PatientModel(
        model_client_config=ModelClientConfig(
            client_provider="OpenAI",
            api_key=api_key,
            api_base=base_url,
            timeout=config.TASK_TIMEOUT_SECONDS,
            stream_first_chunk_timeout=config.TASK_TIMEOUT_SECONDS,
        ),
        model_config=ModelRequestConfig(model=model_name, max_tokens=config.SOLVER_MAX_TOKENS),
    )


async def solve_task(task, model_name, workdir, guard, store, ledger, base_url, api_key, cost_allowance):
    """One attempt at one task. The answer is saved by `store` as soon as it is accepted."""
    attempt = Attempt(task_id=task.task_id, model=model_name)

    async def accept(answer):
        answer = str(answer).strip()
        if not answer:
            return "Not accepted: the answer is empty."
        if task.check == "unittest":
            failures = await guard.failures()
            if failures is not None:
                log.info("%s: answer rejected, tests still fail", task.task_id)
                return f"Not accepted, tests still fail:\n{failures}"
        store.save(task, answer)
        attempt.submitted = True
        attempt.answer = answer
        log.info("%s: answer accepted: %s", task.task_id, answer[:60])
        return "Accepted."

    @tool(name="bash", description="Run a bash command in the working directory and return its output.")
    async def bash(command: str) -> str:
        log.info("%s: bash %s", task.task_id, " ".join(str(command).split())[:150])
        if "task.json" in command:
            return "Refused: task.json is not part of the task data."
        code, output = await run_command([shell.find_bash(), "-c", shell.PYTHON3_PREAMBLE + command], workdir,
                                         config.BASH_TIMEOUT_SECONDS)
        return truncate(output) if code == 0 else f"exit code {code}\n{truncate(output)}"

    @tool(name="submit", description="Record the final answer and end the task.")
    async def submit(answer: str) -> str:
        return await accept(answer)

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as agent_home:
        agent = create_deep_agent(
            build_model(base_url, api_key, model_name),
            card=AgentCard(id=f"{task.task_id}-{uuid.uuid4().hex[:8]}", name="task_solver"),
            workspace=agent_home,  # DeepAgent keeps its own files here, away from the task folder
            tools=[bash, submit],
            rails=[MeterRail(attempt, ledger, cost_allowance)],
            system_prompt=SYSTEM_PROMPT,
            max_iterations=config.SOLVER_MAX_ITERATIONS,
            parallel_tool_calls=False,
            prompt_mode="none",
            # These defaults add a paid image-support probe, prompt text and unmetered retries.
            enable_read_image_multimodal=False,
            enable_security_rail=False,
            enable_model_anomaly_detection_rail=False,
        )
        inputs = {"query": task.instruction, "conversation_id": f"{task.task_id}-{uuid.uuid4().hex[:8]}"}
        try:
            await asyncio.wait_for(agent.invoke(inputs), config.TASK_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            log.warning("%s: attempt with %s timed out after %ss", task.task_id, model_name,
                        config.TASK_TIMEOUT_SECONDS)
        except Exception as err:  # e.g. the iteration limit; provider errors are also flagged by the rail
            if attempt.llm_error and any(marker in str(err) for marker in MODEL_UNAVAILABLE_MARKERS):
                attempt.model_unavailable = True
                ledger.forgive_llm_error()
                log.warning("%s: model %s is not available to this key: %s", task.task_id, model_name,
                            str(err).split("Error code:")[-1][:160])
            else:
                log.warning("%s: attempt with %s ended with %s: %s", task.task_id, model_name, type(err).__name__,
                            str(err)[:200])

    # The instructions ask for answer.txt, so a model may write it itself instead of calling submit.
    written = workdir / "answer.txt"
    if not attempt.submitted and written.is_file():
        await accept(written.read_text(encoding="utf-8", errors="replace"))
    return attempt
