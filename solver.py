"""Solves one task with an openJiuwen agent in as few model calls as possible.

Every call resends the conversation, so the cost of a task is mostly its number of calls. Three things keep
that number low:
- the task's files are put into the first message, so the agent does not spend calls on ls and cat;
- bash can end the task itself (final=true), so computing an answer and submitting it is one call;
- give_up ends an unsolvable task at once instead of letting the agent run to its cost cap.
The tool set is kept tiny on purpose: tool schemas are resent with every model call, and openJiuwen's stock
tools cost about 2800 input tokens per call.
"""
import asyncio
import logging
import os
import re
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
import models  # noqa: E402
import precheck  # noqa: E402
import ratelimit  # noqa: E402
import shell  # noqa: E402

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You solve one small task. Its files are shown below the task; bash runs a command in their directory. "
    "Finish in as few steps as you can, ideally one, and do not explain.\n"
    "- The answer is already visible: call submit.\n"
    "- The answer must be computed: call bash with final=true and a command that prints only the answer; "
    "its output is submitted for you.\n"
    "- The task asks to change files: make the change with bash and submit DONE; one command with final=true "
    "that edits the file and then prints DONE does both.\n"
    "submit takes the bare value, for example 42 or Osaka. Never write answer.txt yourself.\n"
    "First check that the task can be solved at all. Call give_up at once, without computing anything, when:\n"
    "- a quantity it needs is not given as a number (\"some\", \"several\", \"a few\");\n"
    "- a step is undefined, such as splitting or dividing by zero;\n"
    "- a file, column, field, person or record it asks about is not in Files (a Note lists names found in no "
    "file);\n"
    "- the record it needs lacks the field asked for, or points to a record that is not there;\n"
    "- two tests require different results from the same call.\n"
    "Never guess, never skip a broken step, and never answer 0, none or a sentence for something that is not there."
)
MAX_TOOL_OUTPUT_CHARS = 2000
# Signs in a failed call that this model is not available to the key, as opposed to the proxy being down.
MODEL_UNAVAILABLE_MARKERS = ("Error code: 403", "Error code: 404", "PermissionDenied", "NotFoundError")


def call_cost(usage):
    """Competition cost of one call: output tokens + input tokens / 10. Reasoning counts as output.

    Providers report reasoning either inside the output tokens or next to them. The total settles which:
    whatever it holds beyond the input was generated. Comparing the two counts alone is not enough, because a
    call that is all reasoning can report a few reasoning tokens more than output tokens, and adding them then
    bills the call twice.
    """
    input_tokens = usage.input_tokens or 0
    output = usage.output_tokens or 0
    reasoning = usage.reasoning_tokens or 0
    total = getattr(usage, "total_tokens", 0) or 0
    if total >= input_tokens + output:
        output = total - input_tokens
    if reasoning > output * 1.05 + 16:  # clearly more than the output can hold: counted apart, and not in the total
        output += reasoning
    return input_tokens / 10 + output


class Ledger:
    """Spending shared by all sessions of a run.

    The budget is a hard limit: every model call first reserves its worst-case cost, and a call is made only
    when the money spent, the reservations of the calls in flight and its own reservation all fit. Without
    that, the calls already in flight when the budget runs out push the run over it.
    """

    def __init__(self, budget):
        self.budget = budget
        self.spent = 0.0
        self.reserved = 0.0  # worst-case cost of the calls in flight
        self.calls = 0
        self.consecutive_llm_errors = 0

    def charge(self, cost):
        self.spent += cost
        self.calls += 1
        self.consecutive_llm_errors = 0

    @property
    def available(self):
        """What a new call may still cost."""
        return self.budget - self.spent - self.reserved

    def fits(self, amount):
        """Whether a call of this worst-case cost can still be paid once the calls in flight are settled."""
        return self.spent + amount <= self.budget

    async def reserve(self, amount):
        """Set the amount aside for a call. False when it can no longer fit into the budget.

        When only the reservations of other calls stand in the way, waits for them: they are worst cases, and
        most calls cost a fraction of theirs.
        """
        while True:
            if self.proxy_down or not self.fits(amount):
                return False
            if amount <= self.available:
                self.reserved += amount
                return True
            await asyncio.sleep(0.1)

    def release(self, amount):
        self.reserved = max(0.0, self.reserved - amount)

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
    gave_up: bool = False  # the agent found the task unsolvable


REFUSAL = re.compile(
    r"\b(cannot|can't|can not|unable|impossible|unsolvable|infeasible|insufficient|not enough|missing|undefined"
    r"|no answer|does not exist|doesn't exist|not (?:found|available|present|possible|solvable|given|provided|specified"
    r"|determin\w+)|n/a)\b|^(none|null|nan|unknown|error)$", re.IGNORECASE)


def is_refusal(answer):
    """Whether the text says the task cannot be answered instead of answering it."""
    return bool(REFUSAL.search(answer))


def normalize_answer(answer):
    """Trim the answer; a whole number written as a float (3570.0) becomes 3570, which is what the check expects."""
    answer = str(answer).strip()
    if re.fullmatch(r"[-+]?\d+\.0+", answer):
        answer = answer.split(".")[0]
    return answer


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


def _read_text(path):
    """The file's text, or None for a binary or oversized file."""
    try:
        if path.stat().st_size > 1_000_000:
            return None
        return path.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def workspace_context(workdir):
    """The task's files as text for the first message; reading them here costs no model call.

    Small workspaces are shown whole. Larger ones show the first lines of each file, which is enough to write
    the command that computes the answer.
    """
    workdir = Path(workdir)
    paths = sorted(path for path in workdir.rglob("*")
                   if path.is_file() and path.name != "answer.txt" and "__pycache__" not in path.parts)
    if not paths:
        return "none"
    files = [(path.relative_to(workdir).as_posix(), path.stat().st_size, _read_text(path)) for path in paths]

    whole = "\n".join(f"== {name} ==\n{text.rstrip()}" if text is not None else f"== {name} == ({size} bytes, binary)"
                      for name, size, text in files)
    if len(whole) <= config.CONTEXT_MAX_CHARS:
        return whole

    parts = []
    for name, size, text in files[:config.CONTEXT_MAX_FILES]:
        if text is None:
            parts.append(f"== {name} == ({size} bytes, binary)")
            continue
        lines = text.splitlines()
        head = "\n".join(line[:200] for line in lines[:config.CONTEXT_HEAD_LINES])
        parts.append(f"== {name} == ({len(lines)} lines, first {min(len(lines), config.CONTEXT_HEAD_LINES)} shown)\n{head}")
    rest = files[config.CONTEXT_MAX_FILES:]
    if rest:
        parts.append(truncate(f"and {len(rest)} more files: " + ", ".join(name for name, _, _ in rest), 600))
    return "\n".join(parts)


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

    def __init__(self, attempt, ledger, cost_allowance, first_input_chars, max_tokens):
        super().__init__()
        self.attempt = attempt
        self.ledger = ledger
        self.allowance = cost_allowance
        self.max_tokens = max_tokens
        self.next_input_tokens = first_input_chars / 2.5 + 500  # the chat template adds the tool instructions
        self.reservation = 0.0

    def worst_case_cost(self):
        """Upper bound of the next call: its input, a full-length output and a margin for the input guess."""
        return self.next_input_tokens / 10 + self.max_tokens + config.CALL_RESERVE_MARGIN

    def settle(self):
        """Give back the reservation of a call that has ended, whatever its outcome."""
        self.ledger.release(self.reservation)
        self.reservation = 0.0

    async def before_model_call(self, ctx: AgentCallbackContext):
        self.settle()
        stop = None
        if self.attempt.cost >= self.allowance:
            stop = "task cap reached"
        elif self.ledger.proxy_down:
            stop = "proxy stopped answering"
        else:
            amount = self.worst_case_cost()
            if await self.ledger.reserve(amount):
                self.reservation = amount
            else:
                stop = f"a call of up to {amount:.0f} no longer fits the budget"
        if stop:
            log.warning("%s: stopping, %s (task %.0f, run %.0f/%.0f)", self.attempt.task_id, stop,
                        self.attempt.cost, self.ledger.spent, self.ledger.budget)
            ctx.request_force_finish({"output": "", "result_type": "stopped"})

    async def after_model_call(self, ctx: AgentCallbackContext):
        if not isinstance(ctx.inputs, ModelCallInputs):
            return
        self.settle()
        if ctx.inputs.response is None:  # the call failed; on_model_exception records that
            return
        usage = getattr(ctx.inputs.response, "usage_metadata", None)
        if usage is None:
            log.warning("%s: response without usage data, spending is undercounted", self.attempt.task_id)
            return
        cost = call_cost(usage)
        # The next call resends this one's input and output plus one tool result.
        self.next_input_tokens = ((usage.input_tokens or 0) + (usage.output_tokens or 0)
                                  + MAX_TOOL_OUTPUT_CHARS / 2.5 + 100)
        self.ledger.charge(cost)
        self.attempt.cost += cost
        self.attempt.calls += 1
        log.info("%s [%s] call %d: %d in, %d out -> %.0f (task %.0f, run %.0f/%.0f)", self.attempt.task_id,
                 self.attempt.model, self.attempt.calls, usage.input_tokens, usage.output_tokens, cost,
                 self.attempt.cost, self.ledger.spent, self.ledger.budget)

    async def on_model_exception(self, ctx: AgentCallbackContext):
        self.settle()
        self.attempt.llm_error = True
        self.ledger.consecutive_llm_errors += 1

    async def after_tool_call(self, ctx: AgentCallbackContext):
        # Ending right after an accepted answer skips a closing model call that would resend the whole context.
        if isinstance(ctx.inputs, ToolCallInputs) and (self.attempt.submitted or self.attempt.gave_up):
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


def max_tokens_for(task, has_tests):
    """Output limit of one call. A coding task rewrites a file in one call, on top of the model's reasoning."""
    task_type = task.task_type or ("coding" if has_tests else "")
    return config.MAX_TOKENS_BY_TASK_TYPE.get(task_type, config.SOLVER_MAX_TOKENS)


def build_model(base_url, api_key, model_name, max_tokens):
    api_name, extra_body = models.request_model(model_name)
    request = {"model": api_name, "max_tokens": max_tokens}
    if extra_body:
        request["extra_body"] = extra_body
    return PatientModel(
        model_client_config=ModelClientConfig(
            client_provider="OpenAI",
            api_key=api_key,
            api_base=base_url,
            timeout=config.TASK_TIMEOUT_SECONDS,
            stream_first_chunk_timeout=config.TASK_TIMEOUT_SECONDS,
        ),
        model_config=ModelRequestConfig(**request),
    )


async def solve_task(task, model_name, workdir, guard, store, ledger, base_url, api_key, cost_allowance):
    """One attempt at one task. The answer is saved by `store` as soon as it is accepted."""
    attempt = Attempt(task_id=task.task_id, model=model_name)

    async def accept(answer, from_command=False):
        answer = normalize_answer(answer)
        if not answer:
            return "Not accepted: the answer is empty."
        if is_refusal(answer):
            if from_command:  # a command that found nothing; the agent decides what that means
                return (f"Not taken as the answer: {answer[:100]}\n"
                        "Submit the value, or give_up if the data does not contain it.")
            # "The problem has missing information" sent as the answer: a give_up in other words.
            attempt.gave_up = True
            log.info("%s: answer is a refusal, taken as giving up: %s", task.task_id, answer[:100])
            return "Ended."
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

    answer_file = Path(workdir) / "answer.txt"
    try:
        answer_file.unlink(missing_ok=True)  # one left by an earlier run would be taken for this run's answer
    except OSError:
        pass

    def take_written_answer():
        """The content of an answer.txt the agent wrote itself; the file is removed, accept() saves the real one."""
        try:
            text = answer_file.read_text(encoding="utf-8", errors="replace").strip()
            answer_file.unlink()
        except OSError:
            return ""
        return text if len(text) <= config.FINAL_ANSWER_MAX_CHARS and "\n" not in text else ""

    @tool(name="bash", description="Run a bash command in the task's directory and return its output.",
          input_params={"type": "object", "required": ["command"], "additionalProperties": False, "properties": {
              "command": {"type": "string"},
              "final": {"type": "boolean", "description": "true: the command's output is the final answer"},
          }})
    async def bash(command: str, final: bool = False) -> str:
        log.info("%s: bash%s %s", task.task_id, " (final)" if final else "", " ".join(str(command).split())[:150])
        if "task.json" in command:
            return "Refused: task.json is not part of the task data."
        code, output = await run_command([shell.find_bash(), "-c", shell.PYTHON3_PREAMBLE + command], workdir,
                                         config.BASH_TIMEOUT_SECONDS)
        # The task text asks for answer.txt, so the command may have written the answer there, whatever it printed.
        written = take_written_answer()
        if written:
            verdict = await accept(written, from_command=True)
            return verdict if attempt.submitted else f"{truncate(output)}\n{verdict}"
        if code != 0:
            return f"exit code {code}\n{truncate(output)}"
        if not final:
            return truncate(output)
        answer = output.strip()
        if not answer or len(answer) > config.FINAL_ANSWER_MAX_CHARS or "\n" in answer:
            shown = truncate(output, 300) if answer else "(no output)"
            return (f"Not taken as the answer, it must be one short line:\n{shown}\n"
                    "Submit the value, or give_up if the data does not contain it.")
        return await accept(answer, from_command=True)

    @tool(name="submit", description="Record the final answer and end the task.",
          input_params={"type": "object", "required": ["answer"], "additionalProperties": False,
                        "properties": {"answer": {"type": "string"}}})
    async def submit(answer: str) -> str:
        return await accept(answer)

    @tool(name="give_up", description="End the task as unsolvable.",
          input_params={"type": "object", "required": ["reason"], "additionalProperties": False,
                        "properties": {"reason": {"type": "string", "description": "a few words"}}})
    async def give_up(reason: str = "") -> str:
        attempt.gave_up = True
        log.info("%s: agent gave up: %s", task.task_id, " ".join(str(reason).split())[:120])
        return "Ended."

    # Everything the agent would otherwise spend its first calls on: the files, and for coding tasks the tests' output.
    query = f"{task.instruction}\n\nFiles:\n{workspace_context(workdir)}"
    absent = precheck.absent_terms(task.instruction, workdir)
    if absent:
        query += "\n\nNote: found in no file: " + ", ".join(absent)
    if guard.originals:
        failures = await guard.failures()
        query += f"\n\nOutput of the tests now:\n{failures if failures is not None else 'all tests pass'}"

    max_tokens = max_tokens_for(task, bool(guard.originals))
    meter = MeterRail(attempt, ledger, cost_allowance, len(SYSTEM_PROMPT) + len(query), max_tokens)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as agent_home:
        agent = create_deep_agent(
            build_model(base_url, api_key, model_name, max_tokens),
            card=AgentCard(id=f"{task.task_id}-{uuid.uuid4().hex[:8]}", name="task_solver"),
            workspace=agent_home,  # DeepAgent keeps its own files here, away from the task folder
            tools=[bash, submit, give_up],
            rails=[meter],
            system_prompt=SYSTEM_PROMPT,
            max_iterations=config.SOLVER_MAX_ITERATIONS,
            parallel_tool_calls=False,
            prompt_mode="none",
            # These defaults add a paid image-support probe, prompt text and unmetered retries.
            enable_read_image_multimodal=False,
            enable_security_rail=False,
            enable_model_anomaly_detection_rail=False,
        )
        inputs = {"query": query, "conversation_id": f"{task.task_id}-{uuid.uuid4().hex[:8]}"}
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
        finally:
            meter.settle()  # a timeout or an error can end the attempt in the middle of a call

    # Written by a command that was cut off before its result came back, for example by a timeout.
    if not attempt.submitted and not attempt.gave_up:
        written = take_written_answer()
        if written:
            await accept(written, from_command=True)
    return attempt
