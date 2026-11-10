JUDGE_MODEL_NAME="qwen3.6-35b-a3b-noreason"
MODEL_NAME_LIST=["qwen3.8-27b","qwen3.8-27b-noreason","qwen3.6-35b-a3b","qwen3.6-35b-a3b-noreason"]
TIME_LIMIT_IN_SECONDS=3600

SYSTEM = (
    "You estimate how a given LLM would perform on a task. "
    "Given a task description and a model name, return:"
    "The expected total token usage, and the expected wall-clock time in seconds. "
    "If the task with given tools cannot be completed return (-1,-1)"
    "Give your best single-number estimates; never return ranges."
)

# Solving the planned tasks
PARALLEL_SESSIONS=5  # the rules allow up to 5 agent sessions at once; PARALLEL_SESSIONS in .env overrides it
# Tried in this order when the planned model gives no accepted answer; models the key cannot use are skipped.
FALLBACK_MODELS=["qwen3.6-35b-a3b", "qwen3.8-27b", "qwen3.6-35b-a3b-noreason", "qwen3.8-27b-noreason"]
MAX_MODELS_PER_TASK=2  # the planned model plus one fallback
SOLVER_MAX_ITERATIONS=8  # agent steps per attempt
SOLVER_MAX_TOKENS=2000  # output tokens per model call
TASK_COST_CAP=6000  # stop-loss per task, in competition units (output + input / 10)
TASK_TIMEOUT_SECONDS=300
BASH_TIMEOUT_SECONDS=60
MAX_CONSECUTIVE_LLM_ERRORS=6  # this many failures in a row means the proxy has cut us off
LLM_ERROR_RETRIES=1  # repeats of an attempt that failed on a model call error

# Rate limits (HTTP 429): wait as long as the proxy asks, then retry the same call
RATE_LIMIT_MAX_WAITS=12  # waits per model call before giving up
RATE_LIMIT_DEFAULT_WAIT_SECONDS=5  # when the proxy does not say how long
RATE_LIMIT_MAX_WAIT_SECONDS=90
