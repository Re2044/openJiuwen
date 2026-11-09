JUDGE_MODEL_NAME="qwen3.6-35b-a3b-noreason"
MODEL_NAME_LIST=["qwen3.8-27b","qwen3.8-27b-noreason","qwen3.6-35b-a3b","qwen3.6-35b-a3b-noreason"]
TIME_LIMIT_IN_SECONDS=1000

SYSTEM = (
    "You estimate how a given LLM would perform on a task. "
    "Given a task description and a model name, return:"
    "The expected total token usage, and the expected wall-clock time in seconds. "
    "If the task with given tools cannot be completed return (-1,-1)"
    "Give your best single-number estimates; never return ranges."
)