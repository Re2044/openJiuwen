JUDGE_MODEL_NAME="qwen3.6-35b-a3b-noreason"
MODEL_NAME_LIST=["qwen3.8-27b","qwen3.8-27b-noreason","qwen3.6-35b-a3b","qwen3.6-35b-a3b-noreason"]

SYSTEM = (
    "You estimate how a given LLM would perform on a task. "
    "Given a task description and a model name, return: the model name, "
    "the probability (0-100) that it completes the task successfully, "
    "the expected total token usage, and the expected wall-clock time in seconds. "
    "Give your best single-number estimates; never return ranges."
)