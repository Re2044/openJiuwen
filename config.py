JUDGE_MODEL_NAME="qwen3.6-35b-a3b-noreason"
MODEL_NAME_LIST=["qwen3.8-27b","qwen3.8-27b-noreason","qwen3.6-35b-a3b","qwen3.6-35b-a3b-noreason"]
TIME_LIMIT_IN_SECONDS=3600

# The judge is asked once per task for the number of model calls (steps) the solving agent needs; tokens, cost and
# time per model then follow from the per-call costs measured in earlier runs (see estimate_from_verdict).
JUDGE_SYSTEM = (
    "You predict how many model calls an AI agent needs to finish a task.\n"
    "The agent is a small LLM with two tools: bash, which runs one shell command in the task folder and returns at "
    "most 2000 characters of output, and submit, which records the final answer. Every call uses one tool; the "
    "agent may make at most 8 calls. It sees the task files only by listing or reading them with bash, and it "
    "can run python3 there.\n"
    "Typical counts from past runs: an arithmetic word problem computed with one python3 command takes 1-2 calls; "
    "a question about a CSV file 2-3 (look at the header, compute, answer); a lookup in a folder of markdown "
    "files 3-5, more for questions that chain several records or for large folders; fixing bugs in a small "
    "Python module until its tests pass 3-5, up to 8 for several subtle bugs. Writing the answer usually happens "
    "in the last command.\n"
    "Judge by the actual work, not by the wording: long or technical descriptions of simple work are still simple.\n"
    "You may also get a summary of the task's files made by a script: file counts and sizes, CSV columns with their "
    "values, Python functions and how many visible tests fail, the text of very small files, and files the "
    "instruction names that are missing. Use it: many files to search, records that point to other records, "
    "messy data (repeated rows, names that differ in case or spaces) and more failing tests all mean more steps. "
    "When the data the task needs is missing, unreadable, withheld or redacted, or the files only point to each "
    "other without the needed value, the task cannot be done and solvable_probability must be low.\n"
    "Return: difficulty (trivial, easy, medium or hard); steps, a whole number from 1 to 8; "
    "solvable_probability from 0 to 1, low only when the instruction itself lacks information or contradicts "
    "itself; needs_reasoning_model, true only when a model that does not reason step by step would likely fail."
)
# Input tokens of the first call and growth per further call, by task type (each call resends the history).
STEP_INPUT_TOKENS = {"math": (620, 60), "data_analysis": (620, 200), "search": (620, 230), "coding": (620, 250)}
DEFAULT_STEP_INPUT_TOKENS = (620, 220)
# Mean output tokens per call (reasoning included), by model.
OUTPUT_TOKENS_PER_CALL = {"qwen3.8-27b": 110, "qwen3.8-27b-noreason": 52, "qwen3.6-35b-a3b": 219,
                          "qwen3.6-35b-a3b-noreason": 67}
DEFAULT_OUTPUT_TOKENS_PER_CALL = 150
SECONDS_PER_CALL = 7  # model call plus the bash command it runs
# Steps kept within (min, max) per task type, and the default when the judge gives no usable answer.
STEP_LIMITS = {"math": (1, 4, 2), "data_analysis": (2, 6, 3), "search": (3, 8, 5), "coding": (2, 8, 4)}
DEFAULT_STEP_LIMITS = (1, 8, 4)
MIN_SOLVABLE_PROBABILITY = 0.1  # below this the task is left out of the plan
# The judge also gets a script-made summary of the task's files (workspace_summary.py); no model call is spent on it.
JUDGE_WORKSPACE_SUMMARY = True
WORKSPACE_SUMMARY_MAX_CHARS = 1800  # about 450 input tokens, so about 45 cost units at most
JUDGE_RUN_TESTS = True  # run the visible tests on a copy of the workspace to report how many fail
JUDGE_TEST_TIMEOUT_SECONDS = 20
# The summary's warnings (missing or withheld data) are also added to the solving agent's first message, so it can
# check them and give up early. SOLVER_WARNINGS=0 in the environment turns this off.
SOLVER_WARNINGS = True

# Solving the planned tasks
PARALLEL_SESSIONS=5  # the rules allow up to 5 agent sessions at once; PARALLEL_SESSIONS in .env overrides it
# Tried in this order when the planned model gives no accepted answer; models the key cannot use are skipped.
FALLBACK_MODELS=["qwen3.6-35b-a3b", "qwen3.8-27b", "qwen3.6-35b-a3b-noreason", "qwen3.8-27b-noreason"]
MAX_MODELS_PER_TASK=2  # the planned model plus one fallback
# Fallback models are tried after model call errors. Also trying them when the planned model simply found no answer
# cost about 23k per 100 tasks in test runs and rescued only unsolvable tasks, so it is off.
FALLBACK_ON_NO_ANSWER=False
ACCEPT_ANSWER_FILE=True  # take answer.txt written by the agent as its answer at once, saving a submit call
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
