JUDGE_MODEL_NAME="qwen3.6-35b-a3b-noreason"
MODEL_NAME_LIST=["qwen3.8-27b","qwen3.8-27b-noreason","qwen3.6-35b-a3b","qwen3.6-35b-a3b-noreason"]
TIME_LIMIT_IN_SECONDS=3600
DEFAULT_BUDGET=1000000  # used when BUDGET is not set; too high is safer than too low, see README

# Planning
# Checks made before planning, without model calls (precheck.py). A task they find unsolvable is not planned.
# On a generated 100-task set they found 9 of 20 unsolvable tasks and flagged none of the 80 solvable ones.
PRECHECK_CONTRADICTORY_TESTS=True  # two tests demand different results from the same call
PRECHECK_MISSING_INPUTS=True  # a file or folder the task tells to read is not there
RUN_SAFETY_MARGIN_SECONDS=300  # kept free at the end of the hour: the container is killed at the limit
WORKSPACE_SUMMARY_MAX_FILES=20  # files of the task's workspace shown to the judge
# With USE_JUDGE only (these per-step costs were measured on the earlier agent and are stale).
# The judge only says whether a task is feasible and how many agent steps it takes; its guesses of the cost
# itself were 30-1000 times too low. Cost and time come from the steps and these per-step averages, in budget
# units (output tokens + input tokens / 10). Model names are looked up without a provider prefix like "qwen/".
COST_PER_STEP={
    "qwen3.8-27b": 190,  # measured on the 8 example tasks (27 calls)
    "qwen3.6-35b-a3b": 170,  # measured on the 8 example tasks (28 calls)
    "qwen3.8-27b-noreason": 150,  # not measured yet
    "qwen3.6-35b-a3b-noreason": 135,  # not measured yet
}
DEFAULT_COST_PER_STEP=190
SECONDS_PER_STEP=3
# The judge is off: in a 100-task run it cost 5313 units and caught 4 of 20 unsolvable tasks, because most of
# them show only in the files, which it does not read. The agent now sees the files in its first call and gives
# up there. Estimates come from this table instead, at no cost. USE_JUDGE=1 in .env turns the judge back on.
USE_JUDGE=False
# Average cost and time of a task by its type, in budget units (output tokens + input tokens / 10) and
# seconds. Measured on a 100-task run of the current agent with qwen3.6-35b-a3b, 25 tasks of each type, over
# all of them: solved, failed and unsolvable alike, because the plan cannot tell them apart beforehand.
# Replace with the averages of a newer run recorded with TRAJECTORIES whenever the agent changes.
COST_BY_TASK_TYPE={"math": 720, "data_analysis": 1000, "search": 1220, "coding": 1240}
SECONDS_BY_TASK_TYPE={"math": 7, "data_analysis": 8, "search": 8, "coding": 12}
DEFAULT_TASK_COST=1050  # the average over all four types, for a type that is not in the table
DEFAULT_TASK_SECONDS=9
# Cost of a model relative to qwen3.6-35b-a3b. Model names are looked up without a provider prefix like "qwen/".
MODEL_COST_FACTOR={
    "qwen3.6-35b-a3b": 1.0,
    "qwen3.8-27b": 1.1,  # from the per-step costs of the earlier agent (190 against 170)
    "qwen3.8-27b-noreason": 0.9,  # not measured yet
    "qwen3.6-35b-a3b-noreason": 0.8,  # not measured yet
}
DEFAULT_STEPS=3  # with USE_JUDGE only: when the judge gives no usable answer

SYSTEM = (
    "You plan work for an agent that solves small tasks. The agent works in steps: each step is one model call "
    "that either runs one bash command or submits the answer. It has bash with python3, the files listed under "
    "Workspace, and no internet access.\n"
    "First decide whether the task can be completed at all with these files and tools: set feasible to false "
    "when needed data is missing, the request contradicts itself, or it needs something the agent does not have.\n"
    "Then estimate the number of steps, counting the final submit. A pure calculation takes 2 or 3 steps; "
    "reading or searching files usually takes 3 to 5; fixing code until its tests pass usually takes 6 to 8. "
    "Give a single number, never a range."
)

# Solving the planned tasks
PARALLEL_SESSIONS=5  # the rules allow up to 5 agent sessions at once; PARALLEL_SESSIONS in .env overrides it
# Tried in this order when the planned model gives no accepted answer; models the key cannot use are skipped.
FALLBACK_MODELS=["qwen3.6-35b-a3b", "qwen3.8-27b", "qwen3.6-35b-a3b-noreason", "qwen3.8-27b-noreason"]
MAX_MODELS_PER_TASK=2  # the planned model plus one fallback
# After the plan, the budget that is left goes to tasks that ended without an answer (one more attempt, with
# another model when there is one) and to tasks the plan left out. Points decide the result; tokens only
# break ties.
SECOND_PASS=True
# The task's files go into the first message. Up to this many characters the workspace is shown whole;
# a larger one shows the first lines of up to CONTEXT_MAX_FILES files. Input costs a tenth of output, and a
# call saved is worth far more than the text resent.
CONTEXT_MAX_CHARS=6000
CONTEXT_MAX_FILES=20
CONTEXT_HEAD_LINES=4
FINAL_ANSWER_MAX_CHARS=200  # a longer output of a final command is not an answer
SOLVER_MAX_ITERATIONS=8  # agent steps per attempt
# Output tokens per model call. In a 100-task run the largest call of a correctly solved task used 1012;
# calls that hit the old limit of 2000 were all reasoning on unsolvable tasks.
SOLVER_MAX_TOKENS=1200
# A coding task needs more: in the first run of the one-call agent 10 of 20 solvable coding tasks were cut off
# at 1200 in the middle of rewriting the file, reasoning included.
MAX_TOKENS_BY_TASK_TYPE={"coding": 2500}
# Stop-loss per task, in competition units (output + input / 10). In the same run the dearest correctly solved
# task cost 1631, while unsolvable tasks ran up to the old cap of 6000.
TASK_COST_CAP=2000
TASK_COST_CAP_BY_TYPE={"coding": 3500}  # room for one long call and a correction after failing tests
# A second model is tried only when the first one gave up this cheaply. Of 14 second attempts in that run the
# only one that helped followed a first attempt of 241 units; the other 13 were unsolvable tasks.
FALLBACK_MAX_SPENT=400
# Added to the worst-case cost reserved for a call, to cover a low guess of its input tokens.
CALL_RESERVE_MARGIN=50
TASK_TIMEOUT_SECONDS=300
BASH_TIMEOUT_SECONDS=60
MAX_CONSECUTIVE_LLM_ERRORS=6  # this many failures in a row means the proxy has cut us off
LLM_ERROR_RETRIES=1  # repeats of an attempt that failed on a model call error

# Rate limits (HTTP 429): wait as long as the proxy asks, then retry the same call
RATE_LIMIT_MAX_WAITS=12  # waits per model call before giving up
RATE_LIMIT_DEFAULT_WAIT_SECONDS=5  # when the proxy does not say how long
RATE_LIMIT_MAX_WAIT_SECONDS=90
