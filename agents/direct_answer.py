"""Example without an agent loop: one model call per task, and a plan() that chooses tasks.

Shows the other building blocks: env.ask, env.bash, env.submit. Without tools it
cannot fix code, so plan() leaves coding tasks out instead of paying for them.
"""

MODEL = "qwen3.8-27b-noreason"


def plan(tasks):
    """Choose and order tasks. Anything left out is reported as not_run and costs nothing."""
    chosen = [task for task in tasks if task.task_type != "coding"]
    return sorted(chosen, key=lambda task: -task.reward)


async def solve(env):
    # Commands cost no tokens, so facts can be gathered before asking the model.
    files = await env.bash("ls")
    answer = await env.ask(
        f"{env.instruction}\nFiles here: {files}\nReply with only the final answer.",
        model=MODEL,
        max_tokens=200,
    )
    await env.submit(answer)
