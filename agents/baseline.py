"""Baseline architecture: one openJiuwen agent with the bash and submit tools.

Copy this file to try your own idea, then run:
    python -m bench --agent agents/my_agent.py --only math_0000 --parallel 1
"""
from harness.agent import SYSTEM_PROMPT

MODEL = "qwen3.8-27b-noreason"


async def solve(env):
    agent = env.make_agent(model=MODEL, system_prompt=SYSTEM_PROMPT, max_iterations=8)
    await env.run_agent(agent)
