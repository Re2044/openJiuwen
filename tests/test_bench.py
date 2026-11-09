"""Runs of the test bench on the stub. Run with: python -m unittest tests.test_bench"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


class BenchRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="bench-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def bench(self, *args: str) -> tuple[subprocess.CompletedProcess, dict, Path]:
        env = {k: v for k, v in os.environ.items() if k not in ("LLM_BASE_URL", "LLM_API_KEY")}
        proc = subprocess.run(
            [sys.executable, "-m", "bench", "--stub", "--runs-dir", str(self.tmp / "runs"), *args],
            cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        run_dir = next((self.tmp / "runs").iterdir())
        return proc, json.loads((run_dir / "summary.json").read_text(encoding="utf-8")), run_dir

    def statuses(self, summary: dict) -> dict[str, str]:
        return {task["task_id"]: task["status"] for task in summary["tasks"]}

    def test_baseline_on_examples_and_own_tasks(self):
        before = (ROOT / "task-dir" / "coding_0003" / "workspace" / "mylib.py").read_bytes()
        proc, summary, run_dir = self.bench("--agent", "agents/baseline.py")

        statuses = self.statuses(summary)
        self.assertEqual(len(statuses), 10, proc.stdout)
        self.assertEqual(statuses.pop("my_impossible_0001"), "skipped_ok")
        self.assertEqual(set(statuses.values()), {"solved"}, proc.stdout)
        self.assertEqual(summary["points"], 22)  # 20 from the examples, 2 from my_math_0001
        self.assertGreater(summary["cost"], 0)

        # The originals stay untouched and the project root gets no logs.
        self.assertEqual((ROOT / "task-dir" / "coding_0003" / "workspace" / "mylib.py").read_bytes(), before)
        self.assertFalse((ROOT / "task-dir" / "math_0000" / "answer.txt").exists())
        self.assertFalse((ROOT / "logs").exists())

        # The transcript shows what the model saw and did, without reference answers.
        transcript = (run_dir / "transcripts" / "math_0000.jsonl").read_text(encoding="utf-8")
        self.assertIn("submit", transcript)
        self.assertIn("warehouse", transcript)
        self.assertNotIn('"expected"', transcript)

    def test_plan_hook_and_direct_calls(self):
        proc, summary, _ = self.bench("--agent", "agents/direct_answer.py", "--tasks", "task-dir")
        statuses = self.statuses(summary)
        self.assertEqual(statuses["coding_0003"], "not_run", proc.stdout)
        self.assertEqual(statuses["math_0000"], "solved", proc.stdout)
        self.assertEqual(statuses["search_0001"], "solved", proc.stdout)

    def test_filters(self):
        _, summary, _ = self.bench("--only", "math_0000,math_0004")
        self.assertEqual(set(self.statuses(summary)), {"math_0000", "math_0004"})
        shutil.rmtree(self.tmp / "runs")
        _, summary, _ = self.bench("--type", "search", "--tasks", "task-dir")
        self.assertEqual(set(self.statuses(summary)), {"search_0001", "search_0005"})

    def test_bug_in_an_agent_is_reported_per_task(self):
        agent = self.tmp / "broken.py"
        agent.write_text("async def solve(env):\n    raise ValueError('boom')\n", encoding="utf-8")
        proc, summary, run_dir = self.bench("--agent", str(agent), "--only", "math_0000,math_0004")
        self.assertEqual(set(self.statuses(summary).values()), {"error"}, proc.stdout)
        self.assertIn("boom", proc.stdout)
        self.assertIn("ValueError", (run_dir / "transcripts" / "math_0000.jsonl").read_text(encoding="utf-8"))

    def test_budget_limits_a_run(self):
        proc, summary, _ = self.bench("--budget", "100", "--parallel", "1", "--tasks", "task-dir")
        self.assertIn("not_run", summary["statuses"], proc.stdout)

    def test_refuses_to_start_without_a_model(self):
        env = {k: v for k, v in os.environ.items() if k not in ("LLM_BASE_URL", "LLM_API_KEY")}
        proc = subprocess.run([sys.executable, "-m", "bench"], cwd=ROOT, env=env, capture_output=True, text=True,
                              timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("--stub", proc.stderr)


if __name__ == "__main__":
    unittest.main()
