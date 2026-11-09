"""End-to-end runs of run_agent.py against the stub proxy, on a copy of the example tasks.

Run with: python -m unittest discover -s tests -v
"""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "tests"), str(ROOT / "scripts")]
os.chdir(tempfile.gettempdir())  # importing openJiuwen below creates ./logs in the cwd

from evaluate import evaluate  # noqa: E402
from harness.agent import GuardedTests  # noqa: E402
from harness.runner import plan  # noqa: E402
from harness.tasks import load_tasks  # noqa: E402
from stub_llm import StubLLM  # noqa: E402

EXAMPLES = Path(os.getenv("EXAMPLE_TASKS", ROOT / "task-dir"))
TASK_COUNT = len(list(EXAMPLES.glob("*/task.json")))


class HarnessRun(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="harness-test-"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.tasks = self.tmp / "tasks"
        shutil.copytree(EXAMPLES, self.tasks, ignore=shutil.ignore_patterns(".DS_Store", "__pycache__"))
        self.out = self.tmp / "out"
        self.out.mkdir()
        self.cwd = self.tmp / "cwd"
        self.cwd.mkdir()

    def run_harness(self, stub: StubLLM, timeout: float = 120, **env) -> subprocess.CompletedProcess:
        self.addCleanup(stub.stop)
        environment = {
            **os.environ,
            "LLM_BASE_URL": stub.start(),
            "LLM_API_KEY": "stub-key",
            "OUT": str(self.out),
            "HARNESS_WORK_DIR": str(self.tmp / "work"),
            "PYTHONIOENCODING": "utf-8",
            **env,
        }
        return subprocess.run(
            [sys.executable, str(ROOT / "run_agent.py"), "--tasks-dir", str(self.tasks)],
            env=environment, cwd=self.cwd, capture_output=True, text=True, encoding="utf-8", errors="replace",
            timeout=timeout,
        )

    def trajectory(self) -> list[dict]:
        path = self.tmp / "work" / "trajectories.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def test_solves_every_example_within_limits(self):
        stub = StubLLM(self.tasks, latency=0.2)
        proc = self.run_harness(stub)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(stub.errors, [])

        report = evaluate(self.tasks)
        self.assertEqual(report["failed"], [], proc.stderr[-2000:])
        solutions = json.loads((self.out / "solutions.json").read_text(encoding="utf-8"))
        self.assertEqual(len(solutions), TASK_COUNT)

        # Answers exist both where the rules and where the instructions put them.
        self.assertTrue((self.tasks / "search_0001" / "answer.txt").exists())
        self.assertTrue((self.tasks / "search_0001" / "workspace" / "answer.txt").exists())

        # Parallel, but within the session limit.
        self.assertGreater(stub.max_in_flight, 1)
        self.assertLessEqual(stub.max_in_flight, 5)

        # Reference answers and the proxy key never reach the model; no stock tools or image probe are sent.
        for body in stub.requests:
            text = json.dumps(body["messages"])
            self.assertNotIn('"expected"', text)
            self.assertEqual({t["function"]["name"] for t in body["tools"]}, {"bash", "submit"})
            self.assertEqual(body.get("max_tokens"), 2000)

        # Submit ends the attempt: no closing call after it.
        by_task: dict[str, int] = {}
        for event in self.trajectory():
            if event["event"] == "call":
                by_task[event["task_id"]] = by_task.get(event["task_id"], 0) + 1
        self.assertEqual(by_task["math_0000"], 2)
        self.assertEqual(by_task["coding_0003"], 4)

        # The harness's own count matches the proxy's.
        spent = sum(e["cost"] for e in self.trajectory() if e["event"] == "call")
        self.assertAlmostEqual(spent, stub.spent, delta=stub.spent * 0.02)

        # Nothing is written to the cwd, which is read-only in the container.
        self.assertEqual(list(self.cwd.iterdir()), [])

    def test_higher_reward_per_token_runs_first(self):
        order = [task.task_id for task in plan(load_tasks(self.tasks))]
        self.assertEqual(order[0], "math_0000")  # reward 5 at the lowest estimated cost
        self.assertEqual(order[-2:], ["coding_0003", "coding_0007"])  # reward 2 at the highest; ties go by task id
        self.assertEqual(len(order), TASK_COUNT)

    def test_budget_cutoff_keeps_saved_answers_and_exits(self):
        stub = StubLLM(self.tasks, budget=800, on_exhausted="429", latency=0.05)
        started = time.time()
        proc = self.run_harness(stub)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertLess(time.time() - started, 90)

        report = evaluate(self.tasks)
        self.assertGreater(len(report["solved"]), 0, proc.stderr[-2000:])
        self.assertGreater(len(report["failed"]), 0)
        solutions = json.loads((self.out / "solutions.json").read_text(encoding="utf-8"))
        self.assertEqual(set(solutions), set(report["solved"]))
        self.assertIn("'proxy_down': True", proc.stderr)

    def test_proxy_that_stops_answering_does_not_hang_the_run(self):
        stub = StubLLM(self.tasks, budget=400, on_exhausted="hang", latency=0.05)
        proc = self.run_harness(stub, timeout=150, HARNESS_LLM_TIMEOUT_SEC="2")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertGreater(len(evaluate(self.tasks)["solved"]), 0, proc.stderr[-2000:])

    def test_falls_back_to_next_model(self):
        stub = StubLLM(self.tasks, refuse_models={"qwen3.8-27b-noreason"})
        proc = self.run_harness(stub)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(evaluate(self.tasks)["failed"], [], proc.stderr[-2000:])
        submitted = [e["model"] for e in self.trajectory() if e["event"] == "task" and e["submitted"]]
        self.assertEqual(set(submitted), {"qwen3.8-27b"})

    def test_model_that_never_submits_is_not_a_proxy_failure(self):
        stub = StubLLM(self.tasks, loop_models={"qwen3.8-27b-noreason"})
        proc = self.run_harness(stub)
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("'proxy_down': False", proc.stderr)
        self.assertEqual(evaluate(self.tasks)["failed"], [], proc.stderr[-2000:])

        # The looping model is stopped at the iteration limit and not retried.
        looping = [e for e in self.trajectory() if e["event"] == "call" and e["model"] == "qwen3.8-27b-noreason"]
        per_task: dict[str, int] = {}
        for event in looping:
            per_task[event["task_id"]] = per_task.get(event["task_id"], 0) + 1
        self.assertLessEqual(max(per_task.values()), 8)

    def test_edited_tests_are_restored(self):
        workdir = self.tasks / "coding_0003" / "workspace"
        guard = GuardedTests(workdir)
        test_file = workdir / "test_mylib.py"
        original = test_file.read_bytes()
        test_file.write_text("import sys; sys.exit(0)\n", encoding="utf-8")
        failures = asyncio.run(guard.failures(60))
        self.assertIsNotNone(failures)  # the buggy code still fails against the restored tests
        self.assertEqual(test_file.read_bytes(), original)

    def test_known_budget_stops_before_overspending(self):
        budget = 300
        stub = StubLLM(self.tasks)
        proc = self.run_harness(stub, TOKEN_BUDGET=str(budget))
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertGreater(len(evaluate(self.tasks)["failed"]), 0)
        # No new call starts after the limit; up to five sessions may already have one in flight.
        call_costs = [e["cost"] for e in self.trajectory() if e["event"] == "call"]
        self.assertLessEqual(stub.spent, budget + 5 * max(call_costs))

    def test_contacts_only_the_proxy(self):
        stub = StubLLM(self.tasks)
        self.addCleanup(stub.stop)
        hosts_log = self.tmp / "hosts.txt"
        # A hostname, because IP literals skip resolution: seeing it logged proves the recording works.
        base_url = stub.start().replace("127.0.0.1", "localhost")
        env = {**os.environ, "LLM_BASE_URL": base_url, "LLM_API_KEY": "stub-key",
               "HARNESS_WORK_DIR": str(self.tmp / "work"), "PYTHONIOENCODING": "utf-8"}
        proc = subprocess.run(
            [sys.executable, str(ROOT / "tests" / "record_hosts.py"), str(hosts_log), str(ROOT / "run_agent.py"),
             "--tasks-dir", str(self.tasks)],
            env=env, cwd=self.cwd, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertEqual(evaluate(self.tasks)["failed"], [], proc.stderr[-2000:])
        hosts = set(hosts_log.read_text(encoding="utf-8").split())
        self.assertEqual(hosts, {"localhost"})

    def test_missing_settings_fail_fast(self):
        env = {k: v for k, v in os.environ.items() if k not in ("LLM_BASE_URL", "LLM_API_KEY")}
        proc = subprocess.run([sys.executable, str(ROOT / "run_agent.py"), "--tasks-dir", str(self.tasks)],
                              env=env, cwd=self.cwd, capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 2)
        self.assertIn("LLM_BASE_URL", proc.stderr)


if __name__ == "__main__":
    unittest.main()
