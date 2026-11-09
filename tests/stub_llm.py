"""OpenAI-compatible stand-in for the organizers' proxy, for tests and rehearsals. Uses no real model.

It follows a fixed script per task, reads reference answers from task.json, and
can cut the run off at a token budget the way the real proxy is described to.
"""
from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ALLOWED_MODELS = {"qwen3.8-27b", "qwen3.8-27b-noreason", "qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason"}
FIX_MYLIB = (
    "sed -i -e 's/sorted(s.split())/reversed(s.split())/' -e 's/s -= x/s += x/' "
    "-e 's/x < best/x > best/' -e 's/% 2 == 1/% 2 == 0/' mylib.py"
)


class StubLLM:
    def __init__(self, tasks_dir: Path, budget: float | None = None, on_exhausted: str = "429",
                 refuse_models: set[str] | None = None, loop_models: set[str] | None = None,
                 latency: float = 0.0) -> None:
        self.by_instruction: dict[str, list[dict]] = {}
        for path in sorted(tasks_dir.glob("*/task.json")):
            meta = json.loads(path.read_text(encoding="utf-8"))
            self.by_instruction.setdefault(meta["instruction"], []).append(meta)
        self.budget = budget
        self.on_exhausted = on_exhausted
        self.refuse_models = refuse_models or set()
        self.loop_models = loop_models or set()
        self.latency = latency
        self.lock = threading.Lock()
        self.spent = 0.0
        self.requests: list[dict] = []
        self.errors: list[str] = []
        self.in_flight = 0
        self.max_in_flight = 0
        self._server: ThreadingHTTPServer | None = None

    def start(self, port: int = 0, host: str = "127.0.0.1") -> str:
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                stub.handle(self, body)

            def do_GET(self):
                self.send_json(200, {"spent": stub.spent, "requests": len(stub.requests), "errors": stub.errors,
                                     "max_in_flight": stub.max_in_flight})

            def send_json(self, status: int, payload: dict) -> None:
                data = json.dumps(payload).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer((host, port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return f"http://{host}:{self._server.server_port}/v1"

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()
            self._server.server_close()

    def handle(self, handler, body: dict) -> None:
        with self.lock:
            self.requests.append(body)
            exhausted = self.budget is not None and self.spent >= self.budget
            self.in_flight += 1
            self.max_in_flight = max(self.max_in_flight, self.in_flight)
        try:
            if body.get("model") not in ALLOWED_MODELS:
                handler.send_json(400, {"error": {"message": f"unknown model {body.get('model')}"}})
                return
            if exhausted:
                if self.on_exhausted == "hang":
                    time.sleep(3600)
                handler.send_json(429, {"error": {"message": "token budget exhausted"}})
                return
            time.sleep(self.latency)
            message = self.reply(body)
            handler.send_json(200, self.completion(body, message))
        finally:
            with self.lock:
                self.in_flight -= 1

    def completion(self, body: dict, message: dict) -> dict:
        chars = sum(len(json.dumps(m, ensure_ascii=False)) for m in body["messages"])
        prompt = (chars + len(json.dumps(body.get("tools") or []))) // 4
        reasoning = 0 if body["model"].endswith("-noreason") else 50
        completion = 30 + reasoning
        with self.lock:
            self.spent += prompt / 10 + completion
        finish = "tool_calls" if message.get("tool_calls") else "stop"
        return {
            "id": "stub", "object": "chat.completion", "created": 0, "model": body["model"],
            "choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "usage": {"prompt_tokens": prompt, "completion_tokens": completion,
                      "total_tokens": prompt + completion,
                      "completion_tokens_details": {"reasoning_tokens": reasoning}},
        }

    def reply(self, body: dict) -> dict:
        if body["model"] in self.refuse_models:
            return {"role": "assistant", "content": "I cannot solve this."}
        if body["model"] in self.loop_models:
            return self.call("bash", command="pwd")
        messages = body["messages"]
        instruction = next(m["content"] for m in messages if m["role"] == "user")
        candidates = self.by_instruction.get(instruction) or next(
            (metas for known, metas in self.by_instruction.items() if known in instruction), None)
        if candidates is None:
            self.errors.append(f"unknown or mixed-up instruction: {str(instruction)[:80]}")
            return {"role": "assistant", "content": "unknown task"}
        if candidates[0]["check"] == "impossible":
            return {"role": "assistant", "content": "This task cannot be solved."}
        results = [str(m.get("content") or "") for m in messages if m["role"] == "tool"]
        if not body.get("tools"):  # a direct call without tools gets the answer as text
            return {"role": "assistant", "content": str(candidates[0].get("expected", "DONE"))}
        if not results:
            return self.call("bash", command='python3 -c "import os; print(os.getcwd())"')

        # The first tool result is the working directory: it must belong to this task.
        meta = next((c for c in candidates if c["task_id"] in results[0]), None)
        if meta is None:
            self.errors.append(f"tool ran in a foreign directory: {results[0][:120]}")
            return {"role": "assistant", "content": "wrong directory"}
        if meta["check"] != "unittest":
            return self.call("submit", answer=str(meta["expected"]))
        # Submit before fixing to exercise the harness's rejection of failing code, then fix and resubmit.
        if len(results) == 1:
            return self.call("submit", answer="DONE")
        if len(results) == 2:
            if "Not accepted" not in results[1]:
                self.errors.append(f"failing code was accepted: {results[1][:120]}")
            return self.call("bash", command=FIX_MYLIB)
        return self.call("submit", answer="DONE")

    @staticmethod
    def call(name: str, **args) -> dict:
        return {"role": "assistant", "content": None, "tool_calls": [{
            "id": f"call_{time.time_ns()}", "type": "function",
            "function": {"name": name, "arguments": json.dumps(args)},
        }]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--budget", type=float)
    parser.add_argument("--latency", type=float, default=0.2)
    args = parser.parse_args()
    stub = StubLLM(args.tasks_dir, budget=args.budget, latency=args.latency)
    print("stub listening on", stub.start(args.port, "0.0.0.0"), flush=True)
    threading.Event().wait()
