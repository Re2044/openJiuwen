#!/usr/bin/env bash
# Runs the harness without Docker on a throwaway copy of the example tasks, then scores it.
# Needs LLM_BASE_URL and LLM_API_KEY. Usage: scripts/run_local.sh [tasks dir] (default: task-dir)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [ -x "$ROOT/.venv/Scripts/python.exe" ]; then PY="$ROOT/.venv/Scripts/python.exe"; else PY="$ROOT/.venv/bin/python"; fi
WORK="$(mktemp -d)"
if command -v cygpath >/dev/null; then  # Git Bash: Windows Python needs C:/ style paths
  ROOT="$(cygpath -m "$ROOT")"
  WORK="$(cygpath -m "$WORK")"
fi
cp -r "${1:-$ROOT/task-dir}" "$WORK/tasks"

HARNESS_WORK_DIR="$WORK/work" PYTHONIOENCODING=utf-8 \
  "$PY" "$ROOT/run_agent.py" --tasks-dir "$WORK/tasks" --out "$WORK/out"

"$PY" "$ROOT/scripts/evaluate.py" "$WORK/tasks"
echo "trajectories: $WORK/work/trajectories.jsonl"
