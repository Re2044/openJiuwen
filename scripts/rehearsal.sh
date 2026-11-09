#!/usr/bin/env bash
# Runs the exact container command from the rules against the stub proxy, on an isolated network.
# Usage: scripts/rehearsal.sh [stub budget in tokens]
set -euo pipefail

export MSYS_NO_PATHCONV=1  # keep container paths intact under Git Bash on Windows
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PYTHON:-python}"
BUDGET="${1:-}"
TEAM=rehearsal
WORK="$(mktemp -d)"
if command -v cygpath >/dev/null; then  # Git Bash: docker and Windows Python need C:/ style paths
  ROOT="$(cygpath -m "$ROOT")"
  WORK="$(cygpath -m "$WORK")"
fi
TASKS="$WORK/tasks"
OUT="$WORK/out"
cp -r "$ROOT/task-dir" "$TASKS"
chmod -R a+rwX "$TASKS"

cleanup() {
  docker rm -f stub-llm >/dev/null 2>&1 || true
  rm -rf "$WORK" 2>/dev/null || true
}
trap cleanup EXIT

docker build -t hack:latest "$ROOT"
docker images --format '{{.Repository}}:{{.Tag}} -> {{.Size}}' hack:latest

# --internal gives the containers no route to the internet, like the real run.
docker network inspect hack-net >/dev/null 2>&1 || docker network create --internal hack-net >/dev/null

docker rm -f stub-llm >/dev/null 2>&1 || true
docker run -d --name stub-llm --network hack-net --network-alias stub \
  -v "$ROOT/tests:/stub:ro" -v "$ROOT/task-dir:/reference:ro" \
  --entrypoint python hack:latest /stub/stub_llm.py --tasks-dir /reference ${BUDGET:+--budget "$BUDGET"} >/dev/null
sleep 2

export LLM_BASE_URL=http://stub:8000/v1 LLM_API_KEY=rehearsal-key

mkdir -p "$OUT" && chmod 777 "$OUT"
started=$SECONDS
timeout -k 1000s 1h docker run --rm --name "run-${TEAM}" \
  --network hack-net \
  -e LLM_BASE_URL -e LLM_API_KEY \
  -v "$TASKS:/data/tasks:rw" \
  --read-only --tmpfs /tmp:rw,size=2g -e HOME=/tmp \
  --cap-drop ALL --security-opt no-new-privileges \
  --cpus 8 --memory 16g --pids-limit 512 \
  --workdir /app --entrypoint python \
  "hack:latest" \
  /app/run_agent.py --tasks-dir /data/tasks
echo "container exit: $?, took $((SECONDS - started))s"

echo "--- score"
"$PY" "$ROOT/scripts/evaluate.py" "$TASKS"
echo "--- stub stats"
docker logs stub-llm 2>&1 | tail -n 3
docker run --rm --network hack-net --entrypoint python hack:latest -c \
  "import urllib.request; print(urllib.request.urlopen('http://stub:8000/stats').read().decode())"
