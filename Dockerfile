# Harness for the competition: python /app/run_agent.py --tasks-dir /data/tasks (see README.md).
FROM python:3.12.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only the harness code; tasks are mounted at run time, the API key comes from the environment.
COPY run_agent.py main.py config.py data_access.py dp.py executor.py solver.py shell.py ratelimit.py \
     workspace_summary.py evaluate.py ./

ENTRYPOINT ["python", "/app/run_agent.py"]
