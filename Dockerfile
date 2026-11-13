FROM python:3.12.10-slim-bookworm

# The container runs with a read-only file system: nothing may be written next to the code.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt

COPY *.py ./

ENTRYPOINT ["python", "/app/run_agent.py"]
CMD ["--tasks-dir", "/data/tasks"]
