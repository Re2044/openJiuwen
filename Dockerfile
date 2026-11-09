FROM python:3.13.2-slim-bookworm

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY run_agent.py .
COPY harness ./harness

# Run unprivileged. All capabilities are dropped at run time, so root would gain nothing here.
RUN useradd --uid 1000 --no-create-home --shell /usr/sbin/nologin agent
USER 1000

ENTRYPOINT ["python", "/app/run_agent.py"]
