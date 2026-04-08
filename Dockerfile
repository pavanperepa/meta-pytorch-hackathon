ARG BASE_IMAGE=ghcr.io/meta-pytorch/openenv-base:latest
FROM ${BASE_IMAGE} AS builder

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

COPY . /app/env

WORKDIR /app/env

RUN python -m venv .venv && \
    ./.venv/bin/pip install --no-cache-dir --upgrade pip setuptools wheel && \
    ./.venv/bin/pip install --no-cache-dir \
        "openenv-core[core]>=0.2.2" \
        "python-dotenv>=1.2.2" \
        "openai>=1.0.0"

FROM ${BASE_IMAGE}

WORKDIR /app

COPY --from=builder /app/env/.venv /app/.venv
COPY --from=builder /app/env /app/env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app/env:$PYTHONPATH"

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && /app/.venv/bin/python -m uvicorn server.app:app --host 0.0.0.0 --port 8000"]
