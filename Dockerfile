# syntax=docker/dockerfile:1.6
# =============================================================================
#  AI Coding Agent — multi-stage Dockerfile
#  Produces a slim image that runs the Web UI + CLI from src/ai_env.
# =============================================================================

ARG PYTHON_VERSION=3.11

# -----------------------------------------------------------------------------
# 1) BUILDER — compile Python wheels so the runtime image stays lean
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

WORKDIR /build

# Build deps for packages with C extensions (torch, sentencepiece, etc.)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Build wheels for everything (including heavy training deps).
# We use --no-cache-dir here to keep the builder image small.
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# -----------------------------------------------------------------------------
# 2) RUNTIME — minimal image, runs as non-root, exposes the web UI
# -----------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # Defaults — overridden by docker-compose env block or `--env`
    OLLAMA_HOST=ollama \
    OLLAMA_PORT=11434 \
    WEB_HOST=0.0.0.0 \
    WEB_PORT=7861 \
    HF_HOME=/app/checkpoints/hf_cache \
    TRANSFORMERS_CACHE=/app/checkpoints/hf_cache

# Runtime OS deps: curl for healthchecks, tini for proper signal handling
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    curl \
    tini \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --system --gid 1000 aiuser \
    && useradd  --system --uid 1000 --gid aiuser --create-home --shell /bin/bash aiuser

WORKDIR /app

# Install Python deps from prebuilt wheels (no compiler needed in this stage)
COPY --from=builder /wheels /wheels
COPY requirements.txt .
RUN pip install --no-cache-dir --no-index --find-links=/wheels -r requirements.txt \
    && rm -rf /wheels

# Copy the application source AND the packaging descriptor, then install
# the package in editable mode so that `python -m ai_env ...` and the
# `ai-env` / `ai-env-web` console scripts resolve inside the container.
COPY --chown=aiuser:aiuser pyproject.toml /app/pyproject.toml
COPY --chown=aiuser:aiuser src/        /app/src/
COPY --chown=aiuser:aiuser scripts/    /app/scripts/
COPY --chown=aiuser:aiuser launch.py   /app/launch.py
COPY --chown=aiuser:aiuser README.md   /app/README.md
RUN pip install --no-cache-dir -e .

# Persistent directories
RUN mkdir -p /app/workspace /app/checkpoints /app/training_data \
    && chown -R aiuser:aiuser /app

USER aiuser

EXPOSE 7861

# tini gives us clean PID 1 / signal forwarding
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command: serve the Web UI on container port 7861
CMD ["python", "launch.py", \
    "--host", "0.0.0.0", \
    "--port", "7861", \
    "--no-browser", \
    "--root", "/app/workspace"]

# Container-level healthcheck — uses our own /api/status endpoint.
# start-period is generous (120s) because the first launch on a cold
# volume may need to wait for the model file to be available on the
# shared volume; subsequent boots are <10s.
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -fsS http://localhost:7861/api/status || exit 1

# Image metadata
LABEL org.opencontainers.image.title="ai-env" \
    org.opencontainers.image.description="Local AI coding agent Web UI (Ollama + Roo Code)" \
    org.opencontainers.image.source="https://github.com/local/ai-env" \
    org.opencontainers.image.licenses="MIT"
