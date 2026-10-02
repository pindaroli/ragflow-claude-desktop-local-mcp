FROM python:3.12-slim

WORKDIR /app

# Prevent Python from writing pyc files and buffer stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install nodejs, npm (for supergateway) and wget (for container healthchecks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    nodejs \
    npm \
    wget \
    git \
    && npm install -g supergateway \
    && pip install --no-cache-dir uv \
    && rm -rf /var/lib/apt/lists/*

# Create unprivileged non-root app user
RUN useradd -u 1000 -m -s /bin/bash appuser && \
    chown -R appuser:appuser /app

# Copy project manifests and source code
COPY --chown=appuser:appuser pyproject.toml uv.lock README.md /app/
COPY --chown=appuser:appuser src/ /app/src/

# Install dependencies into virtual environment
RUN uv sync --no-dev

USER appuser

# Expose internal port for supergateway SSE bridge
EXPOSE 8080

ENTRYPOINT ["supergateway", "--port", "8080", "--ssePath", "/mcp", "--messagePath", "/message", "--stdio", "uv run ragflow-claude-mcp"]
