# syntax=docker/dockerfile:1
FROM python:3.12-slim

# LightGBM needs the OpenMP runtime available at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Bring in the uv binary from its official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies in their own layer so that source changes do not
# invalidate the cached dependency install.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Install the project itself.
COPY README.md ./
COPY src/ ./src/
RUN uv sync --frozen --no-dev

# Copy the self-contained serving bundle: the exported model, the feature
# table and the metadata. The MLflow registry (mlflow.db, mlartifacts) never
# enters the image.
COPY data/processed/ranking/serving/ ./serving/

ENV GEOPLAY_SERVING_DIR=/app/serving \
    GEOPLAY_MODEL_URI=/app/serving/model

EXPOSE 8000
CMD ["uv", "run", "--no-sync", "uvicorn", "geoplay.ranking.api:app", \
     "--host", "0.0.0.0", "--port", "8000"]
