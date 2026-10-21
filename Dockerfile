FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

COPY anomaly anomaly
COPY web web
COPY data data

# 7860 is the port Hugging Face Spaces expects; override with PORT
ENV PORT=7860
EXPOSE 7860
CMD ["sh", "-c", "uv run --no-dev uvicorn anomaly.server:app --host 0.0.0.0 --port ${PORT}"]
