# Build stage
FROM python:3.13-slim AS builder
WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml ./
RUN uv sync --no-dev

# Production stage
FROM python:3.13-slim
WORKDIR /app

COPY --from=builder /app/.venv /app/.venv
COPY . .

ENV PYTHONUNBUFFERED=1
ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
