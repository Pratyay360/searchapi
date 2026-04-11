FROM ghcr.io/astral-sh/uv:alpine3.22

WORKDIR /app

ENV UV_NO_DEV=1 \
    PYTHONUNBUFFERED=1

# Cache dependency install layer separately from source
COPY . ./
RUN uv sync --frozen
EXPOSE 8000
CMD ["uvx", "fastapi", "run"]
