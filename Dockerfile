# Build stage
FROM python:3.11-slim AS builder
WORKDIR /app

COPY requirements.txt pyproject.toml* setup.py* ./
RUN pip install --no-cache-dir build && \
    if [ -f pyproject.toml ]; then pip install .; elif [ -f requirements.txt ]; then pip install -r requirements.txt; fi

# Production stage
FROM python:3.11-slim
WORKDIR /app

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY . .

ENV PYTHONUNBUFFERED=1
EXPOSE 8000

CMD ["python", "app.py"]
