FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir \
    temporalio \
    anthropic \
    pydantic \
    python-dotenv \
    httpx \
    fastapi \
    "uvicorn[standard]" \
    websockets \
    jinja2 \
    aiofiles \
    pytest \
    pytest-asyncio

COPY src/ ./src/
COPY static/ ./static/
COPY tests/ ./tests/

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

CMD ["python", "-m", "src.temporal_activities"]
