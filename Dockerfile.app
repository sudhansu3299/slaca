FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY static/ ./static/
COPY tests/ ./tests/
COPY prompts/ ./prompts/
COPY scripts/ ./scripts/
COPY admin_server.py .

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1
ENV MONGO_URI=mongodb://mongodb:27017
ENV MONGODB_DB=collections_ai