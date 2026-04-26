# AI Collections System

Self-learning 3-agent collections workflow with a chat UI, admin panel, Temporal orchestration, and evaluation pipelines.

## What You Need

1. Docker + Docker Compose
2. A valid OpenAI API key if you want live model calls

## 1) Set Your Environment

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Open `.env` and set at least:

```bash
OPENAI_API_KEY=your_openai_key_here
```

Notes:

1. Mock mode can run without a real key in some paths, but set `OPENAI_API_KEY` anyway so live/eval paths work cleanly.
2. If you plan to use live voice integrations, also set `VAPI_*` keys from `docker-compose.yml` comments.

## 2) Start the Stack with Docker Compose

From project root:

```bash
docker compose up --build
```

This starts:

1. Temporal + Temporal UI
2. Redis
3. MongoDB
4. Postgres (Temporal + app DB)
5. Worker
6. Chat/API server

## 3) Open the UI and Admin Panel

After containers are healthy:

1. Main start/chat UI: [http://localhost:8000](http://localhost:8000)
2. Admin panel: [http://localhost:8000/admin](http://localhost:8000/admin)
3. Temporal UI: [http://localhost:8233](http://localhost:8233)

## 4) Live vs Mock Modes

Default `docker compose up --build` runs with mock worker defaults from compose.

If you want full live profile (real worker/voice path), run:

```bash
docker compose --profile live up --build
```

If you want the smoke harness profile:

```bash
docker compose --profile test up --build
```

## 5) Useful Commands

Tail logs:

```bash
docker compose logs -f chat-server
docker compose logs -f worker
```

Restart after env changes:

```bash
docker compose down
docker compose up --build
```

## 6) Quick Troubleshooting

If `localhost:8000` is not loading:

1. Check `chat-server` logs.
2. Confirm required services are healthy with `docker compose ps`.

If model calls fail:

1. Confirm `.env` has `OPENAI_API_KEY`.
2. Recreate containers so env is reloaded.

If admin panel looks stale:

1. Hard refresh browser.
2. Ensure you are using the same running stack (avoid mixing old local processes with Docker stack).

## 7) Key Paths in This Repo

1. Workflow orchestration: `src/temporal_workflow.py`
2. Agents: `src/agents/`
3. Handoff/context: `src/handoff.py`
4. Self-learning: `src/self_learning/`
5. Chat server + admin routes: `src/chat_server.py`
6. Admin UI: `static/admin.html`
7. Docker setup: `docker-compose.yml`, `Dockerfile`
