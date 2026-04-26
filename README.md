# AI Collections System

Self-learning 3-agent collections workflow with a chat UI, admin panel, Temporal orchestration, and evaluation pipelines.

## What You Need

1. Docker + Docker Compose
2. A valid OpenAI API key if you want live model calls

## 1) Set Your Environment

Before running anything, create a new `.env` file in the project root:

```bash
cp .env.example .env
```

Then open `.env` and set at least:

```bash
OPENAI_API_KEY=your_openai_key_here
```

Notes:

1. Mock mode can run without a real key in some paths, but set `OPENAI_API_KEY` anyway so live/eval paths work cleanly.
2. If you plan to use live voice integrations, also set `VAPI_*` keys in `.env`.
3. `BORROWER_PHONE` is read from `.env` and should be set to your target number for live calling tests.
4. Current Twilio/Vapi integration in this repo is wired for a single configured borrower number flow, so use your own number in `BORROWER_PHONE`.
5. `.env.example` now includes the key variables you can copy and edit.

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

# UI Screenshots
Home Page:

<img width="1062" height="828" alt="image" src="https://github.com/user-attachments/assets/e936b763-b202-46f7-8a31-b50b5ea7ade1" />

1. Assessment Agent:

<img width="1878" height="963" alt="image" src="https://github.com/user-attachments/assets/ba7f5059-1117-4498-bfcd-aa0202f2b424" />

2. Final Notice Agent:

<img width="797" height="490" alt="image" src="https://github.com/user-attachments/assets/1a030a6a-2196-47e1-b7a5-aaa7e55464bd" />
<img width="1261" height="742" alt="image" src="https://github.com/user-attachments/assets/007053aa-aa03-423f-af21-fe667b1c318e" />


## Admin Panel

Chat-wise scores and trends
<img width="1771" height="963" alt="image" src="https://github.com/user-attachments/assets/3614d778-f7a4-4f55-937c-b7f6080d438d" />

Synthetic conversations creator
<img width="1641" height="939" alt="image" src="https://github.com/user-attachments/assets/963219a5-8b56-4d6b-9473-13ac8f4b3693" />

Self-Improvement Pipeline
<img width="1436" height="960" alt="image" src="https://github.com/user-attachments/assets/842bcfab-07ab-431c-97de-7fa28f6c7631" />

Meta-Evaluator
<img width="1575" height="1022" alt="image" src="https://github.com/user-attachments/assets/fdb8e6ec-9b7b-40a5-956c-3332ef7fe068" />

Agent Analytics
<img width="1615" height="668" alt="image" src="https://github.com/user-attachments/assets/a97e2455-9b8c-46f2-8a42-4792b7736ede" />



### Cost per Self-Learning Loop
<img width="1324" height="664" alt="image" src="https://github.com/user-attachments/assets/8d0f6399-8b1c-449b-aacf-1462c8a544b6" />


