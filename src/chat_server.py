"""
Chat Server — FastAPI + WebSocket.

Endpoints:
  GET  /                          → redirect to /chat
  GET  /chat/{workflow_id}        → serve the chat UI HTML
  WS   /ws/{workflow_id}          → real-time borrower ↔ agent conversation
  POST /start                     → launch a new collections workflow
  GET  /status/{workflow_id}      → workflow status + conversation history
  GET  /health                    → health check

Flow:
  1. POST /start  → creates Temporal workflow, returns workflow_id + chat URL
  2. Borrower opens /chat/{workflow_id} in browser
  3. Browser connects WebSocket /ws/{workflow_id}
  4. Activity writes agent message → push_agent_message() → WebSocket → browser
  5. Borrower types → WebSocket → push_message() → activity reads it
  6. When stage=FINAL_NOTICE, same flow but different agent personality
"""

from __future__ import annotations

import asyncio
import json
import os
import uuid
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware

from contextlib import asynccontextmanager

from temporalio.client import Client

from src.temporal_workflow import CollectionsInput, CollectionsWorkflow
from src.message_bus import (
    deliver_borrower_message,
    register_push_callback,
    unregister_push_callback,
    get_message_buffer,
)
from src.admin_api import router as admin_router


# ── Lifespan (embedded worker) ────────────────────────────────── #

@asynccontextmanager
async def lifespan(app_):
    """Start embedded Temporal worker in same process as chat server."""
    worker_task = None
    if os.getenv("EMBEDDED_WORKER", "1") != "0":
        from src.temporal_activities import start_worker, TASK_QUEUE
        task_queue = os.getenv("TASK_QUEUE_LIVE", TASK_QUEUE)
        worker_task = asyncio.create_task(start_worker(task_queue))
        print(f"[chat_server] embedded worker started on '{task_queue}'", flush=True)
    yield
    if worker_task:
        worker_task.cancel()
        try:
            await worker_task
        except asyncio.CancelledError:
            pass


# ── App ──────────────────────────────────────────────────────── #

app = FastAPI(title="Collections Chat", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_methods=["*"], allow_headers=["*"])
app.include_router(admin_router)

# ── In-process connection registry ─────────────────────────── #
_connections: dict[str, set[WebSocket]] = {}
_history: dict[str, list[dict]] = {}


async def _broadcast(workflow_id: str, data: dict) -> None:
    """Send a message to all connected WebSocket clients for this workflow."""
    if "ts" not in data:
        data["ts"] = datetime.now(timezone.utc).isoformat()
    _history.setdefault(workflow_id, []).append(data)
    conns = _connections.get(workflow_id, set())
    dead = set()
    for ws in conns:
        try:
            await ws.send_json(data)
        except Exception:
            dead.add(ws)
    for ws in dead:
        conns.discard(ws)


# ── WebSocket ─────────────────────────────────────────────── #

@app.websocket("/ws/{workflow_id}")
async def websocket_endpoint(ws: WebSocket, workflow_id: str):
    await ws.accept()
    _connections.setdefault(workflow_id, set()).add(ws)

    # Register as a push target for this workflow so activities can reach us
    async def _push_callback(entry: dict):
        try:
            await ws.send_json(entry)
        except Exception:
            pass

    register_push_callback(workflow_id, _push_callback)

    # Replay: merge stored history + message bus buffer, deduplicate by ts
    seen_ts = set()
    replay = []
    for msg in (_history.get(workflow_id, []) + get_message_buffer(workflow_id)):
        key = (msg.get("role"), msg.get("ts", ""), msg.get("content", "")[:20])
        if key not in seen_ts:
            seen_ts.add(key)
            replay.append(msg)
    replay.sort(key=lambda m: m.get("ts", ""))
    for msg in replay:
        try:
            await ws.send_json(msg)
        except Exception:
            break

    try:
        while True:
            data = await ws.receive_text()
            try:
                payload = json.loads(data)
                text = payload.get("message", data)
            except (json.JSONDecodeError, AttributeError):
                text = data

            if not text.strip():
                continue

            # Store and echo borrower turn
            entry = {
                "role": "borrower",
                "content": text,
                "ts": datetime.now(timezone.utc).isoformat(),
            }
            _history.setdefault(workflow_id, []).append(entry)
            await _broadcast(workflow_id, entry)

            # Deliver to waiting activity
            await deliver_borrower_message(workflow_id, text)

    except WebSocketDisconnect:
        _connections.get(workflow_id, set()).discard(ws)
        unregister_push_callback(workflow_id, _push_callback)


# ── REST endpoints ─────────────────────────────────────────── #

class StartRequest(BaseModel):
    borrower_id: Optional[str] = None
    loan_id: Optional[str] = None
    borrower_name: str = ""
    phone_number: Optional[str] = None
    principal_amount: float = 100_000
    outstanding_amount: float = 85_000
    days_past_due: int = 90
    persona: str = "cooperative"


class StartResponse(BaseModel):
    workflow_id: str
    chat_url: str
    status: str = "started"


@app.post("/start", response_model=StartResponse)
async def start_workflow(req: StartRequest):
    """Launch a new collections workflow and return the chat URL."""
    borrower_id = req.borrower_id or f"BRW-{uuid.uuid4().hex[:8].upper()}"
    loan_id = req.loan_id or f"LN-{uuid.uuid4().hex[:6].upper()}"
    workflow_id = f"collections-{borrower_id}"
    phone_number = (req.phone_number or "").strip() or os.getenv("BORROWER_PHONE", "").strip()
    if not phone_number:
        raise HTTPException(
            status_code=400,
            detail="Phone number is required. Provide one in the form or set BORROWER_PHONE in env.",
        )

    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    task_queue = os.getenv("TASK_QUEUE_LIVE", "collections-queue-live")

    try:
        client = await Client.connect(address)
        await client.start_workflow(
            CollectionsWorkflow.run,
            CollectionsInput(
                borrower_id=borrower_id,
                loan_id=loan_id,
                workflow_id=workflow_id,        # ← KEY FIX: full ID passed through
                borrower_name=req.borrower_name or borrower_id,
                phone_number=phone_number,
                principal_amount=req.principal_amount,
                outstanding_amount=req.outstanding_amount,
                days_past_due=req.days_past_due,
                persona=req.persona,
                use_real_chat=True,
            ),
            id=workflow_id,
            task_queue=task_queue,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    host = os.getenv("CHAT_HOST", "localhost:8000")
    return StartResponse(
        workflow_id=workflow_id,
        chat_url=f"http://{host}/chat/{workflow_id}",
    )


@app.get("/status/{workflow_id}")
async def get_status(workflow_id: str):
    # Merge chat history + message bus buffer
    combined = {}
    for msg in (_history.get(workflow_id, []) + get_message_buffer(workflow_id)):
        key = (msg.get("role"), msg.get("ts", ""), msg.get("content", "")[:20])
        combined[key] = msg
    history = sorted(combined.values(), key=lambda m: m.get("ts", ""))
    return {
        "workflow_id": workflow_id,
        "history": history,
        "active_connections": len(_connections.get(workflow_id, set())),
    }


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/internal/push/{workflow_id}")
async def internal_push(workflow_id: str, payload: dict):
    """
    Called by worker activities (multi-process mode) to push agent messages
    to the chat UI. Also registers the message in history.
    """
    await _broadcast(workflow_id, payload)
    return {"ok": True}


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTMLResponse(_START_PAGE)


@app.get("/start", response_class=HTMLResponse)
async def start_page():
    return HTMLResponse(_START_PAGE)


_START_PAGE = """<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"UTF-8\">
<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">
<title>Collections — New Case</title>
<link rel=\"preconnect\" href=\"https://fonts.googleapis.com\">
<link rel=\"preconnect\" href=\"https://fonts.gstatic.com\" crossorigin>
<link href=\"https://fonts.googleapis.com/css2?family=Public+Sans:wght@400;500;600;700&family=Spectral:ital,wght@0,500;0,600;1,500&display=swap\" rel=\"stylesheet\">
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg: oklch(97% 0.01 240);
  --card: oklch(96% 0.015 230);
  --card-border: oklch(88% 0.018 230);
  --ink: oklch(28% 0.02 250);
  --muted: oklch(52% 0.02 250);
  --accent: oklch(62% 0.11 242);
  --accent-strong: oklch(56% 0.13 242);
  --accent-soft: oklch(92% 0.04 242);
  --critical: oklch(70% 0.14 30);
  --space-xs: 6px;
  --space-sm: 12px;
  --space-md: 20px;
  --space-lg: 32px;
  --r-lg: 20px;
  --r-md: 12px;
  --font-ui: 'Public Sans', 'Helvetica Neue', sans-serif;
  --font-serif: 'Spectral', 'Times New Roman', serif;
}
body {
  min-height: 100vh;
  background: var(--bg);
  font-family: var(--font-ui);
  color: var(--ink);
  padding: clamp(24px, 5vw, 60px);
  display: flex;
  align-items: center;
  justify-content: center;
}
.shell {
  width: min(1040px, 100%);
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
  gap: var(--space-lg);
  align-items: stretch;
}
.panel {
  background: var(--card);
  border: 1px solid var(--card-border);
  border-radius: var(--r-lg);
  padding: var(--space-lg);
  box-shadow: 0 20px 60px rgba(15, 37, 66, 0.08);
}
.intro {
  display: flex;
  flex-direction: column;
  gap: var(--space-md);
}
.badge {
  align-self: flex-start;
  padding: 6px 14px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent-strong);
  font-size: 11px;
  letter-spacing: 0.2em;
  font-weight: 600;
  text-transform: uppercase;
}
.logo-mark {
  width: 56px;
  height: 56px;
  border-radius: 16px;
  border: 1px solid var(--card-border);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: var(--font-serif);
  font-size: 26px;
}
.intro h1 {
  font-size: clamp(28px, 4vw, 40px);
  letter-spacing: -0.01em;
  font-weight: 600;
}
.intro p {
  color: var(--muted);
  line-height: 1.7;
  font-size: 15px;
  max-width: 46ch;
}
.timeline {
  border: 1px solid var(--card-border);
  border-radius: var(--r-md);
  padding: var(--space-sm);
  display: flex;
  gap: var(--space-sm);
  flex-wrap: wrap;
  background: white;
}
.stage {
  flex: 1;
  min-width: 140px;
  padding: 10px 12px;
  border-radius: 12px;
  border: 1px solid var(--card-border);
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.stage small {
  font-size: 11px;
  letter-spacing: 0.12em;
  color: var(--muted);
  text-transform: uppercase;
}
.stage strong { font-size: 14px; }
.stage span { font-size: 12px; color: var(--muted); }
.intel {
  display: grid;
  grid-template-columns: repeat(auto-fit,minmax(140px,1fr));
  gap: var(--space-sm);
}
.intel-card {
  border-radius: var(--r-md);
  border: 1px dashed var(--card-border);
  padding: 14px;
}
.intel-card .label {
  font-size: 11px;
  letter-spacing: 0.16em;
  color: var(--muted);
  text-transform: uppercase;
}
.intel-card .value {
  font-size: 20px;
  font-weight: 600;
  margin-top: 6px;
}
.panel form {
  margin-top: var(--space-md);
  display: flex;
  flex-direction: column;
  gap: 18px;
}
label {
  font-size: 11px;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--muted);
  margin-bottom: 6px;
  font-weight: 600;
}
.field {
  display: flex;
  flex-direction: column;
}
.row {
  display: grid;
  grid-template-columns: repeat(auto-fit,minmax(220px,1fr));
  gap: var(--space-sm);
}
.phone-group {
  display: grid;
  grid-template-columns: 110px 1fr;
  gap: 8px;
}
input, select {
  width: 100%;
  border: 1px solid var(--card-border);
  background: white;
  border-radius: 12px;
  padding: 12px 16px;
  font-size: 15px;
  font-family: var(--font-ui);
  color: var(--ink);
  transition: border-color 0.2s, box-shadow 0.2s;
}
input:focus, select:focus {
  border-color: var(--accent);
  box-shadow: 0 0 0 3px color-mix(in oklch, var(--accent) 40%, transparent);
  outline: none;
}
.btn {
  width: 100%;
  border: none;
  border-radius: 14px;
  padding: 16px;
  font-size: 16px;
  font-weight: 600;
  color: white;
  background: var(--accent);
  cursor: pointer;
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  transition: background 0.2s, transform 0.2s;
}
.btn:hover:not(:disabled) { background: var(--accent-strong); transform: translateY(-1px); }
.btn:disabled { opacity: 0.45; cursor: not-allowed; }
.err {
  margin-top: 10px;
  color: var(--critical);
  font-size: 13px;
  display: none;
}
.footnote {
  margin-top: 18px;
  font-size: 12px;
  color: var(--muted);
}
</style>
</head>
<body>
<div class=\"shell\">
  <section class=\"panel intro\">
    <div class=\"badge\">Collections Desk</div>
    <div class=\"logo-mark\">Σ</div>
    <h1>Open a new collection case</h1>
    <p>Kick off a supervised chat session in under a minute. We preload balances and loan terms from the ledger once you supply the borrower identity.</p>
    <div class=\"timeline\">
      <div class=\"stage\">
        <small>Stage 01</small>
        <strong>Assessment</strong>
        <span>Chat · verify, gather intent</span>
      </div>
      <div class=\"stage\">
        <small>Stage 02</small>
        <strong>Resolution</strong>
        <span>Voice · structure proposal</span>
      </div>
      <div class=\"stage\">
        <small>Stage 03</small>
        <strong>Final Notice</strong>
        <span>Chat · documentation + deadline</span>
      </div>
    </div>
    <div class=\"intel\">
      <div class=\"intel-card\">
        <div class=\"label\">SLA</div>
        <div class=\"value\">&lt; 4 min to connect</div>
      </div>
      <div class=\"intel-card\">
        <div class=\"label\">Coverage</div>
        <div class=\"value\">APAC · GCC · NA</div>
      </div>
      <div class=\"intel-card\">
        <div class=\"label\">Compliance</div>
        <div class=\"value\">Every call recorded</div>
      </div>
    </div>
  </section>
  <section class=\"panel\">
    <div class=\"badge\" style=\"font-size:10px;letter-spacing:0.3em;\">Intake</div>
    <h2 style=\"margin-top:12px;font-size:22px;font-weight:600;\">Borrower details</h2>
    <p style=\"margin-top:4px;color:var(--muted);font-size:14px;\">Only the caller information is required to initiate the workflow. Ledger fields hydrate downstream.</p>
    <form id=\"frm\" onsubmit=\"go(event)\">
      <div class=\"row\">
        <div class=\"field\">
          <label>Name</label>
          <input name=\"borrower_name\" placeholder=\"Rahul Sharma\" required>
        </div>
        <div class=\"field\">
          <label>Phone</label>
          <div class=\"phone-group\">
            <select name=\"country_code\">
              <option value=\"+91\" selected>🇮🇳 +91</option>
              <option value=\"+1\">🇺🇸 +1</option>
              <option value=\"+44\">🇬🇧 +44</option>
              <option value=\"+61\">🇦🇺 +61</option>
              <option value=\"+65\">🇸🇬 +65</option>
            </select>
            <input name=\"phone_number\" type=\"tel\" placeholder=\"7008098779\">
          </div>
        </div>
      </div>
      <button class=\"btn\" id=\"btn\" type=\"submit\">
        <span id=\"btn-text\">Start Chat Session →</span>
        <span id=\"spin\" style=\"display:none\">⏳</span>
      </button>
      <div class=\"err\" id=\"err\"></div>
      <p class=\"footnote\">Submitting triggers a supervised channel and records the attempt in the audit ledger.</p>
    </form>
  </section>
</div>

<script>
async function go(e) {
  e.preventDefault();
  const btn = document.getElementById('btn');
  const spin = document.getElementById('spin');
  const err  = document.getElementById('err');
  document.getElementById('btn-text').textContent = 'Starting…';
  btn.disabled = true; spin.style.display = 'inline';

  const fd = new FormData(e.target);
  const body = Object.fromEntries(fd.entries());
  const cc = (body.country_code || '+91').trim();
  const rawPhone = (body.phone_number || '').trim();
  if (rawPhone) {
    const compact = rawPhone.replace(/\\s+/g, '');
    body.phone_number = compact.startsWith('+')
      ? compact
      : `${cc}${compact.replace(/^0+/, '')}`;
  } else {
    // Backend will fall back to BORROWER_PHONE from env.
    delete body.phone_number;
  }
  delete body.country_code;
  try {
    const res = await fetch('/start', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(body)
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || JSON.stringify(data));
    window.location.href = data.chat_url;
  } catch(ex) {
    err.textContent = '❌ ' + ex.message;
    err.style.display = 'block';
    document.getElementById('btn-text').textContent = 'Start Chat Session →';
    btn.disabled = false; spin.style.display = 'none';
  }
}
</script>
</body>
</html>
"""


# ── Chat UI page ─────────────────────────────────────────────── #

@app.get("/admin", response_class=HTMLResponse)
async def admin_page():
    with open(os.path.join(os.path.dirname(__file__), "../static/admin.html")) as f:
        return HTMLResponse(
            f.read(),
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            },
        )


@app.get("/chat/{workflow_id}", response_class=HTMLResponse)
async def chat_page(workflow_id: str):
    with open(os.path.join(os.path.dirname(__file__), "../static/chat.html")) as f:
        html = f.read()
    return HTMLResponse(html.replace("{{WORKFLOW_ID}}", workflow_id))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
