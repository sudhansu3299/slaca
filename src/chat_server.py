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
    phone_number: str
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
                phone_number=req.phone_number,
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
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Collections — New Case</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: #0d0f1a; color: #dde3f0;
  font-family: -apple-system, "Segoe UI", sans-serif;
  min-height: 100vh; display: flex; align-items: center;
  justify-content: center; padding: 24px;
}
.card {
  background: #161927; border: 1px solid #252839;
  border-radius: 18px; padding: 36px 32px; width: 100%; max-width: 520px;
}
.logo { font-size: 34px; margin-bottom: 6px; }
h1 { font-size: 21px; font-weight: 700; margin-bottom: 4px; }
.sub { font-size: 12px; color: #7a849a; margin-bottom: 28px; line-height: 1.6; }
.pipeline {
  display: flex; gap: 0; margin-bottom: 28px; border-radius: 10px;
  overflow: hidden; border: 1px solid #252839; font-size: 11px;
}
.step {
  flex: 1; padding: 8px 6px; text-align: center; font-weight: 600;
  letter-spacing: .3px;
}
.step.a { background:#0d2a4a; color:#60a5fa; }
.step.b { background:#2d1660; color:#c084fc; border-left:1px solid #252839; border-right:1px solid #252839; }
.step.c { background:#3a0f0f; color:#f87171; }
.arrow { padding-top:8px; color:#7a849a; font-size:12px; align-self:center; }
label {
  display: block; font-size: 11px; color: #7a849a;
  font-weight: 700; text-transform: uppercase; letter-spacing: .5px;
  margin: 16px 0 6px;
}
label:first-of-type { margin-top: 0; }
input, select {
  width: 100%; background: #0d0f1a; border: 1px solid #252839;
  border-radius: 10px; color: #dde3f0; padding: 10px 14px;
  font-size: 14px; outline: none; transition: border-color .2s;
}
input:focus, select:focus { border-color: #4c82f7; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.btn {
  margin-top: 24px; width: 100%; background: #4c82f7; border: none;
  border-radius: 10px; color: #fff; font-size: 15px; font-weight: 700;
  padding: 14px; cursor: pointer; transition: background .2s;
  display: flex; align-items: center; justify-content: center; gap: 8px;
}
.btn:hover:not(:disabled) { background: #3464c8; }
.btn:disabled { opacity: .45; cursor: not-allowed; }
.err { margin-top: 12px; color: #f87171; font-size: 12px; display: none; }
</style>
</head>
<body>
<div class="card">
  <div class="logo">🏦</div>
  <h1>New Collection Case</h1>
  <p class="sub">Starts a live 3-stage pipeline — borrower chats first, then gets a voice call, then final written notice.</p>

  <div class="pipeline">
    <div class="step a">💬 Assessment<br><span style="font-weight:400;font-size:10px">Chat · identity + financials</span></div>
    <div class="step b">📞 Resolution<br><span style="font-weight:400;font-size:10px">Voice · offer + commitment</span></div>
    <div class="step c">📋 Final Notice<br><span style="font-weight:400;font-size:10px">Chat · consequences + deadline</span></div>
  </div>

  <form id="frm" onsubmit="go(event)">
    <div class="row">
      <div>
        <label>Borrower name</label>
        <input name="borrower_name" placeholder="Rahul Sharma" required>
      </div>
      <div>
        <label>Phone (for voice call)</label>
        <input name="phone_number" type="tel" placeholder="+917008098779" value="+917008098779" required>
      </div>
    </div>

    <div class="row">
      <div>
        <label>Outstanding amount (₹)</label>
        <input name="outstanding_amount" type="number" value="85000" min="1" required>
      </div>
      <div>
        <label>Days past due</label>
        <input name="days_past_due" type="number" value="90" min="0" required>
      </div>
    </div>

    <div class="row">
      <div>
        <label>Principal (₹)</label>
        <input name="principal_amount" type="number" value="100000" min="1" required>
      </div>
      <div>
        <label>Borrower persona</label>
        <select name="persona">
          <option value="cooperative">Cooperative</option>
          <option value="hostile">Hostile</option>
          <option value="broke">Broke</option>
          <option value="strategic_defaulter">Strategic</option>
        </select>
      </div>
    </div>

    <button class="btn" id="btn" type="submit">
      <span id="btn-text">Start Chat Session →</span>
      <span id="spin" style="display:none">⏳</span>
    </button>
  </form>
  <div class="err" id="err"></div>
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
  body.outstanding_amount = parseFloat(body.outstanding_amount);
  body.principal_amount   = parseFloat(body.principal_amount);
  body.days_past_due      = parseInt(body.days_past_due);

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

@app.get("/chat/{workflow_id}", response_class=HTMLResponse)
async def chat_page(workflow_id: str):
    with open(os.path.join(os.path.dirname(__file__), "../static/chat.html")) as f:
        html = f.read()
    return HTMLResponse(html.replace("{{WORKFLOW_ID}}", workflow_id))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
