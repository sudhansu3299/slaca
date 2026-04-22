"""
Message bus — bridges worker activities ↔ chat server WebSocket.

Architecture:
  - Both worker and chat server import this module
  - In single-process mode: in-memory queues shared directly
  - In multi-process mode: worker calls chat server HTTP endpoints to push messages,
    chat server calls worker HTTP endpoint to deliver borrower input

Env vars:
  MESSAGE_BUS_URL  — if set, use HTTP bridge (multi-process)
                     e.g. http://localhost:8000 (chat server address)
  WORKER_BUS_URL   — if set, chat server calls this to deliver borrower messages
                     e.g. http://localhost:8001 (worker address)
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

# ── In-process queues (single-process or same-process testing) ── #
_message_queues: dict[str, asyncio.Queue] = {}

# Buffer of agent messages for late-connecting UIs (replay on WebSocket connect)
_message_buffer: dict[str, list[dict]] = {}


def get_message_buffer(workflow_id: str) -> list[dict]:
    """Return buffered agent messages for this workflow (for replay)."""
    return _message_buffer.get(workflow_id, [])


def get_borrower_queue(workflow_id: str) -> asyncio.Queue:
    if workflow_id not in _message_queues:
        _message_queues[workflow_id] = asyncio.Queue()
    return _message_queues[workflow_id]


async def deliver_borrower_message(workflow_id: str, text: str) -> None:
    """UI → Activity: borrower typed something."""
    q = get_borrower_queue(workflow_id)
    await q.put(text)


async def wait_for_borrower_message(workflow_id: str, timeout: float = 300.0) -> Optional[str]:
    """Activity → waits for borrower input. Returns None on timeout."""
    q = get_borrower_queue(workflow_id)
    try:
        return await asyncio.wait_for(q.get(), timeout=timeout)
    except asyncio.TimeoutError:
        return None


# ── Agent → UI push ─────────────────────────────────────────── #

# In-process callbacks registered by the chat server
_agent_push_callbacks: dict[str, list] = {}   # workflow_id → list of async callbacks


def register_push_callback(workflow_id: str, callback) -> None:
    _agent_push_callbacks.setdefault(workflow_id, []).append(callback)


def unregister_push_callback(workflow_id: str, callback) -> None:
    cbs = _agent_push_callbacks.get(workflow_id, [])
    if callback in cbs:
        cbs.remove(callback)


async def push_agent_message(workflow_id: str, message: str, stage: str) -> None:
    """
    Activity → UI: agent produced a message.
    In single-process mode: calls registered WebSocket callbacks directly.
    Messages are also buffered so late-connecting UIs get full history.
    In multi-process mode: HTTP POST to chat server.
    """
    from datetime import datetime, timezone
    entry = {
        "role": "agent",
        "content": message,
        "stage": stage,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    bus_url = os.getenv("MESSAGE_BUS_URL", "")
    if bus_url:
        await _http_push(bus_url, workflow_id, entry)
        return

    # Buffer for replay on connect
    _message_buffer.setdefault(workflow_id, []).append(entry)

    # Push to active WebSocket callbacks
    for cb in list(_agent_push_callbacks.get(workflow_id, [])):
        try:
            await cb(entry)
        except Exception:
            pass


async def push_stage_event(workflow_id: str, event: str, message: str) -> None:
    """Activity → UI: stage transition event."""
    from datetime import datetime, timezone
    entry = {
        "role": "system",
        "event": event,
        "content": message,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    bus_url = os.getenv("MESSAGE_BUS_URL", "")
    if bus_url:
        await _http_push(bus_url, workflow_id, entry)
        return

    # Buffer for replay
    _message_buffer.setdefault(workflow_id, []).append(entry)

    for cb in list(_agent_push_callbacks.get(workflow_id, [])):
        try:
            await cb(entry)
        except Exception:
            pass


async def _http_push(base_url: str, workflow_id: str, payload: dict) -> None:
    import httpx
    try:
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{base_url.rstrip('/')}/internal/push/{workflow_id}",
                json=payload,
                timeout=5,
            )
    except Exception:
        pass  # best-effort
