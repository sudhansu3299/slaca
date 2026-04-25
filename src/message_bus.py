"""
Message bus — bridges worker activities ↔ chat server WebSocket.

Architecture:
  - Both worker and chat server import this module
  - In single-process mode (EMBEDDED_WORKER=1, the default):
      in-memory asyncio queues shared directly — zero latency, no deps
  - In multi-process mode (MESSAGE_BUS_URL is set):
      Agent → UI:       worker HTTP POSTs to /internal/push/{wf_id}
      Borrower → Worker: chat server RPUSH to Redis list;
                         worker BLPOP with short polling loops to avoid
                         clashing with the shared Redis socket_timeout

Env vars:
  MESSAGE_BUS_URL  — if set, switches to multi-process HTTP bridge
                     e.g. http://localhost:8000
  REDIS_URL        — used for borrower→worker queue in multi-process mode
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# ── In-process queues (single-process / embedded-worker mode) ──── #
_message_queues: dict[str, asyncio.Queue] = {}
_message_buffer: dict[str, list[dict]] = {}

_BORROWER_KEY_TTL    = 600   # 10 minutes
_BORROWER_KEY_PREFIX = "borrower_msg:"


def _redis_key(workflow_id: str) -> str:
    return f"{_BORROWER_KEY_PREFIX}{workflow_id}"


def _is_multiprocess() -> bool:
    return bool(os.getenv("MESSAGE_BUS_URL", "").strip())


def get_message_buffer(workflow_id: str) -> list[dict]:
    """Return buffered agent messages for late-connecting UIs (replay)."""
    return _message_buffer.get(workflow_id, [])


def get_borrower_queue(workflow_id: str) -> asyncio.Queue:
    if workflow_id not in _message_queues:
        _message_queues[workflow_id] = asyncio.Queue()
    return _message_queues[workflow_id]


# ── Borrower → Activity ─────────────────────────────────────────── #

async def deliver_borrower_message(workflow_id: str, text: str) -> None:
    """
    Chat server → Activity: borrower typed something.

    Embedded mode:   put into local asyncio queue (same process).
    Multi-process:   RPUSH to Redis list so the worker can poll it.
    """
    if _is_multiprocess():
        await _redis_push_borrower(workflow_id, text)
    else:
        await get_borrower_queue(workflow_id).put(text)


async def wait_for_borrower_message(
    workflow_id: str, timeout: float = 300.0
) -> Optional[str]:
    """
    Activity: wait for the next borrower message.

    Embedded mode:   asyncio queue with timeout.
    Multi-process:   poll Redis list in 5-second chunks.
    """
    if _is_multiprocess():
        return await _redis_pop_borrower(workflow_id, timeout)

    try:
        return await asyncio.wait_for(
            get_borrower_queue(workflow_id).get(), timeout=timeout
        )
    except asyncio.TimeoutError:
        return None


# ── Redis helpers (multi-process mode only) ─────────────────────── #

async def _redis_push_borrower(workflow_id: str, text: str) -> None:
    """RPUSH borrower message onto a Redis list."""
    try:
        from src.data_layer import get_redis
        r = await get_redis()
        if r is None:
            log.warning("[bus] Redis unavailable — cannot deliver borrower message")
            return
        key = _redis_key(workflow_id)
        await r.rpush(key, text)
        await r.expire(key, _BORROWER_KEY_TTL)
        log.debug("[bus] pushed borrower msg → %s", key)
    except Exception as e:
        log.warning("[bus] _redis_push_borrower failed: %s", e)


async def _redis_pop_borrower(
    workflow_id: str, timeout: float = 300.0
) -> Optional[str]:
    """
    Poll Redis for a borrower message, total wait up to `timeout` seconds.

    Uses short 5-second BLPOP chunks on a dedicated connection (no socket_timeout)
    to avoid clashing with the 2-second socket_timeout on the shared data_layer
    Redis client.
    """
    import redis.asyncio as aioredis

    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    key = _redis_key(workflow_id)
    loop = asyncio.get_event_loop()
    deadline = loop.time() + timeout

    try:
        r = aioredis.from_url(
            redis_url,
            encoding="utf-8",
            decode_responses=True,
            socket_connect_timeout=5,
            # No socket_timeout — required for BLPOP to block
        )
    except Exception as e:
        log.warning("[bus] could not create blocking Redis connection: %s", e)
        return None

    try:
        while loop.time() < deadline:
            chunk = min(5.0, deadline - loop.time())
            if chunk <= 0:
                break
            try:
                result = await r.blpop(key, timeout=int(chunk) or 1)
                if result is not None:
                    _, value = result
                    log.debug("[bus] popped borrower msg ← %s", key)
                    return value if isinstance(value, str) else value.decode()
            except Exception as e:
                log.warning("[bus] blpop error (retrying): %s", e)
                await asyncio.sleep(1)
        return None
    finally:
        try:
            await r.aclose()
        except Exception:
            pass


# ── Agent → UI push ─────────────────────────────────────────────── #

_agent_push_callbacks: dict[str, list] = {}


def register_push_callback(workflow_id: str, callback) -> None:
    _agent_push_callbacks.setdefault(workflow_id, []).append(callback)


def unregister_push_callback(workflow_id: str, callback) -> None:
    cbs = _agent_push_callbacks.get(workflow_id, [])
    if callback in cbs:
        cbs.remove(callback)


async def push_agent_message(workflow_id: str, message: str, stage: str) -> None:
    """
    Activity → UI: agent produced a message.

    Multi-process:  HTTP POST to chat server /internal/push/{wf_id}.
    Embedded:       call registered WebSocket callbacks + buffer for replay.
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

    _message_buffer.setdefault(workflow_id, []).append(entry)
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
