"""
Regression Monitor — watches post-patch performance and auto-rollbacks on regression.

Flow:
  1. On patch ADOPTION → start_monitoring(agent, baseline_rate, version, backup_filename)
     Saves a monitoring window to Redis: next MONITOR_WINDOW conversations are tracked.
  2. On every conversation end → record_outcome(agent, resolved)
     Increments counter. When window is full, checks rate vs baseline.
  3. If post_rate < baseline × (1 − REGRESSION_THRESHOLD) → auto-rollback + log event.

Configuration (change these constants):
  MONITOR_WINDOW        — how many conversations to observe after a patch
  REGRESSION_THRESHOLD  — relative drop that triggers rollback (0.15 = 15%)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────── #
MONITOR_WINDOW: int       = 10    # conversations to observe post-patch
REGRESSION_THRESHOLD: float = 0.15  # 15% relative drop triggers rollback

_KEY  = "regression:monitor:{agent}"
_TTL  = 7 * 24 * 3600   # 7 days


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ─────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────

async def start_monitoring(
    agent_name: str,
    baseline_rate: float,
    version: str,
    backup_filename: str,
) -> None:
    """
    Called immediately after a patch is adopted.
    Resets the monitoring window for this agent.
    """
    from src.data_layer import get_redis
    r = await get_redis()
    if r is None:
        log.warning("[regression] Redis unavailable — monitoring disabled for %s", agent_name)
        return

    state = {
        "agent_name":       agent_name,
        "baseline_rate":    round(baseline_rate, 4),
        "version":          version,
        "backup_filename":  backup_filename,
        "count":            0,
        "resolved":         0,
        "started_at":       _now(),
        "status":           "watching",   # watching | passed | regressed
    }
    key = _KEY.format(agent=agent_name)
    await r.setex(key, _TTL, json.dumps(state))
    log.info("[regression] started monitoring %s: baseline=%.2f%% version=%s window=%d",
             agent_name, baseline_rate * 100, version, MONITOR_WINDOW)

    # Persist start event to MongoDB
    _fire(_persist_event(agent_name, "monitoring_started", state))


async def record_outcome(agent_name: str, resolved: bool) -> None:
    """
    Call after every completed conversation.
    When MONITOR_WINDOW is reached, evaluates for regression.
    Safe to call even when no monitoring is active — no-ops silently.
    """
    from src.data_layer import get_redis
    r = await get_redis()
    if r is None:
        return

    key = _KEY.format(agent=agent_name)
    raw = await r.get(key)
    if not raw:
        return   # no active monitoring window for this agent

    state = json.loads(raw)
    if state.get("status") != "watching":
        return   # window already closed (passed or regressed)

    state["count"]   += 1
    state["resolved"] += 1 if resolved else 0

    if state["count"] < MONITOR_WINDOW:
        await r.setex(key, _TTL, json.dumps(state))
        log.debug("[regression] %s progress %d/%d resolved=%d",
                  agent_name, state["count"], MONITOR_WINDOW, state["resolved"])
        return

    # Window complete — evaluate
    post_rate = state["resolved"] / state["count"]
    baseline  = state["baseline_rate"]
    drop      = baseline - post_rate
    rel_drop  = drop / baseline if baseline > 0 else 0

    log.info("[regression] %s window complete: baseline=%.1f%% post=%.1f%% drop=%.1f%%",
             agent_name, baseline * 100, post_rate * 100, rel_drop * 100)

    if rel_drop > REGRESSION_THRESHOLD:
        state["status"] = "regressed"
        state["post_rate"] = round(post_rate, 4)
        state["rel_drop"]  = round(rel_drop, 4)
        await r.setex(key, _TTL, json.dumps(state))
        log.warning("[regression] REGRESSION DETECTED in %s — triggering auto-rollback", agent_name)
        asyncio.create_task(_auto_rollback(agent_name, state, post_rate))
    else:
        state["status"] = "passed"
        state["post_rate"] = round(post_rate, 4)
        await r.setex(key, _TTL, json.dumps(state))
        log.info("[regression] %s passed monitoring — no regression", agent_name)
        _fire(_persist_event(agent_name, "monitoring_passed", state))


async def get_status(agent_name: str) -> Optional[dict]:
    """Return current monitoring state for an agent (None if not active)."""
    from src.data_layer import get_redis
    r = await get_redis()
    if r is None:
        return None
    raw = await r.get(_KEY.format(agent=agent_name))
    return json.loads(raw) if raw else None


async def get_all_statuses() -> dict:
    """Return monitoring state for all three agents."""
    from src.self_learning.feeder import ALL_AGENTS
    return {a: await get_status(a) for a in ALL_AGENTS}


async def list_regression_events(limit: int = 50) -> list[dict]:
    """Return recent regression events from MongoDB."""
    from src.data_layer import get_mongo
    db = await get_mongo()
    if db is None:
        return []
    cursor = db.regression_events.find({}, {"_id": 0}).sort("timestamp", -1).limit(limit)
    return await cursor.to_list(length=limit)


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

async def _auto_rollback(agent_name: str, state: dict, post_rate: float) -> None:
    from src.self_learning.improvement_pipeline import rollback_prompt
    backup = state.get("backup_filename", "")
    if not backup:
        log.error("[regression] no backup filename stored — cannot auto-rollback %s", agent_name)
        return

    result = await rollback_prompt(agent_name, backup)
    success = result.get("success", False)
    log.info("[regression] rollback %s for %s: %s",
             "SUCCESS" if success else "FAILED", agent_name, result)

    await _persist_event(agent_name, "regression_rollback", {
        **state,
        "post_rate":     round(post_rate, 4),
        "rollback_ok":   success,
        "rollback_info": result,
    })


async def _persist_event(agent_name: str, event_type: str, state: dict) -> None:
    from src.data_layer import get_mongo
    db = await get_mongo()
    if db is None:
        return
    try:
        await db.regression_events.insert_one({
            "agent_name":    agent_name,
            "event_type":    event_type,
            "baseline_rate": state.get("baseline_rate"),
            "post_rate":     state.get("post_rate"),
            "rel_drop":      state.get("rel_drop"),
            "version":       state.get("version"),
            "backup_filename": state.get("backup_filename"),
            "window_count":  state.get("count"),
            "window_resolved": state.get("resolved"),
            "status":        state.get("status"),
            "timestamp":     _now(),
        })
    except Exception as e:
        log.warning("[regression] persist_event failed: %s", e)


def _fire(coro) -> None:
    try:
        asyncio.get_event_loop().create_task(coro)
    except RuntimeError:
        pass
