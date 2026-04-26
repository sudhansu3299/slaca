"""
Data Layer — Redis (ephemeral) + MongoDB (persistent).

Redis patterns (all with TTL):
  session:{borrower_id}   → current stage/status/attempts   TTL 24h
  summary:{borrower_id}   → agent summaries list             TTL 1h
  retry:{borrower_id}     → retry count                      TTL 24h
  lock:{borrower_id}      → distributed lock (SET NX EX)     TTL 30s

MongoDB collections:
  borrower_cases    → one doc per borrower, current status
  interactions      → one doc per agent turn (full audit trail)
  outcomes          → one doc per workflow completion
  eval_results      → LLM-as-judge scores per interaction

All MongoDB writes are fire-and-forget (async, non-blocking to the workflow).
Redis is the source of truth during active execution.
On Redis miss → reconstruct from last N interactions in MongoDB.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

log = logging.getLogger(__name__)

# ── Connection strings (read from env, fall back to localhost) ── #
REDIS_URL   = os.getenv("REDIS_URL",   "redis://localhost:6379/0")
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017")
MONGODB_DB  = os.getenv("MONGODB_DB",  "collections_ai")

# ── TTLs (seconds) ──────────────────────────────────────────── #
SESSION_TTL = 24 * 3600   # 24 hours
SUMMARY_TTL = 3600        # 1 hour
RETRY_TTL   = 24 * 3600   # 24 hours
LOCK_TTL    = 30          # 30 seconds


# ── Redis client (lazy singleton) ───────────────────────────── #

_redis = None

async def get_redis():
    global _redis
    if _redis is None:
        try:
            import redis.asyncio as aioredis
            _redis = aioredis.from_url(
                REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            await _redis.ping()
            log.info(f"[redis] connected to {REDIS_URL}")
        except Exception as e:
            log.warning(f"[redis] unavailable ({e}) — running without Redis cache")
            _redis = None
    return _redis


# ── MongoDB client (lazy singleton) ─────────────────────────── #

_mongo_client = None
_mongo_db     = None

async def get_mongo():
    global _mongo_client, _mongo_db
    if _mongo_db is None:
        try:
            from motor.motor_asyncio import AsyncIOMotorClient
            _mongo_client = AsyncIOMotorClient(
                MONGODB_URL,
                serverSelectionTimeoutMS=2000,
            )
            _mongo_db = _mongo_client[MONGODB_DB]
            # Ensure indexes
            await _ensure_indexes(_mongo_db)
            log.info(f"[mongo] connected to {MONGODB_URL}/{MONGODB_DB}")
        except Exception as e:
            log.warning(f"[mongo] unavailable ({e}) — running without persistence")
            _mongo_db = None
    return _mongo_db


async def _ensure_indexes(db) -> None:
    try:
        await db.borrower_cases.create_index("borrower_id", unique=True)
        await db.interactions.create_index([("borrower_id", 1), ("timestamp", -1)])
        await db.interactions.create_index("trace_id", sparse=True)
        await db.outcomes.create_index("borrower_id")
        await db.eval_results.create_index("interaction_id")
        await db.eval_results.create_index("borrower_id")
        await db.eval_pipeline.create_index([("agent_target", 1), ("started_at", -1)])
        await db.prompt_changes.create_index([("agent_name", 1), ("timestamp", -1)])
        await db.regression_events.create_index([("agent_name", 1), ("timestamp", -1)])
        await db.meta_eval_runs.create_index("run_id", unique=True)
        await db.meta_eval_runs.create_index([("started_at", -1)])
        await db.meta_eval_runs.create_index([("target_agents", 1), ("started_at", -1)])
    except Exception as e:
        log.warning(f"[mongo] index creation failed: {e}")


# ══════════════════════════════════════════════════════════════
# REDIS OPERATIONS
# ══════════════════════════════════════════════════════════════

# ── session:{borrower_id} ─────────────────────────────────── #

async def session_get(borrower_id: str) -> Optional[dict]:
    r = await get_redis()
    if not r:
        return None
    try:
        raw = await r.get(f"session:{borrower_id}")
        return json.loads(raw) if raw else None
    except Exception as e:
        log.warning(f"[redis] session_get failed: {e}")
        return None


async def session_set(borrower_id: str, data: dict) -> None:
    r = await get_redis()
    if not r:
        return
    try:
        data["updated_at"] = _now()
        await r.setex(f"session:{borrower_id}", SESSION_TTL, json.dumps(data))
    except Exception as e:
        log.warning(f"[redis] session_set failed: {e}")


async def session_update(borrower_id: str, **kwargs) -> None:
    """Patch specific fields without overwriting the whole session."""
    current = await session_get(borrower_id) or {}
    current.update(kwargs)
    await session_set(borrower_id, current)


# ── summary:{borrower_id} ─────────────────────────────────── #

async def summary_get(borrower_id: str) -> list[dict]:
    r = await get_redis()
    if not r:
        return []
    try:
        raw = await r.get(f"summary:{borrower_id}")
        return json.loads(raw) if raw else []
    except Exception as e:
        log.warning(f"[redis] summary_get failed: {e}")
        return []


async def summary_append(borrower_id: str, entry: dict) -> None:
    r = await get_redis()
    if not r:
        return
    try:
        current = await summary_get(borrower_id)
        current.append(entry)
        await r.setex(f"summary:{borrower_id}", SUMMARY_TTL, json.dumps(current))
    except Exception as e:
        log.warning(f"[redis] summary_append failed: {e}")


# ── retry:{borrower_id} ───────────────────────────────────── #

async def retry_count(borrower_id: str) -> int:
    r = await get_redis()
    if not r:
        return 0
    try:
        val = await r.get(f"retry:{borrower_id}")
        return int(val) if val else 0
    except Exception as e:
        log.warning(f"[redis] retry_count failed: {e}")
        return 0


async def retry_increment(borrower_id: str) -> int:
    r = await get_redis()
    if not r:
        return 0
    try:
        key = f"retry:{borrower_id}"
        n = await r.incr(key)
        await r.expire(key, RETRY_TTL)
        return n
    except Exception as e:
        log.warning(f"[redis] retry_increment failed: {e}")
        return 0


async def retry_reset(borrower_id: str) -> None:
    r = await get_redis()
    if not r:
        return
    try:
        await r.delete(f"retry:{borrower_id}")
    except Exception as e:
        log.warning(f"[redis] retry_reset failed: {e}")


# ── lock:{borrower_id} ────────────────────────────────────── #

async def lock_acquire(borrower_id: str) -> bool:
    """Returns True if lock acquired (SET NX EX). False if already locked."""
    r = await get_redis()
    if not r:
        return True   # degraded mode: no locking
    try:
        result = await r.set(f"lock:{borrower_id}", "locked",
                             nx=True, ex=LOCK_TTL)
        return result is True
    except Exception as e:
        log.warning(f"[redis] lock_acquire failed: {e}")
        return True


async def lock_release(borrower_id: str) -> None:
    r = await get_redis()
    if not r:
        return
    try:
        await r.delete(f"lock:{borrower_id}")
    except Exception as e:
        log.warning(f"[redis] lock_release failed: {e}")


# ══════════════════════════════════════════════════════════════
# MONGODB OPERATIONS (all async, fire-and-forget safe)
# ══════════════════════════════════════════════════════════════

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fire(coro) -> None:
    """Schedule a coroutine without awaiting — used for non-blocking writes."""
    try:
        loop = asyncio.get_event_loop()
        loop.create_task(coro)
    except RuntimeError:
        pass   # no event loop — skip


# ── borrower_cases ────────────────────────────────────────── #

async def upsert_borrower_case(
    borrower_id: str,
    stage: str,
    status: str,
    attempts_used: int = 1,
) -> None:
    db = await get_mongo()
    if db is None:
        return
    try:
        await db.borrower_cases.update_one(
            {"borrower_id": borrower_id},
            {"$set": {
                "current_stage": stage,
                "status": status,
                "attempts_used": attempts_used,
                "updated_at": _now(),
            }, "$setOnInsert": {
                "created_at": _now(),
            }},
            upsert=True,
        )
    except Exception as e:
        log.warning(f"[mongo] upsert_borrower_case failed: {e}")


# ── interactions ──────────────────────────────────────────── #

async def log_interaction(
    borrower_id: str,
    agent_name: str,
    agent_version: str,
    prompt_version: str,
    model: str,
    model_params: dict,
    input_text: str,
    output_text: str,
    input_summary: str = "",
    structured_context: Optional[dict] = None,
    decision: str = "",
    confidence: float = 0.0,
    reasoning_summary: str = "",
    trace_id: Optional[str] = None,
) -> str:
    """Write an interaction record. Returns interaction_id."""
    interaction_id = f"int-{uuid.uuid4().hex[:12]}"
    db = await get_mongo()
    if db is None:
        return interaction_id
    try:
        await db.interactions.insert_one({
            "interaction_id": interaction_id,
            "borrower_id": borrower_id,
            "agent_name": agent_name,
            "agent_version": agent_version,
            "prompt_version": prompt_version,
            "model": model,
            "model_params": model_params,
            "input": input_text,
            "output": output_text,
            "input_summary": input_summary,
            "structured_context": structured_context or {},
            "decision": decision,
            "confidence": confidence,
            "reasoning_summary": reasoning_summary,
            "trace_id": trace_id or interaction_id,
            "timestamp": _now(),
        })
    except Exception as e:
        log.warning(f"[mongo] log_interaction failed: {e}")
    return interaction_id


# ── outcomes ──────────────────────────────────────────────── #

async def log_outcome(
    borrower_id: str,
    outcome: str,                           # AGREEMENT | NO_DEAL | LEGAL
    agent_versions: dict,
    metadata: Optional[dict] = None,
    experiment_id: str = "default",
) -> None:
    db = await get_mongo()
    if db is None:
        return
    try:
        await db.outcomes.update_one(
            {"borrower_id": borrower_id},
            {"$set": {
                "outcome": outcome,
                "agent_versions": agent_versions,
                "experiment_id": experiment_id,
                "metadata": metadata or {},
                "timestamp": _now(),
            }},
            upsert=True,
        )
    except Exception as e:
        log.warning(f"[mongo] log_outcome failed: {e}")


# ── eval_results ──────────────────────────────────────────── #

async def log_eval_result(
    interaction_id: str,
    borrower_id: str,
    judge_model: str,
    score: float,
    dimension_scores: dict,
    feedback: str,
    experiment_id: str = "default",
) -> None:
    db = await get_mongo()
    if db is None:
        return
    try:
        await db.eval_results.insert_one({
            "interaction_id": interaction_id,
            "borrower_id": borrower_id,
            "judge_model": judge_model,
            "score": score,
            "dimension_scores": dimension_scores,
            "feedback": feedback,
            "experiment_id": experiment_id,
            "timestamp": _now(),
        })
    except Exception as e:
        log.warning(f"[mongo] log_eval_result failed: {e}")


# ── Reconstruction (Redis miss → MongoDB fallback) ─────────── #

async def reconstruct_summary_from_mongo(
    borrower_id: str, last_n: int = 5
) -> list[dict]:
    """
    On Redis miss, rebuild summary from last N interactions in MongoDB.
    Called automatically when summary_get returns [].
    """
    db = await get_mongo()
    if db is None:
        return []
    try:
        cursor = db.interactions.find(
            {"borrower_id": borrower_id},
            {"agent_name": 1, "output": 1, "structured_context": 1,
             "timestamp": 1, "_id": 0}
        ).sort("timestamp", -1).limit(last_n)
        docs = await cursor.to_list(length=last_n)
        return [
            {
                "agent": d["agent_name"],
                "agent_version": "reconstructed",
                "summary": d["output"][:200],
                "structured_context": d.get("structured_context", {}),
                "timestamp": d["timestamp"],
            }
            for d in reversed(docs)
        ]
    except Exception as e:
        log.warning(f"[mongo] reconstruct_summary failed: {e}")
        return []


async def get_summary_with_fallback(borrower_id: str) -> list[dict]:
    """
    Redis-first, MongoDB fallback.
    Populates Redis cache after reconstruction.
    """
    summaries = await summary_get(borrower_id)
    if summaries:
        return summaries
    # Redis miss — reconstruct from MongoDB
    summaries = await reconstruct_summary_from_mongo(borrower_id)
    if summaries:
        r = await get_redis()
        if r:
            try:
                await r.setex(
                    f"summary:{borrower_id}", SUMMARY_TTL,
                    json.dumps(summaries)
                )
            except Exception:
                pass
    return summaries
