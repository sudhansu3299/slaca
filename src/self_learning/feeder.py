"""
Auto-feeder for the self-improvement pipeline.

Tracks completed conversations via a Redis counter and automatically triggers
the 6-stage improvement pipeline for ALL agents every N conversations.
Also exposes trigger_now() for on-demand / manual runs.

Additionally tracks per-agent transcript counters in Redis so auto-triggering
uses real per-agent data volume rather than only a global call count.

Configurable constants at the top of this file:
  TRIGGER_EVERY_N_CONVERSATIONS  — auto-fire threshold (default: 30)
  TRANSCRIPT_BATCH_SIZE          — how many recent transcripts to feed
  ALL_AGENTS                     — which agents get a pipeline run
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Optional

log = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────── #
def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


TRIGGER_EVERY_N_CONVERSATIONS: int = max(1, _int_env("AUTO_PIPELINE_TRIGGER_EVERY", 30))
TRANSCRIPT_BATCH_SIZE: int = max(1, _int_env("AUTO_PIPELINE_TRANSCRIPT_BATCH", 30))
ALL_AGENTS: list[str] = [
    "AssessmentAgent",
    "ResolutionAgent",
    "FinalNoticeAgent",
]

_COUNTER_KEY = "pipeline:call_count"
_COUNTER_TTL = 7 * 24 * 3600   # 7 days — survives restarts
_AGENT_COUNTER_PREFIX = "pipeline:agent_count:"


def _agent_counter_key(agent_name: str) -> str:
    return f"{_AGENT_COUNTER_PREFIX}{agent_name}"


async def _read_agent_counters(r) -> dict[str, int]:
    counts: dict[str, int] = {}
    for agent in ALL_AGENTS:
        raw = await r.get(_agent_counter_key(agent))
        counts[agent] = int(raw) if raw else 0
    return counts


async def on_conversation_complete(agents_seen: Optional[list[str]] = None) -> None:
    """
    Call after every completed conversation (any outcome).
    Increments the Redis counter; fires the pipeline when threshold is reached.
    Safe to call fire-and-forget — all errors are swallowed.
    """
    from src.data_layer import get_redis

    r = await get_redis()
    if r is None:
        log.warning("[feeder] Redis unavailable — skipping auto-trigger check")
        return

    try:
        count = await r.incr(_COUNTER_KEY)
        await r.expire(_COUNTER_KEY, _COUNTER_TTL)

        valid_agents = sorted({
            a for a in (agents_seen or []) if a in ALL_AGENTS
        })
        for agent in valid_agents:
            await r.incr(_agent_counter_key(agent))
            await r.expire(_agent_counter_key(agent), _COUNTER_TTL)

        per_agent = await _read_agent_counters(r)
        log.info(
            "[feeder] call_count=%d per_agent=%s trigger_at=%d",
            count,
            per_agent,
            TRIGGER_EVERY_N_CONVERSATIONS,
        )

        trigger_agents = [
            agent for agent, n in per_agent.items()
            if n >= TRIGGER_EVERY_N_CONVERSATIONS
        ]
        if trigger_agents:
            for agent in trigger_agents:
                overflow = max(0, per_agent.get(agent, 0) - TRIGGER_EVERY_N_CONVERSATIONS)
                if overflow:
                    await r.setex(_agent_counter_key(agent), _COUNTER_TTL, str(overflow))
                else:
                    await r.delete(_agent_counter_key(agent))

            # Keep this counter as "since last auto-trigger run" for UX clarity.
            await r.delete(_COUNTER_KEY)
            log.info(
                "[feeder] threshold reached for agents=%s — auto-triggering pipeline",
                trigger_agents,
            )
            asyncio.create_task(_fetch_and_run(triggered_by="auto", agents=trigger_agents))

    except Exception as e:
        log.warning("[feeder] on_conversation_complete error: %s", e)


async def trigger_now(
    agent_name: Optional[str] = None,
    triggered_by: str = "manual",
) -> dict:
    """
    On-demand trigger. Runs pipeline for one specific agent or all agents.
    Returns immediately — pipeline runs as a background task.
    """
    agents = [agent_name] if agent_name else ALL_AGENTS
    log.info("[feeder] on-demand trigger agents=%s by=%s", agents, triggered_by)
    asyncio.create_task(_fetch_and_run(triggered_by=triggered_by, agents=agents))
    return {
        "triggered": agents,
        "triggered_by": triggered_by,
        "transcript_batch_size": TRANSCRIPT_BATCH_SIZE,
        "status": "started",
    }


async def get_counter() -> int:
    """Return current call counter value (useful for status endpoints)."""
    from src.data_layer import get_redis

    r = await get_redis()
    if r is None:
        return -1
    try:
        val = await r.get(_COUNTER_KEY)
        return int(val) if val else 0
    except Exception:
        return -1


async def get_agent_counters() -> dict[str, int]:
    """Return per-agent transcript counters used by auto-trigger logic."""
    from src.data_layer import get_redis

    r = await get_redis()
    if r is None:
        return {agent: -1 for agent in ALL_AGENTS}

    try:
        return await _read_agent_counters(r)
    except Exception:
        return {agent: -1 for agent in ALL_AGENTS}


async def _fetch_and_run(
    triggered_by: str = "auto",
    agents: Optional[list[str]] = None,
) -> None:
    """Fetch last N transcripts from MongoDB, run pipeline for each agent."""
    from src.self_learning.improvement_pipeline import run_improvement_pipeline
    from src.admin_api import _fetch_last_n_transcripts

    if agents is None:
        agents = ALL_AGENTS

    try:
        # Default to real v2 replay (live LLM); pipeline rejects non-real mode.
        os.environ.setdefault("REAL_V2_EXECUTION_MODE", "real")
        os.environ.setdefault("REPLAY_BORROWER_MODE", "simulator")

        transcripts = await _fetch_last_n_transcripts(TRANSCRIPT_BATCH_SIZE)
        if not transcripts:
            log.warning("[feeder] no transcripts in MongoDB — skipping pipeline run")
            return

        log.info("[feeder] fetched %d transcripts, running for agents=%s",
                 len(transcripts), agents)

        for agent in agents:
            try:
                await run_improvement_pipeline(transcripts, agent, triggered_by)
                log.info("[feeder] pipeline complete for %s", agent)
            except Exception as e:
                log.exception("[feeder] pipeline failed for %s: %s", agent, e)

    except Exception as e:
        log.exception("[feeder] _fetch_and_run failed: %s", e)
