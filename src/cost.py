"""
Cost Optimization (MS10)

Strategies:
1. Model tiering      — strong model (Opus) for agents, cheap model (Haiku) for simulation & eval
2. Summary caching    — cache handoff summaries by context hash to avoid recomputation
3. Batch processing   — process multiple borrowers in parallel
4. Token compression  — PromptBuilder trims history before API calls

Cost report:
  - Per-agent token usage
  - Simulation vs production split
  - LLM judge cost (cheap model)
  - Total cost vs $20 budget
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional

from src.token_budget import (
    CostTracker, TokenUsage,
    OPUS_INPUT_COST_PER_M, OPUS_OUTPUT_COST_PER_M,
    TOTAL_COST_BUDGET_USD,
)

# Model pricing (per million tokens)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-4-5": {
        "input": OPUS_INPUT_COST_PER_M,
        "output": OPUS_OUTPUT_COST_PER_M,
    },
    "claude-haiku-4-5": {
        "input": 1.0,     # ~15x cheaper than Opus
        "output": 5.0,
    },
    "claude-sonnet-4-5": {
        "input": 3.0,
        "output": 15.0,
    },
}

# Model assignments
AGENT_MODEL = "claude-opus-4-5"          # production agents
SIMULATION_MODEL = "claude-haiku-4-5"     # synthetic borrower simulation
EVAL_MODEL = "claude-haiku-4-5"           # LLM-as-judge evaluation


def cost_for_model(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["claude-opus-4-5"])
    return (
        input_tokens / 1_000_000 * pricing["input"]
        + output_tokens / 1_000_000 * pricing["output"]
    )


# ──────────────────────────────────────────────────────────── #
# Summary Cache
# ──────────────────────────────────────────────────────────── #

class SummaryCache:
    """
    LRU-style cache for HandoffSummary prompt blocks.
    Key: hash of (borrower_id + question_state + resolution_offer).
    Saves token computation on repeated builds (same context = same summary).
    """

    def __init__(self, max_size: int = 500):
        self._cache: dict[str, str] = {}
        self._hits = 0
        self._misses = 0
        self.max_size = max_size

    def _make_key(self, borrower_id: str, context_fingerprint: str) -> str:
        raw = f"{borrower_id}:{context_fingerprint}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, borrower_id: str, context_fingerprint: str) -> Optional[str]:
        key = self._make_key(borrower_id, context_fingerprint)
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def set(self, borrower_id: str, context_fingerprint: str, prompt_block: str) -> None:
        key = self._make_key(borrower_id, context_fingerprint)
        if len(self._cache) >= self.max_size:
            # Evict oldest (first) entry
            oldest = next(iter(self._cache))
            del self._cache[oldest]
        self._cache[key] = prompt_block

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def stats(self) -> str:
        return f"cache: hits={self._hits} misses={self._misses} hit_rate={self.hit_rate:.1%} size={len(self._cache)}"


# Global singleton cache
SUMMARY_CACHE = SummaryCache()


# ──────────────────────────────────────────────────────────── #
# Extended cost tracker with model tiering
# ──────────────────────────────────────────────────────────── #

@dataclass
class TieredCostTracker:
    """
    Tracks cost separately for:
    - Production agent calls (Opus)
    - Simulation calls (Haiku)
    - Evaluation calls (Haiku)
    """
    production: CostTracker = field(default_factory=CostTracker)
    simulation: CostTracker = field(default_factory=CostTracker)
    evaluation: CostTracker = field(default_factory=CostTracker)

    def record_production(self, agent: str, usage: TokenUsage) -> None:
        self.production.record(agent, usage)

    def record_simulation(self, usage: TokenUsage) -> None:
        self.simulation.record("SimulationEngine", usage)

    def record_evaluation(self, usage: TokenUsage) -> None:
        self.evaluation.record("LLMJudge", usage)

    @property
    def total_cost_usd(self) -> float:
        return (
            self.production.total_cost_usd
            + cost_for_model(
                SIMULATION_MODEL,
                self.simulation.total_tokens.input_tokens,
                self.simulation.total_tokens.output_tokens,
            )
            + cost_for_model(
                EVAL_MODEL,
                self.evaluation.total_tokens.input_tokens,
                self.evaluation.total_tokens.output_tokens,
            )
        )

    def within_budget(self) -> bool:
        return self.total_cost_usd <= TOTAL_COST_BUDGET_USD

    def full_report(self) -> str:
        lines = [
            "=== COST REPORT ===",
            f"Budget: ${TOTAL_COST_BUDGET_USD}",
            f"",
            "--- Production (Opus) ---",
        ]
        for agent, usage in self.production.per_agent.items():
            lines.append(
                f"  {agent}: {usage.input_tokens} in / {usage.output_tokens} out"
                f" = ${usage.cost_usd:.4f}"
            )
        lines.append(f"  Subtotal: ${self.production.total_cost_usd:.4f}")

        lines += [
            f"",
            "--- Simulation (Haiku) ---",
            f"  Tokens: {self.simulation.total_tokens.input_tokens} in / {self.simulation.total_tokens.output_tokens} out",
            f"  Subtotal: ${cost_for_model(SIMULATION_MODEL, self.simulation.total_tokens.input_tokens, self.simulation.total_tokens.output_tokens):.4f}",
            f"",
            "--- Evaluation (Haiku) ---",
            f"  Tokens: {self.evaluation.total_tokens.input_tokens} in / {self.evaluation.total_tokens.output_tokens} out",
            f"  Subtotal: ${cost_for_model(EVAL_MODEL, self.evaluation.total_tokens.input_tokens, self.evaluation.total_tokens.output_tokens):.4f}",
            f"",
            f"TOTAL: ${self.total_cost_usd:.4f} / ${TOTAL_COST_BUDGET_USD}",
            f"STATUS: {'✓ WITHIN BUDGET' if self.within_budget() else '✗ OVER BUDGET'}",
            f"REMAINING: ${TOTAL_COST_BUDGET_USD - self.total_cost_usd:.4f}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────── #
# Batch processor
# ──────────────────────────────────────────────────────────── #

@dataclass
class BatchResult:
    total_borrowers: int
    completed: int
    failed: int
    total_cost_usd: float
    elapsed_seconds: float

    @property
    def cost_per_borrower(self) -> float:
        return self.total_cost_usd / max(self.completed, 1)

    def summary(self) -> str:
        return (
            f"Batch: {self.completed}/{self.total_borrowers} completed, "
            f"{self.failed} failed, "
            f"total=${self.total_cost_usd:.4f}, "
            f"per_borrower=${self.cost_per_borrower:.4f}, "
            f"elapsed={self.elapsed_seconds:.1f}s"
        )


async def run_batch(
    profiles: list,          # list[BorrowerProfile]
    pipeline_factory,        # callable that returns CollectionsPipeline
    input_provider,          # callable(ctx, agent_name) -> str
    max_concurrent: int = 5,
) -> BatchResult:
    """
    Process multiple borrowers in parallel (up to max_concurrent).
    Returns BatchResult.
    """
    import asyncio
    start = time.time()
    semaphore = asyncio.Semaphore(max_concurrent)
    completed = 0
    failed = 0
    total_cost = 0.0

    async def _run_one(profile):
        nonlocal completed, failed, total_cost
        async with semaphore:
            try:
                pipeline = pipeline_factory()
                result = await pipeline.run(
                    borrower_id=profile.borrower_id,
                    loan_id=profile.loan_id,
                    principal_amount=profile.principal_amount,
                    outstanding_amount=profile.outstanding_amount,
                    days_past_due=profile.days_past_due,
                    input_provider=input_provider,
                )
                completed += 1
                total_cost += result.total_cost_usd
            except Exception:
                failed += 1

    await asyncio.gather(*[_run_one(p) for p in profiles])

    return BatchResult(
        total_borrowers=len(profiles),
        completed=completed,
        failed=failed,
        total_cost_usd=total_cost,
        elapsed_seconds=time.time() - start,
    )
