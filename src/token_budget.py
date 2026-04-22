"""Token budget enforcement and cost tracking for all agents."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

# Claude Opus 4 pricing (per million tokens, as of 2025)
# Input: $15/M  Output: $75/M
OPUS_INPUT_COST_PER_M = 15.0
OPUS_OUTPUT_COST_PER_M = 75.0

MAX_TOKENS_PER_AGENT = 2000       # max output tokens per agent turn
MAX_TOKENS_HANDOFF = 500          # max tokens for handoff summary
TOTAL_COST_BUDGET_USD = 20.0      # hard cap across entire run


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def cost_usd(self) -> float:
        return (
            self.input_tokens / 1_000_000 * OPUS_INPUT_COST_PER_M
            + self.output_tokens / 1_000_000 * OPUS_OUTPUT_COST_PER_M
        )

    def add(self, other: "TokenUsage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens


@dataclass
class CostTracker:
    """Singleton-style tracker; pass one instance through the whole run."""

    per_agent: dict[str, TokenUsage] = field(default_factory=dict)
    _total: TokenUsage = field(default_factory=TokenUsage)

    def record(self, agent: str, usage: TokenUsage) -> None:
        if agent not in self.per_agent:
            self.per_agent[agent] = TokenUsage()
        self.per_agent[agent].add(usage)
        self._total.add(usage)

    @property
    def total_cost_usd(self) -> float:
        return self._total.cost_usd

    @property
    def total_tokens(self) -> TokenUsage:
        return self._total

    def budget_remaining(self) -> float:
        return TOTAL_COST_BUDGET_USD - self.total_cost_usd

    def check_budget(self) -> None:
        if self.total_cost_usd > TOTAL_COST_BUDGET_USD:
            raise BudgetExceededError(
                f"Total LLM cost ${self.total_cost_usd:.4f} exceeds "
                f"${TOTAL_COST_BUDGET_USD} budget"
            )

    def report(self) -> str:
        lines = ["=== Cost Report ==="]
        for agent, usage in self.per_agent.items():
            lines.append(
                f"  {agent}: {usage.input_tokens} in / {usage.output_tokens} out "
                f"= ${usage.cost_usd:.4f}"
            )
        lines.append(
            f"  TOTAL: ${self._total.cost_usd:.4f} / ${TOTAL_COST_BUDGET_USD} budget"
        )
        lines.append(f"  REMAINING: ${self.budget_remaining():.4f}")
        return "\n".join(lines)


class BudgetExceededError(RuntimeError):
    pass


class TokenLimitError(RuntimeError):
    pass


def enforce_output_limit(tokens: int, limit: int = MAX_TOKENS_PER_AGENT, label: str = "") -> None:
    """Raise if actual output exceeded limit."""
    if tokens > limit:
        raise TokenLimitError(
            f"{label} output {tokens} tokens exceeds limit {limit}"
        )


def clamp_max_tokens(requested: int, tracker: CostTracker, label: str = "") -> int:
    """
    Return safe max_tokens based on remaining cost budget.
    Approximates by assuming 3 output tokens cost per input token equivalent.
    """
    remaining_usd = tracker.budget_remaining()
    if remaining_usd <= 0:
        raise BudgetExceededError(f"No budget remaining before {label} call")
    # Conservative: remaining_usd / (OPUS_OUTPUT_COST_PER_M / 1_000_000)
    affordable = int(remaining_usd / (OPUS_OUTPUT_COST_PER_M / 1_000_000))
    return min(requested, affordable, MAX_TOKENS_PER_AGENT)
