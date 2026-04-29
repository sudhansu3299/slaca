"""Token budget enforcement and cost tracking for all agents."""

from __future__ import annotations

from dataclasses import dataclass, field

# Claude Opus 4 pricing (per million tokens, as of 2025)
# Input: $15/M  Output: $75/M
OPUS_INPUT_COST_PER_M = 15.0
OPUS_OUTPUT_COST_PER_M = 75.0

# Per chat.completions call: prompt_tokens + completion_tokens must not exceed this.
MAX_TOTAL_TOKENS_PER_AGENT_TURN = 2000
# Back-compat name: same cap, applies to input + output together (not output alone).
MAX_TOKENS_PER_AGENT = MAX_TOTAL_TOKENS_PER_AGENT_TURN

# When sizing prompts, reserve at least this many tokens for the model's reply so
# prompt + max_output can stay within MAX_TOTAL_TOKENS_PER_AGENT_TURN.
MIN_RESERVED_OUTPUT_TOKENS = 64

MAX_TOKENS_HANDOFF = 500          # max tokens for handoff summary
TOTAL_COST_BUDGET_USD = 20.0      # hard cap across entire run

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Rough token count for budgeting (matches PromptBuilder / handoff heuristics)."""
    return max(1, len(text) // CHARS_PER_TOKEN)


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
    """Raise if actual output exceeded limit (output-only check; prefer enforce_total_turn_limit)."""
    if tokens > limit:
        raise TokenLimitError(
            f"{label} output {tokens} tokens exceeds limit {limit}"
        )


def enforce_total_turn_limit(
    input_tokens: int,
    output_tokens: int,
    limit: int = MAX_TOTAL_TOKENS_PER_AGENT_TURN,
    label: str = "",
) -> None:
    """Raise if this API call's prompt + completion usage exceeded the per-turn cap."""
    total = input_tokens + output_tokens
    if total > limit:
        prefix = f"{label} " if label else ""
        raise TokenLimitError(
            f"{prefix}total context {total} tokens (in={input_tokens}, out={output_tokens}) "
            f"exceeds per-turn limit {limit}"
        )


def clamp_max_tokens(
    requested: int,
    tracker: CostTracker,
    label: str = "",
    *,
    estimated_input_tokens: int = 0,
) -> int:
    """
    Return safe max_tokens (completion budget) for the next API call:
    - remaining USD budget
    - requested ceiling
    - room left under MAX_TOTAL_TOKENS_PER_AGENT_TURN for the estimated prompt size
    """
    remaining_usd = tracker.budget_remaining()
    if remaining_usd <= 0:
        raise BudgetExceededError(f"No budget remaining before {label} call")
    # Conservative: remaining_usd / (OPUS_OUTPUT_COST_PER_M / 1_000_000)
    affordable = int(remaining_usd / (OPUS_OUTPUT_COST_PER_M / 1_000_000))
    output_cap = max(1, MAX_TOTAL_TOKENS_PER_AGENT_TURN - max(0, estimated_input_tokens))
    return min(requested, affordable, output_cap)
