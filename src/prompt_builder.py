"""
PromptBuilder — enforces per-agent prompt token limits.

From milestone spec:
  Agent 2 (Resolution/Voice) → max 1500 prompt tokens
  Agent 3 (FinalNotice/Chat) → max 1500 prompt tokens
  Handoff summary            → max 500 tokens
  Agent output               → max 2000 tokens

Strategy when context overflows:
  1. Keep system prompt (non-negotiable)
  2. Keep handoff summary (non-negotiable)
  3. Truncate conversation history from the oldest end
  4. If still over, compress financial facts to single line
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.models import ConversationContext, Stage
from src.handoff import HandoffSummary
from src.token_budget import MAX_TOKENS_HANDOFF

# Per-agent prompt (input) token limits
PROMPT_TOKEN_LIMITS: dict[str, int] = {
    "AssessmentAgent":  2500,   # chat — more context is useful
    "ResolutionAgent":  1500,   # voice — tight, keep it short
    "FinalNoticeAgent": 1500,   # chat — tight, consequence-focused
}

CHARS_PER_TOKEN = 4             # same estimate used in handoff.py


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // CHARS_PER_TOKEN)


@dataclass
class PromptComponents:
    system_prompt: str
    context_block: str          # handoff + known facts
    history_block: str          # trimmed conversation history
    total_estimated_tokens: int
    was_trimmed: bool = False


class PromptBuilder:
    """
    Builds the prompt for a given agent, enforcing the token limit.
    Trims history from oldest end when over budget.
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.limit = PROMPT_TOKEN_LIMITS.get(agent_name, 2000)

    def build(
        self,
        system_prompt: str,
        context: ConversationContext,
        handoff_summary: Optional[HandoffSummary],
        user_input: str,
    ) -> PromptComponents:
        """
        Returns trimmed prompt components that fit within the token limit.
        """
        system_tokens = estimate_tokens(system_prompt)
        user_tokens = estimate_tokens(user_input)

        # Build handoff/context block
        context_block = self._build_context_block(context, handoff_summary)
        context_tokens = estimate_tokens(context_block)

        # Budget remaining for history
        history_budget = self.limit - system_tokens - context_tokens - user_tokens
        if history_budget < 0:
            # Even without history, we're over — compress context block
            context_block = self._compress_context_block(context, handoff_summary)
            context_tokens = estimate_tokens(context_block)
            history_budget = self.limit - system_tokens - context_tokens - user_tokens

        history_block, trimmed = self._trim_history(
            context.conversation_history, history_budget
        )

        total = system_tokens + context_tokens + estimate_tokens(history_block) + user_tokens

        return PromptComponents(
            system_prompt=system_prompt,
            context_block=context_block,
            history_block=history_block,
            total_estimated_tokens=total,
            was_trimmed=trimmed,
        )

    def _build_context_block(
        self,
        context: ConversationContext,
        handoff_summary: Optional[HandoffSummary],
    ) -> str:
        parts = []

        if handoff_summary:
            parts.append(handoff_summary.to_prompt_block())
        else:
            # No handoff: build minimal context from ConversationContext
            parts.append(f"[Borrower: {context.borrower_id} | Loan: {context.loan_id}]")
            if context.assessment_data:
                ad = context.assessment_data
                parts.append(
                    f"[Outstanding: {ad.outstanding_amount} | DPD: {ad.days_past_due} | "
                    f"Path: {ad.resolution_path.value if ad.resolution_path else 'TBD'}]"
                )

        if context.question_state:
            answered = {
                k: v["value"]
                for k, v in context.question_state.items()
                if v.get("answered") and v.get("value")
            }
            if answered:
                facts = ", ".join(f"{k}={v}" for k, v in answered.items())
                parts.append(f"[Known facts — DO NOT re-ask: {facts}]")

        return "\n".join(parts)

    def _compress_context_block(
        self,
        context: ConversationContext,
        handoff_summary: Optional[HandoffSummary],
    ) -> str:
        """Ultra-compact fallback when even the normal block is too large."""
        parts = [f"[Borrower:{context.borrower_id} Loan:{context.loan_id}]"]
        if context.assessment_data:
            ad = context.assessment_data
            parts.append(f"[Owed:{ad.outstanding_amount} DPD:{ad.days_past_due}]")
        if handoff_summary and handoff_summary.identity_verified:
            parts.append("[ID:verified]")
        if handoff_summary and handoff_summary.resolution_path:
            parts.append(f"[Path:{handoff_summary.resolution_path}]")
        return " ".join(parts)

    def _trim_history(
        self, history: list[dict], budget_tokens: int
    ) -> tuple[str, bool]:
        """
        Return the most recent turns that fit in budget_tokens.
        Returns (formatted_history, was_trimmed).
        Uses a 5% safety margin to account for join overhead.
        """
        if budget_tokens <= 0:
            return "", True

        # 5% safety margin
        safe_budget = int(budget_tokens * 0.95)

        lines = [
            f"{'Borrower' if m['role'] == 'user' else 'Agent'}: {m.get('content', '')}"
            for m in history
        ]

        # Start from most recent, include as many as fit
        included = []
        for line in reversed(lines):
            candidate = "\n".join([line] + included)
            if estimate_tokens(candidate) > safe_budget:
                break
            included.insert(0, line)

        trimmed = len(included) < len(lines)
        return "\n".join(included), trimmed

    def assert_within_limit(self, components: PromptComponents) -> None:
        """Raise if assembled prompt is over limit."""
        if components.total_estimated_tokens > self.limit:
            raise ValueError(
                f"{self.agent_name} prompt ~{components.total_estimated_tokens} tokens "
                f"exceeds limit {self.limit}"
            )
