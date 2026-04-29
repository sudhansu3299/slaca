"""
PromptBuilder — per-agent input budgets for LLM turns.

Policy:
  - Agent 1: no prior handoff; system prompt may use up to 2000 tokens (trimmed to fit
    the total turn budget with history + user).
  - Agent 2: ≤500 tokens for structured handoff (Agent 1 chat summary); system ≤1500.
  - Agent 3: ≤500 tokens for structured handoff (A1 chat + A2 voice via HandoffBuilder);
    system ≤1500.

Total estimated prompt (system + user message body) stays within
MAX_TOTAL_TOKENS_PER_AGENT_TURN - MIN_RESERVED_OUTPUT_TOKENS so completion can stay
within the API per-call cap.

Strategy when context overflows:
  1. Trim conversation history from the oldest end
  2. Compress supplemental context (non-handoff user block)
  3. Shrink handoff slice
  4. Shrink system below its cap as needed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from src.models import ConversationContext
from src.handoff import HandoffSummary
from src.token_budget import (
    MAX_TOKENS_HANDOFF,
    MAX_TOTAL_TOKENS_PER_AGENT_TURN,
    MIN_RESERVED_OUTPUT_TOKENS,
    estimate_tokens,
)


@dataclass(frozen=True)
class AgentPromptBudget:
    """Caps for one agent. total_limit applies to system + full user content."""

    total_limit: int
    max_system: int
    max_handoff: int  # 0 = Agent 1 (no handoff slice)


# All text sent as system + single user message must fit here (room for completion).
_MAX_AGENT_BODY_TOKENS = MAX_TOTAL_TOKENS_PER_AGENT_TURN - MIN_RESERVED_OUTPUT_TOKENS

AGENT_PROMPT_BUDGETS: dict[str, AgentPromptBudget] = {
    "AssessmentAgent": AgentPromptBudget(_MAX_AGENT_BODY_TOKENS, 2000, 0),
    "ResolutionAgent": AgentPromptBudget(_MAX_AGENT_BODY_TOKENS, 1500, MAX_TOKENS_HANDOFF),
    "FinalNoticeAgent": AgentPromptBudget(_MAX_AGENT_BODY_TOKENS, 1500, MAX_TOKENS_HANDOFF),
}

# Back-compat for tests / introspection (total body budget per agent)
PROMPT_TOKEN_LIMITS: dict[str, int] = {
    name: cfg.total_limit for name, cfg in AGENT_PROMPT_BUDGETS.items()
}


def _shrink_chars_to_token_budget(text: str, max_tokens: int) -> str:
    if max_tokens <= 0:
        return ""
    if estimate_tokens(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        chunk = text[:mid]
        if estimate_tokens(chunk) <= max_tokens:
            best = chunk
            lo = mid + 1
        else:
            hi = mid - 1
    if not best:
        return ""
    suffix = "…" if len(best) < len(text) else ""
    return best + suffix


@dataclass
class PromptComponents:
    system_prompt: str
    context_block: str          # handoff + supplemental (no history)
    history_block: str
    user_input: str
    total_estimated_tokens: int
    was_trimmed: bool = False

    def user_message_body(self) -> str:
        """Single user-turn text for chat.completions."""
        parts: list[str] = []
        if self.context_block.strip():
            parts.append(self.context_block.strip())
        if self.history_block.strip():
            parts.append("--- Conversation so far ---")
            parts.append(self.history_block.strip())
        parts.append(f"Borrower says: {self.user_input}")
        return "\n\n".join(parts)


def build_llm_turn(
    agent_name: str,
    raw_system_prompt: str,
    context: ConversationContext,
    handoff_summary: Optional[HandoffSummary],
    user_input: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Returns (system_prompt, messages) ready for BaseAgent._call_claude[_with_tools].
    """
    pb = PromptBuilder(agent_name)
    comp = pb.build(raw_system_prompt, context, handoff_summary, user_input)
    return comp.system_prompt, [{"role": "user", "content": comp.user_message_body()}]


class PromptBuilder:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.cfg = AGENT_PROMPT_BUDGETS.get(
            agent_name,
            AgentPromptBudget(_MAX_AGENT_BODY_TOKENS, 1500, MAX_TOKENS_HANDOFF),
        )
        self.limit = self.cfg.total_limit

    def build(
        self,
        system_prompt: str,
        context: ConversationContext,
        handoff_summary: Optional[HandoffSummary],
        user_input: str,
    ) -> PromptComponents:
        cfg = self.cfg
        user_line = f"Borrower says: {user_input}"
        slack = 8  # formatting between sections

        handoff_text = ""
        if handoff_summary is not None and cfg.max_handoff > 0:
            handoff_text = handoff_summary.to_prompt_block()
            handoff_text = _shrink_chars_to_token_budget(handoff_text, cfg.max_handoff)

        system = _shrink_chars_to_token_budget(system_prompt, cfg.max_system)
        supplemental, supp_trimmed = self._build_supplemental(context, handoff_summary)
        context_core = "\n\n".join(x for x in (handoff_text, supplemental) if x.strip())

        history_block, hist_trimmed = self._trim_history(
            context.conversation_history,
            self._history_budget(system, context_core, user_line, slack),
        )

        trimmed = hist_trimmed or supp_trimmed
        est = self._total_estimate(system, context_core, history_block, user_line, slack)
        if est > cfg.total_limit:
            system, context_core, history_block, packed = self._fit_turn_to_limit(
                context=context,
                handoff_summary=handoff_summary,
                system=system,
                handoff_raw=handoff_text,
                supplemental=supplemental,
                history_block=history_block,
                user_line=user_line,
                slack=slack,
                cfg=cfg,
            )
            trimmed = trimmed or packed

        return PromptComponents(
            system_prompt=system,
            context_block=context_core,
            history_block=history_block,
            user_input=user_input,
            total_estimated_tokens=self._total_estimate(
                system, context_core, history_block, user_line, slack
            ),
            was_trimmed=trimmed,
        )

    def _total_estimate(
        self, system: str, context_core: str, history_block: str, user_line: str, slack: int
    ) -> int:
        return (
            estimate_tokens(system)
            + estimate_tokens(context_core)
            + estimate_tokens(history_block)
            + estimate_tokens(user_line)
            + slack
        )

    def _history_budget(self, system: str, context_core: str, user_line: str, slack: int) -> int:
        used = (
            estimate_tokens(system)
            + estimate_tokens(context_core)
            + estimate_tokens(user_line)
            + slack
        )
        return self.cfg.total_limit - used

    def _fit_turn_to_limit(
        self,
        *,
        context: ConversationContext,
        handoff_summary: Optional[HandoffSummary],
        system: str,
        handoff_raw: str,
        supplemental: str,
        history_block: str,
        user_line: str,
        slack: int,
        cfg: AgentPromptBudget,
    ) -> tuple[str, str, str, bool]:
        """Shrink pieces until estimated total ≤ cfg.total_limit."""
        trimmed = True

        def total(sys: str, core: str, hist: str) -> int:
            return self._total_estimate(sys, core, hist, user_line, slack)

        handoff_text = handoff_raw
        supp = supplemental
        core = "\n\n".join(x for x in (handoff_text, supp) if x.strip())

        history_block = ""
        if total(system, core, history_block) > cfg.total_limit:
            supp = self._compress_supplemental(context, handoff_summary)
            core = "\n\n".join(x for x in (handoff_text, supp) if x.strip())

        if total(system, core, history_block) > cfg.total_limit and cfg.max_handoff > 0:
            cap = cfg.max_handoff
            while total(system, core, history_block) > cfg.total_limit and cap >= 32:
                handoff_text = _shrink_chars_to_token_budget(handoff_raw, cap)
                core = "\n\n".join(x for x in (handoff_text, supp) if x.strip())
                if cap == 32:
                    break
                cap = max(32, cap // 2)

        if total(system, core, history_block) > cfg.total_limit:
            sys_cap = cfg.max_system
            while sys_cap >= 128 and total(system, core, history_block) > cfg.total_limit:
                system = _shrink_chars_to_token_budget(system, sys_cap)
                if sys_cap == 128:
                    break
                sys_cap = max(128, int(sys_cap * 0.85))

        if total(system, core, history_block) > cfg.total_limit:
            room = self.cfg.total_limit - estimate_tokens(system) - estimate_tokens(history_block)
            room -= estimate_tokens(user_line) + slack
            core = _shrink_chars_to_token_budget(core, max(16, room))

        return system, core, history_block, trimmed

    def _build_supplemental(
        self, context: ConversationContext, handoff_summary: Optional[HandoffSummary]
    ) -> tuple[str, bool]:
        """
        Non-handoff context lines in the user message (stage, ids, offer snapshot, …).
        """
        parts: list[str] = [f"[Stage: {context.current_stage.value}]"]
        parts.append(f"[Borrower ID: {context.borrower_id} | Loan ID: {context.loan_id}]")

        if handoff_summary is None:
            if context.assessment_data:
                ad = context.assessment_data
                parts.append(
                    f"[Outstanding: {ad.outstanding_amount} | DPD: {ad.days_past_due} | "
                    f"Path: {ad.resolution_path.value if ad.resolution_path else 'TBD'}]"
                )
        elif context.resolution_offer:
            parts.append(f"[Resolution offer: {context.resolution_offer.model_dump_json()}]")

        if context.assessment_data and context.assessment_data.resolution_path and handoff_summary is not None:
            parts.append(
                f"[Recommended path: {context.assessment_data.resolution_path.value}]"
            )

        # Compact facts only when not redundant with structured handoff
        if handoff_summary is None and context.question_state:
            answered = {
                k: v["value"]
                for k, v in context.question_state.items()
                if v.get("answered") and v.get("value")
            }
            if answered:
                facts = ", ".join(f"{k}={v}" for k, v in sorted(answered.items())[:24])
                parts.append(f"[Known facts — DO NOT re-ask: {facts}]")

        text = "\n".join(parts)
        max_supp = max(64, self.cfg.total_limit // 4)
        if estimate_tokens(text) > max_supp:
            return self._compress_supplemental(context, handoff_summary), True
        return text, False

    def _compress_supplemental(
        self, context: ConversationContext, handoff_summary: Optional[HandoffSummary]
    ) -> str:
        parts = [
            f"[Stage: {context.current_stage.value}]",
            f"[{context.borrower_id}|{context.loan_id}]",
        ]
        if context.assessment_data:
            ad = context.assessment_data
            parts.append(f"[Owed:{ad.outstanding_amount} DPD:{ad.days_past_due}]")
        if handoff_summary and handoff_summary.identity_verified:
            parts.append("[ID:verified]")
        return " ".join(parts)

    def _trim_history(
        self, history: list[dict], budget_tokens: int
    ) -> tuple[str, bool]:
        if budget_tokens <= 0:
            return "", True
        safe_budget = max(1, int(budget_tokens * 0.95))
        lines = [
            f"{'Borrower' if m.get('role') == 'user' else 'Agent'}: {m.get('content', '')}"
            for m in history
        ]
        included: list[str] = []
        for line in reversed(lines):
            candidate = "\n".join([line] + included)
            if estimate_tokens(candidate) > safe_budget:
                break
            included.insert(0, line)
        trimmed = len(included) < len(lines)
        return "\n".join(included), trimmed

    def assert_within_limit(self, components: PromptComponents) -> None:
        if components.total_estimated_tokens > self.limit:
            raise ValueError(
                f"{self.agent_name} prompt ~{components.total_estimated_tokens} tokens "
                f"exceeds limit {self.limit}"
            )