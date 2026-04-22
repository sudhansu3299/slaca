from __future__ import annotations

import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from anthropic import AsyncAnthropic
from pydantic import BaseModel

from src.models import ConversationContext
from src.token_budget import (
    CostTracker, TokenUsage, clamp_max_tokens, enforce_output_limit,
    MAX_TOKENS_PER_AGENT,
)
from src.question_tracker import QuestionTracker, FactKey

logger = logging.getLogger(__name__)

def _first_nonempty_env(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""

def _normalize_base_url(url: str) -> str:
    if not url:
        return ""
    # Allow env values copied with inline comments, e.g. "host # note".
    url = url.split("#", 1)[0].strip()
    if not url:
        return ""
    if url.startswith(("http://", "https://")):
        return url
    return f"https://{url.lstrip('/')}"


class AgentResponse(BaseModel):
    message: str
    context_update: dict[str, Any] = {}
    should_advance: bool = False
    metadata: dict[str, Any] = {}
    tokens_used: Optional[TokenUsage] = None


class BaseAgent(ABC):
    MODEL = "claude-opus-4-5"

    def __init__(self, name: str, cost_tracker: Optional[CostTracker] = None):
        self.name = name
        self.model = self.MODEL
        self.cost_tracker = cost_tracker or CostTracker()

    # ------------------------------------------------------------------ #
    # Subclasses implement these two
    # ------------------------------------------------------------------ #
    @abstractmethod
    async def process(self, context: ConversationContext, user_input: str) -> AgentResponse:
        pass

    @abstractmethod
    def get_system_prompt(self, context: ConversationContext) -> str:
        """Return system prompt. Context passed so agents can inline known facts."""
        pass

    # ------------------------------------------------------------------ #
    # Shared Claude call with token enforcement
    # ------------------------------------------------------------------ #
    async def _call_claude(
        self,
        system: str,
        messages: list[dict],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, TokenUsage]:
        """
        Call Claude Opus with token budget enforcement.
        Returns (text, usage).
        Raises BudgetExceededError / TokenLimitError on violation.
        """
        self.cost_tracker.check_budget()

        safe_max = clamp_max_tokens(
            max_tokens or MAX_TOKENS_PER_AGENT,
            self.cost_tracker,
            label=self.name,
        )

        opencode_key = _first_nonempty_env("OPENCODE_API_KEY")
        openai_key = _first_nonempty_env("OPENAI_API_KEY")
        anthropic_key = _first_nonempty_env("ANTHROPIC_API_KEY")
        opencode_base_url = _normalize_base_url(_first_nonempty_env("OPENCODE_BASE_URL"))
        openai_base_url = _normalize_base_url(_first_nonempty_env("OPENAI_BASE_URL"))
        anthropic_base_url = _normalize_base_url(_first_nonempty_env("ANTHROPIC_BASE_URL"))

        for env_name in ("OPENCODE_BASE_URL", "OPENAI_BASE_URL", "ANTHROPIC_BASE_URL"):
            if not os.getenv(env_name, "").strip():
                os.environ.pop(env_name, None)

        key_source = (
            "opencode"
            if opencode_key
            else ("openai" if openai_key else ("anthropic" if anthropic_key else "none"))
        )
        base_url = (
            opencode_base_url
            if opencode_key
            else (openai_base_url if openai_key else anthropic_base_url)
        )
        logger.info(
            "[llm] provider_key=%s base_url_override=%s",
            key_source,
            "yes" if bool(base_url) else "no",
        )
        if not (opencode_key or openai_key or anthropic_key):
            raise RuntimeError(
                "Missing LLM API key. Set one of OPENCODE_API_KEY, OPENAI_API_KEY, or ANTHROPIC_API_KEY."
            )

        api_key = opencode_key or openai_key or anthropic_key
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncAnthropic(**client_kwargs)
        response = await client.messages.create(
            model=self.model,
            max_tokens=safe_max,
            temperature=temperature,
            system=system,
            messages=messages,
        )
        text = response.content[0].text
        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )

        enforce_output_limit(usage.output_tokens, MAX_TOKENS_PER_AGENT, label=self.name)
        self.cost_tracker.record(self.name, usage)

        return text, usage

    # ------------------------------------------------------------------ #
    # Context helpers
    # ------------------------------------------------------------------ #
    def _get_question_tracker(self, context: ConversationContext) -> QuestionTracker:
        """Deserialise QuestionTracker from context.question_state."""
        qt = QuestionTracker()
        for key_str, state in context.question_state.items():
            try:
                fk = FactKey(key_str)
                if state.get("asked"):
                    qt.mark_asked(fk, state.get("asked_by", self.name), state.get("stage", ""))
                if state.get("answered") and state.get("value") is not None:
                    qt.mark_answered(fk, state["value"])
            except ValueError:
                pass
        return qt

    def _save_question_tracker(self, context: ConversationContext, qt: QuestionTracker) -> None:
        context.question_state = {
            k: {
                "asked": v.asked,
                "answered": v.answered,
                "value": v.value,
                "asked_by": v.asked_by,
                "stage": v.stage,
            }
            for k, v in qt.facts.items()
        }

    def format_context_for_agent(self, context: ConversationContext) -> str:
        """Build the user-turn context block injected before each agent call."""
        history = context.conversation_history[-8:]   # last 8 turns to stay lean
        history_str = "\n".join(
            f"{'Borrower' if m['role'] == 'user' else 'Agent'}: {m['content']}"
            for m in history
        )

        qt = self._get_question_tracker(context)
        known_facts = qt.as_context_str()

        parts = [
            f"[Stage: {context.current_stage.value}]",
            f"[Borrower ID: {context.borrower_id} | Loan ID: {context.loan_id}]",
            "",
            known_facts,
            "",
            "--- Conversation so far ---",
            history_str,
        ]

        if context.assessment_data and context.assessment_data.resolution_path:
            parts += ["", f"[Recommended path: {context.assessment_data.resolution_path.value}]"]

        if context.resolution_offer:
            parts += ["", f"[Resolution offer: {context.resolution_offer.model_dump_json()}]"]

        return "\n".join(parts)
