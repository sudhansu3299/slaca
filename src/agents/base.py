from __future__ import annotations

import json
import os
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel

from src.models import ConversationContext
from src.token_budget import (
    CostTracker,
    TokenUsage,
    clamp_max_tokens,
    enforce_total_turn_limit,
    MAX_TOKENS_PER_AGENT,
    estimate_tokens,
)
from src.question_tracker import QuestionTracker, FactKey

logger = logging.getLogger(__name__)

AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.2"))

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


def _resolve_llm_provider() -> tuple[str, str, str]:
    """
    Resolve (provider, api_key, base_url) for agent LLM calls.
    Priority keeps DeepSeek/OpenRouter opt-in simple for replay runs.
    """
    deepseek_key = _first_nonempty_env("DEEPSEEK_API_KEY")
    opencode_key = _first_nonempty_env("OPENCODE_API_KEY")
    openai_key = _first_nonempty_env("OPENAI_API_KEY")
    if deepseek_key:
        base_url = _normalize_base_url(
            _first_nonempty_env("DEEPSEEK_BASE_URL") or "https://openrouter.ai/api/v1"
        )
        return "deepseek", deepseek_key, base_url
    if opencode_key:
        return "opencode", opencode_key, _normalize_base_url(_first_nonempty_env("OPENCODE_BASE_URL"))
    if openai_key:
        return "openai", openai_key, _normalize_base_url(_first_nonempty_env("OPENAI_BASE_URL"))
    return "none", "", ""


def _build_chat_extra_body(model: str) -> dict[str, Any]:
    if "deepseek" in str(model).lower():
        return {"reasoning": {"enabled": True}}
    return {}


class AgentResponse(BaseModel):
    message: str
    context_update: dict[str, Any] = {}
    should_advance: bool = False
    metadata: dict[str, Any] = {}
    tokens_used: Optional[TokenUsage] = None


class BaseAgent(ABC):
    MODEL = os.getenv("AGENT_MODEL", "gpt-4o")

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
    # Shared LLM call with token enforcement
    # ------------------------------------------------------------------ #
    def _build_openai_messages(self, system: str, messages: list[dict]) -> list[dict[str, Any]]:
        built: list[dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, list):
                content = json.dumps(content)
            built.append({"role": m.get("role", "user"), "content": content})
        return built

    def _estimate_openai_prompt_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Heuristic prompt size for budgeting (API tokenizer may differ slightly)."""
        total = 0
        for m in messages:
            c = m.get("content", "")
            if isinstance(c, list):
                c = json.dumps(c)
            elif c is None:
                c = ""
            elif not isinstance(c, str):
                c = str(c)
            total += estimate_tokens(c)
            tc = m.get("tool_calls")
            if tc:
                total += estimate_tokens(json.dumps(tc))
        return total

    def _build_openai_tools(self, tools: list[dict]) -> list[dict[str, Any]]:
        openai_tools: list[dict[str, Any]] = []
        for tool in tools:
            openai_tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("input_schema", {"type": "object", "properties": {}}),
                    },
                }
            )
        return openai_tools

    @staticmethod
    def _content_to_text(content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts: list[str] = []
            for part in content:
                if isinstance(part, str):
                    parts.append(part)
                elif isinstance(part, dict) and isinstance(part.get("text"), str):
                    parts.append(part["text"])
            return "\n".join(parts)
        return str(content)

    async def _call_claude(
        self,
        system: str,
        messages: list[dict],
        temperature: float = AGENT_TEMPERATURE,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, TokenUsage]:
        """
        Call configured chat model with token budget enforcement.
        Returns (text, usage).
        Raises BudgetExceededError / TokenLimitError on violation.
        """
        self.cost_tracker.check_budget()

        built = self._build_openai_messages(system, messages)
        est_in = self._estimate_openai_prompt_tokens(built)
        safe_max = clamp_max_tokens(
            max_tokens or MAX_TOKENS_PER_AGENT,
            self.cost_tracker,
            label=self.name,
            estimated_input_tokens=est_in,
        )

        _provider, api_key, base_url = _resolve_llm_provider()
        for env_name in ("DEEPSEEK_BASE_URL", "OPENCODE_BASE_URL", "OPENAI_BASE_URL"):
            if not os.getenv(env_name, "").strip():
                os.environ.pop(env_name, None)
        logger.info(
            "[llm] provider_key=%s base_url_override=%s",
            provider,
            "yes" if bool(base_url) else "no",
        )
        if not api_key:
            raise RuntimeError(
                "Missing LLM API key. Set one of DEEPSEEK_API_KEY, OPENCODE_API_KEY, or OPENAI_API_KEY."
            )

        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncOpenAI(**client_kwargs)
        request_kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": safe_max,
            "temperature": temperature,
            "messages": built,
        }
        extra_body = _build_chat_extra_body(self.model)
        if extra_body:
            request_kwargs["extra_body"] = extra_body
        response = await client.chat.completions.create(**request_kwargs)
        message = response.choices[0].message if response.choices else None
        text = self._content_to_text(message.content if message else "")
        usage = TokenUsage(
            input_tokens=(response.usage.prompt_tokens if response.usage else 0),
            output_tokens=(response.usage.completion_tokens if response.usage else 0),
        )

        if response.usage:
            enforce_total_turn_limit(
                response.usage.prompt_tokens or 0,
                response.usage.completion_tokens or 0,
                label=self.name,
            )
        self.cost_tracker.record(self.name, usage)

        return text, usage

    async def _call_claude_with_tools(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict],
        temperature: float = AGENT_TEMPERATURE,
        max_tokens: Optional[int] = None,
    ) -> tuple[str, TokenUsage]:
        """
        Call chat model with tool-use support.

        Runs the full agentic loop:
          1. Send messages + tool schemas to the model
          2. If model returns tool calls, execute each tool
          3. Append tool results to the message list and re-call Claude
          4. Repeat until Claude returns a plain text response (no tool calls)

        Returns (final_text, aggregated_usage).
        """
        from src.agent_tools import execute_tool

        # Compatibility flag for test/mocked paths that patch _call_claude
        # but do not execute the structured tool loop.
        self._tool_loop_fallback = False

        self.cost_tracker.check_budget()

        provider, api_key, base_url = _resolve_llm_provider()
        if not api_key:
            self._tool_loop_fallback = True
            logger.info(
                "[llm] no api key for tool loop; falling back to _call_claude"
            )
            return await self._call_claude(
                system,
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        client_kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncOpenAI(**client_kwargs)

        total_input = 0
        total_output = 0
        current_messages = self._build_openai_messages(system, messages)
        openai_tools = self._build_openai_tools(tools)

        # Agentic tool loop — max 5 rounds to prevent runaway calls
        for _round in range(5):
            tools_blob = json.dumps(openai_tools)
            est_in = self._estimate_openai_prompt_tokens(current_messages) + estimate_tokens(tools_blob)
            safe_max = clamp_max_tokens(
                max_tokens or MAX_TOKENS_PER_AGENT,
                self.cost_tracker,
                label=self.name,
                estimated_input_tokens=est_in,
            )
            request_kwargs = {
                "model": self.model,
                "max_tokens": safe_max,
                "temperature": temperature,
                "messages": current_messages,
                "tools": openai_tools,
                "tool_choice": "auto",
            }
            extra_body = _build_chat_extra_body(self.model)
            if extra_body:
                request_kwargs["extra_body"] = extra_body
            response = await client.chat.completions.create(**request_kwargs)
            if response.usage:
                enforce_total_turn_limit(
                    response.usage.prompt_tokens or 0,
                    response.usage.completion_tokens or 0,
                    label=self.name,
                )
                total_input += response.usage.prompt_tokens or 0
                total_output += response.usage.completion_tokens or 0

            message = response.choices[0].message if response.choices else None
            tool_calls = list(message.tool_calls or []) if message else []

            if not tool_calls:
                # No more tool calls — extract the final text and stop
                final_text = self._content_to_text(message.content if message else "")
                break

            # Execute all tool calls in this round
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [tc.model_dump() for tc in tool_calls],
            }
            reasoning_details = getattr(message, "reasoning_details", None)
            if reasoning_details is not None:
                assistant_msg["reasoning_details"] = reasoning_details
            current_messages.append(assistant_msg)

            tool_results = []
            for call in tool_calls:
                name = call.function.name if call.function else ""
                args_raw = call.function.arguments if call.function else "{}"
                logger.info("[tool_call] %s → %s", self.name, name)
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    args = {}
                result = await execute_tool(name, args)
                # Store every tool result on self so subclasses can inspect them
                if not hasattr(self, "_tool_results"):
                    self._tool_results = {}
                self._tool_results[name] = result
                tool_results.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result),
                })

            # Append tool results to history for the next model turn
            current_messages.extend(tool_results)

        else:
            # Exhausted rounds without a plain-text response — use last text found
            final_text = self._content_to_text(message.content if message else "")

        usage = TokenUsage(input_tokens=total_input, output_tokens=total_output)
        self.cost_tracker.record(self.name, usage)

        return final_text, usage

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
