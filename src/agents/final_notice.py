from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

from src.agent_tools import FINAL_NOTICE_TOOLS
from src.agents.base import BaseAgent, AgentResponse
from src.models import ConversationContext, Stage
from src.token_budget import CostTracker, TokenUsage
from src.prompts import final_notice_system_prompt
from src.handoff import HandoffBuilder
from src.prompt_builder import build_llm_turn

CONSEQUENCES = [
    "Immediate credit bureau reporting (7-year negative record)",
    "Legal notice issued within 7 days",
    "Court summons initiated within 30 days",
    "Wage garnishment order sought",
    "Lien placed on registered property/assets",
    "Employment reference contact (if legally permitted)",
]

ESCALATION_SIGNALS = [
    "refuse", "won't pay", "not paying", "cannot pay", "don't care",
    "take me to court", "sue me", "call my lawyer", "talk to my attorney",
    "i have nothing", "bankrupt",
]

ACCEPTANCE_SIGNALS = [
    "yes", "i accept", "accept", "i agree", "agreed", "okay", "ok", "confirmed",
]

REJECTION_SIGNALS = [
    "no", "nope", "not accept", "do not accept", "don't accept", "i refuse",
    "won't pay", "not paying", "cannot pay", "can't pay", "decline",
]

RENEGOTIATION_SIGNALS = [
    "discount", "reduce", "lower", "less", "waive", "better offer",
    "negotiate", "extend", "extra time", "more time", "change terms",
    "another plan", "can you do", "can we do", "rework", "modify",
]


class FinalNoticeAgent(BaseAgent):
    """
    Agent 3 — Consequence-driven closer.

    - States exactly what happens next if payment not received
    - One final offer with hard expiry (reuses the committed offer)
    - Does not argue, persuade, or sympathise
    - Routes to COMPLETE (paid/committed) or ESCALATED (refused/silent)
    """

    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        super().__init__("FinalNoticeAgent", cost_tracker)

    def _compose_system_prompt(self, context: ConversationContext, voice_handoff_block: str) -> str:
        qt = self._get_question_tracker(context)
        known_facts = qt.as_context_str()
        offer_block = self._format_final_offer(context)
        guidance = getattr(self, "_injected_guidance", "")
        return final_notice_system_prompt(
            known_facts,
            offer_block,
            voice_handoff_block,
            guidance,
        )

    def get_system_prompt(self, context: ConversationContext) -> str:
        return self._compose_system_prompt(context, self._format_voice_handoff(context))

    async def process(self, context: ConversationContext, user_input: str) -> AgentResponse:
        stage_open = user_input.strip().startswith("[STAGE_OPEN:")

        # Backward compatibility for existing contexts created before attempt counter.
        if context.final_notice_confirmation_asked and context.final_notice_confirmation_attempts <= 0:
            context.final_notice_confirmation_attempts = 1

        # Code-level confirmation guard (3 attempts total):
        # 1) opening ask
        # 2) re-ask
        # 3) final warning ask (renegotiation rejected here)
        # then escalate if still not accepted
        if not stage_open and context.final_notice_confirmation_asked:
            if self._is_acceptance_signal(user_input):
                contract_result = await self._generate_contract(context)
                contract_html = contract_result.get("contract_html", "")
                context.final_notice_outcome = "resolved"
                if contract_result.get("required_docs"):
                    context.settlement_documents = contract_result["required_docs"]
                return AgentResponse(
                    message="Confirmed. Your settlement is accepted. Goodbye.",
                    should_advance=True,
                    context_update={"current_stage": Stage.COMPLETE},
                    tokens_used=TokenUsage(input_tokens=0, output_tokens=0),
                    metadata={"contract_html": contract_html} if contract_html else {},
                )

            # Explicit disagreement should immediately escalate and trigger
            # the legal notice PDF (handled by temporal layer).
            if self._is_rejection_signal(user_input) or self._is_escalation_signal(user_input):
                context.final_notice_outcome = "escalated"
                return AgentResponse(
                    message=self._build_escalation_message(context),
                    should_advance=True,
                    context_update={"current_stage": Stage.ESCALATED},
                    tokens_used=TokenUsage(input_tokens=0, output_tokens=0),
                )

            attempts = max(context.final_notice_confirmation_attempts, 1)

            if attempts == 1:
                context.final_notice_confirmation_attempts = 2
                context.final_notice_reask_used = True
                return AgentResponse(
                    message=self._build_attempt_two_message(context),
                    should_advance=False,
                    tokens_used=TokenUsage(input_tokens=0, output_tokens=0),
                )

            if attempts == 2:
                context.final_notice_confirmation_attempts = 3
                context.final_notice_reask_used = True
                return AgentResponse(
                    message=self._build_attempt_three_message(
                        context,
                        renegotiation=self._is_renegotiation_signal(user_input),
                    ),
                    should_advance=False,
                    tokens_used=TokenUsage(input_tokens=0, output_tokens=0),
                )

            context.final_notice_outcome = "escalated"
            return AgentResponse(
                message=self._build_escalation_message(context),
                should_advance=True,
                context_update={"current_stage": Stage.ESCALATED},
                tokens_used=TokenUsage(input_tokens=0, output_tokens=0),
            )

        system = self._compose_system_prompt(context, "")
        summary = HandoffBuilder.build(context, Stage.RESOLUTION, Stage.FINAL_NOTICE)
        system, messages = build_llm_turn(self.name, system, context, summary, user_input)

        # Reset tool results for this turn so stale results from previous turns
        # don't bleed into the contract metadata check below
        self._tool_results = {}
        # Tool loop: Claude calls generate_settlement_document when borrower agrees
        text, usage = await self._call_claude_with_tools(
            system, messages, FINAL_NOTICE_TOOLS, temperature=0.05
        )

        outcome = self._parse_outcome(text)
        clean_message = (
            text
            .replace("COLLECTIONS_COMPLETE", "")
            .replace("COLLECTIONS_ESCALATED", "")
            .strip()
        )

        # Stage-open should only present the notice and request explicit
        # confirmation in chat. Never auto-close from handoff context alone.
        if stage_open:
            outcome = None

        asked_before = context.final_notice_confirmation_asked

        # Opening notice must ask confirmation and initialize attempt counter.
        if stage_open and outcome is None:
            if not self._contains_accept_question(clean_message):
                clean_message = f"{clean_message}\n\nDo you agree to this?".strip()
            context.final_notice_confirmation_asked = True
            context.final_notice_confirmation_attempts = max(
                context.final_notice_confirmation_attempts, 1
            )
        elif self._contains_accept_question(clean_message):
            if asked_before:
                # Prevent repeated confirmation prompts unless this is the explicit re-ask path.
                clean_message = self._strip_accept_question(clean_message).strip()
            else:
                context.final_notice_confirmation_asked = True
                context.final_notice_confirmation_attempts = max(
                    context.final_notice_confirmation_attempts, 1
                )

        # Also check for explicit escalation signals in user input
        if outcome is None and self._is_escalation_signal(user_input):
            if context.final_notice_confirmation_asked and context.final_notice_confirmation_attempts == 1:
                context.final_notice_confirmation_attempts = 2
                context.final_notice_reask_used = True
                return AgentResponse(
                    message=self._build_attempt_two_message(context),
                    should_advance=False,
                    tokens_used=usage,
                )
            if context.final_notice_confirmation_asked and context.final_notice_confirmation_attempts == 2:
                context.final_notice_confirmation_attempts = 3
                context.final_notice_reask_used = True
                return AgentResponse(
                    message=self._build_attempt_three_message(context, renegotiation=True),
                    should_advance=False,
                    tokens_used=usage,
                )
            outcome = "escalated"
            clean_message = self._build_escalation_message(context)

        if outcome == "complete":
            context.final_notice_outcome = "resolved"
            # Pull the contract from the tool result and attach it to metadata
            contract_result = self._tool_results.get("generate_settlement_document", {})
            contract_html   = contract_result.get("contract_html", "")
            if not contract_html:
                contract_result = await self._generate_contract(context)
                contract_html = contract_result.get("contract_html", "")
            if contract_html:
                context.settlement_documents = contract_result.get("required_docs", [])
            return AgentResponse(
                message=clean_message or "Confirmed. Your settlement is accepted. Goodbye.",
                should_advance=True,
                context_update={"current_stage": Stage.COMPLETE},
                tokens_used=usage,
                metadata={"contract_html": contract_html} if contract_html else {},
            )

        if outcome == "escalated":
            context.final_notice_outcome = "escalated"
            return AgentResponse(
                message=clean_message,
                should_advance=True,
                context_update={"current_stage": Stage.ESCALATED},
                tokens_used=usage,
            )

        return AgentResponse(
            message=clean_message,
            should_advance=False,
            tokens_used=usage,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _parse_outcome(self, text: str) -> Optional[str]:
        if "COLLECTIONS_COMPLETE" in text:
            return "complete"
        if "COLLECTIONS_ESCALATED" in text:
            return "escalated"
        return None

    def _is_escalation_signal(self, user_input: str) -> bool:
        text = user_input.lower()
        return any(signal in text for signal in ESCALATION_SIGNALS)

    def _is_acceptance_signal(self, user_input: str) -> bool:
        text = user_input.lower().strip()
        if re.fullmatch(r"\s*(yes|y|ok|okay)\s*[.!]?\s*", text):
            return True
        return any(signal in text for signal in ACCEPTANCE_SIGNALS)

    def _is_rejection_signal(self, user_input: str) -> bool:
        text = user_input.lower().strip()
        if re.fullmatch(r"\s*(no|n|nope)\s*[.!]?\s*", text):
            return True
        return any(signal in text for signal in REJECTION_SIGNALS)

    def _is_renegotiation_signal(self, user_input: str) -> bool:
        text = user_input.lower()
        return any(signal in text for signal in RENEGOTIATION_SIGNALS)

    def _contains_accept_question(self, text: str) -> bool:
        t = text.lower()
        return (
            "do you accept" in t
            or "accept this offer" in t
            or "can you confirm you agree" in t
            or "do you agree to this" in t
        )

    def _strip_accept_question(self, text: str) -> str:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        kept = [
            ln for ln in lines
            if not self._contains_accept_question(ln)
        ]
        return "\n".join(kept)

    def _offer_summary(self, context: ConversationContext) -> str:
        offer = context.resolution_offer
        if not offer:
            return "the final offer"

        try:
            deadline = datetime.fromisoformat(offer.valid_until).strftime("%d %b %Y %H:%M UTC")
        except (TypeError, ValueError):
            deadline = "the stated deadline"

        if offer.monthly_payment:
            return (
                f"₹{offer.upfront_required:,.0f} down + ₹{offer.monthly_payment:,.0f}/month "
                f"for {offer.tenure_months} months, deadline {deadline}"
            )
        return f"₹{offer.upfront_required:,.0f} one-time payment, deadline {deadline}"

    def _consequence_document(self) -> str:
        lines = ["LEGAL CONSEQUENCES NOTICE:"]
        for i, item in enumerate(CONSEQUENCES, start=1):
            lines.append(f"{i}. {item}")
        return "\n".join(lines)

    def _build_attempt_two_message(self, context: ConversationContext) -> str:
        return (
            f"Final offer remains unchanged: {self._offer_summary(context)}. "
            "Do you agree to this?"
        )

    def _build_attempt_three_message(self, context: ConversationContext, renegotiation: bool) -> str:
        if renegotiation:
            lead = (
                "Renegotiation is not available at this stage. "
                "If not confirmed this time, this will be escalated immediately."
            )
        else:
            lead = "This is the final confirmation before immediate escalation."
        return (
            f"{lead}\n\n"
            f"Final offer: {self._offer_summary(context)}\n\n"
            f"{self._consequence_document()}\n\n"
            "Do you agree to this now?"
        )

    def _build_escalation_message(self, context: ConversationContext) -> str:
        return (
            "No confirmation received. This account is now escalated to legal. Goodbye.\n\n"
            f"{self._consequence_document()}"
        )

    async def _generate_contract(self, context: ConversationContext) -> dict:
        if not context.resolution_offer or not context.assessment_data:
            return {}

        from src.agent_tools import handle_generate_settlement_document

        offer = context.resolution_offer
        try:
            expiry = datetime.fromisoformat(offer.valid_until).strftime("%d %B %Y, %H:%M UTC")
        except (TypeError, ValueError):
            expiry = offer.valid_until or ""

        return await handle_generate_settlement_document({
            "borrower_id": context.borrower_id,
            "loan_id": context.loan_id,
            "resolution_path": offer.path.value,
            "outstanding_amount": context.assessment_data.outstanding_amount,
            "upfront_amount": offer.upfront_required,
            "monthly_amount": offer.monthly_payment,
            "tenure_months": offer.tenure_months,
            "offer_expiry": expiry,
            "accepted": True,
        })

    def _format_final_offer(self, context: ConversationContext) -> str:
        offer = context.resolution_offer
        if not offer:
            deadline_str = "48 hours from now"
            return (
                f"FINAL OFFER: Full outstanding amount due.\n"
                f"Deadline: {deadline_str}\n"
                "No further extensions available."
            )

        try:
            deadline = datetime.fromisoformat(offer.valid_until)
        except (ValueError, TypeError):
            deadline = None

        deadline_str = deadline.strftime("%d %b %Y %H:%M UTC") if deadline else "as stated"

        lines = [
            f"FINAL OFFER — EXPIRES {deadline_str}",
            f"Type: {offer.path.value.upper()}",
        ]
        if offer.discount_percentage:
            lines.append(f"Settlement: ₹{offer.upfront_required:,.0f} (discount: {offer.discount_percentage}%)")
        if offer.monthly_payment:
            lines.append(
                f"Or: ₹{offer.upfront_required:,.0f} down + "
                f"₹{offer.monthly_payment:,.0f}/month × {offer.tenure_months} months"
            )
        lines.append("This offer will NOT be renewed after expiry.")
        return "\n".join(lines)

    def _format_voice_handoff(self, context: ConversationContext) -> str:
        """
        Provide an explicit handoff block to Agent 3 so the final notice
        can reference resolution call context without re-asking.
        """
        handoff_entry = None
        for msg in reversed(context.conversation_history):
            if msg.get("source") == "voice_call_handoff":
                handoff_entry = msg
                break

        if handoff_entry and handoff_entry.get("content"):
            return f"[VOICE CALL HANDOFF]\n{handoff_entry['content']}"

        # Fallback if explicit handoff message is unavailable.
        resolution_turns = [
            m for m in context.conversation_history
            if m.get("stage") == Stage.RESOLUTION.value
        ]
        if not resolution_turns:
            return "[VOICE CALL HANDOFF]\nNo voice transcript available."

        borrower_lines = [
            m.get("content", "").strip()
            for m in resolution_turns
            if m.get("role") == "user"
        ]
        last_borrower = borrower_lines[-1] if borrower_lines else "No borrower transcript line available."

        return (
            "[VOICE CALL HANDOFF]\n"
            f"Resolution outcome: {context.resolution_outcome or 'unknown'}\n"
            f"Last borrower line on call: {last_borrower[:300]}"
        )
