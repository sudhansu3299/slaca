from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from src.agents.base import BaseAgent, AgentResponse
from src.models import ConversationContext, Stage
from src.token_budget import CostTracker
from src.prompts import final_notice_system_prompt

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

    def get_system_prompt(self, context: ConversationContext) -> str:
        qt = self._get_question_tracker(context)
        known_facts = qt.as_context_str()
        offer_block = self._format_final_offer(context)
        voice_handoff_block = self._format_voice_handoff(context)
        guidance = getattr(self, "_injected_guidance", "")
        return final_notice_system_prompt(
            known_facts,
            offer_block,
            voice_handoff_block,
            guidance,
        )

    async def process(self, context: ConversationContext, user_input: str) -> AgentResponse:
        system = self.get_system_prompt(context)
        context_str = self.format_context_for_agent(context)

        messages = [
            {"role": "user", "content": f"{context_str}\n\nBorrower says: {user_input}"}
        ]

        text, usage = await self._call_claude(system, messages, temperature=0.05)

        outcome = self._parse_outcome(text)
        clean_message = (
            text
            .replace("COLLECTIONS_COMPLETE", "")
            .replace("COLLECTIONS_ESCALATED", "")
            .strip()
        )

        # Also check for explicit escalation signals in user input
        if outcome is None and self._is_escalation_signal(user_input):
            outcome = "escalated"
            clean_message = (
                "This account will now be referred to our legal team. "
                "All consequences outlined above will proceed automatically."
            )

        if outcome == "complete":
            context.final_notice_outcome = "resolved"
            return AgentResponse(
                message=clean_message,
                should_advance=True,
                context_update={"current_stage": Stage.COMPLETE},
                tokens_used=usage,
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
