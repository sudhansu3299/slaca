from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from src.agent_tools import RESOLUTION_TOOLS
from src.agents.base import BaseAgent, AgentResponse
from src.models import (
    ConversationContext, Stage, ResolutionPath, ResolutionOffer, BorrowerBehaviour
)
from src.token_budget import CostTracker
from src.prompts import resolution_system_prompt

# Policy-defined offer ranges
OFFER_POLICY = {
    ResolutionPath.LUMP_SUM: {
        "discount_pct_min": 10,
        "discount_pct_max": 25,
        "deadline_days": 5,
    },
    ResolutionPath.INSTALLMENT: {
        "down_pct": 15,
        "monthly_divisor": 10,   # outstanding / 10
        "tenure_months": 10,
        "deadline_days": 3,
    },
    ResolutionPath.HARDSHIP: {
        "down_pct": 5,
        "monthly_divisor": 24,
        "tenure_months": 6,
        "deadline_days": 3,
    },
    ResolutionPath.LEGAL: {
        "deadline_days": 7,
    },
}


class ResolutionAgent(BaseAgent):
    """
    Agent 2 — Voice-based transactional dealmaker.

    - Picks up the conversation seamlessly from Assessment handoff
    - Presents policy-bound offer (lump sum / installment / hardship)
    - Handles objections by restating terms, not comforting
    - Pushes for explicit commitment
    - Hard cap: 2000 output tokens per turn
    """

    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        super().__init__("ResolutionAgent", cost_tracker)

    def get_system_prompt(self, context: ConversationContext) -> str:
        qt = self._get_question_tracker(context)
        known_facts = qt.as_context_str()
        offer_block = self._format_offer_for_prompt(context)
        guidance = getattr(self, "_injected_guidance", "")
        return resolution_system_prompt(known_facts, offer_block, guidance)

    def generate_offer(self, context: ConversationContext) -> ResolutionOffer:
        ad = context.assessment_data
        if not ad or not ad.resolution_path:
            path = ResolutionPath.LEGAL
            outstanding = ad.outstanding_amount if ad else 0
        else:
            path = ad.resolution_path
            outstanding = ad.outstanding_amount

        policy = OFFER_POLICY.get(path, OFFER_POLICY[ResolutionPath.LEGAL])
        deadline_days = policy["deadline_days"]
        valid_until = (datetime.now(timezone.utc) + timedelta(days=deadline_days)).isoformat()

        if path == ResolutionPath.LUMP_SUM:
            # Discount: scale within 10–25% based on amount
            discount_pct = min(
                policy["discount_pct_max"],
                max(policy["discount_pct_min"], int(outstanding / 10_000))
            )
            return ResolutionOffer(
                path=path,
                discount_percentage=float(discount_pct),
                upfront_required=round(outstanding * (1 - discount_pct / 100), 2),
                deadline_days=deadline_days,
                valid_until=valid_until,
            )

        elif path == ResolutionPath.INSTALLMENT:
            down = round(outstanding * policy["down_pct"] / 100, 2)
            remaining = outstanding - down
            monthly = round(remaining / policy["monthly_divisor"], 2)
            return ResolutionOffer(
                path=path,
                upfront_required=down,
                monthly_payment=monthly,
                tenure_months=policy["tenure_months"],
                deadline_days=deadline_days,
                valid_until=valid_until,
            )

        elif path == ResolutionPath.HARDSHIP:
            down = round(outstanding * policy["down_pct"] / 100, 2)
            remaining = outstanding - down
            monthly = round(remaining / policy["monthly_divisor"], 2)
            return ResolutionOffer(
                path=path,
                upfront_required=down,
                monthly_payment=monthly,
                tenure_months=policy["tenure_months"],
                deadline_days=deadline_days,
                valid_until=valid_until,
            )

        else:  # LEGAL
            return ResolutionOffer(
                path=ResolutionPath.LEGAL,
                deadline_days=deadline_days,
                valid_until=valid_until,
            )

    async def process(self, context: ConversationContext, user_input: str) -> AgentResponse:
        # Ensure offer exists
        if not context.resolution_offer:
            context.resolution_offer = self.generate_offer(context)

        system = self.get_system_prompt(context)
        context_str = self.format_context_for_agent(context)

        messages = [
            {"role": "user", "content": f"{context_str}\n\nBorrower says: {user_input}"}
        ]

        # Voice turns should be short: cap at 300 tokens.
        # Tool loop is used so Claude can call classify_borrower_behaviour when ready.
        text, usage = await self._call_claude_with_tools(
            system, messages, RESOLUTION_TOOLS, temperature=0.15, max_tokens=300
        )

        should_advance, refused = self._parse_outcome(text)
        clean_message = (
            text
            .replace("RESOLUTION_COMPLETE", "")
            .replace("RESOLUTION_REFUSED", "")
            .strip()
        )

        # Persist behaviour classification extracted by the tool call
        behaviour_label = getattr(self, "_last_behaviour", None)
        if behaviour_label:
            try:
                context.borrower_behaviour = BorrowerBehaviour(behaviour_label)
            except ValueError:
                pass

        if should_advance:
            context.resolution_outcome = "committed"
            return AgentResponse(
                message=clean_message,
                should_advance=True,
                context_update={
                    "current_stage": Stage.FINAL_NOTICE,
                    "resolution_outcome": "committed",
                },
                tokens_used=usage,
            )

        if refused:
            context.resolution_outcome = "refused"
            return AgentResponse(
                message=clean_message,
                should_advance=True,
                context_update={
                    "current_stage": Stage.FINAL_NOTICE,
                    "resolution_outcome": "refused",
                },
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

    def _parse_outcome(self, text: str) -> tuple[bool, bool]:
        """Returns (committed, refused)."""
        committed = "RESOLUTION_COMPLETE" in text
        refused = "RESOLUTION_REFUSED" in text
        return committed, refused

    def _format_offer_for_prompt(self, context: ConversationContext) -> str:
        offer = context.resolution_offer
        if not offer:
            return "[Offer not yet generated — generate before first turn]"

        lines = [f"CURRENT OFFER ({offer.path.value.upper()}):"]
        if offer.discount_percentage:
            lines.append(f"  Settlement discount: {offer.discount_percentage}%")
            lines.append(f"  Amount due: {offer.upfront_required}")
        if offer.monthly_payment:
            lines.append(f"  Down payment: {offer.upfront_required}")
            lines.append(f"  Monthly: {offer.monthly_payment} x {offer.tenure_months} months")
        lines.append(f"  Deadline: {offer.deadline_days} days ({offer.valid_until[:10]})")
        return "\n".join(lines)


class VoiceProvider:
    """Voice provider integration — pluggable (Retell / Vapi / Bland)."""

    def __init__(self, provider: str = "retell"):
        self.provider = provider

    async def initiate_call(
        self, phone_number: str, agent_id: str, context: dict
    ) -> str:
        if self.provider == "retell":
            return await self._retell_call(phone_number, agent_id, context)
        elif self.provider == "vapi":
            return await self._vapi_call(phone_number, agent_id, context)
        elif self.provider == "bland":
            return await self._bland_call(phone_number, agent_id, context)
        raise ValueError(f"Unknown voice provider: {self.provider}")

    async def _retell_call(self, phone_number: str, agent_id: str, context: dict) -> str:
        import httpx, os
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.retellai.com/v2/call",
                json={
                    "from_number": os.getenv("CALLER_NUMBER"),
                    "to_number": phone_number,
                    "agent_id": agent_id,
                    "metadata": context,
                },
                headers={"Authorization": f"Bearer {os.getenv('RETELL_API_KEY')}"},
            )
            return r.json().get("call_id", "")

    async def _vapi_call(self, phone_number: str, agent_id: str, context: dict) -> str:
        import httpx, os
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.vapi.ai/call",
                json={"to": phone_number, "assistant_id": agent_id, "metadata": context},
                headers={"Authorization": f"Bearer {os.getenv('VAPI_API_KEY')}"},
            )
            return r.json().get("id", "")

    async def _bland_call(self, phone_number: str, agent_id: str, context: dict) -> str:
        import httpx, os
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.bland.ai/call",
                json={"phone_number": phone_number, "agent_id": agent_id, "metadata": context},
                headers={"Authorization": f"Bearer {os.getenv('BLAND_API_KEY')}"},
            )
            return r.json().get("call_id", "")

    def _get_key(self, provider: str) -> str:
        import os
        return os.getenv(f"{provider.upper()}_API_KEY", "")
