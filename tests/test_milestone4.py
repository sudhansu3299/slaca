"""
Milestone 4 validation tests — Resolution Agent.
"""

import os
import pytest
from unittest.mock import patch

from src.agents.resolution import ResolutionAgent, OFFER_POLICY
from src.models import (
    ConversationContext, Stage, AssessmentData, ResolutionPath, ResolutionOffer
)
from src.token_budget import CostTracker, TokenUsage


def mock_claude_response(text: str, input_tokens: int = 80, output_tokens: int = 40):
    usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    async def _mock(*args, **kwargs):
        return text, usage
    return _mock


def make_assessed_context(path: ResolutionPath = ResolutionPath.INSTALLMENT) -> ConversationContext:
    ctx = ConversationContext(borrower_id="B001", loan_id="L001")
    ctx.current_stage = Stage.RESOLUTION
    ctx.assessment_data = AssessmentData(
        borrower_id="B001",
        loan_id="L001",
        principal_amount=100_000,
        outstanding_amount=85_000,
        days_past_due=90,
        identity_verified=True,
        resolution_path=path,
    )
    ctx.question_state = {
        "monthly_income": {"asked": True, "answered": True, "value": "50000", "asked_by": "AssessmentAgent", "stage": "assessment"},
        "employment_status": {"asked": True, "answered": True, "value": "salaried", "asked_by": "AssessmentAgent", "stage": "assessment"},
    }
    return ctx


# ------------------------------------------------------------------ #
# Offer generation
# ------------------------------------------------------------------ #

class TestOfferGeneration:
    def test_installment_offer_fields(self):
        agent = ResolutionAgent()
        ctx = make_assessed_context(ResolutionPath.INSTALLMENT)
        offer = agent.generate_offer(ctx)
        assert offer.path == ResolutionPath.INSTALLMENT
        assert offer.upfront_required > 0
        assert offer.monthly_payment > 0
        assert offer.tenure_months == OFFER_POLICY[ResolutionPath.INSTALLMENT]["tenure_months"]
        assert offer.deadline_days == OFFER_POLICY[ResolutionPath.INSTALLMENT]["deadline_days"]

    def test_lump_sum_offer_discount_in_range(self):
        agent = ResolutionAgent()
        ctx = make_assessed_context(ResolutionPath.LUMP_SUM)
        offer = agent.generate_offer(ctx)
        assert offer.path == ResolutionPath.LUMP_SUM
        policy = OFFER_POLICY[ResolutionPath.LUMP_SUM]
        assert policy["discount_pct_min"] <= offer.discount_percentage <= policy["discount_pct_max"]
        assert offer.upfront_required < ctx.assessment_data.outstanding_amount

    def test_hardship_offer_lower_down(self):
        agent = ResolutionAgent()
        ctx = make_assessed_context(ResolutionPath.HARDSHIP)
        offer = agent.generate_offer(ctx)
        assert offer.path == ResolutionPath.HARDSHIP
        # Down payment should be only 5%
        expected_down = round(85_000 * 0.05, 2)
        assert abs(offer.upfront_required - expected_down) < 1.0

    def test_legal_offer_no_payment_fields(self):
        agent = ResolutionAgent()
        ctx = make_assessed_context(ResolutionPath.LEGAL)
        offer = agent.generate_offer(ctx)
        assert offer.path == ResolutionPath.LEGAL
        assert offer.monthly_payment == 0.0

    def test_offer_has_valid_until(self):
        agent = ResolutionAgent()
        ctx = make_assessed_context()
        offer = agent.generate_offer(ctx)
        assert offer.valid_until is not None
        assert len(offer.valid_until) > 0

    def test_no_offer_with_no_assessment(self):
        agent = ResolutionAgent()
        ctx = ConversationContext(borrower_id="B1", loan_id="L1")
        ctx.current_stage = Stage.RESOLUTION
        offer = agent.generate_offer(ctx)
        assert offer.path == ResolutionPath.LEGAL


# ------------------------------------------------------------------ #
# Outcome parsing
# ------------------------------------------------------------------ #

class TestOutcomeParsing:
    def test_parse_complete(self):
        agent = ResolutionAgent()
        committed, refused = agent._parse_outcome("Thank you. RESOLUTION_COMPLETE")
        assert committed is True
        assert refused is False

    def test_parse_refused(self):
        agent = ResolutionAgent()
        committed, refused = agent._parse_outcome("Noted. RESOLUTION_REFUSED")
        assert committed is False
        assert refused is True

    def test_parse_neither(self):
        agent = ResolutionAgent()
        committed, refused = agent._parse_outcome("The offer is 85,000 upfront.")
        assert committed is False
        assert refused is False


# ------------------------------------------------------------------ #
# Process (mocked)
# ------------------------------------------------------------------ #

class TestResolutionAgentProcess:
    @pytest.mark.asyncio
    async def test_process_generates_offer_if_missing(self):
        agent = ResolutionAgent()
        ctx = make_assessed_context()
        assert ctx.resolution_offer is None

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "The offer is 12,750 upfront plus 8,000 per month for 10 months."
        )):
            response = await agent.process(ctx, "Okay, what are my options?")

        assert ctx.resolution_offer is not None
        assert response.should_advance is False

    @pytest.mark.asyncio
    async def test_process_advances_on_commit(self):
        agent = ResolutionAgent()
        ctx = make_assessed_context()

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "Great. RESOLUTION_COMPLETE"
        )):
            response = await agent.process(ctx, "Fine, I agree to the installment plan.")

        assert response.should_advance is True
        assert ctx.resolution_outcome == "committed"

    @pytest.mark.asyncio
    async def test_process_advances_on_refusal(self):
        agent = ResolutionAgent()
        ctx = make_assessed_context()

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "Understood. RESOLUTION_REFUSED"
        )):
            response = await agent.process(ctx, "I refuse to pay anything.")

        assert response.should_advance is True
        assert ctx.resolution_outcome == "refused"

    @pytest.mark.asyncio
    async def test_process_no_repeat_question(self):
        """Agent must not re-ask income — already in question_state."""
        agent = ResolutionAgent()
        ctx = make_assessed_context()

        # System prompt should include "Already known: monthly_income: 50000"
        system_prompt = agent.get_system_prompt(ctx)
        assert "50000" in system_prompt or "monthly_income" in system_prompt

    @pytest.mark.asyncio
    async def test_token_usage_returned(self):
        agent = ResolutionAgent()
        ctx = make_assessed_context()

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "The offer stands at these terms.", input_tokens=150, output_tokens=25
        )):
            response = await agent.process(ctx, "Can you repeat the offer?")

        assert response.tokens_used.input_tokens == 150
        assert response.tokens_used.output_tokens == 25

    @pytest.mark.asyncio
    async def test_marker_stripped_from_message(self):
        agent = ResolutionAgent()
        ctx = make_assessed_context()

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "Confirmed. RESOLUTION_COMPLETE"
        )):
            response = await agent.process(ctx, "Yes, I agree.")

        assert "RESOLUTION_COMPLETE" not in response.message


# ------------------------------------------------------------------ #
# Offer within policy bounds (invariant)
# ------------------------------------------------------------------ #

class TestOfferPolicyBounds:
    def test_lump_sum_upfront_less_than_outstanding(self):
        agent = ResolutionAgent()
        for outstanding in [50_000, 100_000, 250_000]:
            ctx = ConversationContext(borrower_id="B", loan_id="L")
            ctx.assessment_data = AssessmentData(
                borrower_id="B", loan_id="L",
                principal_amount=outstanding,
                outstanding_amount=outstanding,
                days_past_due=30,
                resolution_path=ResolutionPath.LUMP_SUM,
            )
            offer = agent.generate_offer(ctx)
            assert offer.upfront_required < outstanding, \
                f"upfront {offer.upfront_required} should be < outstanding {outstanding}"

    def test_installment_total_reasonable(self):
        agent = ResolutionAgent()
        ctx = make_assessed_context(ResolutionPath.INSTALLMENT)
        offer = agent.generate_offer(ctx)
        # total = down + monthly * tenure
        total = offer.upfront_required + offer.monthly_payment * offer.tenure_months
        assert total <= ctx.assessment_data.outstanding_amount * 1.10, \
            "Installment total should not exceed outstanding + 10%"
