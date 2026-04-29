"""
Milestone 5 validation tests — Final Notice Agent.
"""

import pytest
from unittest.mock import AsyncMock, patch

from src.agents.final_notice import FinalNoticeAgent, CONSEQUENCES, ESCALATION_SIGNALS
from src.models import (
    ConversationContext, Stage, AssessmentData, ResolutionPath, ResolutionOffer
)
from src.token_budget import CostTracker, TokenUsage


def mock_claude_response(text: str, input_tokens: int = 80, output_tokens: int = 60):
    usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    async def _mock(*args, **kwargs):
        return text, usage
    return _mock


def make_final_notice_context(
    resolution_outcome: str = "committed",
    path: ResolutionPath = ResolutionPath.INSTALLMENT,
) -> ConversationContext:
    ctx = ConversationContext(borrower_id="B001", loan_id="L001")
    ctx.current_stage = Stage.FINAL_NOTICE
    ctx.assessment_data = AssessmentData(
        borrower_id="B001",
        loan_id="L001",
        principal_amount=100_000,
        outstanding_amount=85_000,
        days_past_due=90,
        identity_verified=True,
        resolution_path=path,
    )
    ctx.resolution_offer = ResolutionOffer(
        path=path,
        upfront_required=12_750,
        monthly_payment=7_225,
        tenure_months=10,
        deadline_days=3,
        valid_until="2026-04-24T00:00:00+00:00",
    )
    ctx.resolution_outcome = resolution_outcome
    ctx.question_state = {
        "monthly_income": {"asked": True, "answered": True, "value": "50000",
                          "asked_by": "AssessmentAgent", "stage": "assessment"},
        "identity_last4": {"asked": True, "answered": True, "value": "7823",
                          "asked_by": "AssessmentAgent", "stage": "assessment"},
    }
    return ctx


# ------------------------------------------------------------------ #
# Outcome parsing
# ------------------------------------------------------------------ #

class TestOutcomeParsing:
    def test_parse_complete(self):
        agent = FinalNoticeAgent()
        assert agent._parse_outcome("Thank you. COLLECTIONS_COMPLETE") == "complete"

    def test_parse_escalated(self):
        agent = FinalNoticeAgent()
        assert agent._parse_outcome("COLLECTIONS_ESCALATED noted.") == "escalated"

    def test_parse_none(self):
        agent = FinalNoticeAgent()
        assert agent._parse_outcome("The offer expires on 24 Apr.") is None


# ------------------------------------------------------------------ #
# Escalation signal detection
# ------------------------------------------------------------------ #

class TestEscalationSignals:
    @pytest.mark.parametrize("phrase", [
        "I refuse to pay", "I won't pay", "I'm not paying",
        "take me to court", "sue me", "I don't care",
        "call my lawyer",
    ])
    def test_detects_escalation(self, phrase):
        agent = FinalNoticeAgent()
        assert agent._is_escalation_signal(phrase)

    def test_no_false_positive(self):
        agent = FinalNoticeAgent()
        assert not agent._is_escalation_signal("I'd like to understand the terms better")


# ------------------------------------------------------------------ #
# System prompt
# ------------------------------------------------------------------ #

class TestSystemPrompt:
    def test_consequences_in_prompt(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()
        prompt = agent.get_system_prompt(ctx)
        # Check canonical consequence keywords (from prompts.py)
        consequence_keywords = ["credit", "legal notice", "court", "garnishment"]
        for kw in consequence_keywords:
            assert kw in prompt.lower(), f"Missing consequence keyword: {kw}"

    def test_offer_in_prompt(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()
        prompt = agent.get_system_prompt(ctx)
        assert "12,750" in prompt or "12750" in prompt

    def test_no_repeat_in_prompt(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()
        prompt = agent.get_system_prompt(ctx)
        assert "DO NOT ask again" in prompt or "Already known" in prompt

    def test_deadline_in_offer_block(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()
        block = agent._format_final_offer(ctx)
        assert "2026" in block or "Apr" in block or "24" in block


# ------------------------------------------------------------------ #
# Process (mocked)
# ------------------------------------------------------------------ #

class TestFinalNoticeProcess:
    @pytest.mark.asyncio
    async def test_stage_open_never_auto_resolves(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "Confirmed. COLLECTIONS_COMPLETE"
        )):
            response = await agent.process(ctx, "[STAGE_OPEN: Generate the final notice now.]")

        assert response.should_advance is False
        assert ctx.final_notice_outcome is None
        assert "do you agree to this" in response.message.lower()

    @pytest.mark.asyncio
    async def test_resolves_on_complete_marker(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "Noted. COLLECTIONS_COMPLETE"
        )):
            response = await agent.process(ctx, "Okay, I'll pay.")

        assert response.should_advance is True
        assert ctx.final_notice_outcome == "resolved"
        assert response.context_update.get("current_stage") == Stage.COMPLETE

    @pytest.mark.asyncio
    async def test_escalates_on_escalated_marker(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "COLLECTIONS_ESCALATED"
        )):
            response = await agent.process(ctx, "I have no intention of paying.")

        assert response.should_advance is True
        assert ctx.final_notice_outcome == "escalated"
        assert response.context_update.get("current_stage") == Stage.ESCALATED

    @pytest.mark.asyncio
    async def test_escalates_on_signal_in_user_input(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()

        # Claude says nothing special, but user input is a hard refusal
        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "This account will be referred to legal."
        )):
            response = await agent.process(ctx, "Sue me. I refuse to pay.")

        assert response.should_advance is True
        assert ctx.final_notice_outcome == "escalated"

    @pytest.mark.asyncio
    async def test_continues_without_resolution(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "The offer expires on 24 April. Do you accept?"
        )):
            response = await agent.process(ctx, "I need more time to think.")

        assert response.should_advance is False
        assert ctx.final_notice_outcome is None

    @pytest.mark.asyncio
    async def test_marker_stripped_from_message(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "Payment confirmed. COLLECTIONS_COMPLETE"
        )):
            response = await agent.process(ctx, "Yes, I accept.")

        assert "COLLECTIONS_COMPLETE" not in response.message

    @pytest.mark.asyncio
    async def test_token_usage_returned(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "The offer expires on 24 April.", input_tokens=200, output_tokens=40
        )):
            response = await agent.process(ctx, "What happens if I don't pay?")

        assert response.tokens_used.input_tokens == 200
        assert response.tokens_used.output_tokens == 40

    @pytest.mark.asyncio
    async def test_refusal_allows_three_attempts_then_escalates(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()
        ctx.final_notice_confirmation_asked = True
        ctx.final_notice_confirmation_attempts = 1

        llm_mock = AsyncMock(return_value=("unused", TokenUsage(input_tokens=1, output_tokens=1)))
        with patch.object(agent, "_call_claude_with_tools", llm_mock):
            first = await agent.process(ctx, "no")

        # Explicit "no" should immediately escalate (binary agreement flow).
        assert first.should_advance is True
        assert ctx.final_notice_outcome == "escalated"
        assert "legal consequences notice" in first.message.lower()
        llm_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_second_attempt_renegotiation_gets_hard_warning(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()
        ctx.final_notice_confirmation_asked = True
        ctx.final_notice_confirmation_attempts = 2

        llm_mock = AsyncMock(return_value=("unused", TokenUsage(input_tokens=1, output_tokens=1)))
        with patch.object(agent, "_call_claude_with_tools", llm_mock):
            response = await agent.process(ctx, "Can you reduce the monthly amount?")

        assert response.should_advance is False
        assert "renegotiation is not available" in response.message.lower()
        assert "escalated" in response.message.lower()
        assert "legal consequences notice" in response.message.lower()
        llm_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_yes_after_confirmation_resolves_and_returns_contract(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()
        ctx.final_notice_confirmation_asked = True

        llm_mock = AsyncMock(return_value=("unused", TokenUsage(input_tokens=1, output_tokens=1)))
        with patch.object(agent, "_call_claude_with_tools", llm_mock):
            response = await agent.process(ctx, "yes")

        assert response.should_advance is True
        assert ctx.final_notice_outcome == "resolved"
        assert response.metadata.get("contract_html")
        llm_mock.assert_not_called()


# ------------------------------------------------------------------ #
# No repeat questions
# ------------------------------------------------------------------ #

class TestNoRepeatQuestions:
    def test_known_facts_present_in_context_block(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()
        context_block = agent.format_context_for_agent(ctx)
        assert "monthly_income" in context_block or "50000" in context_block

    def test_agent_does_not_ask_income_again(self):
        agent = FinalNoticeAgent()
        ctx = make_final_notice_context()
        from src.question_tracker import QuestionTracker, FactKey
        qt = agent._get_question_tracker(ctx)
        violations = qt.validate_no_repeat("What is your monthly income?")
        assert "monthly_income" in violations
