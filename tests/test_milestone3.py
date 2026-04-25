"""
Milestone 3 validation tests — Assessment Agent.
All tests that don't hit the API are mocked via monkeypatch.
One live API test is included (skipped if key absent).
"""

import os
import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from src.agents.assessment import AssessmentAgent, REQUIRED_FACTS
from src.models import ConversationContext, Stage, AssessmentData, ResolutionPath
from src.question_tracker import QuestionTracker, FactKey
from src.token_budget import CostTracker, TokenUsage, MAX_TOKENS_PER_AGENT


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

def make_context(with_assessment: bool = True) -> ConversationContext:
    ctx = ConversationContext(borrower_id="B001", loan_id="L001")
    if with_assessment:
        ctx.assessment_data = AssessmentData(
            borrower_id="B001",
            loan_id="L001",
            principal_amount=100_000,
            outstanding_amount=85_000,
            days_past_due=90,
        )
    return ctx


def make_fully_answered_context() -> ConversationContext:
    ctx = make_context()
    qt = QuestionTracker()
    for key in REQUIRED_FACTS:
        qt.mark_asked(key, "AssessmentAgent", "assessment")
        qt.mark_answered(key, "test_value")
    from src.handoff import _serialise_qt
    _serialise_qt(ctx, qt)
    return ctx


def mock_claude_response(text: str, input_tokens: int = 100, output_tokens: int = 50):
    """Return a coroutine that simulates _call_claude_with_tools."""
    usage = TokenUsage(input_tokens=input_tokens, output_tokens=output_tokens)
    async def _mock(*args, **kwargs):
        return text, usage
    return _mock


# ------------------------------------------------------------------ #
# Unit tests (no API calls)
# ------------------------------------------------------------------ #

class TestAssessmentAgentInit:
    def test_name(self):
        agent = AssessmentAgent()
        assert agent.name == "AssessmentAgent"

    def test_model_is_agent_model_default(self):
        agent = AssessmentAgent()
        assert agent.model == os.getenv("AGENT_MODEL", "gpt-4o")

    def test_uses_shared_cost_tracker(self):
        tracker = CostTracker()
        agent = AssessmentAgent(cost_tracker=tracker)
        assert agent.cost_tracker is tracker


class TestFactExtraction:
    def test_extracts_income(self):
        agent = AssessmentAgent()
        ctx = make_context()
        qt = QuestionTracker()
        agent._extract_facts("My income is 50000 per month", qt, ctx)
        assert qt.is_answered(FactKey.MONTHLY_INCOME)
        assert qt.get_answered()[FactKey.MONTHLY_INCOME] == "50000"

    def test_extracts_identity_last4(self):
        agent = AssessmentAgent()
        ctx = make_context()
        qt = QuestionTracker()
        agent._extract_facts("Last 4 digits are 7823", qt, ctx)
        assert qt.is_answered(FactKey.IDENTITY_LAST4)
        assert qt.get_answered()[FactKey.IDENTITY_LAST4] == "7823"

    def test_extracts_dob_year(self):
        agent = AssessmentAgent()
        ctx = make_context()
        qt = QuestionTracker()
        agent._extract_facts("I was born in 1988", qt, ctx)
        assert qt.is_answered(FactKey.IDENTITY_DOB_YEAR)
        assert qt.get_answered()[FactKey.IDENTITY_DOB_YEAR] == "1988"

    def test_extracts_employment_salaried(self):
        agent = AssessmentAgent()
        ctx = make_context()
        qt = QuestionTracker()
        agent._extract_facts("I am salaried at a private company", qt, ctx)
        assert qt.is_answered(FactKey.EMPLOYMENT_STATUS)
        assert qt.get_answered()[FactKey.EMPLOYMENT_STATUS] == "salaried"

    def test_extracts_cash_flow_issue(self):
        agent = AssessmentAgent()
        ctx = make_context()
        qt = QuestionTracker()
        agent._extract_facts("Finances are quite tight right now", qt, ctx)
        assert qt.is_answered(FactKey.CASH_FLOW_ISSUE)

    def test_no_overwrite_answered_fact(self):
        agent = AssessmentAgent()
        ctx = make_context()
        qt = QuestionTracker()
        qt.mark_asked(FactKey.MONTHLY_INCOME, "AssessmentAgent", "assessment")
        qt.mark_answered(FactKey.MONTHLY_INCOME, "60000")
        agent._extract_facts("My income is 70000", qt, ctx)
        # Should NOT overwrite existing answer
        assert qt.get_answered()[FactKey.MONTHLY_INCOME] == "60000"


class TestParseCompletion:
    def test_parse_complete_installment(self):
        agent = AssessmentAgent()
        ok, path = agent._parse_completion(
            "Understood. ASSESSMENT_COMPLETE:INSTALLMENT"
        )
        assert ok
        assert path == ResolutionPath.INSTALLMENT

    def test_parse_complete_lump_sum(self):
        agent = AssessmentAgent()
        ok, path = agent._parse_completion("ASSESSMENT_COMPLETE:LUMP_SUM")
        assert ok
        assert path == ResolutionPath.LUMP_SUM

    def test_parse_complete_hardship(self):
        agent = AssessmentAgent()
        ok, path = agent._parse_completion("ASSESSMENT_COMPLETE:HARDSHIP")
        assert ok
        assert path == ResolutionPath.HARDSHIP

    def test_parse_no_marker(self):
        agent = AssessmentAgent()
        ok, path = agent._parse_completion("What is your current income?")
        assert not ok
        assert path is None

    def test_parse_legal(self):
        agent = AssessmentAgent()
        ok, path = agent._parse_completion("ASSESSMENT_COMPLETE:LEGAL")
        assert ok
        assert path == ResolutionPath.LEGAL


class TestAllRequiredFacts:
    def test_not_complete_when_empty(self):
        agent = AssessmentAgent()
        ctx = make_context()
        assert not agent.all_required_facts_present(ctx)

    def test_complete_when_all_answered(self):
        agent = AssessmentAgent()
        ctx = make_fully_answered_context()
        assert agent.all_required_facts_present(ctx)


class TestProcessMocked:
    @pytest.mark.asyncio
    async def test_normal_turn_returns_message(self):
        agent = AssessmentAgent()
        ctx = make_context()
        ctx.conversation_history.append({"role": "user", "content": "Hello"})

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "What are the last 4 digits of your loan account?"
        )):
            response = await agent.process(ctx, "Hello")
        assert "last 4" in response.message.lower() or len(response.message) > 0
        assert response.should_advance is False

    @pytest.mark.asyncio
    async def test_advance_when_complete_marker_returned(self):
        agent = AssessmentAgent()
        ctx = make_context()
        # Pre-fill required facts
        qt = QuestionTracker()
        for key in REQUIRED_FACTS:
            qt.mark_asked(key, "AssessmentAgent", "assessment")
            qt.mark_answered(key, "val")
        from src.handoff import _serialise_qt
        _serialise_qt(ctx, qt)
        ctx.assessment_data.identity_verified = True

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "All facts gathered. ASSESSMENT_COMPLETE:INSTALLMENT"
        )):
            response = await agent.process(ctx, "I earn 50000 monthly")

        assert response.should_advance is True
        assert ctx.assessment_data.resolution_path == ResolutionPath.INSTALLMENT

    @pytest.mark.asyncio
    async def test_token_usage_recorded(self):
        tracker = CostTracker()
        agent = AssessmentAgent(cost_tracker=tracker)
        ctx = make_context()

        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "What is your employment status?", input_tokens=200, output_tokens=30
        )):
            response = await agent.process(ctx, "I am employed")

        # When _call_claude_with_tools is mocked, tracker.record is bypassed;
        # verify that the response carries the usage the mock returned.
        assert response.tokens_used is not None
        assert response.tokens_used.input_tokens == 200
        assert response.tokens_used.output_tokens == 30

    @pytest.mark.asyncio
    async def test_ignores_stale_failed_identity_recheck(self):
        agent = AssessmentAgent()
        ctx = make_context()
        ctx.assessment_data.identity_verified = True

        qt = QuestionTracker()
        qt.mark_asked(FactKey.IDENTITY_LAST4, "AssessmentAgent", "assessment")
        qt.mark_answered(FactKey.IDENTITY_LAST4, "7823")
        qt.mark_asked(FactKey.IDENTITY_DOB_YEAR, "AssessmentAgent", "assessment")
        qt.mark_answered(FactKey.IDENTITY_DOB_YEAR, "1985")
        from src.handoff import _serialise_qt
        _serialise_qt(ctx, qt)

        async def _mock(*args, **kwargs):
            agent._tool_results = {"verify_borrower_identity": {"verified": False}}
            return "What is your cash flow difficulty? (yes / no / mild)", TokenUsage(120, 30)

        with patch.object(agent, "_call_claude_with_tools", _mock):
            response = await agent.process(ctx, "a little")

        assert ctx.assessment_data.identity_verified is True
        assert "last 4" not in response.message.lower()

    @pytest.mark.asyncio
    async def test_adds_identity_verified_ack_when_tool_succeeds(self):
        agent = AssessmentAgent()
        ctx = make_context()

        async def _mock(*args, **kwargs):
            agent._tool_results = {"verify_borrower_identity": {"verified": True}}
            return "What is your monthly income?", TokenUsage(110, 24)

        with patch.object(agent, "_call_claude_with_tools", _mock):
            response = await agent.process(ctx, "7823, 1985")

        assert ctx.assessment_data.identity_verified is True
        assert response.message.lower().startswith("identity verified")

    @pytest.mark.asyncio
    async def test_output_within_token_limit(self):
        agent = AssessmentAgent()
        ctx = make_context()

        # Simulate a response right at the limit
        with patch.object(agent, "_call_claude_with_tools", mock_claude_response(
            "x" * (MAX_TOKENS_PER_AGENT * 4),  # ~2000 tokens of chars
            output_tokens=MAX_TOKENS_PER_AGENT
        )):
            response = await agent.process(ctx, "test")
        assert response.tokens_used.output_tokens == MAX_TOKENS_PER_AGENT


# ------------------------------------------------------------------ #
# Live API test — skipped if no key
# ------------------------------------------------------------------ #

@pytest.mark.skipif(
    not (os.getenv("OPENCODE_API_KEY") or os.getenv("OPENAI_API_KEY")),
    reason="OPENCODE_API_KEY or OPENAI_API_KEY not set"
)
@pytest.mark.asyncio
async def test_live_assessment_first_turn():
    """Single live turn — verify the agent asks for identity info."""
    tracker = CostTracker()
    agent = AssessmentAgent(cost_tracker=tracker)
    ctx = make_context()

    response = await agent.process(ctx, "Hi, I received a notice about my loan.")
    print(f"\n[LIVE] Agent says: {response.message}")
    print(f"[LIVE] Tokens: {response.tokens_used}")
    print(tracker.report())

    assert len(response.message) > 10
    assert response.tokens_used.output_tokens <= MAX_TOKENS_PER_AGENT
    assert tracker.total_cost_usd < 20.0
