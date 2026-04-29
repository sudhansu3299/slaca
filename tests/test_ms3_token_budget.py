"""
MS3 validation: Token budget enforcement layer.
- PromptBuilder enforces per-agent limits
- Overflow scenario trims history correctly
- No agent exceeds its prompt token limit
- Summary compressor keeps handoff ≤500 tokens under pressure
"""

import pytest
from src.prompt_builder import PromptBuilder, PROMPT_TOKEN_LIMITS, AGENT_PROMPT_BUDGETS
from src.token_budget import MAX_TOKENS_HANDOFF, MAX_TOTAL_TOKENS_PER_AGENT_TURN, MIN_RESERVED_OUTPUT_TOKENS, estimate_tokens
from src.question_tracker import QuestionTracker, FactKey
from src.handoff import _serialise_qt, HandoffBuilder
from src.models import ConversationContext, Stage, AssessmentData, ResolutionPath


_MAX_BODY = MAX_TOTAL_TOKENS_PER_AGENT_TURN - MIN_RESERVED_OUTPUT_TOKENS


def make_context_with_history(n_turns: int = 5) -> ConversationContext:
    ctx = ConversationContext(borrower_id="B1", loan_id="L1")
    ctx.assessment_data = AssessmentData(
        borrower_id="B1", loan_id="L1",
        principal_amount=100_000, outstanding_amount=85_000,
        days_past_due=90, resolution_path=ResolutionPath.INSTALLMENT
    )
    for i in range(n_turns):
        ctx.conversation_history.append({"role": "user", "content": f"Borrower message {i} " + "x" * 50})
        ctx.conversation_history.append({"role": "assistant", "content": f"Agent reply {i} " + "x" * 80})
    return ctx


class TestPromptLimits:
    def test_agent2_system_cap_is_1500(self):
        assert AGENT_PROMPT_BUDGETS["ResolutionAgent"].max_system == 1500
        assert AGENT_PROMPT_BUDGETS["ResolutionAgent"].max_handoff == 500

    def test_agent3_system_cap_is_1500(self):
        assert AGENT_PROMPT_BUDGETS["FinalNoticeAgent"].max_system == 1500
        assert AGENT_PROMPT_BUDGETS["FinalNoticeAgent"].max_handoff == 500

    def test_agent1_system_cap_is_2000_no_handoff(self):
        assert AGENT_PROMPT_BUDGETS["AssessmentAgent"].max_system == 2000
        assert AGENT_PROMPT_BUDGETS["AssessmentAgent"].max_handoff == 0

    def test_shared_total_body_budget(self):
        for cfg in AGENT_PROMPT_BUDGETS.values():
            assert cfg.total_limit == _MAX_BODY


class TestPromptBuilder:
    def test_normal_context_within_limit(self):
        pb = PromptBuilder("ResolutionAgent")
        ctx = make_context_with_history(3)
        system = "You are the resolution agent. Keep it short."
        components = pb.build(system, ctx, None, "What are my options?")
        assert components.total_estimated_tokens <= PROMPT_TOKEN_LIMITS["ResolutionAgent"]
        assert components.was_trimmed is False

    def test_overflow_trims_history(self):
        pb = PromptBuilder("ResolutionAgent")
        # 40 turns = ~40*130 chars = ~5200 chars = ~1300 tokens of history alone
        ctx = make_context_with_history(40)
        system = "Short system prompt."
        components = pb.build(system, ctx, None, "Question?")
        # Should be within limit after trimming
        assert components.total_estimated_tokens <= PROMPT_TOKEN_LIMITS["ResolutionAgent"]
        assert components.was_trimmed is True

    def test_assert_within_limit_passes(self):
        pb = PromptBuilder("FinalNoticeAgent")
        ctx = make_context_with_history(2)
        components = pb.build("System.", ctx, None, "Hello.")
        pb.assert_within_limit(components)  # should not raise

    def test_assert_within_limit_fails_on_override(self):
        from src.prompt_builder import PromptComponents
        pb = PromptBuilder("ResolutionAgent")
        over = PromptComponents(
            system_prompt="x",
            context_block="x",
            history_block="x",
            user_input="y",
            total_estimated_tokens=5000,
        )
        with pytest.raises(ValueError, match="exceeds limit"):
            pb.assert_within_limit(over)

    def test_handoff_block_in_context(self):
        pb = PromptBuilder("ResolutionAgent")
        ctx = make_context_with_history(2)
        qt = QuestionTracker()
        qt.mark_asked(FactKey.MONTHLY_INCOME, "AssessmentAgent", "assessment")
        qt.mark_answered(FactKey.MONTHLY_INCOME, "50000")
        _serialise_qt(ctx, qt)

        ctx.assessment_data.identity_verified = True
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        components = pb.build("System.", ctx, summary, "What's the offer?")
        assert "HANDOFF" in components.context_block or "identity" in components.context_block.lower()

    def test_compressed_block_on_extreme_overflow(self):
        """Even with giant system prompt, result stays within limit."""
        pb = PromptBuilder("FinalNoticeAgent")
        giant_system = "X " * 2000  # ~4000 chars ≈ 1000 tokens
        ctx = make_context_with_history(10)
        components = pb.build(giant_system, ctx, None, "Do you accept?")
        assert components.total_estimated_tokens <= PROMPT_TOKEN_LIMITS["FinalNoticeAgent"]

    def test_history_most_recent_kept(self):
        pb = PromptBuilder("ResolutionAgent")
        ctx = make_context_with_history(20)
        components = pb.build("System.", ctx, None, "Q?")
        # Most recent message should appear in history
        if components.history_block:
            assert "Agent reply 19" in components.history_block or "Borrower message 19" in components.history_block


class TestSummaryCompressor:
    """Handoff summary must stay ≤500 tokens even with many objections/offers."""

    def test_many_objections_still_within_limit(self):
        ctx = ConversationContext(borrower_id="B", loan_id="L")
        ctx.assessment_data = AssessmentData(
            borrower_id="B", loan_id="L",
            principal_amount=50_000, outstanding_amount=40_000,
            days_past_due=60, resolution_path=ResolutionPath.INSTALLMENT,
        )
        # Add 20 objection-containing messages
        for i in range(20):
            ctx.conversation_history.append({
                "role": "user",
                "content": f"I cannot pay this amount, it is too expensive for me {i}"
            })
        s = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert s.estimated_tokens() <= MAX_TOKENS_HANDOFF

    def test_long_conversation_still_within_limit(self):
        ctx = ConversationContext(borrower_id="B", loan_id="L")
        ctx.assessment_data = AssessmentData(
            borrower_id="B", loan_id="L",
            principal_amount=100_000, outstanding_amount=80_000,
            days_past_due=120,
        )
        for i in range(30):
            ctx.conversation_history.append({
                "role": "user", "content": "Some long borrower message " * 5
            })
            ctx.conversation_history.append({
                "role": "assistant", "content": "Agent response " * 5
            })
        s = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert s.estimated_tokens() <= MAX_TOKENS_HANDOFF


class TestTokenCounter:
    def test_estimate_tokens_4_chars_per_token(self):
        assert estimate_tokens("aaaa") == 1
        assert estimate_tokens("a" * 400) == 100

    def test_estimate_tokens_empty(self):
        assert estimate_tokens("") >= 0

    def test_estimate_tokens_long_text(self):
        text = "word " * 1000   # ~5000 chars
        assert estimate_tokens(text) > 1000
