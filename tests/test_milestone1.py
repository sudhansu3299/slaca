"""
Milestone 1 validation tests.
- Token budget enforcement
- Cost tracking
- No-repeat question tracker
- EvaluationResult reserved-word fix
- ConversationContext carries question_state and token_usage
"""

import pytest

from src.token_budget import (
    CostTracker, TokenUsage, BudgetExceededError, TokenLimitError,
    enforce_output_limit, enforce_total_turn_limit, clamp_max_tokens, TOTAL_COST_BUDGET_USD,
    MAX_TOKENS_PER_AGENT, MAX_TOKENS_HANDOFF,
)
from src.question_tracker import QuestionTracker, FactKey
from src.models import ConversationContext, Stage, AssessmentData
from src.self_learning.eval import EvaluationResult, EvaluationCriteria


# ------------------------------------------------------------------ #
# TOKEN BUDGET
# ------------------------------------------------------------------ #

class TestTokenBudget:
    def test_cost_tracking_records_per_agent(self):
        tracker = CostTracker()
        tracker.record("AssessmentAgent", TokenUsage(input_tokens=1000, output_tokens=500))
        tracker.record("ResolutionAgent", TokenUsage(input_tokens=800, output_tokens=400))
        assert "AssessmentAgent" in tracker.per_agent
        assert "ResolutionAgent" in tracker.per_agent
        assert tracker.per_agent["AssessmentAgent"].input_tokens == 1000

    def test_total_cost_is_sum(self):
        tracker = CostTracker()
        tracker.record("A", TokenUsage(input_tokens=100_000, output_tokens=0))
        tracker.record("B", TokenUsage(input_tokens=0, output_tokens=100_000))
        # 100k input @ $15/M = $1.50; 100k output @ $75/M = $7.50
        assert abs(tracker.total_cost_usd - 9.0) < 0.01

    def test_budget_exceeded_raises(self):
        tracker = CostTracker()
        # Inject more than $20 of output tokens (~267k output @ $75/M ≈ $20)
        tracker.record("X", TokenUsage(input_tokens=0, output_tokens=270_000))
        with pytest.raises(BudgetExceededError):
            tracker.check_budget()

    def test_budget_remaining_decreases(self):
        tracker = CostTracker()
        before = tracker.budget_remaining()
        tracker.record("A", TokenUsage(input_tokens=10_000, output_tokens=5_000))
        after = tracker.budget_remaining()
        assert after < before

    def test_enforce_total_turn_limit_pass(self):
        enforce_total_turn_limit(1200, 799, label="TestAgent")

    def test_enforce_total_turn_limit_fail(self):
        with pytest.raises(TokenLimitError):
            enforce_total_turn_limit(2001, 0, label="X")

    def test_enforce_output_limit_pass(self):
        enforce_output_limit(1999, MAX_TOKENS_PER_AGENT)  # should not raise

    def test_enforce_output_limit_fail(self):
        with pytest.raises(TokenLimitError):
            enforce_output_limit(2001, MAX_TOKENS_PER_AGENT)

    def test_clamp_max_tokens_respects_per_turn_total(self):
        tracker = CostTracker()
        clamped = clamp_max_tokens(9999, tracker, estimated_input_tokens=1700)
        assert clamped <= MAX_TOKENS_PER_AGENT

    def test_clamp_max_tokens_respects_limit(self):
        tracker = CostTracker()
        clamped = clamp_max_tokens(9999, tracker)
        assert clamped <= MAX_TOKENS_PER_AGENT

    def test_clamp_max_tokens_zero_budget(self):
        tracker = CostTracker()
        tracker.record("X", TokenUsage(input_tokens=0, output_tokens=270_000))
        with pytest.raises(BudgetExceededError):
            clamp_max_tokens(2000, tracker)

    def test_max_tokens_per_agent_constant(self):
        assert MAX_TOKENS_PER_AGENT == 2000

    def test_max_tokens_handoff_constant(self):
        assert MAX_TOKENS_HANDOFF == 500

    def test_total_budget_constant(self):
        assert TOTAL_COST_BUDGET_USD == 20.0


# ------------------------------------------------------------------ #
# QUESTION TRACKER
# ------------------------------------------------------------------ #

class TestQuestionTracker:
    def test_all_facts_initialised(self):
        qt = QuestionTracker()
        for key in FactKey:
            assert key.value in qt.facts

    def test_mark_asked_and_check(self):
        qt = QuestionTracker()
        qt.mark_asked(FactKey.MONTHLY_INCOME, agent="AssessmentAgent", stage="assessment")
        assert qt.is_asked(FactKey.MONTHLY_INCOME)
        assert not qt.is_answered(FactKey.MONTHLY_INCOME)

    def test_mark_answered(self):
        qt = QuestionTracker()
        qt.mark_asked(FactKey.MONTHLY_INCOME, "AssessmentAgent", "assessment")
        qt.mark_answered(FactKey.MONTHLY_INCOME, "50000")
        assert qt.is_answered(FactKey.MONTHLY_INCOME)
        assert qt.get_answered()[FactKey.MONTHLY_INCOME] == "50000"

    def test_unanswered_returns_all_initially(self):
        qt = QuestionTracker()
        unanswered = qt.get_unanswered()
        assert len(unanswered) == len(FactKey)

    def test_as_context_str_empty(self):
        qt = QuestionTracker()
        result = qt.as_context_str()
        assert "No facts" in result

    def test_as_context_str_with_answers(self):
        qt = QuestionTracker()
        qt.mark_asked(FactKey.MONTHLY_INCOME, "AssessmentAgent", "assessment")
        qt.mark_answered(FactKey.MONTHLY_INCOME, "60000")
        ctx = qt.as_context_str()
        assert "monthly_income" in ctx
        assert "60000" in ctx
        assert "DO NOT ask again" in ctx

    def test_validate_no_repeat_clean(self):
        qt = QuestionTracker()
        violations = qt.validate_no_repeat("Please tell me your monthly income.")
        assert violations == []  # not yet answered, so no violation

    def test_validate_no_repeat_violation(self):
        qt = QuestionTracker()
        qt.mark_asked(FactKey.MONTHLY_INCOME, "AssessmentAgent", "assessment")
        qt.mark_answered(FactKey.MONTHLY_INCOME, "50000")
        violations = qt.validate_no_repeat("What is your monthly income?")
        assert "monthly_income" in violations

    def test_no_violation_for_unanswered(self):
        qt = QuestionTracker()
        # employment_status not yet answered
        violations = qt.validate_no_repeat("Are you currently employed?")
        assert "employment_status" not in violations


# ------------------------------------------------------------------ #
# CONVERSATION CONTEXT CARRIES NEW FIELDS
# ------------------------------------------------------------------ #

class TestConversationContext:
    def test_context_has_question_state(self):
        ctx = ConversationContext(borrower_id="B1", loan_id="L1")
        assert hasattr(ctx, "question_state")
        assert isinstance(ctx.question_state, dict)

    def test_context_has_token_usage(self):
        ctx = ConversationContext(borrower_id="B1", loan_id="L1")
        assert hasattr(ctx, "token_usage")

    def test_context_has_total_cost(self):
        ctx = ConversationContext(borrower_id="B1", loan_id="L1")
        assert ctx.total_cost_usd == 0.0


# ------------------------------------------------------------------ #
# EVALUATION RESULT — reserved-word fix
# ------------------------------------------------------------------ #

class TestEvaluationResult:
    def test_passed_field_not_pass(self):
        result = EvaluationResult(
            criteria=[EvaluationCriteria(metric="x", threshold=0.5)],
            scores={"x": 0.8},
            passed=True,
        )
        assert result.passed is True

    def test_failed_flag(self):
        result = EvaluationResult(
            criteria=[EvaluationCriteria(metric="x", threshold=0.5)],
            scores={"x": 0.3},
            passed=False,
        )
        assert result.passed is False

    def test_no_syntax_error_importing(self):
        from src.self_learning.eval import EvaluationResult  # noqa: F401
