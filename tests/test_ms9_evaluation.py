"""
MS9 validation: Evaluation System.
- Rule-based checks: repeat questions, tone, resolution push, length, outcome marker
- LLM-as-judge stub (skipped by default)
- EvalReport structure
"""

import os
import pytest

# Ensure LLM judge is skipped in unit tests
os.environ.setdefault("SKIP_LLM_JUDGE", "1")

from src.evaluator import RuleBasedEvaluator, LLMJudge, EvalReport, RuleCheckResult
from src.models import ConversationContext, Stage, AssessmentData, ResolutionPath
from src.question_tracker import QuestionTracker, FactKey
from src.handoff import _serialise_qt


def make_assessment_context() -> ConversationContext:
    ctx = ConversationContext(borrower_id="B1", loan_id="L1")
    ctx.assessment_data = AssessmentData(
        borrower_id="B1", loan_id="L1",
        principal_amount=100_000, outstanding_amount=85_000,
        days_past_due=90,
    )
    qt = QuestionTracker()
    qt.mark_asked(FactKey.MONTHLY_INCOME, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.MONTHLY_INCOME, "50000")
    _serialise_qt(ctx, qt)
    return ctx


class TestRuleChecks:
    def test_no_repeat_clean(self):
        evaluator = RuleBasedEvaluator()
        ctx = make_assessment_context()
        history = [
            {"role": "user", "content": "Last 4 are 7823"},
            {"role": "assistant", "content": "Employment status?", "stage": "assessment"},
        ]
        result = evaluator._check_no_repeat_questions("AssessmentAgent", history, ctx)
        assert result.passed
        assert result.score == 1.0

    def test_no_repeat_violation_detected(self):
        evaluator = RuleBasedEvaluator()
        ctx = make_assessment_context()
        # income is already answered — agent asking again is a violation
        history = [
            {"role": "assistant", "content": "What is your monthly income?", "stage": "assessment"},
        ]
        result = evaluator._check_no_repeat_questions("AssessmentAgent", history, ctx)
        assert not result.passed

    def test_tone_clean(self):
        evaluator = RuleBasedEvaluator()
        history = [
            {"role": "assistant", "content": "Provide last 4 digits of account."},
            {"role": "assistant", "content": "State your monthly income."},
        ]
        result = evaluator._check_tone_compliance("AssessmentAgent", history)
        assert result.passed
        assert result.score == 1.0

    def test_tone_violation_detected(self):
        evaluator = RuleBasedEvaluator()
        history = [
            {"role": "assistant", "content": "I understand how difficult this must be for you."},
        ]
        result = evaluator._check_tone_compliance("AssessmentAgent", history)
        assert not result.passed

    def test_resolution_push_assessment_always_passes(self):
        evaluator = RuleBasedEvaluator()
        result = evaluator._check_resolution_push("AssessmentAgent", [])
        assert result.passed
        assert result.score == 1.0

    def test_resolution_push_resolution_agent_has_keywords(self):
        evaluator = RuleBasedEvaluator()
        history = [
            {"role": "assistant", "content": "Can you commit to this installment plan? Deadline April 24."},
        ]
        result = evaluator._check_resolution_push("ResolutionAgent", history)
        assert result.score > 0.0

    def test_response_length_voice_too_long(self):
        evaluator = RuleBasedEvaluator()
        long_msg = "word " * 60  # 60 words, over 40 limit for voice
        history = [{"role": "assistant", "content": long_msg}]
        result = evaluator._check_response_length("ResolutionAgent", history)
        assert not result.passed

    def test_response_length_voice_short_passes(self):
        evaluator = RuleBasedEvaluator()
        short_msg = "The offer is ₹12,750 down. Deadline April 24. Can you commit?"
        history = [{"role": "assistant", "content": short_msg}]
        result = evaluator._check_response_length("ResolutionAgent", history)
        assert result.passed

    def test_outcome_marker_found(self):
        """Agents strip markers from `content` before storing — the evaluator
        now checks the `advanced` flag instead."""
        evaluator = RuleBasedEvaluator()
        history = [
            {"role": "assistant", "content": "All facts confirmed.", "advanced": True},
        ]
        result = evaluator._check_outcome_marker("AssessmentAgent", history)
        assert result.passed
        assert result.score == 1.0

    def test_outcome_marker_missing(self):
        evaluator = RuleBasedEvaluator()
        history = [
            {"role": "assistant", "content": "What is your income?", "advanced": False},
        ]
        result = evaluator._check_outcome_marker("AssessmentAgent", history)
        assert not result.passed
        assert result.score == 0.0


class TestRuleBasedEvaluator:
    def test_full_eval_good_assessment(self):
        evaluator = RuleBasedEvaluator()
        ctx = make_assessment_context()
        history = [
            {"role": "user", "content": "7823, born 1985"},
            {"role": "assistant", "content": "Employment status?", "stage": "assessment"},
            {"role": "user", "content": "Salaried"},
            {"role": "assistant", "content": "ASSESSMENT_COMPLETE:INSTALLMENT", "stage": "assessment"},
        ]
        report = evaluator.evaluate("AssessmentAgent", history, ctx)
        assert isinstance(report, EvalReport)
        assert report.overall_score > 0.0
        assert len(report.checks) == 5

    def test_full_eval_returns_report(self):
        evaluator = RuleBasedEvaluator()
        report = evaluator.evaluate("ResolutionAgent", [], None)
        assert report.agent_name == "ResolutionAgent"
        assert 0.0 <= report.overall_score <= 1.0

    def test_report_summary_non_empty(self):
        evaluator = RuleBasedEvaluator()
        report = evaluator.evaluate("FinalNoticeAgent", [
            {"role": "assistant", "content": "Final offer expires April 24. Credit reporting follows. COLLECTIONS_COMPLETE"}
        ], None)
        summary = report.summary()
        assert "FinalNoticeAgent" in summary
        assert "PASS" in summary or "FAIL" in summary

    def test_five_checks_always_run(self):
        evaluator = RuleBasedEvaluator()
        for agent in ["AssessmentAgent", "ResolutionAgent", "FinalNoticeAgent"]:
            report = evaluator.evaluate(agent, [], None)
            assert len(report.checks) == 5


class TestEvalReport:
    def test_overall_passed_when_high_score(self):
        checks = [RuleCheckResult(f"check_{i}", True, 1.0) for i in range(5)]
        report = EvalReport("TestAgent", True, 1.0, checks)
        assert report.overall_passed is True

    def test_overall_failed_when_low_score(self):
        checks = [RuleCheckResult(f"check_{i}", False, 0.0) for i in range(5)]
        report = EvalReport("TestAgent", False, 0.0, checks)
        assert report.overall_passed is False


class TestLLMJudge:
    @pytest.mark.asyncio
    async def test_judge_returns_score_and_reason(self):
        judge = LLMJudge()
        history = [
            {"role": "assistant", "content": "Employment status?"},
            {"role": "user", "content": "Salaried"},
        ]
        score, reason = await judge.judge("AssessmentAgent", history)
        # SKIP_LLM_JUDGE is set — returns hardcoded 0.85
        assert 0.0 <= score <= 1.0
        assert isinstance(reason, str)
        assert len(reason) > 0

    @pytest.mark.asyncio
    async def test_judge_skipped_when_env_set(self):
        judge = LLMJudge()
        score, reason = await judge.judge("ResolutionAgent", [])
        assert score == 0.85
        assert "skipped" in reason.lower()
