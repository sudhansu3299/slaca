"""
Milestone 6 validation tests — Self-learning loop.

Tests:
1. Scores are computed correctly for each agent type
2. Methodology version bumps after threshold evolution
3. Thresholds relax when metric keeps failing
4. Thresholds tighten when metric keeps passing
5. Learnings are recorded and retrievable
6. get_injected_guidance returns compact string
7. Methodology summary is human-readable
"""

import pytest
from src.self_learning.eval import SelfLearningLoop, EvaluationResult, EvaluationCriteria


# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

def assessment_history_complete() -> list[dict]:
    return [
        {"role": "assistant", "content": "Provide last 4 digits of your loan account."},
        {"role": "user", "content": "Last 4 are 7823 and I was born in 1988"},
        {"role": "assistant", "content": "What is your monthly income?"},
        {"role": "user", "content": "My income is 50000 and I spend about 35000 monthly"},
        {"role": "assistant", "content": "Employment status?"},
        {"role": "user", "content": "I am salaried, employed at a private firm"},
    ]


def resolution_history_committed() -> list[dict]:
    return [
        {"role": "assistant", "content": "The offer is ₹12,750 down plus ₹7,225/month for 10 months."},
        {"role": "user", "content": "That seems too high."},
        {"role": "assistant", "content": "The installment terms are fixed. ₹12,750 down by April 24. Can you commit?"},
        {"role": "user", "content": "Okay fine, I agree."},
    ]


def final_notice_history_resolved() -> list[dict]:
    return [
        {"role": "assistant", "content": "Final offer expires April 24. Legal proceedings will begin, credit bureau reporting will follow."},
        {"role": "user", "content": "Okay, I'll pay the amount."},
    ]


# ------------------------------------------------------------------ #
# 1. Score computation
# ------------------------------------------------------------------ #

class TestScoreComputation:
    @pytest.mark.asyncio
    async def test_assessment_identity_verified(self):
        loop = SelfLearningLoop("AssessmentAgent")
        result = await loop.evaluate_interaction(assessment_history_complete(), "success")
        assert result.scores["identity_verified"] == 1.0

    @pytest.mark.asyncio
    async def test_assessment_financial_completeness(self):
        loop = SelfLearningLoop("AssessmentAgent")
        result = await loop.evaluate_interaction(assessment_history_complete(), "success")
        # History covers income, expense, employed → 3/6 keywords = 0.5
        assert result.scores["financial_completeness"] >= 0.3

    @pytest.mark.asyncio
    async def test_assessment_no_repeat_clean(self):
        loop = SelfLearningLoop("AssessmentAgent")
        result = await loop.evaluate_interaction(assessment_history_complete(), "success")
        assert result.scores["no_repeat_questions"] == 1.0

    @pytest.mark.asyncio
    async def test_resolution_offer_presented(self):
        loop = SelfLearningLoop("ResolutionAgent")
        result = await loop.evaluate_interaction(resolution_history_committed(), "committed")
        assert result.scores["offer_presented"] == 1.0

    @pytest.mark.asyncio
    async def test_resolution_commitment_rate_success(self):
        loop = SelfLearningLoop("ResolutionAgent")
        result = await loop.evaluate_interaction(resolution_history_committed(), "committed")
        assert result.scores["commitment_rate"] == 1.0

    @pytest.mark.asyncio
    async def test_resolution_commitment_rate_failure(self):
        loop = SelfLearningLoop("ResolutionAgent")
        result = await loop.evaluate_interaction(resolution_history_committed(), "refused")
        assert result.scores["commitment_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_final_notice_consequences_stated(self):
        loop = SelfLearningLoop("FinalNoticeAgent")
        result = await loop.evaluate_interaction(final_notice_history_resolved(), "resolved")
        assert result.scores["consequences_stated"] > 0.0

    @pytest.mark.asyncio
    async def test_final_notice_deadline_stated(self):
        loop = SelfLearningLoop("FinalNoticeAgent")
        result = await loop.evaluate_interaction(final_notice_history_resolved(), "resolved")
        assert result.scores["deadline_stated"] == 1.0

    @pytest.mark.asyncio
    async def test_final_notice_resolution_rate(self):
        loop = SelfLearningLoop("FinalNoticeAgent")
        result = await loop.evaluate_interaction(final_notice_history_resolved(), "resolved")
        assert result.scores["resolution_rate"] == 1.0


# ------------------------------------------------------------------ #
# 2. Version bumps on evolution
# ------------------------------------------------------------------ #

class TestMethodologyEvolution:
    @pytest.mark.asyncio
    async def test_version_bumps_after_10_results(self):
        loop = SelfLearningLoop("AssessmentAgent")
        initial_version = loop.methodology.version

        # Feed 10 results that keep failing one metric
        bad_history = [{"role": "user", "content": "nothing useful"}]
        for _ in range(10):
            await loop.evaluate_interaction(bad_history, outcome=None)

        # Should have evolved
        assert loop.methodology.version != initial_version

    @pytest.mark.asyncio
    async def test_version_stable_under_10_results(self):
        loop = SelfLearningLoop("ResolutionAgent")
        initial_version = loop.methodology.version

        for _ in range(9):
            await loop.evaluate_interaction([], outcome=None)

        assert loop.methodology.version == initial_version


# ------------------------------------------------------------------ #
# 3. Thresholds relax on repeated failure
# ------------------------------------------------------------------ #

class TestThresholdRelaxation:
    @pytest.mark.asyncio
    async def test_threshold_relaxes_for_always_failing_metric(self):
        loop = SelfLearningLoop("AssessmentAgent")
        # "identity_verified" metric — always 0 on empty history
        initial = next(c for c in loop.methodology.criteria if c.metric == "identity_verified")
        initial_threshold = initial.threshold  # 1.0

        bad_history = [{"role": "user", "content": "no digits here"}]
        for _ in range(10):
            await loop.evaluate_interaction(bad_history, outcome=None)

        updated = next(c for c in loop.methodology.criteria if c.metric == "identity_verified")
        assert updated.threshold < initial_threshold


# ------------------------------------------------------------------ #
# 4. Thresholds tighten on repeated success
# ------------------------------------------------------------------ #

class TestThresholdTightening:
    @pytest.mark.asyncio
    async def test_threshold_tightens_for_always_passing_metric(self):
        loop = SelfLearningLoop("ResolutionAgent")
        # "offer_presented" — always 1.0 with good history, initial threshold = 1.0
        # We need it to start < 1.0 to have room to tighten
        for c in loop.methodology.criteria:
            if c.metric == "offer_presented":
                c.threshold = 0.80  # set artificially low so it can tighten
        initial_threshold = 0.80

        good_history = resolution_history_committed()
        for _ in range(10):
            await loop.evaluate_interaction(good_history, "committed")

        updated = next(c for c in loop.methodology.criteria if c.metric == "offer_presented")
        assert updated.threshold > initial_threshold


# ------------------------------------------------------------------ #
# 5. Learnings
# ------------------------------------------------------------------ #

class TestLearnings:
    @pytest.mark.asyncio
    async def test_success_learning_recorded(self):
        loop = SelfLearningLoop("ResolutionAgent")
        await loop.evaluate_interaction(resolution_history_committed(), "committed")
        successes = [l for l in loop.learnings if l.outcome == "success"]
        assert len(successes) >= 1

    @pytest.mark.asyncio
    async def test_failure_learning_recorded(self):
        loop = SelfLearningLoop("FinalNoticeAgent")
        await loop.evaluate_interaction(
            [{"role": "assistant", "content": "Pay or else."}, {"role": "user", "content": "No."}],
            "refused"
        )
        failures = [l for l in loop.learnings if l.outcome == "failure"]
        assert len(failures) >= 1

    @pytest.mark.asyncio
    async def test_learnings_capped_at_100(self):
        loop = SelfLearningLoop("AssessmentAgent")
        good_history = assessment_history_complete()
        for _ in range(60):
            await loop.evaluate_interaction(good_history, "success")
        assert len(loop.learnings) <= 100


# ------------------------------------------------------------------ #
# 6. Injected guidance
# ------------------------------------------------------------------ #

class TestInjectedGuidance:
    @pytest.mark.asyncio
    async def test_guidance_empty_with_no_learnings(self):
        loop = SelfLearningLoop("AssessmentAgent")
        guidance = loop.get_injected_guidance()
        assert guidance == ""

    @pytest.mark.asyncio
    async def test_guidance_non_empty_after_successes(self):
        loop = SelfLearningLoop("ResolutionAgent")
        await loop.evaluate_interaction(resolution_history_committed(), "committed")
        guidance = loop.get_injected_guidance(top_n=2)
        assert len(guidance) > 0
        assert "Learned" in guidance or "pattern" in guidance.lower()


# ------------------------------------------------------------------ #
# 7. Summary
# ------------------------------------------------------------------ #

class TestMethodologySummary:
    def test_summary_contains_version(self):
        loop = SelfLearningLoop("AssessmentAgent")
        summary = loop.methodology_summary()
        assert "1.0.0" in summary

    def test_summary_contains_agent_name(self):
        loop = SelfLearningLoop("AssessmentAgent")
        summary = loop.methodology_summary()
        assert "AssessmentAgent" in summary

    def test_summary_lists_metrics(self):
        loop = SelfLearningLoop("FinalNoticeAgent")
        summary = loop.methodology_summary()
        assert "resolution_rate" in summary
