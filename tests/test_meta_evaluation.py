"""Meta-evaluation tests."""

import pytest
from src.self_learning.eval import SelfLearningLoop, EvaluationResult, EvaluationCriteria
from src.self_learning.loop import MetaEvaluator, MetaEvalReport, RunMetrics


class TestMetaEvaluator:
    @pytest.mark.asyncio
    async def test_returns_needs_data_with_few_evals(self):
        loop = SelfLearningLoop("AssessmentAgent")
        meta = MetaEvaluator()
        report = meta.assess(loop, [])
        assert report.verdict == "needs_data"
        assert report.num_evaluations < MetaEvaluator.MIN_EVALUATIONS

    @pytest.mark.asyncio
    async def test_detects_ineffective_eval_with_no_variance(self):
        loop = SelfLearningLoop("AssessmentAgent")
        # Feed 10 identical failing results — zero variance
        bad = [{"role": "user", "content": "nothing"}]
        for _ in range(10):
            await loop.evaluate_interaction(bad)
        meta = MetaEvaluator()
        report = meta.assess(loop, [])
        assert report.verdict == "ineffective"
        assert report.score_variance < 0.1

    @pytest.mark.asyncio
    async def test_detects_effective_eval_with_variance(self):
        loop = SelfLearningLoop("ResolutionAgent")
        good = [
            {"role": "assistant", "content": "The offer: ₹12,750 down. Can you commit?"},
            {"role": "user", "content": "Okay I agree"},
        ]
        bad = [
            {"role": "assistant", "content": "Offer presented."},
            {"role": "user", "content": "No."},
        ]

        run_metrics = []
        for i in range(12):
            if i % 2 == 0:
                await loop.evaluate_interaction(good, "committed")
                run_metrics.append(RunMetrics(f"r{i}", "resolved", recovery_rate=1.0))
            else:
                await loop.evaluate_interaction(bad, "refused")
                run_metrics.append(RunMetrics(f"r{i}", "escalated", recovery_rate=0.0))

        meta = MetaEvaluator()
        report = meta.assess(loop, run_metrics)
        assert report.num_evaluations == 12
        assert report.score_variance > 0.0

    def test_report_fields(self):
        report = MetaEvalReport(
            evaluator_name="AssessmentAgent",
            num_evaluations=10,
            score_variance=0.15,
            outcome_correlation=0.85,
            versions_evolved=2,
            verdict="effective",
        )
        assert report.verdict == "effective"
        assert report.versions_evolved == 2
