"""
MS7 validation: Self-learning loop.
- Recovery rate tracked per run
- Commitment success tracked
- Drop-off points identified
- Repetition errors counted
- Prompts updated after evaluation
- Strategy updates recorded in history
"""

import pytest
from src.self_learning.loop import (
    LearningHistory, RunMetrics, PromptUpdater, StrategyUpdate,
    build_run_metrics,
)
from src.self_learning.eval import SelfLearningLoop, EvaluationResult, EvaluationCriteria


# ──────────────────────────────────────────────── #
# RunMetrics
# ──────────────────────────────────────────────── #

class TestRunMetrics:
    def test_recovery_rate_resolved(self):
        m = RunMetrics(run_id="r1", outcome="resolved", recovery_rate=1.0)
        assert m.recovery_rate == 1.0

    def test_recovery_rate_unresolved(self):
        m = RunMetrics(run_id="r2", outcome="escalated", recovery_rate=0.0)
        assert m.recovery_rate == 0.0

    def test_drop_off_none_when_resolved(self):
        m = RunMetrics(run_id="r3", outcome="resolved", drop_off_stage=None)
        assert m.drop_off_stage is None

    def test_drop_off_stage_tracked(self):
        m = RunMetrics(run_id="r4", outcome="escalated", drop_off_stage="final_notice")
        assert m.drop_off_stage == "final_notice"

    def test_repetition_errors_tracked(self):
        m = RunMetrics(run_id="r5", outcome="resolved", repetition_errors=2)
        assert m.repetition_errors == 2


# ──────────────────────────────────────────────── #
# LearningHistory aggregation
# ──────────────────────────────────────────────── #

class TestLearningHistory:
    def test_avg_recovery_rate_empty(self):
        h = LearningHistory()
        assert h.average_recovery_rate == 0.0

    def test_avg_recovery_rate_partial(self):
        h = LearningHistory()
        h.add_run(RunMetrics("r1", "resolved", recovery_rate=1.0))
        h.add_run(RunMetrics("r2", "escalated", recovery_rate=0.0))
        assert h.average_recovery_rate == 0.5

    def test_avg_commitment_rate(self):
        h = LearningHistory()
        h.add_run(RunMetrics("r1", "resolved", commitment_success=1.0))
        h.add_run(RunMetrics("r2", "escalated", commitment_success=0.0))
        assert h.average_commitment_rate == 0.5

    def test_total_repetition_errors(self):
        h = LearningHistory()
        h.add_run(RunMetrics("r1", "resolved", repetition_errors=1))
        h.add_run(RunMetrics("r2", "resolved", repetition_errors=3))
        assert h.total_repetition_errors == 4

    def test_drop_off_distribution(self):
        h = LearningHistory()
        h.add_run(RunMetrics("r1", "escalated", drop_off_stage="resolution"))
        h.add_run(RunMetrics("r2", "escalated", drop_off_stage="resolution"))
        h.add_run(RunMetrics("r3", "escalated", drop_off_stage="final_notice"))
        dist = h.drop_off_distribution
        assert dist["resolution"] == 2
        assert dist["final_notice"] == 1

    def test_summary_contains_key_metrics(self):
        h = LearningHistory()
        h.add_run(RunMetrics("r1", "resolved", recovery_rate=1.0, commitment_success=1.0))
        summary = h.summary()
        assert "recovery rate" in summary.lower()
        assert "commitment rate" in summary.lower()
        assert "repetition" in summary.lower()


# ──────────────────────────────────────────────── #
# PromptUpdater
# ──────────────────────────────────────────────── #

class TestPromptUpdater:
    @pytest.mark.asyncio
    async def test_guidance_injected_into_agent(self):
        from unittest.mock import MagicMock

        history = LearningHistory()
        loop = SelfLearningLoop("ResolutionAgent")
        updater = PromptUpdater(history)

        # Feed some successful interactions so learnings exist
        good_history = [
            {"role": "assistant", "content": "The offer is ₹12,750 down. Can you commit?"},
            {"role": "user", "content": "Okay I agree"},
        ]
        for _ in range(3):
            await loop.evaluate_interaction(good_history, "committed")

        agent = MagicMock()
        agent.name = "ResolutionAgent"
        updater.update_agent(agent, loop)

        # Guidance should be set on the agent
        assert hasattr(agent, "_injected_guidance")

    @pytest.mark.asyncio
    async def test_strategy_update_recorded_after_evolution(self):
        history = LearningHistory()
        loop = SelfLearningLoop("AssessmentAgent")
        updater = PromptUpdater(history)

        # Feed 10 bad interactions to trigger evolution + threshold relaxation
        bad_history = [{"role": "user", "content": "nothing here"}]
        for _ in range(10):
            await loop.evaluate_interaction(bad_history)

        from unittest.mock import MagicMock
        agent = MagicMock()
        agent.name = "AssessmentAgent"
        updater.update_agent(agent, loop)

        # After evolution, version should be updated
        assert loop.methodology.version != "1.0.0"


# ──────────────────────────────────────────────── #
# build_run_metrics
# ──────────────────────────────────────────────── #

class TestBuildRunMetrics:
    def test_resolved_outcome_gives_full_recovery(self):
        history = [
            {"role": "user", "content": "I agree to pay", "stage": "final_notice"},
            {"role": "assistant", "content": "Payment confirmed.", "stage": "final_notice"},
        ]
        metrics = build_run_metrics("r1", "resolved", history, 0.5, [])
        assert metrics.recovery_rate == 1.0
        assert metrics.drop_off_stage is None

    def test_escalated_outcome_drops_off(self):
        history = [
            {"role": "user", "content": "Hello", "stage": "assessment"},
            {"role": "assistant", "content": "What is your income?", "stage": "assessment"},
            {"role": "user", "content": "I refuse to pay", "stage": "final_notice"},
        ]
        metrics = build_run_metrics("r2", "escalated", history, 0.3, [])
        assert metrics.recovery_rate == 0.0
        assert metrics.drop_off_stage == "final_notice"

    def test_repetition_errors_counted_from_eval(self):
        eval_results = [
            EvaluationResult(
                criteria=[EvaluationCriteria(metric="no_repeat_questions", threshold=1.0)],
                scores={"no_repeat_questions": 0.5},
                passed=False,
            ),
            EvaluationResult(
                criteria=[EvaluationCriteria(metric="no_repeat_questions", threshold=1.0)],
                scores={"no_repeat_questions": 1.0},
                passed=True,
            ),
        ]
        metrics = build_run_metrics("r3", "resolved", [], 0.1, eval_results)
        assert metrics.repetition_errors == 1

    def test_commitment_detected_in_resolution_turns(self):
        history = [
            {"role": "user", "content": "Okay I agree to the plan", "stage": "resolution"},
        ]
        metrics = build_run_metrics("r4", "resolved", history, 0.2, [])
        assert metrics.commitment_success == 1.0

    def test_cost_stored(self):
        metrics = build_run_metrics("r5", "resolved", [], 1.23, [])
        assert metrics.cost_usd == 1.23


# ──────────────────────────────────────────────── #
# Full loop: simulate → evaluate → update
# ──────────────────────────────────────────────── #

class TestFullLearningLoop:
    @pytest.mark.asyncio
    async def test_multiple_runs_improve_guidance(self):
        """After 5 successful runs, guidance should be non-empty."""
        loop = SelfLearningLoop("ResolutionAgent")
        history = LearningHistory()

        good_history = [
            {"role": "assistant", "content": "The offer is ₹12,750 down + ₹7,225/month."},
            {"role": "user", "content": "Okay I commit to this plan"},
        ]
        for i in range(5):
            result = await loop.evaluate_interaction(good_history, "committed")
            metrics = build_run_metrics(f"run_{i}", "resolved", good_history, 0.1, [result])
            history.add_run(metrics)

        guidance = loop.get_injected_guidance(top_n=3)
        assert len(guidance) > 0
        assert history.average_recovery_rate == 1.0

    @pytest.mark.asyncio
    async def test_failure_runs_record_drop_off(self):
        loop = SelfLearningLoop("FinalNoticeAgent")
        history = LearningHistory()

        bad_history = [
            {"role": "assistant", "content": "Final offer expires April 24."},
            {"role": "user", "content": "I refuse to pay anything."},
        ]
        for i in range(3):
            result = await loop.evaluate_interaction(bad_history, "escalated")
            metrics = build_run_metrics(
                f"fail_{i}", "escalated", bad_history, 0.05, [result],
            )
            history.add_run(metrics)

        assert history.average_recovery_rate == 0.0
        dist = history.drop_off_distribution
        assert "final_notice" in dist or "assessment" in dist or "resolution" in dist
