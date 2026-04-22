"""
MS10 validation: Cost optimization.
- Model tiering: Opus for production, Haiku for simulation/eval
- Summary cache: hit rate, eviction
- TieredCostTracker: separate buckets, total within budget
- Cost report: human-readable, contains all sections
- Batch processor: runs multiple borrowers, returns BatchResult
"""

import pytest
from src.cost import (
    SummaryCache, TieredCostTracker, BatchResult, cost_for_model,
    AGENT_MODEL, SIMULATION_MODEL, EVAL_MODEL,
    MODEL_PRICING,
)
from src.token_budget import TokenUsage, TOTAL_COST_BUDGET_USD


class TestModelTiering:
    def test_agent_model_is_opus(self):
        assert "opus" in AGENT_MODEL.lower()

    def test_simulation_model_is_cheaper(self):
        opus_out = MODEL_PRICING[AGENT_MODEL]["output"]
        haiku_out = MODEL_PRICING[SIMULATION_MODEL]["output"]
        assert haiku_out < opus_out

    def test_eval_model_is_cheap(self):
        assert MODEL_PRICING[EVAL_MODEL]["output"] < MODEL_PRICING[AGENT_MODEL]["output"]

    def test_cost_for_haiku_less_than_opus(self):
        tokens = 10_000
        opus_cost = cost_for_model(AGENT_MODEL, tokens, tokens)
        haiku_cost = cost_for_model(SIMULATION_MODEL, tokens, tokens)
        assert haiku_cost < opus_cost

    def test_cost_for_model_computation(self):
        cost = cost_for_model("claude-opus-4-20250514", 1_000_000, 0)
        assert abs(cost - 15.0) < 0.01


class TestSummaryCache:
    def test_miss_on_first_lookup(self):
        cache = SummaryCache()
        result = cache.get("B1", "fp1")
        assert result is None

    def test_hit_after_set(self):
        cache = SummaryCache()
        cache.set("B1", "fp1", "my prompt block")
        result = cache.get("B1", "fp1")
        assert result == "my prompt block"

    def test_different_fingerprint_is_miss(self):
        cache = SummaryCache()
        cache.set("B1", "fp1", "block1")
        assert cache.get("B1", "fp2") is None

    def test_hit_rate_increases(self):
        cache = SummaryCache()
        cache.set("B1", "fp1", "block")
        cache.get("B1", "fp1")
        cache.get("B1", "fp1")
        assert cache.hit_rate > 0.5

    def test_eviction_at_max_size(self):
        cache = SummaryCache(max_size=3)
        for i in range(4):
            cache.set("B", f"fp{i}", f"block{i}")
        assert len(cache._cache) == 3

    def test_stats_string(self):
        cache = SummaryCache()
        cache.set("B", "fp", "block")
        cache.get("B", "fp")
        stats = cache.stats()
        assert "hit_rate" in stats
        assert "hits=1" in stats


class TestTieredCostTracker:
    def test_production_tracked_separately(self):
        tracker = TieredCostTracker()
        tracker.record_production("AssessmentAgent", TokenUsage(1000, 200))
        assert tracker.production.per_agent["AssessmentAgent"].input_tokens == 1000
        assert tracker.simulation.total_tokens.input_tokens == 0

    def test_simulation_tracked_separately(self):
        tracker = TieredCostTracker()
        tracker.record_simulation(TokenUsage(5000, 1000))
        assert tracker.simulation.total_tokens.input_tokens == 5000
        assert tracker.production.total_tokens.input_tokens == 0

    def test_evaluation_tracked_separately(self):
        tracker = TieredCostTracker()
        tracker.record_evaluation(TokenUsage(2000, 500))
        assert tracker.evaluation.total_tokens.input_tokens == 2000

    def test_total_cost_is_sum_of_tiers(self):
        tracker = TieredCostTracker()
        tracker.record_production("A", TokenUsage(0, 10_000))
        tracker.record_simulation(TokenUsage(0, 10_000))
        tracker.record_evaluation(TokenUsage(0, 10_000))
        # Production output @ $75/M = $0.75
        # Simulation output @ $5/M = $0.05
        # Eval output @ $5/M = $0.05
        assert abs(tracker.total_cost_usd - 0.85) < 0.01

    def test_within_budget_true_when_zero(self):
        tracker = TieredCostTracker()
        assert tracker.within_budget() is True

    def test_within_budget_false_when_over(self):
        tracker = TieredCostTracker()
        # Put $21 of Opus output
        tracker.record_production("X", TokenUsage(0, 280_001))
        assert tracker.within_budget() is False

    def test_full_report_contains_sections(self):
        tracker = TieredCostTracker()
        tracker.record_production("AssessmentAgent", TokenUsage(500, 100))
        tracker.record_simulation(TokenUsage(1000, 200))
        tracker.record_evaluation(TokenUsage(300, 50))
        report = tracker.full_report()
        assert "Production" in report
        assert "Simulation" in report
        assert "Evaluation" in report
        assert "TOTAL" in report
        assert "REMAINING" in report
        assert "WITHIN BUDGET" in report or "OVER BUDGET" in report


class TestCostReport:
    def test_report_shows_budget(self):
        tracker = TieredCostTracker()
        report = tracker.full_report()
        assert f"${TOTAL_COST_BUDGET_USD}" in report

    def test_report_shows_agent_breakdown(self):
        tracker = TieredCostTracker()
        tracker.record_production("AssessmentAgent", TokenUsage(1000, 200))
        tracker.record_production("ResolutionAgent", TokenUsage(800, 150))
        report = tracker.full_report()
        assert "AssessmentAgent" in report
        assert "ResolutionAgent" in report

    def test_20_simulations_within_budget(self):
        """20 simulations × ~600 tokens/sim on Haiku should be < $1."""
        tracker = TieredCostTracker()
        for _ in range(20):
            tracker.record_simulation(TokenUsage(400, 200))
        sim_cost = cost_for_model(
            SIMULATION_MODEL,
            tracker.simulation.total_tokens.input_tokens,
            tracker.simulation.total_tokens.output_tokens,
        )
        assert sim_cost < 1.0, f"20 simulations cost ${sim_cost:.4f}, expected < $1"


class TestBatchResult:
    def test_cost_per_borrower(self):
        result = BatchResult(
            total_borrowers=10, completed=10, failed=0,
            total_cost_usd=2.0, elapsed_seconds=5.0
        )
        assert result.cost_per_borrower == 0.2

    def test_summary_string(self):
        result = BatchResult(
            total_borrowers=5, completed=4, failed=1,
            total_cost_usd=0.5, elapsed_seconds=3.0
        )
        s = result.summary()
        assert "4/5" in s
        assert "failed" in s
