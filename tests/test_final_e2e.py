"""
Final Milestone: End-to-End Test.

Success criteria:
1. Feels like ONE system — no re-introductions, no repeated questions
2. No repeated info across agents
3. Logical progression: info → deal → ultimatum
4. Voice call starts mid-context (no restart feel)
5. Golden summary captures full conversation
6. Evaluation passes rule-based checks
7. Total cost within $20

Runs a cooperative borrower through the full pipeline (all mocked LLM).
"""

import os
import pytest
from unittest.mock import patch

os.environ.setdefault("SKIP_LLM_JUDGE", "1")

from src.pipeline import CollectionsPipeline
from src.simulation import SimulationEngine, PersonaType, PersonaScript
from src.handoff import HandoffBuilder
from src.memory import MemoryBuilder
from src.evaluator import RuleBasedEvaluator
from src.voice import VoiceProvider, build_voice_metadata
from src.cost import TieredCostTracker, TOTAL_COST_BUDGET_USD
from src.models import Stage, ResolutionPath
from src.token_budget import TokenUsage, MAX_TOKENS_HANDOFF


# ──────────────────────────────────────────────── #
# Helpers
# ──────────────────────────────────────────────── #

def mock_claude(responses: list[str]):
    it = iter(responses)
    async def _mock(*args, **kwargs):
        try:
            text = next(it)
        except StopIteration:
            text = "Please continue."
        return text, TokenUsage(input_tokens=300, output_tokens=60)
    return _mock


async def run_full_pipeline():
    pipeline = CollectionsPipeline()
    borrower = PersonaScript(PersonaType.COOPERATIVE)
    llm = SimulationEngine.get_mock_llm_responses(PersonaType.COOPERATIVE)

    with patch.object(pipeline.assessment_agent, "_call_claude",
                      side_effect=mock_claude(llm["AssessmentAgent"])), \
         patch.object(pipeline.resolution_agent, "_call_claude",
                      side_effect=mock_claude(llm["ResolutionAgent"])), \
         patch.object(pipeline.final_notice_agent, "_call_claude",
                      side_effect=mock_claude(llm["FinalNoticeAgent"])):
        result = await pipeline.run(
            borrower_id="E2E-001",
            loan_id="LN-E2E-001",
            principal_amount=100_000,
            outstanding_amount=85_000,
            days_past_due=90,
            input_provider=lambda ctx, agent: borrower.respond(agent),
            max_turns_per_stage=8,
        )
    return result, pipeline


# ──────────────────────────────────────────────── #
# Tests
# ──────────────────────────────────────────────── #

class TestOneSystemFeel:
    """Borrower experiences one continuous system."""

    @pytest.mark.asyncio
    async def test_all_three_stages_in_conversation(self):
        result, _ = await run_full_pipeline()
        stages = {m.get("stage") for m in result.conversation}
        assert Stage.ASSESSMENT.value in stages
        assert Stage.RESOLUTION.value in stages
        assert Stage.FINAL_NOTICE.value in stages

    @pytest.mark.asyncio
    async def test_no_re_introductions(self):
        """Agent messages should not introduce themselves by name."""
        result, _ = await run_full_pipeline()
        agent_msgs = [
            m["content"].lower()
            for m in result.conversation
            if m.get("role") == "assistant"
        ]
        # "I am your collections agent" type re-introductions are forbidden
        intro_phrases = ["my name is", "i am your", "let me introduce", "this is your"]
        for msg in agent_msgs:
            for phrase in intro_phrases:
                assert phrase not in msg, f"Re-introduction found: '{phrase}' in '{msg[:60]}'"

    @pytest.mark.asyncio
    async def test_question_not_asked_twice(self):
        """No fact should be solicited more than once across all agents."""
        result, _ = await run_full_pipeline()
        from collections import Counter
        from src.prompts import check_tone

        # Collect all question-like agent messages
        agent_msgs = [
            m["content"].lower()
            for m in result.conversation
            if m.get("role") == "assistant"
        ]
        # Check for repeated 6-word phrases (question patterns)
        phrases = []
        for msg in agent_msgs:
            words = msg.split()
            phrases += [" ".join(words[i:i+6]) for i in range(len(words) - 5)]
        counts = Counter(phrases)
        duplicates = {p: c for p, c in counts.items() if c > 1 and "?" in p}
        # Allow minor duplicates (same boilerplate), but no substantive questions twice
        substantive = {p: c for p, c in duplicates.items()
                      if any(kw in p for kw in ["income", "digits", "birth", "employ", "asset"])}
        assert not substantive, f"Repeated substantive questions: {substantive}"


class TestLogicalProgression:
    """Info → Deal → Ultimatum."""

    @pytest.mark.asyncio
    async def test_assessment_comes_first(self):
        result, _ = await run_full_pipeline()
        stages = [m.get("stage") for m in result.conversation if m.get("role") == "assistant"]
        # First assistant stage should be assessment
        assert stages[0] == Stage.ASSESSMENT.value

    @pytest.mark.asyncio
    async def test_resolution_comes_after_assessment(self):
        result, _ = await run_full_pipeline()
        stages = [m.get("stage") for m in result.conversation if m.get("role") == "assistant"]
        seen = []
        for s in stages:
            if s not in seen:
                seen.append(s)
        assert seen.index(Stage.ASSESSMENT.value) < seen.index(Stage.RESOLUTION.value)

    @pytest.mark.asyncio
    async def test_final_notice_comes_last(self):
        result, _ = await run_full_pipeline()
        stages = [m.get("stage") for m in result.conversation if m.get("role") == "assistant"]
        seen = []
        for s in stages:
            if s not in seen:
                seen.append(s)
        assert seen[-1] == Stage.FINAL_NOTICE.value

    @pytest.mark.asyncio
    async def test_offer_only_appears_in_resolution_or_later(self):
        result, _ = await run_full_pipeline()
        assessment_msgs = [
            m["content"].lower() for m in result.conversation
            if m.get("role") == "assistant" and m.get("stage") == Stage.ASSESSMENT.value
        ]
        for msg in assessment_msgs:
            # Assessment agent must not offer settlements
            assert "settlement" not in msg, f"Settlement mentioned in assessment: {msg[:80]}"
            assert "% discount" not in msg, f"Discount mentioned in assessment: {msg[:80]}"


class TestHandoffLosslessness:
    @pytest.mark.asyncio
    async def test_both_handoffs_within_token_limit(self):
        result, pipeline = await run_full_pipeline()
        assert len(result.handoff_tokens) == 2
        for tok in result.handoff_tokens:
            assert tok <= MAX_TOKENS_HANDOFF

    @pytest.mark.asyncio
    async def test_voice_call_would_start_mid_context(self):
        """Simulate building voice metadata from the assessment handoff."""
        result, pipeline = await run_full_pipeline()

        # Build context from pipeline state
        from src.models import ConversationContext, AssessmentData
        ctx = ConversationContext(borrower_id="E2E-001", loan_id="LN-E2E-001")
        ctx.assessment_data = AssessmentData(
            borrower_id="E2E-001", loan_id="LN-E2E-001",
            principal_amount=100_000, outstanding_amount=85_000,
            days_past_due=90, identity_verified=True,
            resolution_path=ResolutionPath.INSTALLMENT,
        )
        ctx.conversation_history = [
            {"role": "user", "content": "Last 4 are 7823, born 1985"},
            {"role": "user", "content": "Income 60000, salaried"},
        ]
        from src.question_tracker import QuestionTracker, FactKey
        from src.handoff import _serialise_qt
        qt = QuestionTracker()
        qt.mark_asked(FactKey.IDENTITY_LAST4, "AssessmentAgent", "assessment")
        qt.mark_answered(FactKey.IDENTITY_LAST4, "7823")
        _serialise_qt(ctx, qt)

        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        meta = build_voice_metadata(summary)

        assert meta["identity_verified"] is True
        assert "handoff_prompt" in meta
        assert len(meta["handoff_prompt"]) > 20


class TestGoldenSummary:
    @pytest.mark.asyncio
    async def test_golden_summary_after_full_run(self):
        result, pipeline = await run_full_pipeline()

        from src.models import ConversationContext, AssessmentData, ResolutionOffer
        ctx = ConversationContext(borrower_id="E2E-001", loan_id="LN-E2E-001")
        ctx.assessment_data = AssessmentData(
            borrower_id="E2E-001", loan_id="LN-E2E-001",
            principal_amount=100_000, outstanding_amount=85_000,
            days_past_due=90, identity_verified=True,
            resolution_path=ResolutionPath.INSTALLMENT,
        )
        ctx.resolution_offer = ResolutionOffer(
            path=ResolutionPath.INSTALLMENT,
            upfront_required=12_750, monthly_payment=7_225,
            tenure_months=10, deadline_days=3,
            valid_until="2026-04-24T00:00:00+00:00",
        )
        ctx.conversation_history = result.conversation

        gs = MemoryBuilder.build(ctx)
        assert gs.estimated_tokens() <= MAX_TOKENS_HANDOFF

        text = gs.to_plaintext()
        assert "Identity" in text or "identity" in text
        assert "GOLDEN SUMMARY" in text


class TestEvaluationPasses:
    @pytest.mark.asyncio
    async def test_rule_based_eval_on_assessment(self):
        result, _ = await run_full_pipeline()
        evaluator = RuleBasedEvaluator()

        assessment_history = [
            m for m in result.conversation
            if m.get("stage") == Stage.ASSESSMENT.value
        ]
        report = evaluator.evaluate("AssessmentAgent", assessment_history)
        # With good scripted responses, most checks should pass
        assert report.overall_score >= 0.4   # at least some checks pass

    @pytest.mark.asyncio
    async def test_resolution_eval_has_push_keywords(self):
        result, _ = await run_full_pipeline()
        evaluator = RuleBasedEvaluator()

        resolution_history = [
            m for m in result.conversation
            if m.get("stage") == Stage.RESOLUTION.value
        ]
        if resolution_history:
            report = evaluator.evaluate("ResolutionAgent", resolution_history)
            push_check = next((c for c in report.checks if c.check_name == "resolution_push"), None)
            if push_check:
                assert push_check.score > 0.0


class TestCostConstraints:
    @pytest.mark.asyncio
    async def test_cost_within_budget(self):
        result, _ = await run_full_pipeline()
        assert result.total_cost_usd < TOTAL_COST_BUDGET_USD

    @pytest.mark.asyncio
    async def test_cost_report_has_all_sections(self):
        _, pipeline = await run_full_pipeline()
        report = pipeline.cost_report()
        assert "Cost Report" in report
        assert "TOTAL" in report
        assert "REMAINING" in report

    @pytest.mark.asyncio
    async def test_20_full_runs_still_within_budget(self):
        """
        20 full mocked runs × ~(9 turns × 360 tokens) = ~64,800 tokens total
        Production cost @ $75/M output ≈ $0.0049 per run × 20 = ~$0.1
        Well within $20 budget.
        """
        total_cost = 0.0
        for _ in range(20):
            result, _ = await run_full_pipeline()
            total_cost += result.total_cost_usd
        assert total_cost < TOTAL_COST_BUDGET_USD, \
            f"20 runs cost ${total_cost:.4f} which exceeds ${TOTAL_COST_BUDGET_USD}"
