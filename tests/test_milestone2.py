"""
Milestone 2 validation tests.
- HandoffSummary renders within 500 tokens
- All financial facts carry through losslessly
- Identity, loan, offer fields survive serialisation
- apply_to_context prevents repeat questions in receiving agent
- Oversized summary raises ValueError
"""

import pytest

from src.handoff import HandoffSummary, HandoffBuilder, _estimate_tokens
from src.models import (
    ConversationContext, Stage, AssessmentData, ResolutionOffer, ResolutionPath
)
from src.question_tracker import QuestionTracker, FactKey
from src.token_budget import MAX_TOKENS_HANDOFF


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def make_assessed_context() -> ConversationContext:
    ctx = ConversationContext(borrower_id="B001", loan_id="L001")
    ctx.assessment_data = AssessmentData(
        borrower_id="B001",
        loan_id="L001",
        principal_amount=100_000,
        outstanding_amount=85_000,
        days_past_due=90,
        identity_verified=True,
        resolution_path=ResolutionPath.INSTALLMENT,
    )
    # Populate question tracker
    qt = QuestionTracker()
    qt.mark_asked(FactKey.MONTHLY_INCOME, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.MONTHLY_INCOME, "50000")
    qt.mark_asked(FactKey.MONTHLY_EXPENSES, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.MONTHLY_EXPENSES, "35000")
    qt.mark_asked(FactKey.EMPLOYMENT_STATUS, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.EMPLOYMENT_STATUS, "salaried")
    qt.mark_asked(FactKey.ASSETS_VALUE, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.ASSETS_VALUE, "200000")
    qt.mark_asked(FactKey.LIABILITIES_VALUE, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.LIABILITIES_VALUE, "30000")
    qt.mark_asked(FactKey.CASH_FLOW_ISSUE, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.CASH_FLOW_ISSUE, "mild")
    from src.handoff import _serialise_qt
    _serialise_qt(ctx, qt)
    return ctx


def add_offer(ctx: ConversationContext) -> None:
    ctx.resolution_offer = ResolutionOffer(
        path=ResolutionPath.INSTALLMENT,
        upfront_required=12_750,
        monthly_payment=8_500,
        tenure_months=10,
        deadline_days=3,
        valid_until="2026-04-24T00:00:00",
    )
    ctx.resolution_outcome = "committed"


# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #

class TestHandoffTokenBudget:
    def test_assessment_to_resolution_within_500_tokens(self):
        ctx = make_assessed_context()
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert summary.estimated_tokens() <= MAX_TOKENS_HANDOFF

    def test_resolution_to_final_notice_within_500_tokens(self):
        ctx = make_assessed_context()
        add_offer(ctx)
        summary = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
        assert summary.estimated_tokens() <= MAX_TOKENS_HANDOFF

    def test_oversized_summary_raises(self):
        """Artificially long objection should be truncated, not raise — because we truncate at 120 chars."""
        # The to_prompt_block enforces the limit. This tests our _enforce_token_limit directly.
        from src.handoff import _enforce_token_limit
        long_text = "x" * (MAX_TOKENS_HANDOFF * 4 + 100)
        with pytest.raises(ValueError, match="exceeds limit"):
            _enforce_token_limit(long_text, "test")


class TestHandoffLosslessness:
    def test_financial_facts_preserved(self):
        ctx = make_assessed_context()
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert summary.monthly_income == "50000"
        assert summary.monthly_expenses == "35000"
        assert summary.employment_status == "salaried"
        assert summary.assets_value == "200000"
        assert summary.liabilities_value == "30000"
        assert summary.cash_flow_issue == "mild"

    def test_identity_verified_preserved(self):
        ctx = make_assessed_context()
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert summary.identity_verified is True

    def test_loan_facts_preserved(self):
        ctx = make_assessed_context()
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert summary.outstanding_amount == 85_000
        assert summary.days_past_due == 90
        assert summary.principal_amount == 100_000

    def test_resolution_path_preserved(self):
        ctx = make_assessed_context()
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert summary.resolution_path == ResolutionPath.INSTALLMENT.value

    def test_offer_preserved(self):
        ctx = make_assessed_context()
        add_offer(ctx)
        summary = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
        assert summary.offer_path == ResolutionPath.INSTALLMENT.value
        assert summary.offer_monthly == 8_500
        assert summary.offer_tenure_months == 10
        assert summary.resolution_committed is True

    def test_prompt_block_contains_key_facts(self):
        ctx = make_assessed_context()
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        block = summary.to_prompt_block()
        assert "50000" in block
        assert "installment" in block.lower()
        assert "₹85,000" in block


class TestApplyToContext:
    def test_apply_populates_question_state(self):
        ctx = make_assessed_context()
        receiving_ctx = ConversationContext(borrower_id="B001", loan_id="L001")

        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        HandoffBuilder.apply_to_context(summary, receiving_ctx)

        qt = QuestionTracker()
        for key_str, state in receiving_ctx.question_state.items():
            try:
                fk = FactKey(key_str)
                if state.get("answered") and state.get("value"):
                    qt.mark_asked(fk, "", "")
                    qt.mark_answered(fk, state["value"])
            except ValueError:
                pass

        assert qt.is_answered(FactKey.MONTHLY_INCOME)
        assert qt.get_answered()[FactKey.MONTHLY_INCOME] == "50000"

    def test_receiving_agent_no_repeat_after_apply(self):
        """After apply, no-repeat check should flag monthly_income question."""
        ctx = make_assessed_context()
        receiving_ctx = ConversationContext(borrower_id="B001", loan_id="L001")
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        HandoffBuilder.apply_to_context(summary, receiving_ctx)

        # Build tracker from receiving context
        qt = QuestionTracker()
        for key_str, state in receiving_ctx.question_state.items():
            try:
                fk = FactKey(key_str)
                if state.get("answered") and state.get("value"):
                    qt.mark_asked(fk, "", "")
                    qt.mark_answered(fk, state["value"])
            except ValueError:
                pass

        violations = qt.validate_no_repeat("What is your monthly income?")
        assert "monthly_income" in violations

    def test_identity_verified_applied(self):
        ctx = make_assessed_context()
        receiving_ctx = ConversationContext(borrower_id="B001", loan_id="L001")
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        HandoffBuilder.apply_to_context(summary, receiving_ctx)

        qt = QuestionTracker()
        for key_str, state in receiving_ctx.question_state.items():
            try:
                fk = FactKey(key_str)
                if state.get("answered"):
                    qt.mark_asked(fk, "", "")
                    qt.mark_answered(fk, state.get("value", ""))
            except ValueError:
                pass

        assert qt.is_answered(FactKey.IDENTITY_LAST4)


class TestEstimatedTokens:
    def test_estimate_tokens_basic(self):
        # 400 chars → ~100 tokens
        text = "a" * 400
        assert _estimate_tokens(text) == 100

    def test_short_summary_well_under_limit(self):
        summary = HandoffSummary(
            from_stage="assessment",
            to_stage="resolution",
            borrower_id="B1",
            loan_id="L1",
        )
        assert summary.estimated_tokens() <= MAX_TOKENS_HANDOFF
