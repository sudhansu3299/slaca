"""
MS2 validation: Cross-modal handoff design.
Validates the exact spec fields: identity_verified, debt_amount (outstanding_amount),
financial_state, intent, offers_presented, objections.
Also validates Agent2 never re-asks identity and Agent3 reflects voice context.
"""

import pytest
from src.handoff import HandoffSummary, HandoffBuilder, _collect_objections, _infer_intent
from src.models import (
    ConversationContext, Stage, AssessmentData, ResolutionOffer, ResolutionPath
)
from src.question_tracker import QuestionTracker, FactKey
from src.handoff import _serialise_qt
from src.token_budget import MAX_TOKENS_HANDOFF


def make_full_context() -> ConversationContext:
    ctx = ConversationContext(borrower_id="B001", loan_id="L001")
    ctx.assessment_data = AssessmentData(
        borrower_id="B001", loan_id="L001",
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
    ctx.resolution_outcome = "committed"
    ctx.conversation_history = [
        {"role": "user", "content": "Last 4 digits 7823, born 1988"},
        {"role": "assistant", "content": "Identity confirmed."},
        {"role": "user", "content": "My income is 50000 and I spend 35000"},
        {"role": "assistant", "content": "What is your employment status?"},
        {"role": "user", "content": "I am salaried at a private firm"},
        {"role": "assistant", "content": "The offer is 12750 upfront plus 7225 monthly."},
        {"role": "user", "content": "That seems too expensive for me"},
        {"role": "assistant", "content": "Terms are fixed. Can you commit?"},
        {"role": "user", "content": "Okay, I agree to the installment plan"},
    ]
    qt = QuestionTracker()
    qt.mark_asked(FactKey.IDENTITY_LAST4, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.IDENTITY_LAST4, "7823")
    qt.mark_asked(FactKey.IDENTITY_DOB_YEAR, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.IDENTITY_DOB_YEAR, "1988")
    qt.mark_asked(FactKey.MONTHLY_INCOME, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.MONTHLY_INCOME, "50000")
    qt.mark_asked(FactKey.EMPLOYMENT_STATUS, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.EMPLOYMENT_STATUS, "salaried")
    _serialise_qt(ctx, qt)
    return ctx


class TestSpecFields:
    """Validate the exact fields from milestone.txt spec."""

    def test_identity_verified_field(self):
        ctx = make_full_context()
        s = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert s.identity_verified is True

    def test_debt_amount_field(self):
        """milestone.txt calls it debt_amount — maps to outstanding_amount."""
        ctx = make_full_context()
        s = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert s.outstanding_amount == 85_000

    def test_financial_state_field(self):
        ctx = make_full_context()
        s = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert s.financial_state in ("stable", "unstable_income", "cash_strapped", "crisis")

    def test_intent_field(self):
        ctx = make_full_context()
        s = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
        assert s.intent in ("willing_full", "willing_partial", "resistant", "unknown")

    def test_intent_willing_full_after_agreement(self):
        ctx = make_full_context()
        s = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
        assert s.intent == "willing_full"

    def test_offers_presented_field(self):
        ctx = make_full_context()
        s = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
        assert isinstance(s.offers_presented, list)
        assert len(s.offers_presented) >= 1
        offer = s.offers_presented[0]
        assert "path" in offer
        assert "upfront" in offer

    def test_objections_field(self):
        ctx = make_full_context()
        s = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert isinstance(s.objections, list)
        # "too expensive" should be captured
        combined = " ".join(s.objections)
        assert "expensive" in combined

    def test_objections_empty_when_none(self):
        ctx = ConversationContext(borrower_id="B", loan_id="L")
        ctx.assessment_data = AssessmentData(
            borrower_id="B", loan_id="L",
            principal_amount=50_000, outstanding_amount=40_000,
            days_past_due=30,
        )
        ctx.conversation_history = [
            {"role": "user", "content": "Sure, I understand"},
        ]
        s = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert s.objections == []


class TestNoReVerification:
    """Agent 2 must never ask identity again."""

    def test_agent2_context_knows_identity(self):
        ctx = make_full_context()
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        receiving = ConversationContext(borrower_id="B001", loan_id="L001")
        HandoffBuilder.apply_to_context(summary, receiving)

        qt = QuestionTracker()
        for k, v in receiving.question_state.items():
            if v.get("answered"):
                try:
                    fk = FactKey(k)
                    qt.mark_asked(fk, "", "")
                    qt.mark_answered(fk, v.get("value", ""))
                except ValueError:
                    pass

        assert qt.is_answered(FactKey.IDENTITY_LAST4)
        assert qt.is_answered(FactKey.IDENTITY_DOB_YEAR)

    def test_no_identity_question_violation(self):
        ctx = make_full_context()
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        receiving = ConversationContext(borrower_id="B001", loan_id="L001")
        HandoffBuilder.apply_to_context(summary, receiving)

        qt = QuestionTracker()
        for k, v in receiving.question_state.items():
            if v.get("answered"):
                try:
                    fk = FactKey(k)
                    qt.mark_asked(fk, "", "")
                    qt.mark_answered(fk, v.get("value", ""))
                except ValueError:
                    pass

        violations = qt.validate_no_repeat("Can I confirm your account's last 4 digits?")
        assert "identity_last4" in violations


class TestVoiceKnowsChatContext:
    """Voice agent receives full financial context from chat."""

    def test_financial_facts_in_summary(self):
        ctx = make_full_context()
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        assert summary.monthly_income == "50000"
        assert summary.employment_status == "salaried"

    def test_prompt_block_includes_financial_state(self):
        ctx = make_full_context()
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        block = summary.to_prompt_block()
        assert "financial" in block.lower() or summary.financial_state in block

    def test_prompt_block_includes_intent(self):
        ctx = make_full_context()
        # add intent-indicating message
        ctx.conversation_history.append({"role": "user", "content": "I agree to pay"})
        summary = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
        block = summary.to_prompt_block()
        assert "intent" in block.lower() or summary.intent in block


class TestChatReflectsVoice:
    """Agent 3 (chat) sees what happened in the voice call."""

    def test_offer_carries_through_to_final_notice(self):
        ctx = make_full_context()
        summary = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
        assert summary.offer_path == ResolutionPath.INSTALLMENT.value
        assert summary.offer_monthly == 7_225
        assert summary.resolution_committed is True

    def test_prompt_block_has_offer_detail(self):
        ctx = make_full_context()
        summary = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
        block = summary.to_prompt_block()
        assert "12750" in block or "12,750" in block or "installment" in block.lower()

    def test_objections_from_voice_in_final_notice(self):
        ctx = make_full_context()
        summary = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
        # "too expensive" was said during the call — must appear in objections
        combined = " ".join(summary.objections)
        assert "expensive" in combined


class TestTokenBudget:
    def test_all_handoffs_within_500_tokens(self):
        ctx = make_full_context()
        for from_s, to_s in [
            (Stage.ASSESSMENT, Stage.RESOLUTION),
            (Stage.RESOLUTION, Stage.FINAL_NOTICE),
        ]:
            s = HandoffBuilder.build(ctx, from_s, to_s)
            assert s.estimated_tokens() <= MAX_TOKENS_HANDOFF
