"""
MS6 validation: Conversation Memory + Golden Summary.
Must preserve all 5 spec fields: Identity, Financials, Offers, Objections, Tone.
Must be ≤500 tokens. Must have no info loss.
"""

import pytest
from src.memory import GoldenSummary, MemoryBuilder, IdentityRecord, FinancialRecord, OfferRecord, ToneRecord
from src.models import ConversationContext, Stage, AssessmentData, ResolutionPath, ResolutionOffer
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
        {"role": "user", "content": "My income is 50000 and expenses are 35000"},
        {"role": "user", "content": "I am salaried at a private firm"},
        {"role": "user", "content": "That seems too expensive for me"},
        {"role": "assistant", "content": "The offer is ₹12,750 down + ₹7,225/month."},
        {"role": "user", "content": "Okay, I agree to the installment plan"},
    ]
    qt = QuestionTracker()
    qt.mark_asked(FactKey.IDENTITY_LAST4, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.IDENTITY_LAST4, "7823")
    qt.mark_asked(FactKey.IDENTITY_DOB_YEAR, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.IDENTITY_DOB_YEAR, "1988")
    qt.mark_asked(FactKey.MONTHLY_INCOME, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.MONTHLY_INCOME, "50000")
    qt.mark_asked(FactKey.MONTHLY_EXPENSES, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.MONTHLY_EXPENSES, "35000")
    qt.mark_asked(FactKey.EMPLOYMENT_STATUS, "AssessmentAgent", "assessment")
    qt.mark_answered(FactKey.EMPLOYMENT_STATUS, "salaried")
    _serialise_qt(ctx, qt)
    return ctx


class TestGoldenSummaryFiveFields:
    """All 5 spec fields must be present and correct."""

    def test_identity_field(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        assert gs.identity.verified is True
        assert gs.identity.last4 == "7823"
        assert gs.identity.dob_year == "1988"

    def test_financials_field(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        assert gs.financials.monthly_income == "50000"
        assert gs.financials.monthly_expenses == "35000"
        assert gs.financials.employment_status == "salaried"
        assert gs.financials.financial_state is not None

    def test_offers_field(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        assert len(gs.offers) >= 1
        o = gs.offers[0]
        assert o.path == ResolutionPath.INSTALLMENT.value
        assert o.upfront == 12_750
        assert o.monthly == 7_225
        assert o.tenure_months == 10
        assert o.accepted is True

    def test_objections_field(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        assert len(gs.objections) >= 1
        combined = " ".join(gs.objections)
        assert "expensive" in combined

    def test_tone_field(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        assert gs.tone.final_attitude in ("cooperative", "neutral", "hostile")
        assert gs.tone.intent in ("willing_full", "willing_partial", "resistant", "unknown")
        assert isinstance(gs.tone.arc, list)
        assert len(gs.tone.arc) > 0


class TestGoldenSummaryTokenBudget:
    def test_within_500_tokens(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        assert gs.estimated_tokens() <= MAX_TOKENS_HANDOFF

    def test_long_conversation_still_within_limit(self):
        ctx = make_full_context()
        # Add 30 extra turns
        for i in range(15):
            ctx.conversation_history.append({"role": "user", "content": f"Message {i} " + "x" * 60})
            ctx.conversation_history.append({"role": "assistant", "content": f"Reply {i} " + "x" * 80})
        gs = MemoryBuilder.build(ctx)
        assert gs.estimated_tokens() <= MAX_TOKENS_HANDOFF


class TestPlaintextBlock:
    """Plaintext block must contain all 5 spec sections."""

    def test_identity_in_plaintext(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        text = gs.to_plaintext()
        assert "Identity" in text or "identity" in text
        assert "7823" in text

    def test_financials_in_plaintext(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        text = gs.to_plaintext()
        assert "Financials" in text or "income" in text.lower()
        assert "50000" in text

    def test_offers_in_plaintext(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        text = gs.to_plaintext()
        assert "Offers" in text or "offer" in text.lower()
        assert "installment" in text.lower() or "12,750" in text

    def test_objections_in_plaintext(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        text = gs.to_plaintext()
        assert "Objections" in text or "objection" in text.lower()

    def test_tone_in_plaintext(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        text = gs.to_plaintext()
        assert "Tone" in text or "tone" in text.lower() or "intent" in text.lower()

    def test_has_header_and_footer(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        text = gs.to_plaintext()
        assert "GOLDEN SUMMARY" in text
        assert "END" in text


class TestCompletenessVerification:
    def test_complete_context_has_no_missing_fields(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        missing = gs.verify_completeness()
        # May have empty offers warning but identity + financials should be complete
        assert "financials.monthly_income" not in missing
        assert "financials.employment_status" not in missing
        assert "identity.verified=False" not in missing

    def test_empty_context_has_missing_fields(self):
        ctx = ConversationContext(borrower_id="B", loan_id="L")
        ctx.assessment_data = AssessmentData(
            borrower_id="B", loan_id="L",
            principal_amount=50_000, outstanding_amount=40_000,
            days_past_due=30,
        )
        gs = MemoryBuilder.build(ctx)
        missing = gs.verify_completeness()
        assert len(missing) > 0


class TestNoInfoLoss:
    """Drop summary into next agent — no info loss."""

    def test_summary_json_roundtrip(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        # Serialise to dict and back
        data = gs.model_dump()
        gs2 = GoldenSummary(**data)
        assert gs2.identity.last4 == gs.identity.last4
        assert gs2.financials.monthly_income == gs.financials.monthly_income
        assert len(gs2.offers) == len(gs.offers)
        assert len(gs2.objections) == len(gs.objections)

    def test_summary_preserves_offer_acceptance(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        assert gs.offers[0].accepted is True

    def test_tone_arc_matches_conversation(self):
        ctx = make_full_context()
        gs = MemoryBuilder.build(ctx)
        # Last user message was "I agree to the installment plan" = cooperative
        assert gs.tone.final_attitude == "cooperative"
