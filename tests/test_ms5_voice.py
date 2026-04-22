"""
MS5 validation: Voice integration.
- Call starts mid-context (no restart feel)
- Metadata contains all chat context
- Agent responds to prior chat data naturally
- Provider abstraction works for mock
"""

import pytest
from src.voice import (
    VoiceProvider, VoiceProviderName, CallConfig, CallRecord,
    build_voice_metadata,
)
from src.handoff import HandoffSummary, HandoffBuilder
from src.models import (
    ConversationContext, Stage, AssessmentData, ResolutionPath, ResolutionOffer
)
from src.question_tracker import QuestionTracker, FactKey
from src.handoff import _serialise_qt


def make_assessed_summary() -> HandoffSummary:
    ctx = ConversationContext(borrower_id="B001", loan_id="L001")
    ctx.assessment_data = AssessmentData(
        borrower_id="B001", loan_id="L001",
        principal_amount=100_000, outstanding_amount=85_000,
        days_past_due=90, identity_verified=True,
        resolution_path=ResolutionPath.INSTALLMENT,
    )
    ctx.conversation_history = [
        {"role": "user", "content": "My income is 50000 and I spend 35000"},
        {"role": "user", "content": "I am salaried at a firm"},
        {"role": "user", "content": "That seems too expensive"},
        {"role": "user", "content": "Okay, I can manage the installment"},
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
    return HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)


class TestVoiceMetadata:
    """Metadata injected into call must contain all chat context."""

    def test_identity_verified_in_metadata(self):
        summary = make_assessed_summary()
        meta = build_voice_metadata(summary)
        assert meta["identity_verified"] is True

    def test_financial_context_in_metadata(self):
        summary = make_assessed_summary()
        meta = build_voice_metadata(summary)
        assert meta["monthly_income"] == "50000"
        assert meta["employment_status"] == "salaried"

    def test_financial_state_in_metadata(self):
        summary = make_assessed_summary()
        meta = build_voice_metadata(summary)
        assert meta["financial_state"] in ("stable", "unstable_income", "cash_strapped", "crisis")

    def test_intent_in_metadata(self):
        summary = make_assessed_summary()
        meta = build_voice_metadata(summary)
        assert meta["intent"] in ("willing_full", "willing_partial", "resistant", "unknown")

    def test_resolution_path_in_metadata(self):
        summary = make_assessed_summary()
        meta = build_voice_metadata(summary)
        assert meta["resolution_path"] == ResolutionPath.INSTALLMENT.value

    def test_prior_objections_in_metadata(self):
        summary = make_assessed_summary()
        meta = build_voice_metadata(summary)
        assert isinstance(meta["prior_objections"], str)
        # "too expensive" was in the chat
        assert "expensive" in meta["prior_objections"]

    def test_handoff_prompt_in_metadata(self):
        summary = make_assessed_summary()
        meta = build_voice_metadata(summary)
        assert "HANDOFF" in meta["handoff_prompt"]
        assert len(meta["handoff_prompt"]) > 50

    def test_metadata_has_all_required_keys(self):
        summary = make_assessed_summary()
        meta = build_voice_metadata(summary)
        required = [
            "identity_verified", "borrower_id", "loan_id",
            "outstanding_amount", "days_past_due", "monthly_income",
            "employment_status", "financial_state", "intent",
            "resolution_path", "prior_objections", "handoff_prompt",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"


class TestMockVoiceProvider:
    """Mock provider for local testing without real API."""

    @pytest.mark.asyncio
    async def test_mock_call_initiated(self):
        provider = VoiceProvider()  # defaults to MOCK
        summary = make_assessed_summary()
        record = await provider.initiate_call("+919999999999", summary)
        assert record.status == "in_progress"
        assert record.call_id.startswith("mock_")

    @pytest.mark.asyncio
    async def test_mock_call_has_metadata(self):
        provider = VoiceProvider()
        summary = make_assessed_summary()
        record = await provider.initiate_call("+919999999999", summary)
        assert record.metadata["identity_verified"] is True
        assert record.metadata["monthly_income"] == "50000"

    @pytest.mark.asyncio
    async def test_mock_transcript_retrieval(self):
        provider = VoiceProvider()
        summary = make_assessed_summary()
        record = await provider.initiate_call("+919999999999", summary)
        # Simulate transcript from voice call
        record.transcript = "I agree to the installment plan"
        transcript = await provider.get_transcript(record)
        assert transcript == "I agree to the installment plan"

    @pytest.mark.asyncio
    async def test_mock_call_end(self):
        provider = VoiceProvider()
        summary = make_assessed_summary()
        record = await provider.initiate_call("+919999999999", summary)
        await provider.end_call(record)
        assert record.status == "completed"


class TestMidContextCallStart:
    """Validate that the call starts mid-context — no re-introduction feel."""

    def test_no_identity_reask_needed(self):
        """Since identity is in metadata, voice agent should not ask for it."""
        summary = make_assessed_summary()
        meta = build_voice_metadata(summary)
        # identity_verified=True means the voice agent won't ask again
        assert meta["identity_verified"] is True

    def test_handoff_prompt_references_prior_chat(self):
        """The handoff_prompt injected into voice system references chat facts."""
        summary = make_assessed_summary()
        meta = build_voice_metadata(summary)
        prompt = meta["handoff_prompt"]
        # Should reference the known financial facts from chat
        assert "50000" in prompt or "installment" in prompt.lower()

    def test_objections_from_chat_available_to_voice(self):
        """Voice agent receives objections raised in chat."""
        summary = make_assessed_summary()
        meta = build_voice_metadata(summary)
        # "too expensive" was raised in chat
        assert "expensive" in meta["prior_objections"]

    @pytest.mark.asyncio
    async def test_call_metadata_contains_full_context(self):
        provider = VoiceProvider()
        summary = make_assessed_summary()
        record = await provider.initiate_call("+919999999999", summary)
        # Metadata must include all 12+ keys
        assert len(record.metadata) >= 12


class TestProviderConfig:
    def test_mock_provider_name(self):
        provider = VoiceProvider()
        assert provider.config.provider == VoiceProviderName.MOCK

    def test_retell_config(self):
        config = CallConfig(
            provider=VoiceProviderName.RETELL,
            agent_id="ret_agent_123",
            caller_number="+1234567890",
            api_key="test_key",
        )
        assert config.provider == VoiceProviderName.RETELL
        assert config.agent_id == "ret_agent_123"

    def test_vapi_config(self):
        config = CallConfig(
            provider=VoiceProviderName.VAPI,
            agent_id="vapi_asst_123",
            caller_number="+1234567890",
        )
        assert config.provider == VoiceProviderName.VAPI
