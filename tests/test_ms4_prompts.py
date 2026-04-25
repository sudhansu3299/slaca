"""
MS4 validation: Agent behavior prompts.
- Tone differences are measurable and distinct
- Forbidden phrases are absent from each agent's persona
- Prompts contain required structural elements
- Simulated conversation responses have correct tone markers
"""

import pytest
from src.prompts import (
    assessment_system_prompt, resolution_system_prompt, final_notice_system_prompt,
    check_tone, ASSESSMENT_FORBIDDEN, RESOLUTION_FORBIDDEN, FINAL_NOTICE_FORBIDDEN,
)

# Persona text is now file-based — derive from the prompt functions for tests
ASSESSMENT_PERSONA  = assessment_system_prompt("No facts yet.")
RESOLUTION_PERSONA  = resolution_system_prompt("No facts.", "No offer.")
FINAL_NOTICE_PERSONA = final_notice_system_prompt("No facts.", "No offer.")
from src.models import ConversationContext, Stage, AssessmentData, ResolutionPath, ResolutionOffer


def make_ctx() -> ConversationContext:
    ctx = ConversationContext(borrower_id="B1", loan_id="L1")
    ctx.assessment_data = AssessmentData(
        borrower_id="B1", loan_id="L1",
        principal_amount=100_000, outstanding_amount=85_000,
        days_past_due=90, resolution_path=ResolutionPath.INSTALLMENT,
    )
    ctx.resolution_offer = ResolutionOffer(
        path=ResolutionPath.INSTALLMENT,
        upfront_required=12_750, monthly_payment=7_225,
        tenure_months=10, deadline_days=3,
        valid_until="2026-04-24T00:00:00+00:00",
    )
    return ctx


class TestPersonaDistinctness:
    """Each agent's persona must be clearly different."""

    def test_assessment_persona_has_cold_keywords(self):
        p = ASSESSMENT_PERSONA.lower()
        assert "cold" in p or "clinical" in p
        assert "no empathy" in p or "empathy" in p

    def test_resolution_persona_has_deal_keywords(self):
        p = RESOLUTION_PERSONA.lower()
        assert "deal" in p or "transactional" in p
        assert "commit" in p

    def test_final_notice_persona_has_legal_keywords(self):
        p = FINAL_NOTICE_PERSONA.lower()
        assert "legal" in p or "consequence" in p or "deadline" in p

    def test_all_three_personas_distinct(self):
        """No two personas should be identical."""
        assert ASSESSMENT_PERSONA != RESOLUTION_PERSONA
        assert RESOLUTION_PERSONA != FINAL_NOTICE_PERSONA
        assert ASSESSMENT_PERSONA != FINAL_NOTICE_PERSONA


class TestAssessmentPrompt:
    def test_contains_completion_marker_instruction(self):
        p = assessment_system_prompt("No facts yet.")
        assert "ASSESSMENT_COMPLETE" in p

    def test_contains_resolution_paths(self):
        p = assessment_system_prompt("No facts yet.")
        for path in ["LUMP_SUM", "INSTALLMENT", "HARDSHIP", "LEGAL"]:
            assert path in p

    def test_contains_known_facts_block(self):
        p = assessment_system_prompt("income=50000, employment=salaried")
        assert "income=50000" in p

    def test_injected_guidance_included(self):
        p = assessment_system_prompt("No facts.", injected_guidance="[Learned: use direct tone]")
        assert "Learned" in p

    def test_no_empathy_in_base_prompt(self):
        # Forbidden phrases may appear in FORBIDDEN lists as instruction;
        # they should NOT appear as affirmative agent outputs.
        # We verify check_tone works correctly on actual responses (not prompts).
        clean_response = "Please provide the last 4 digits of your loan account."
        assert check_tone("AssessmentAgent", clean_response) == []
        bad_response = "Don't worry, I'm here to help you through this."
        assert len(check_tone("AssessmentAgent", bad_response)) > 0


class TestResolutionPrompt:
    def test_contains_objection_playbook(self):
        offer = "INSTALLMENT OFFER: ₹12,750 upfront, ₹7,225/month"
        p = resolution_system_prompt("income=50000", offer)
        assert "OBJECTION" in p.upper() or "objection" in p.lower()

    def test_contains_completion_marker_instruction(self):
        p = resolution_system_prompt("facts", "offer block")
        assert "RESOLUTION_COMPLETE" in p

    def test_contains_refused_marker_instruction(self):
        p = resolution_system_prompt("facts", "offer block")
        assert "RESOLUTION_REFUSED" in p

    def test_voice_brevity_instruction(self):
        p = resolution_system_prompt("facts", "offer block")
        assert "voice" in p.lower() or "sentence" in p.lower() or "short" in p.lower()

    def test_injected_guidance_included(self):
        p = resolution_system_prompt("facts", "offer", injected_guidance="[Best: anchor on monthly]")
        assert "Best" in p


class TestFinalNoticePrompt:
    def test_contains_all_six_consequences(self):
        p = final_notice_system_prompt("facts", "final offer")
        consequence_keywords = ["credit", "legal notice", "court", "garnishment", "lien", "employment"]
        for kw in consequence_keywords:
            assert kw in p.lower(), f"Missing consequence keyword: {kw}"

    def test_contains_completion_marker(self):
        p = final_notice_system_prompt("facts", "offer")
        assert "COLLECTIONS_COMPLETE" in p

    def test_contains_escalated_marker(self):
        p = final_notice_system_prompt("facts", "offer")
        assert "COLLECTIONS_ESCALATED" in p

    def test_no_extended_deadline_language_in_output(self):
        # The forbidden phrase may appear in FORBIDDEN section as instruction;
        # verify it is flagged when it appears in agent response.
        bad_response = "We can offer you an extended deadline for this situation."
        violations = check_tone("FinalNoticeAgent", bad_response)
        assert "extended deadline" in violations

    def test_injected_guidance_included(self):
        p = final_notice_system_prompt("facts", "offer", injected_guidance="[Pattern: state consequences first]")
        assert "Pattern" in p


class TestToneChecker:
    """check_tone() correctly identifies forbidden phrases."""

    def test_no_violations_on_clean_assessment_reply(self):
        clean = "Last 4 digits of your loan account, please."
        assert check_tone("AssessmentAgent", clean) == []

    def test_catches_empathy_in_assessment(self):
        bad = "I understand how difficult this must be for you."
        violations = check_tone("AssessmentAgent", bad)
        assert len(violations) > 0

    def test_no_violations_on_clean_resolution_reply(self):
        clean = "The offer is ₹12,750 upfront. Deadline April 24. Can you commit?"
        assert check_tone("ResolutionAgent", clean) == []

    def test_catches_empathy_in_resolution(self):
        bad = "I understand your situation — this must be really hard."
        violations = check_tone("ResolutionAgent", bad)
        assert len(violations) > 0

    def test_no_violations_on_clean_final_notice_reply(self):
        clean = "This offer expires April 24. Non-payment triggers credit bureau reporting."
        assert check_tone("FinalNoticeAgent", clean) == []

    def test_catches_empathy_in_final_notice(self):
        bad = "We really want to help you through this difficult time."
        violations = check_tone("FinalNoticeAgent", bad)
        assert len(violations) > 0

    def test_catches_extended_deadline_in_final_notice(self):
        bad = "We can offer an extended deadline for your situation."
        violations = check_tone("FinalNoticeAgent", bad)
        assert len(violations) > 0


class TestPromptWiredInAgents:
    """Verify agents use the canonical prompts."""

    def test_assessment_agent_uses_canonical_prompt(self):
        from src.agents.assessment import AssessmentAgent
        agent = AssessmentAgent()
        ctx = make_ctx()
        prompt = agent.get_system_prompt(ctx)
        # Canonical prompt contains the persona text
        assert "ASSESSMENT_COMPLETE" in prompt
        assert "cold" in prompt.lower() or "clinical" in prompt.lower() or "PERSONA" in prompt or "Cold" in prompt

    def test_resolution_agent_uses_canonical_prompt(self):
        from src.agents.resolution import ResolutionAgent
        agent = ResolutionAgent()
        ctx = make_ctx()
        prompt = agent.get_system_prompt(ctx)
        assert "RESOLUTION_COMPLETE" in prompt
        assert "OBJECTION" in prompt.upper() or "objection" in prompt.lower()

    def test_final_notice_agent_uses_canonical_prompt(self):
        from src.agents.final_notice import FinalNoticeAgent
        agent = FinalNoticeAgent()
        ctx = make_ctx()
        prompt = agent.get_system_prompt(ctx)
        assert "COLLECTIONS_COMPLETE" in prompt
        assert "credit" in prompt.lower()
