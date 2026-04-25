"""
Canonical agent prompts — versioned and personality-driven.

Each prompt is a pure string function so it can be:
- Tested in isolation
- Updated by the self-learning loop
- Injected with evolved guidance
"""

from __future__ import annotations

from pathlib import Path


_PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"


def _read_prompt_template(filename: str) -> str:
    return (_PROMPTS_DIR / filename).read_text(encoding="utf-8")


def _guidance_block(injected_guidance: str) -> str:
    return f"\n{injected_guidance}" if injected_guidance else ""


def assessment_system_prompt(known_facts_block: str, injected_guidance: str = "") -> str:
    template = _read_prompt_template("assessment_system_prompt.txt")
    return template.format(
        known_facts_block=known_facts_block,
        injected_guidance_block=_guidance_block(injected_guidance),
    )


def resolution_system_prompt(
    known_facts_block: str,
    offer_block: str,
    injected_guidance: str = "",
) -> str:
    template = _read_prompt_template("resolution_system_prompt.txt")
    return template.format(
        known_facts_block=known_facts_block,
        offer_block=offer_block,
        injected_guidance_block=_guidance_block(injected_guidance),
    )


def final_notice_system_prompt(
    known_facts_block: str,
    final_offer_block: str,
    voice_handoff_block: str = "",
    injected_guidance: str = "",
) -> str:
    template = _read_prompt_template("final_notice_system_prompt.txt")
    return template.format(
        known_facts_block=known_facts_block,
        final_offer_block=final_offer_block,
        voice_handoff_block=voice_handoff_block,
        injected_guidance_block=_guidance_block(injected_guidance),
    )


def evaluator_judge_prompt() -> str:
    return _read_prompt_template("evaluator_judge_prompt.txt")


def vapi_first_message_prompt(
    outstanding_amount: str,
    employment_status: str,
    monthly_income: str,
    resolution_path: str,
    offer_upfront: str = "",
    offer_monthly: str = "",
    offer_tenure_months: str = "",
) -> str:
    template = _read_prompt_template("vapi_first_message_prompt.txt")
    return template.format(
        outstanding_amount=outstanding_amount,
        employment_status=employment_status,
        monthly_income=monthly_income,
        resolution_path=resolution_path,
        offer_upfront=offer_upfront or outstanding_amount,
        offer_monthly=offer_monthly,
        offer_tenure_months=offer_tenure_months,
    )


def vapi_system_override_prompt(handoff_prompt: str) -> str:
    template = _read_prompt_template("vapi_system_override_prompt.txt")
    return template.format(handoff_prompt=handoff_prompt)


# ──────────────────────────────────────────────
# Tone validation helpers (used in tests)
# ──────────────────────────────────────────────

EMPATHY_PHRASES = [
    "i understand how",
    "that must be",
    "don't worry",
    "i'm sorry",
    "we want to help",
    "i know this is hard",
    "i can imagine",
    "must be difficult",
]

ASSESSMENT_FORBIDDEN = EMPATHY_PHRASES + [
    "settlement",
    "offer you",
    "legal proceedings",
    "credit score",
]

RESOLUTION_FORBIDDEN = EMPATHY_PHRASES + [
    "i understand your situation",
    "let me explain why",
    "the reason for your debt",
]

FINAL_NOTICE_FORBIDDEN = EMPATHY_PHRASES + [
    "we really want",
    "please don't worry",
    "extended deadline",
    "new offer",
    "different terms",
]


def check_tone(agent_name: str, response_text: str) -> list[str]:
    """
    Returns list of forbidden phrases found in response.
    Empty list = tone is correct.
    """
    text = response_text.lower()
    if agent_name == "AssessmentAgent":
        forbidden = ASSESSMENT_FORBIDDEN
    elif agent_name == "ResolutionAgent":
        forbidden = RESOLUTION_FORBIDDEN
    elif agent_name == "FinalNoticeAgent":
        forbidden = FINAL_NOTICE_FORBIDDEN
    else:
        return []
    return [p for p in forbidden if p in text]
