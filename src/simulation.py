"""
Simulation Engine (MS8)

Generates synthetic borrowers across 4 personas:
1. Cooperative    — responds fully, agrees to terms
2. Hostile        — refuses, uses threatening language
3. Broke          — genuinely can't pay, requests hardship
4. Strategic Defaulter — can pay but avoids, stalls, negotiates aggressively

Each persona has a scripted response sequence per agent stage.
The simulation engine produces RunMetrics for the self-learning loop.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.models import ResolutionPath


class PersonaType(str, Enum):
    COOPERATIVE = "cooperative"
    HOSTILE = "hostile"
    BROKE = "broke"
    STRATEGIC_DEFAULTER = "strategic_defaulter"


@dataclass
class BorrowerProfile:
    borrower_id: str
    loan_id: str
    persona: PersonaType
    phone_number: str = "+919999999999"
    principal_amount: float = 100_000
    outstanding_amount: float = 85_000
    days_past_due: int = 90

    # Expected outcome for this persona (used in test assertions)
    expected_resolution_path: Optional[ResolutionPath] = None
    expected_outcome: Optional[str] = None   # resolved|escalated|hardship


class PersonaScript:
    """
    Scripted response sequences for each persona × agent combination.
    Each script is a list of responses in order.
    Falls back to a neutral reply if the script runs out.
    """

    SCRIPTS: dict[PersonaType, dict[str, list[str]]] = {

        PersonaType.COOPERATIVE: {
            "AssessmentAgent": [
                "Hi, I received the collections notice.",
                "Last 4 digits are 7823, born in 1985.",
                "My monthly income is 60000 and I spend about 40000.",
                "I am salaried at a private company.",
                "I have a vehicle worth about 150000.",
                "No other significant liabilities.",
                "Yes, things are a bit tight but manageable.",
            ],
            "ResolutionAgent": [
                "What options do I have?",
                "The installment plan sounds reasonable.",
                "Yes, I can commit to the 10-month plan. I agree.",
            ],
            "FinalNoticeAgent": [
                "I understand. Can I still take the installment offer?",
                "Yes, I accept the terms. Please proceed.",
            ],
        },

        PersonaType.HOSTILE: {
            "AssessmentAgent": [
                "Why are you calling me?",
                "I'm not giving you any information.",
                "This debt is not valid. I'm consulting my lawyer.",
                "Stop harassing me. I refuse to answer.",
                "7823, fine. 1972. Now leave me alone.",
                "I have no income. Zero. Unemployed.",
                "I have nothing. No assets. Nothing.",
            ],
            "ResolutionAgent": [
                "I'm not agreeing to anything.",
                "Your offer is a joke. I won't pay.",
                "Sue me. I refuse to pay this amount.",
            ],
            "FinalNoticeAgent": [
                "I don't care about your threats.",
                "Take me to court. I won't pay.",
            ],
        },

        PersonaType.BROKE: {
            "AssessmentAgent": [
                "I got the notice. I'm in a really difficult situation.",
                "Last 4 are 4512, born 1990.",
                "I lost my job 3 months ago. No income right now.",
                "I spend about 15000 on basic expenses from savings.",
                "I'm currently unemployed, looking for work.",
                "No significant assets. Just a small bank balance.",
                "I have another loan of 20000 also pending.",
            ],
            "ResolutionAgent": [
                "I genuinely cannot pay the full amount right now.",
                "Even the installment is too high. Can you do a hardship plan?",
                "I'll try to pay 2000 per month if possible.",
            ],
            "FinalNoticeAgent": [
                "I understand the consequences. I really can't pay more.",
                "Is there any hardship program? I'm not refusing, I'm broke.",
            ],
        },

        PersonaType.STRATEGIC_DEFAULTER: {
            "AssessmentAgent": [
                "Yes I got the notice. What exactly is owed?",
                "Last 4 digits are 9901, born 1980.",
                "My income is variable. Around 80000 some months.",
                "Expenses are about 60000 monthly.",
                "I run my own business. Self-employed.",
                "I have property worth 5 lakhs but it's mortgaged.",
                "I have 3 other loans. Total 2 lakhs outstanding.",
            ],
            "ResolutionAgent": [
                "What's the maximum discount you can offer?",
                "I can pay a lump sum but I need 30% off.",
                "If you can do 35% off, I'll pay today. Otherwise I'll wait.",
            ],
            "FinalNoticeAgent": [
                "I know you'll report to credit bureau. I've already factored that in.",
                "My best offer is 50% settlement. Take it or leave it.",
            ],
        },
    }

    FALLBACKS: dict[PersonaType, str] = {
        PersonaType.COOPERATIVE: "I understand. Please continue.",
        PersonaType.HOSTILE: "I have nothing more to say.",
        PersonaType.BROKE: "I really can't afford more than this.",
        PersonaType.STRATEGIC_DEFAULTER: "I'm not accepting these terms.",
    }

    def __init__(self, persona: PersonaType):
        self.persona = persona
        self._turn: dict[str, int] = {}

    def respond(self, agent_name: str) -> str:
        script = self.SCRIPTS.get(self.persona, {}).get(agent_name, [])
        idx = self._turn.get(agent_name, 0)
        self._turn[agent_name] = idx + 1
        if idx < len(script):
            return script[idx]
        return self.FALLBACKS.get(self.persona, "I understand.")

    def reset(self) -> None:
        self._turn = {}


class SimulationEngine:
    """
    Runs simulated conversations for all 4 personas.
    Used by the self-learning loop to generate training data.
    """

    @staticmethod
    def make_profiles() -> list[BorrowerProfile]:
        return [
            BorrowerProfile(
                borrower_id=f"SIM-COOP-{uuid.uuid4().hex[:6]}",
                loan_id="LN-COOP-001",
                persona=PersonaType.COOPERATIVE,
                outstanding_amount=85_000,
                days_past_due=90,
                expected_resolution_path=ResolutionPath.INSTALLMENT,
                expected_outcome="resolved",
            ),
            BorrowerProfile(
                borrower_id=f"SIM-HOST-{uuid.uuid4().hex[:6]}",
                loan_id="LN-HOST-001",
                persona=PersonaType.HOSTILE,
                outstanding_amount=85_000,
                days_past_due=120,
                expected_resolution_path=ResolutionPath.LEGAL,
                expected_outcome="escalated",
            ),
            BorrowerProfile(
                borrower_id=f"SIM-BROK-{uuid.uuid4().hex[:6]}",
                loan_id="LN-BROK-001",
                persona=PersonaType.BROKE,
                outstanding_amount=40_000,
                days_past_due=60,
                expected_resolution_path=ResolutionPath.HARDSHIP,
                expected_outcome="resolved",
            ),
            BorrowerProfile(
                borrower_id=f"SIM-STRT-{uuid.uuid4().hex[:6]}",
                loan_id="LN-STRT-001",
                persona=PersonaType.STRATEGIC_DEFAULTER,
                outstanding_amount=120_000,
                days_past_due=180,
                expected_resolution_path=ResolutionPath.LUMP_SUM,
                expected_outcome="escalated",   # strategic defaulter often escalates
            ),
        ]

    @staticmethod
    def make_script(persona: PersonaType) -> PersonaScript:
        return PersonaScript(persona)

    @staticmethod
    def get_mock_llm_responses(persona: PersonaType) -> dict[str, list[str]]:
        """
        LLM mock responses (agent side) for simulation.
        These are scripted agent responses that trigger the right markers.
        """
        shared_assessment = [
            "Please provide the last 4 digits of your loan account and birth year.",
            "What is your current monthly income?",
            "What are your monthly expenses?",
            "What is your employment status?",
            "Do you have any assets?",
        ]

        if persona == PersonaType.COOPERATIVE:
            return {
                "AssessmentAgent": shared_assessment + [
                    "Thank you. ASSESSMENT_COMPLETE:INSTALLMENT"
                ],
                "ResolutionAgent": [
                    "Your offer: ₹12,750 down + ₹7,225/month × 10 months. Deadline April 24.",
                    "Terms are fixed. Can you commit?",
                    "Confirmed. RESOLUTION_COMPLETE",
                ],
                "FinalNoticeAgent": [
                    "Final offer expires April 24. Credit bureau reporting follows.",
                    "Payment confirmed. COLLECTIONS_COMPLETE",
                ],
            }

        elif persona == PersonaType.HOSTILE:
            return {
                "AssessmentAgent": shared_assessment + [
                    "Understood. ASSESSMENT_COMPLETE:LEGAL"
                ],
                "ResolutionAgent": [
                    "Your settlement: full outstanding ₹85,000 within 7 days.",
                    "Terms stand. Can you commit?",
                    "Noted. RESOLUTION_REFUSED",
                ],
                "FinalNoticeAgent": [
                    "Final offer expires in 48 hours. Legal proceedings follow.",
                    "COLLECTIONS_ESCALATED",
                ],
            }

        elif persona == PersonaType.BROKE:
            return {
                "AssessmentAgent": shared_assessment + [
                    "Assessed. ASSESSMENT_COMPLETE:HARDSHIP"
                ],
                "ResolutionAgent": [
                    "Hardship plan: ₹2,000/month for 6 months. Deadline April 24.",
                    "This is the minimum available. Can you commit?",
                    "Confirmed. RESOLUTION_COMPLETE",
                ],
                "FinalNoticeAgent": [
                    "Final offer: ₹2,000/month hardship plan, expires April 24.",
                    "Payment confirmed. COLLECTIONS_COMPLETE",
                ],
            }

        elif persona == PersonaType.STRATEGIC_DEFAULTER:
            return {
                "AssessmentAgent": shared_assessment + [
                    "Assessed. ASSESSMENT_COMPLETE:LUMP_SUM"
                ],
                "ResolutionAgent": [
                    "Lump sum settlement: ₹70,000 (17.6% discount). Deadline April 24.",
                    "Terms are fixed at policy maximum. Can you commit?",
                    "No further discount. RESOLUTION_REFUSED",
                ],
                "FinalNoticeAgent": [
                    "Final offer: ₹70,000 lump sum, expires April 24. Legal to follow.",
                    "COLLECTIONS_ESCALATED",
                ],
            }

        return {}
