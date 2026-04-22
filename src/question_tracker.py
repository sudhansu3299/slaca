"""
Tracks which questions have already been asked across all agents/modalities.
Prevents repeated borrower questions — one of the core constraints.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class FactKey(str, Enum):
    """Canonical set of facts that can be asked across the pipeline."""
    IDENTITY_LAST4 = "identity_last4"
    IDENTITY_DOB_YEAR = "identity_dob_year"
    MONTHLY_INCOME = "monthly_income"
    MONTHLY_EXPENSES = "monthly_expenses"
    EMPLOYMENT_STATUS = "employment_status"
    ASSETS_VALUE = "assets_value"
    LIABILITIES_VALUE = "liabilities_value"
    CASH_FLOW_ISSUE = "cash_flow_issue"
    PHONE_AVAILABLE = "phone_available"


class FactState(BaseModel):
    key: FactKey
    asked: bool = False
    answered: bool = False
    value: Optional[str] = None
    asked_by: Optional[str] = None   # agent name that asked
    stage: Optional[str] = None      # stage when asked


class QuestionTracker(BaseModel):
    facts: dict[str, FactState] = Field(default_factory=dict)

    def model_post_init(self, __context) -> None:  # noqa: ANN001
        for key in FactKey:
            self.facts[key.value] = FactState(key=key)

    def mark_asked(self, key: FactKey, agent: str, stage: str) -> None:
        self.facts[key.value].asked = True
        self.facts[key.value].asked_by = agent
        self.facts[key.value].stage = stage

    def mark_answered(self, key: FactKey, value: str) -> None:
        self.facts[key.value].answered = True
        self.facts[key.value].value = value

    def is_answered(self, key: FactKey) -> bool:
        return self.facts[key.value].answered

    def is_asked(self, key: FactKey) -> bool:
        return self.facts[key.value].asked

    def get_unanswered(self) -> list[FactKey]:
        return [
            FactKey(k) for k, v in self.facts.items()
            if not v.answered
        ]

    def get_answered(self) -> dict[FactKey, str]:
        return {
            FactKey(k): v.value
            for k, v in self.facts.items()
            if v.answered and v.value is not None
        }

    def as_context_str(self) -> str:
        """Compact string injected into agent context — tells agent what NOT to ask again."""
        answered = self.get_answered()
        if not answered:
            return "No facts collected yet."
        lines = ["Already known (DO NOT ask again):"]
        for key, val in answered.items():
            lines.append(f"  {key.value}: {val}")
        return "\n".join(lines)

    def validate_no_repeat(self, agent_message: str) -> list[str]:
        """
        Heuristic check: returns list of FactKeys that appear to be
        re-asked in an agent message when already answered.
        """
        # Keyword map: fact → phrases that signal the question being asked
        question_signals: dict[FactKey, list[str]] = {
            FactKey.IDENTITY_LAST4: ["last 4", "last four", "account number"],
            FactKey.IDENTITY_DOB_YEAR: ["date of birth", "birth year", "dob", "year you were born"],
            FactKey.MONTHLY_INCOME: ["monthly income", "how much do you earn", "salary", "income per month"],
            FactKey.MONTHLY_EXPENSES: ["monthly expenses", "monthly spending", "expenses per month"],
            FactKey.EMPLOYMENT_STATUS: ["employment status", "are you employed", "do you work", "your job"],
            FactKey.ASSETS_VALUE: ["do you own", "property", "vehicle", "assets", "investments"],
            FactKey.LIABILITIES_VALUE: ["other debts", "other loans", "liabilities"],
            FactKey.CASH_FLOW_ISSUE: ["cash flow", "trouble paying", "financial difficulty"],
        }
        msg_lower = agent_message.lower()
        violations = []
        for key, signals in question_signals.items():
            if self.is_answered(key):
                if any(signal in msg_lower for signal in signals):
                    violations.append(key.value)
        return violations
