from __future__ import annotations

from typing import Optional

from src.agents.base import BaseAgent, AgentResponse
from src.models import ConversationContext, Stage, ResolutionPath, FinancialSituation
from src.question_tracker import QuestionTracker, FactKey
from src.token_budget import CostTracker
from src.prompts import assessment_system_prompt

# Facts required before advancing to Resolution
REQUIRED_FACTS = {
    FactKey.IDENTITY_LAST4,
    FactKey.IDENTITY_DOB_YEAR,
    FactKey.MONTHLY_INCOME,
    FactKey.MONTHLY_EXPENSES,
    FactKey.EMPLOYMENT_STATUS,
}

# Facts that are "nice to have" but not blockers
OPTIONAL_FACTS = {
    FactKey.ASSETS_VALUE,
    FactKey.LIABILITIES_VALUE,
    FactKey.CASH_FLOW_ISSUE,
}


class AssessmentAgent(BaseAgent):
    """
    Agent 1 — Cold, clinical fact-gathering.

    - Verifies identity (last 4 + DOB year)
    - Gathers financial picture
    - Recommends resolution path
    - Tracks all questions via QuestionTracker (no repeats)
    - Hard cap: 2000 output tokens per turn
    """

    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        super().__init__("AssessmentAgent", cost_tracker)

    def get_system_prompt(self, context: ConversationContext) -> str:
        qt = self._get_question_tracker(context)
        known_facts = qt.as_context_str()
        guidance = getattr(self, "_injected_guidance", "")
        return assessment_system_prompt(known_facts, guidance)

    async def process(self, context: ConversationContext, user_input: str) -> AgentResponse:
        qt = self._get_question_tracker(context)

        # Parse user reply and update tracker
        self._extract_facts(user_input, qt, context)
        self._save_question_tracker(context, qt)

        # Build context block
        context_str = self.format_context_for_agent(context)
        system = self.get_system_prompt(context)

        messages = [{"role": "user", "content": f"{context_str}\n\nBorrower says: {user_input}"}]

        text, usage = await self._call_claude(system, messages, temperature=0.1)

        # Check for ASSESSMENT_COMPLETE marker
        should_advance, path = self._parse_completion(text)
        clean_message = text.split("ASSESSMENT_COMPLETE")[0].strip()

        if should_advance and context.assessment_data:
            context.assessment_data.resolution_path = path
            context.assessment_data.identity_verified = qt.is_answered(FactKey.IDENTITY_LAST4)

        # Check for no-repeat violations (guard rail)
        violations = qt.validate_no_repeat(clean_message)
        if violations:
            # Agent tried to re-ask — override with a neutral bridge
            clean_message = (
                f"[System: repeated question guard triggered for {violations}] "
                + clean_message
            )

        # Persist updated tracker
        self._save_question_tracker(context, qt)

        return AgentResponse(
            message=clean_message,
            should_advance=should_advance,
            context_update={"current_stage": Stage.RESOLUTION} if should_advance else {},
            tokens_used=usage,
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _parse_completion(self, text: str) -> tuple[bool, Optional[ResolutionPath]]:
        if "ASSESSMENT_COMPLETE:" not in text:
            return False, None
        try:
            raw = text.split("ASSESSMENT_COMPLETE:")[1].split()[0].strip().upper()
            path = ResolutionPath(raw.lower())
            return True, path
        except (IndexError, ValueError):
            return False, None

    def _extract_facts(
        self, user_input: str, qt: QuestionTracker, context: ConversationContext
    ) -> None:
        """
        Heuristic extraction of facts from borrower's free-text reply.
        Updates QuestionTracker and ConversationContext.assessment_data.
        """
        text = user_input.lower()
        ad = context.assessment_data

        # Identity — look for 4-digit sequences
        if not qt.is_answered(FactKey.IDENTITY_LAST4):
            import re
            digits = re.findall(r"\b\d{4}\b", user_input)
            if digits:
                qt.mark_asked(FactKey.IDENTITY_LAST4, self.name, context.current_stage.value)
                qt.mark_answered(FactKey.IDENTITY_LAST4, digits[0])

        # DOB year — 4-digit year 1940-2005
        if not qt.is_answered(FactKey.IDENTITY_DOB_YEAR):
            import re
            years = re.findall(r"\b(19[4-9]\d|200[0-5])\b", user_input)
            if years:
                qt.mark_asked(FactKey.IDENTITY_DOB_YEAR, self.name, context.current_stage.value)
                qt.mark_answered(FactKey.IDENTITY_DOB_YEAR, years[0])

        # Income — look for number after income/salary/earn keywords
        if not qt.is_answered(FactKey.MONTHLY_INCOME):
            import re
            m = re.search(r"(?:income|earn|salary|make)[^\d]*(\d[\d,]+)", text)
            if m:
                val = m.group(1).replace(",", "")
                qt.mark_asked(FactKey.MONTHLY_INCOME, self.name, context.current_stage.value)
                qt.mark_answered(FactKey.MONTHLY_INCOME, val)
                if ad and ad.financial_situation is None:
                    ad.financial_situation = FinancialSituation(
                        monthly_income=float(val), monthly_expenses=0,
                        outstanding_debt=ad.outstanding_amount, assets_value=0,
                        liabilities_value=0, employment_status="unknown"
                    )
                elif ad and ad.financial_situation:
                    ad.financial_situation.monthly_income = float(val)

        # Expenses
        if not qt.is_answered(FactKey.MONTHLY_EXPENSES):
            import re
            m = re.search(r"(?:spend|expense|expenditure|cost)[^\d]*(\d[\d,]+)", text)
            if m:
                val = m.group(1).replace(",", "")
                qt.mark_asked(FactKey.MONTHLY_EXPENSES, self.name, context.current_stage.value)
                qt.mark_answered(FactKey.MONTHLY_EXPENSES, val)
                if ad and ad.financial_situation:
                    ad.financial_situation.monthly_expenses = float(val)

        # Employment
        if not qt.is_answered(FactKey.EMPLOYMENT_STATUS):
            emp_keywords = {
                "salaried": ["salaried", "employed", "job", "work for", "company"],
                "self_employed": ["self", "business", "own firm", "freelance", "consultant"],
                "unemployed": ["unemployed", "no job", "laid off", "lost job"],
                "retired": ["retired", "pension"],
            }
            for status, keywords in emp_keywords.items():
                if any(kw in text for kw in keywords):
                    qt.mark_asked(FactKey.EMPLOYMENT_STATUS, self.name, context.current_stage.value)
                    qt.mark_answered(FactKey.EMPLOYMENT_STATUS, status)
                    if ad and ad.financial_situation:
                        ad.financial_situation.employment_status = status
                    break

        # Assets
        if not qt.is_answered(FactKey.ASSETS_VALUE):
            import re
            m = re.search(r"(?:property|asset|vehicle|house|flat|car|investment)[^\d]*(\d[\d,]+)", text)
            if m:
                val = m.group(1).replace(",", "")
                qt.mark_asked(FactKey.ASSETS_VALUE, self.name, context.current_stage.value)
                qt.mark_answered(FactKey.ASSETS_VALUE, val)

        # Liabilities
        if not qt.is_answered(FactKey.LIABILITIES_VALUE):
            import re
            m = re.search(r"(?:other loan|other debt|owe|outstanding)[^\d]*(\d[\d,]+)", text)
            if m:
                val = m.group(1).replace(",", "")
                qt.mark_asked(FactKey.LIABILITIES_VALUE, self.name, context.current_stage.value)
                qt.mark_answered(FactKey.LIABILITIES_VALUE, val)

        # Cash flow
        if not qt.is_answered(FactKey.CASH_FLOW_ISSUE):
            if any(w in text for w in ["tight", "difficult", "struggle", "problem", "issue", "can't pay"]):
                qt.mark_asked(FactKey.CASH_FLOW_ISSUE, self.name, context.current_stage.value)
                qt.mark_answered(FactKey.CASH_FLOW_ISSUE, "yes")
            elif any(w in text for w in ["fine", "okay", "no problem", "manage"]):
                qt.mark_asked(FactKey.CASH_FLOW_ISSUE, self.name, context.current_stage.value)
                qt.mark_answered(FactKey.CASH_FLOW_ISSUE, "no")

    def all_required_facts_present(self, context: ConversationContext) -> bool:
        qt = self._get_question_tracker(context)
        return all(qt.is_answered(f) for f in REQUIRED_FACTS)
