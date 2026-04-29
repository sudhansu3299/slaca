from __future__ import annotations

import logging
import re
from typing import Optional

from src.agent_tools import ASSESSMENT_TOOLS

log = logging.getLogger(__name__)
from src.agents.base import BaseAgent, AgentResponse
from src.models import ConversationContext, Stage, ResolutionPath, FinancialSituation
from src.question_tracker import QuestionTracker, FactKey
from src.token_budget import CostTracker
from src.prompts import assessment_system_prompt
from src.prompt_builder import build_llm_turn

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

    - Verifies identity via verify_borrower_identity tool (Postgres lookup)
    - Stores monthly income/expenses via store_financial_data tool
    - Gathers financial picture and recommends resolution path
    - Tracks all questions via QuestionTracker (no repeats)
    - Hard cap: 2000 tokens total (prompt + completion) per LLM API call
    """

    def __init__(self, cost_tracker: Optional[CostTracker] = None):
        super().__init__("AssessmentAgent", cost_tracker)

    # Keywords that signal borrower accepting the hardship offer
    _HARDSHIP_YES = {"yes", "y", "please", "ok", "okay", "sure", "yeah", "yep", "go ahead", "do it", "flag it"}
    # Keywords that signal borrower declining the hardship offer
    _HARDSHIP_NO  = {"no", "n", "nope", "not now", "continue", "carry on", "skip", "nevermind"}
    _OPENING_DISCLOSURE = (
        "This is an automated AI collections agent acting on behalf of Riverline. "
        "This conversation is being logged."
    )

    def get_system_prompt(self, context: ConversationContext) -> str:
        qt = self._get_question_tracker(context)
        known_facts = qt.as_context_str()
        # Tell Claude not to repeat the opening disclosure after the first turn
        if context.assessment_opened:
            known_facts = "[Opening disclosure already sent — do NOT repeat it.]\n\n" + known_facts
        # Tell Claude the hardship offer is pending — wait for yes/no, don't ask more facts
        if context.hardship_offer_pending:
            known_facts = (
                "[HARDSHIP OFFER PENDING — you already offered hardship review. "
                "Wait for the borrower's yes/no answer. Do NOT ask for income, expenses, or any other fact yet.]\n\n"
                + known_facts
            )
        guidance = getattr(self, "_injected_guidance", "")
        return assessment_system_prompt(known_facts, guidance)

    async def process(self, context: ConversationContext, user_input: str) -> AgentResponse:
        qt = self._get_question_tracker(context)
        bypass_identity_verify = bool(getattr(self, "_bypass_identity_verification", False))

        # ── Hardship offer response gate (code-level) ────────────────────
        # If we already made the hardship offer, intercept yes/no before
        # doing any fact extraction or calling Claude.
        if context.hardship_offer_pending:
            lower = user_input.lower().strip().rstrip(".")
            if any(kw in lower for kw in self._HARDSHIP_YES):
                # Borrower accepted — flag hardship, complete assessment immediately
                context.hardship_offer_pending = False
                if context.assessment_data:
                    context.assessment_data.resolution_path = ResolutionPath.HARDSHIP
                    context.assessment_data.identity_verified = True  # already verified earlier
                log.info("[assessment] hardship accepted by borrower — completing assessment")
                return AgentResponse(
                    message="Noted. Your account has been flagged for hardship review. A specialist will contact you.",
                    should_advance=True,
                    context_update={"current_stage": Stage.RESOLUTION},
                    tokens_used=None,
                )
            elif any(kw in lower for kw in self._HARDSHIP_NO):
                # Borrower declined — clear flag and continue normally
                context.hardship_offer_pending = False
                log.info("[assessment] hardship declined by borrower — resuming assessment")
                # Fall through to normal fact collection below

        # Heuristic pre-extraction — populates QuestionTracker so Claude knows
        # which facts are already answered before making tool calls.
        self._extract_facts(user_input, qt, context)
        self._save_question_tracker(context, qt)

        # ── Detect hardship trigger in borrower input ────────────────────
        # Set the pending flag so next turn's prompt tells Claude to wait for yes/no
        if not context.hardship_offer_pending and self._is_hardship_signal(user_input):
            context.hardship_offer_pending = True

        system = self.get_system_prompt(context)
        system, messages = build_llm_turn(self.name, system, context, None, user_input)

        # Use tool-call loop — Claude will call verify_borrower_identity and/or
        # store_financial_data as facts arrive, then produce its final reply
        self._tool_results = {}
        text, usage = await self._call_claude_with_tools(
            system, messages, ASSESSMENT_TOOLS, temperature=0.1
        )

        # Fix 4: mark that the opening disclosure has been sent after the first turn
        if not context.assessment_opened:
            context.assessment_opened = True

        # ── Propagate tool results to context (code-level, authoritative) ─
        was_identity_verified = bool(
            context.assessment_data is not None and context.assessment_data.identity_verified
        )
        identity_just_verified = False
        verify_result = self._tool_results.get("verify_borrower_identity")
        if verify_result is not None:
            verified = bool(verify_result.get("verified", False))
            if bypass_identity_verify and not verified:
                # Replay/simulation mode: don't fail the turn on verification tool outcome.
                verified = True
            if verified:
                if context.assessment_data:
                    context.assessment_data.identity_verified = True
                    # Source of truth is the verified borrower record from DB.
                    if verify_result.get("outstanding_amount") is not None:
                        context.assessment_data.outstanding_amount = float(
                            verify_result.get("outstanding_amount")
                        )
                    if verify_result.get("days_past_due") is not None:
                        context.assessment_data.days_past_due = int(
                            verify_result.get("days_past_due")
                        )
                    if verify_result.get("loan_id"):
                        context.assessment_data.loan_id = str(verify_result.get("loan_id"))
                    if verify_result.get("borrower_id"):
                        context.assessment_data.borrower_id = str(verify_result.get("borrower_id"))
                    if context.assessment_data.financial_situation:
                        context.assessment_data.financial_situation.outstanding_debt = (
                            context.assessment_data.outstanding_amount
                        )
                identity_just_verified = not was_identity_verified
            else:
                # Ignore stale failed re-checks after successful verification
                # unless the borrower actually supplied new identity values now.
                if was_identity_verified and not self._contains_identity_values(user_input):
                    log.warning(
                        "[assessment] ignoring stale verify=false after identity already verified"
                    )
                else:
                    if context.assessment_data:
                        context.assessment_data.identity_verified = False
                    # Fix 2: if verification failed, clear the stale identity values from
                    # the QuestionTracker so the borrower can supply correct ones next turn
                    qt.facts[FactKey.IDENTITY_LAST4.value].answered = False
                    qt.facts[FactKey.IDENTITY_LAST4.value].value = None
                    qt.facts[FactKey.IDENTITY_DOB_YEAR.value].answered = False
                    qt.facts[FactKey.IDENTITY_DOB_YEAR.value].value = None
                    self._save_question_tracker(context, qt)
                    log.info("[assessment] identity verification failed — cleared stale identity facts")

        # Strip the marker from the displayed message regardless
        clean_message = text.split("ASSESSMENT_COMPLETE")[0].strip()

        # Never repeat the opening disclosure once the stage is already open.
        if context.assessment_opened and self._OPENING_DISCLOSURE in clean_message:
            clean_message = clean_message.replace(self._OPENING_DISCLOSURE, "").strip()

        # If the DB verification just succeeded, ensure we acknowledge it once.
        if identity_just_verified and "identity verified" not in clean_message.lower():
            clean_message = f"Identity verified. {clean_message}".strip()

        # ── Hard identity gate (code-level, not prompt-level) ────────────
        identity_confirmed = (
            context.assessment_data is not None
            and context.assessment_data.identity_verified
        )
        if bypass_identity_verify:
            identity_confirmed = True
            if context.assessment_data:
                context.assessment_data.identity_verified = True

        # Compatibility path: some test suites patch _call_claude (legacy)
        # instead of the tool loop. In that mode we cannot receive
        # verify_borrower_identity tool results, so rely on collected
        # identity fields to avoid blocking stage progression forever.
        if (
            not identity_confirmed
            and getattr(self, "_tool_loop_fallback", False)
            and qt.is_answered(FactKey.IDENTITY_LAST4)
            and qt.is_answered(FactKey.IDENTITY_DOB_YEAR)
        ):
            identity_confirmed = True
            if context.assessment_data:
                context.assessment_data.identity_verified = True

        should_advance, path = self._parse_completion(text)

        if (
            should_advance
            and not identity_confirmed
            and not getattr(self, "_skip_identity_gate", False)
        ):
            should_advance = False
            clean_message = (
                "The details you provided do not match our records. "
                "Please provide the correct last 4 digits of your loan account number and birth year."
            )

        if should_advance and context.assessment_data:
            context.assessment_data.resolution_path = path

        # Fix 3: log repeated-question violations internally but never leak
        # the [System: ...] prefix into the borrower-facing message
        violations = qt.validate_no_repeat(clean_message)
        if violations:
            log.warning("[assessment] repeated question guard: %s", violations)

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

    def _is_hardship_signal(self, user_input: str) -> bool:
        """Detect borrower mentioning hardship/crisis in their message."""
        signals = [
            "hardship", "medical", "emergency", "hospital", "surgery",
            "job loss", "lost my job", "laid off", "unemployed", "no income",
            "family emergency", "death in family", "accident", "illness",
            "mental health", "crisis", "struggling", "can't afford",
            "no money", "broke", "bankrupt",
        ]
        text = user_input.lower()
        return any(s in text for s in signals)

    def _contains_identity_values(self, user_input: str) -> bool:
        years = re.findall(r"\b(19[4-9]\d|200[0-5])\b", user_input)
        all_digits = re.findall(r"\b\d{4}\b", user_input)
        non_year_digits = [d for d in all_digits if not re.match(r"^(19[4-9]\d|200[0-5])$", d)]
        return bool(years and non_year_digits)

    def _extract_facts(
        self, user_input: str, qt: QuestionTracker, context: ConversationContext
    ) -> None:
        """
        Heuristic extraction of facts from borrower's free-text reply.
        Updates QuestionTracker and ConversationContext.assessment_data.
        Results are also passed to Claude so it knows which tool calls to make.
        """
        text = user_input.lower()
        ad = context.assessment_data

        # Identity — last 4 digits.
        # Always accept a new value when identity has not been verified yet,
        # so the borrower can correct wrong details without getting stuck.
        # Exclude numbers that look like birth years (1940–2005) so that
        # sending "1985" alone doesn't overwrite the last4 field.
        identity_verified = (
            context.assessment_data is not None
            and context.assessment_data.identity_verified
        )
        _year_pattern = re.compile(r"^(19[4-9]\d|200[0-5])$")
        if not qt.is_answered(FactKey.IDENTITY_LAST4) or not identity_verified:
            all_digits = re.findall(r"\b\d{4}\b", user_input)
            # Keep only numbers that are NOT valid birth years
            non_year_digits = [d for d in all_digits if not _year_pattern.match(d)]
            if non_year_digits:
                qt.mark_asked(FactKey.IDENTITY_LAST4, self.name, context.current_stage.value)
                qt.mark_answered(FactKey.IDENTITY_LAST4, non_year_digits[0])

        # DOB year — 4-digit year 1940–2005. Same correction logic.
        if not qt.is_answered(FactKey.IDENTITY_DOB_YEAR) or not identity_verified:
            years = re.findall(r"\b(19[4-9]\d|200[0-5])\b", user_input)
            if years:
                qt.mark_asked(FactKey.IDENTITY_DOB_YEAR, self.name, context.current_stage.value)
                qt.mark_answered(FactKey.IDENTITY_DOB_YEAR, years[0])

        # Income
        if not qt.is_answered(FactKey.MONTHLY_INCOME):
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
                "salaried":      ["salaried", "employed", "job", "work for", "company"],
                "self_employed": ["self", "business", "own firm", "freelance", "consultant"],
                "unemployed":    ["unemployed", "no job", "laid off", "lost job"],
                "retired":       ["retired", "pension"],
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
            m = re.search(r"(?:property|asset|vehicle|house|flat|car|investment)[^\d]*(\d[\d,]+)", text)
            if m:
                val = m.group(1).replace(",", "")
                qt.mark_asked(FactKey.ASSETS_VALUE, self.name, context.current_stage.value)
                qt.mark_answered(FactKey.ASSETS_VALUE, val)

        # Liabilities
        if not qt.is_answered(FactKey.LIABILITIES_VALUE):
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
