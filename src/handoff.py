"""
Lossless handoff summary system.

Produces a ≤500 token structured block that carries ALL facts collected by
the previous agent into the next one — so the borrower is never asked the
same question twice and no information is lost across modalities.

Token estimation: 1 token ≈ 4 characters (conservative English approximation).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from src.models import ConversationContext, ResolutionPath, Stage
from src.question_tracker import QuestionTracker, FactKey
from src.token_budget import MAX_TOKENS_HANDOFF, estimate_tokens as _estimate_tokens

# 1 token ≈ 4 chars — implemented in src.token_budget.estimate_tokens


class HandoffSummary(BaseModel):
    """
    Compact summary passed between agents.
    All fields are Optional so partial handoffs are valid.
    """
    from_stage: str
    to_stage: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # Identity
    borrower_id: str
    loan_id: str
    identity_verified: bool = False

    # Loan facts
    principal_amount: float = 0.0
    outstanding_amount: float = 0.0
    days_past_due: int = 0

    # Financial facts (from assessment)
    monthly_income: Optional[str] = None
    monthly_expenses: Optional[str] = None
    employment_status: Optional[str] = None
    assets_value: Optional[str] = None
    liabilities_value: Optional[str] = None
    cash_flow_issue: Optional[str] = None

    # Decision
    resolution_path: Optional[str] = None

    # Resolution offer (from voice agent)
    offer_path: Optional[str] = None
    offer_discount_pct: Optional[float] = None
    offer_upfront: Optional[float] = None
    offer_monthly: Optional[float] = None
    offer_tenure_months: Optional[int] = None
    offer_deadline_days: Optional[int] = None
    offer_valid_until: Optional[str] = None
    resolution_committed: bool = False

    # ── Spec-required fields (milestone.txt) ──────────────────────────
    # financial_state: single token classifying the borrower's financial health
    financial_state: Optional[str] = None    # stable | unstable_income | cash_strapped | crisis
    # intent: borrower's indicated willingness to resolve
    intent: Optional[str] = None             # willing_full | willing_partial | resistant | unknown
    # structured list of offers presented (each a compact dict)
    offers_presented: list[dict] = Field(default_factory=list)
    # structured list of objections raised
    objections: list[str] = Field(default_factory=list)
    # ── Legacy tone fields ─────────────────────────────────────────────
    borrower_attitude: Optional[str] = None   # e.g. "cooperative", "hostile"
    key_objection: Optional[str] = None       # last objection raised (compat)

    def to_prompt_block(self) -> str:
        """
        Renders a terse, prompt-ready block that fits in ≤500 tokens.
        """
        lines = [
            "=== HANDOFF CONTEXT ===",
            f"From: {self.from_stage} → To: {self.to_stage}",
            f"Borrower: {self.borrower_id} | Loan: {self.loan_id}",
            f"Identity verified: {self.identity_verified}",
            f"Outstanding: ₹{self.outstanding_amount:,.0f} | DPD: {self.days_past_due}",
        ]

        # Financial facts — only include answered ones
        fin_fields = [
            ("income/mo", self.monthly_income),
            ("expenses/mo", self.monthly_expenses),
            ("employment", self.employment_status),
            ("assets", self.assets_value),
            ("liabilities", self.liabilities_value),
            ("cash_flow_issue", self.cash_flow_issue),
        ]
        answered_fin = [(k, v) for k, v in fin_fields if v is not None]
        if answered_fin:
            lines.append("Financial facts:")
            for k, v in answered_fin:
                lines.append(f"  {k}: {v}")

        if self.resolution_path:
            lines.append(f"Recommended path: {self.resolution_path}")

        # Offer block
        if self.offer_path:
            lines.append(f"Offer: {self.offer_path}")
            if self.offer_discount_pct:
                lines.append(f"  Discount: {self.offer_discount_pct}%")
            if self.offer_upfront:
                lines.append(f"  Upfront: ₹{self.offer_upfront:,.0f}")
            if self.offer_monthly:
                lines.append(
                    f"  Monthly: ₹{self.offer_monthly:,.0f} × {self.offer_tenure_months}mo"
                )
            if self.offer_valid_until:
                lines.append(f"  Valid until: {self.offer_valid_until}")
            lines.append(f"  Committed: {self.resolution_committed}")

        if self.financial_state:
            lines.append(f"Financial state: {self.financial_state}")
        if self.intent:
            lines.append(f"Borrower intent: {self.intent}")
        if self.offers_presented:
            lines.append(f"Offers presented: {len(self.offers_presented)}")
            for o in self.offers_presented[-2:]:   # last 2 only to stay lean
                lines.append(f"  {o}")
        if self.objections:
            lines.append("Objections raised:")
            for obj in self.objections[-3:]:       # last 3 only
                lines.append(f"  - {obj[:80]}")
        if self.borrower_attitude:
            lines.append(f"Attitude: {self.borrower_attitude}")
        if self.key_objection:
            lines.append(f"Last objection: {self.key_objection[:120]}")

        lines.append("=== END HANDOFF ===")
        block = "\n".join(lines)
        _enforce_token_limit(block, label=f"HandoffSummary {self.from_stage}→{self.to_stage}")
        return block

    def estimated_tokens(self) -> int:
        return _estimate_tokens(self.to_prompt_block())


def _enforce_token_limit(text: str, label: str = "") -> None:
    est = _estimate_tokens(text)
    if est > MAX_TOKENS_HANDOFF:
        raise ValueError(
            f"{label} handoff summary ~{est} tokens exceeds limit {MAX_TOKENS_HANDOFF}"
        )


class HandoffBuilder:
    """Builds a HandoffSummary from ConversationContext."""

    @staticmethod
    def build(
        context: ConversationContext,
        from_stage: Stage,
        to_stage: Stage,
    ) -> HandoffSummary:
        qt = _deserialise_qt(context)

        answered = qt.get_answered()

        ad = context.assessment_data
        offer = context.resolution_offer

        summary = HandoffSummary(
            from_stage=from_stage.value,
            to_stage=to_stage.value,
            borrower_id=context.borrower_id,
            loan_id=context.loan_id,
            identity_verified=ad.identity_verified if ad else False,
            principal_amount=ad.principal_amount if ad else 0.0,
            outstanding_amount=ad.outstanding_amount if ad else 0.0,
            days_past_due=ad.days_past_due if ad else 0,
            resolution_path=ad.resolution_path.value if (ad and ad.resolution_path) else None,
            # Financial facts from QuestionTracker
            monthly_income=answered.get(FactKey.MONTHLY_INCOME),
            monthly_expenses=answered.get(FactKey.MONTHLY_EXPENSES),
            employment_status=answered.get(FactKey.EMPLOYMENT_STATUS),
            assets_value=answered.get(FactKey.ASSETS_VALUE),
            liabilities_value=answered.get(FactKey.LIABILITIES_VALUE),
            cash_flow_issue=answered.get(FactKey.CASH_FLOW_ISSUE),
        )

        if offer:
            summary.offer_path = offer.path.value
            summary.offer_discount_pct = offer.discount_percentage or None
            summary.offer_upfront = offer.upfront_required or None
            summary.offer_monthly = offer.monthly_payment or None
            summary.offer_tenure_months = offer.tenure_months or None
            summary.offer_deadline_days = offer.deadline_days
            summary.offer_valid_until = offer.valid_until
            summary.resolution_committed = context.resolution_outcome == "committed"

        # Derive borrower attitude from last few turns
        summary.borrower_attitude = _infer_attitude(context)
        summary.key_objection = _find_last_objection(context)

        # Spec fields
        summary.financial_state = _classify_financial_state(context)
        summary.intent = _infer_intent(context)
        summary.objections = _collect_objections(context)
        if offer:
            summary.offers_presented = [_offer_to_dict(offer)]

        return summary

    @staticmethod
    def apply_to_context(summary: HandoffSummary, context: ConversationContext) -> None:
        """
        Inject handoff facts back into context.question_state so the
        receiving agent's QuestionTracker reflects what is already known.
        """
        qt = _deserialise_qt(context)

        _maybe_answer(qt, FactKey.MONTHLY_INCOME, summary.monthly_income, "HandoffBuilder")
        _maybe_answer(qt, FactKey.MONTHLY_EXPENSES, summary.monthly_expenses, "HandoffBuilder")
        _maybe_answer(qt, FactKey.EMPLOYMENT_STATUS, summary.employment_status, "HandoffBuilder")
        _maybe_answer(qt, FactKey.ASSETS_VALUE, summary.assets_value, "HandoffBuilder")
        _maybe_answer(qt, FactKey.LIABILITIES_VALUE, summary.liabilities_value, "HandoffBuilder")
        _maybe_answer(qt, FactKey.CASH_FLOW_ISSUE, summary.cash_flow_issue, "HandoffBuilder")

        if summary.identity_verified:
            _maybe_answer(qt, FactKey.IDENTITY_LAST4, "verified", "HandoffBuilder")
            _maybe_answer(qt, FactKey.IDENTITY_DOB_YEAR, "verified", "HandoffBuilder")

        _serialise_qt(context, qt)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _deserialise_qt(context: ConversationContext) -> QuestionTracker:
    qt = QuestionTracker()
    for key_str, state in context.question_state.items():
        try:
            fk = FactKey(key_str)
            if state.get("asked"):
                qt.mark_asked(fk, state.get("asked_by", ""), state.get("stage", ""))
            if state.get("answered") and state.get("value") is not None:
                qt.mark_answered(fk, state["value"])
        except ValueError:
            pass
    return qt


def _serialise_qt(context: ConversationContext, qt: QuestionTracker) -> None:
    context.question_state = {
        k: {
            "asked": v.asked,
            "answered": v.answered,
            "value": v.value,
            "asked_by": v.asked_by,
            "stage": v.stage,
        }
        for k, v in qt.facts.items()
    }


def _maybe_answer(qt: QuestionTracker, key: FactKey, value: Optional[str], agent: str) -> None:
    if value is not None and not qt.is_answered(key):
        qt.mark_asked(key, agent, "handoff")
        qt.mark_answered(key, value)


def _classify_financial_state(context: ConversationContext) -> str:
    """Classify borrower financial health into one of four tokens."""
    ad = context.assessment_data
    qt = _deserialise_qt(context)
    answered = qt.get_answered()

    cash_flow = answered.get(FactKey.CASH_FLOW_ISSUE, "")
    employment = answered.get(FactKey.EMPLOYMENT_STATUS, "")

    if employment == "unemployed":
        return "crisis"
    if cash_flow in ("yes", "severe", "critical"):
        return "cash_strapped"
    if cash_flow == "mild" or employment == "self_employed":
        return "unstable_income"
    if ad and ad.resolution_path:
        from src.models import ResolutionPath
        if ad.resolution_path == ResolutionPath.HARDSHIP:
            return "crisis"
        if ad.resolution_path == ResolutionPath.LUMP_SUM:
            return "stable"
    return "unstable_income"


def _infer_intent(context: ConversationContext) -> str:
    """Classify borrower's repayment intent from conversation."""
    attitude = _infer_attitude(context)
    last_user = [
        m.get("content", "").lower()
        for m in context.conversation_history[-6:]
        if m.get("role") == "user"
    ]
    text = " ".join(last_user)

    committed_words = ["agree", "accept", "yes", "okay", "i'll pay", "i can do"]
    partial_words   = ["partial", "some", "half", "part", "maybe", "possible"]
    resist_words    = ["refuse", "won't", "cannot", "no way", "not paying"]

    if any(w in text for w in resist_words) or attitude == "hostile":
        return "resistant"
    if any(w in text for w in committed_words):
        return "willing_full"
    if any(w in text for w in partial_words):
        return "willing_partial"
    return "unknown"


def _collect_objections(context: ConversationContext) -> list[str]:
    """Extract all objections raised by the borrower."""
    objection_words = ["cannot", "won't", "too much", "expensive", "no way",
                       "impossible", "can't afford", "refuse", "not fair"]
    objections = []
    for msg in context.conversation_history:
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if any(w in content.lower() for w in objection_words):
                objections.append(content[:100])
    return objections[-5:]   # cap at last 5


def _offer_to_dict(offer) -> dict:
    """Convert ResolutionOffer to a compact dict for offers_presented list."""
    return {
        "path": offer.path.value,
        "upfront": offer.upfront_required,
        "monthly": offer.monthly_payment,
        "tenure_months": offer.tenure_months,
        "deadline_days": offer.deadline_days,
    }


def _infer_attitude(context: ConversationContext) -> str:
    last_user_msgs = [
        m["content"].lower()
        for m in context.conversation_history[-4:]
        if m.get("role") == "user"
    ]
    hostile_words = ["refuse", "won't", "never", "lawyer", "court", "sue"]
    cooperative_words = ["okay", "sure", "yes", "agree", "understand", "fine"]
    text = " ".join(last_user_msgs)
    hostile = sum(1 for w in hostile_words if w in text)
    cooperative = sum(1 for w in cooperative_words if w in text)
    if hostile > cooperative:
        return "hostile"
    if cooperative > 0:
        return "cooperative"
    return "neutral"


def _find_last_objection(context: ConversationContext) -> Optional[str]:
    objection_words = ["cannot", "won't", "too much", "expensive", "no way", "impossible"]
    for msg in reversed(context.conversation_history):
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if any(w in content.lower() for w in objection_words):
                return content[:120]
    return None
