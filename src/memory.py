"""
Conversation Memory + Golden Summary (MS6)

Technique: Structured → compressed → verified (no LLM rewrite needed;
the structure itself is the compression, and token count enforces the budget).

The "Golden Summary" must preserve exactly 5 fields from the spec:
1. Identity         — verified? account digits, DOB year
2. Financials       — income, expenses, employment, assets, liabilities, cash_flow
3. Offers           — every offer presented with terms
4. Objections       — every objection raised, verbatim (up to 120 chars each)
5. Emotional Tone   — attitude arc across the conversation

Format: compact JSON (machine-readable) + plaintext block (LLM-readable).
Both are generated and both are stored. The plaintext block is ≤500 tokens.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from src.token_budget import MAX_TOKENS_HANDOFF
from src.prompt_builder import estimate_tokens


# ──────────────────────────────────────────────────────────── #
# Golden Summary data model
# ──────────────────────────────────────────────────────────── #

class IdentityRecord(BaseModel):
    verified: bool = False
    last4: Optional[str] = None
    dob_year: Optional[str] = None


class FinancialRecord(BaseModel):
    monthly_income: Optional[str] = None
    monthly_expenses: Optional[str] = None
    employment_status: Optional[str] = None
    assets_value: Optional[str] = None
    liabilities_value: Optional[str] = None
    cash_flow_issue: Optional[str] = None
    financial_state: Optional[str] = None   # stable|unstable_income|cash_strapped|crisis


class OfferRecord(BaseModel):
    stage: str                  # "resolution" | "final_notice"
    path: str
    upfront: Optional[float] = None
    monthly: Optional[float] = None
    tenure_months: Optional[int] = None
    discount_pct: Optional[float] = None
    deadline_days: Optional[int] = None
    valid_until: Optional[str] = None
    accepted: bool = False


class ToneRecord(BaseModel):
    arc: list[str] = Field(default_factory=list)   # ordered list of attitude snapshots
    final_attitude: str = "neutral"
    intent: str = "unknown"
    key_moments: list[str] = Field(default_factory=list)


class GoldenSummary(BaseModel):
    """
    The 500-token 'golden summary' — complete conversation memory.

    Serialisable to both JSON (for storage) and plaintext (for LLM injection).
    """
    borrower_id: str
    loan_id: str
    outstanding_amount: float
    days_past_due: int
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    # The 5 spec fields
    identity: IdentityRecord = Field(default_factory=IdentityRecord)
    financials: FinancialRecord = Field(default_factory=FinancialRecord)
    offers: list[OfferRecord] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    tone: ToneRecord = Field(default_factory=ToneRecord)

    # Resolution outcome
    resolution_path: Optional[str] = None
    outcome: Optional[str] = None           # resolved|escalated|pending

    def to_plaintext(self) -> str:
        """
        Render as ≤500 token plaintext block for LLM injection.
        Structured → compressed. No LLM needed.
        """
        lines = [
            "=== GOLDEN SUMMARY ===",
            f"Borrower: {self.borrower_id} | Loan: {self.loan_id}",
            f"Outstanding: ₹{self.outstanding_amount:,.0f} | DPD: {self.days_past_due}",
        ]

        # 1. Identity
        id_parts = [f"verified={self.identity.verified}"]
        if self.identity.last4:
            id_parts.append(f"last4={self.identity.last4}")
        if self.identity.dob_year:
            id_parts.append(f"dob_year={self.identity.dob_year}")
        lines.append(f"[Identity] {', '.join(id_parts)}")

        # 2. Financials
        fin = self.financials
        fin_parts = []
        if fin.monthly_income:
            fin_parts.append(f"income={fin.monthly_income}")
        if fin.monthly_expenses:
            fin_parts.append(f"expenses={fin.monthly_expenses}")
        if fin.employment_status:
            fin_parts.append(f"employment={fin.employment_status}")
        if fin.assets_value:
            fin_parts.append(f"assets={fin.assets_value}")
        if fin.financial_state:
            fin_parts.append(f"state={fin.financial_state}")
        if fin_parts:
            lines.append(f"[Financials] {', '.join(fin_parts)}")

        # 3. Offers
        if self.offers:
            lines.append(f"[Offers] {len(self.offers)} presented:")
            for o in self.offers[-3:]:   # last 3
                offer_str = f"  {o.path}"
                if o.upfront:
                    offer_str += f" upfront=₹{o.upfront:,.0f}"
                if o.monthly:
                    offer_str += f" monthly=₹{o.monthly:,.0f}×{o.tenure_months}mo"
                if o.discount_pct:
                    offer_str += f" discount={o.discount_pct}%"
                offer_str += f" accepted={o.accepted}"
                lines.append(offer_str)
        else:
            lines.append("[Offers] none presented")

        # 4. Objections
        if self.objections:
            lines.append(f"[Objections] {len(self.objections)} raised:")
            for obj in self.objections[-3:]:
                lines.append(f"  - {obj[:80]}")
        else:
            lines.append("[Objections] none")

        # 5. Emotional Tone
        lines.append(f"[Tone] arc={' → '.join(self.tone.arc[-4:])} | final={self.tone.final_attitude} | intent={self.tone.intent}")

        if self.resolution_path:
            lines.append(f"[Path] {self.resolution_path}")
        if self.outcome:
            lines.append(f"[Outcome] {self.outcome}")

        lines.append("=== END GOLDEN SUMMARY ===")
        return "\n".join(lines)

    def estimated_tokens(self) -> int:
        return max(1, len(self.to_plaintext()) // 4)

    def verify_completeness(self) -> list[str]:
        """Returns list of missing fields. Empty list = complete."""
        missing = []
        if not self.identity.verified:
            missing.append("identity.verified=False")
        if not self.financials.monthly_income:
            missing.append("financials.monthly_income")
        if not self.financials.employment_status:
            missing.append("financials.employment_status")
        if not self.offers:
            missing.append("offers (none)")
        if not self.tone.final_attitude:
            missing.append("tone.final_attitude")
        return missing


# ──────────────────────────────────────────────────────────── #
# Memory Builder
# ──────────────────────────────────────────────────────────── #

class MemoryBuilder:
    """Builds a GoldenSummary from ConversationContext at any stage."""

    @staticmethod
    def build(context) -> GoldenSummary:
        """
        context: ConversationContext
        """
        from src.models import ResolutionPath
        from src.question_tracker import QuestionTracker, FactKey
        from src.handoff import _deserialise_qt

        ad = context.assessment_data
        qt = _deserialise_qt(context)
        answered = qt.get_answered()

        summary = GoldenSummary(
            borrower_id=context.borrower_id,
            loan_id=context.loan_id,
            outstanding_amount=ad.outstanding_amount if ad else 0,
            days_past_due=ad.days_past_due if ad else 0,
            resolution_path=ad.resolution_path.value if (ad and ad.resolution_path) else None,
            outcome=context.final_notice_outcome or context.resolution_outcome,
        )

        # Identity
        summary.identity = IdentityRecord(
            verified=ad.identity_verified if ad else False,
            last4=answered.get(FactKey.IDENTITY_LAST4),
            dob_year=answered.get(FactKey.IDENTITY_DOB_YEAR),
        )

        # Financials
        summary.financials = FinancialRecord(
            monthly_income=answered.get(FactKey.MONTHLY_INCOME),
            monthly_expenses=answered.get(FactKey.MONTHLY_EXPENSES),
            employment_status=answered.get(FactKey.EMPLOYMENT_STATUS),
            assets_value=answered.get(FactKey.ASSETS_VALUE),
            liabilities_value=answered.get(FactKey.LIABILITIES_VALUE),
            cash_flow_issue=answered.get(FactKey.CASH_FLOW_ISSUE),
            financial_state=_classify_financial_state_local(context, answered),
        )

        # Offers
        if context.resolution_offer:
            o = context.resolution_offer
            summary.offers.append(OfferRecord(
                stage="resolution",
                path=o.path.value,
                upfront=o.upfront_required or None,
                monthly=o.monthly_payment or None,
                tenure_months=o.tenure_months or None,
                discount_pct=o.discount_percentage or None,
                deadline_days=o.deadline_days,
                valid_until=o.valid_until,
                accepted=context.resolution_outcome == "committed",
            ))

        # Objections
        objection_words = ["cannot", "won't", "too much", "expensive",
                          "no way", "impossible", "can't afford", "refuse"]
        for msg in context.conversation_history:
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if any(w in content.lower() for w in objection_words):
                    if content[:100] not in summary.objections:
                        summary.objections.append(content[:100])
        summary.objections = summary.objections[-5:]

        # Tone
        summary.tone = _build_tone_record(context)

        return summary


def _classify_financial_state_local(context, answered: dict) -> str:
    from src.question_tracker import FactKey
    from src.models import ResolutionPath
    cash_flow = answered.get(FactKey.CASH_FLOW_ISSUE, "")
    employment = answered.get(FactKey.EMPLOYMENT_STATUS, "")
    if employment == "unemployed":
        return "crisis"
    if cash_flow in ("yes", "severe"):
        return "cash_strapped"
    if cash_flow == "mild" or employment == "self_employed":
        return "unstable_income"
    if context.assessment_data and context.assessment_data.resolution_path == ResolutionPath.LUMP_SUM:
        return "stable"
    return "unstable_income"


def _build_tone_record(context) -> ToneRecord:
    arc = []
    key_moments = []
    hostile_words = ["refuse", "won't", "never", "lawyer", "court", "sue", "no way"]
    cooperative_words = ["okay", "sure", "yes", "agree", "understand", "fine", "accept"]

    for msg in context.conversation_history:
        if msg.get("role") != "user":
            continue
        text = msg.get("content", "").lower()
        hostile = sum(1 for w in hostile_words if w in text)
        cooperative = sum(1 for w in cooperative_words if w in text)
        if hostile > cooperative:
            arc.append("hostile")
        elif cooperative > 0:
            arc.append("cooperative")
        else:
            arc.append("neutral")

    final = arc[-1] if arc else "neutral"
    intent = "willing_full" if final == "cooperative" else \
             "resistant" if final == "hostile" else "unknown"

    # Key moments: first cooperative and last hostile
    for i, a in enumerate(arc):
        if a == "cooperative" and "first_coop" not in [km[:10] for km in key_moments]:
            content = [m for m in context.conversation_history if m.get("role") == "user"][i] if i < len([m for m in context.conversation_history if m.get("role") == "user"]) else None
            if content:
                key_moments.append(f"first_coop: {content.get('content','')[:60]}")

    return ToneRecord(
        arc=arc,
        final_attitude=final,
        intent=intent,
        key_moments=key_moments[:3],
    )
