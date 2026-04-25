from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional
from pydantic import BaseModel, Field


class Stage(str, Enum):
    ASSESSMENT = "assessment"
    RESOLUTION = "resolution"
    FINAL_NOTICE = "final_notice"
    COMPLETE = "complete"
    ESCALATED = "escalated"


class ResolutionPath(str, Enum):
    LUMP_SUM = "lump_sum"
    INSTALLMENT = "installment"
    HARDSHIP = "hardship"
    LEGAL = "legal"


class BorrowerStatus(str, Enum):
    ENGAGED      = "engaged"
    UNRESPONSIVE = "unresponsive"
    COMMITTED    = "committed"
    REFUSED      = "refused"
    HARDSHIP     = "hardship"


class BorrowerBehaviour(str, Enum):
    COOPERATIVE = "cooperative"
    STRATEGIC   = "strategic"
    HOSTILE     = "hostile"
    UNKNOWN     = "unknown"


class FinancialSituation(BaseModel):
    monthly_income: float
    monthly_expenses: float
    outstanding_debt: float
    assets_value: float
    liabilities_value: float
    employment_status: str
    has_cash_flow_issue: bool = False


class AssessmentData(BaseModel):
    borrower_id: str
    loan_id: str
    principal_amount: float
    outstanding_amount: float
    days_past_due: int
    identity_verified: bool = False
    financial_situation: Optional[FinancialSituation] = None
    resolution_path: Optional[ResolutionPath] = None


class ResolutionOffer(BaseModel):
    path: ResolutionPath
    discount_percentage: float = 0.0
    upfront_required: float = 0.0
    monthly_payment: float = 0.0
    tenure_months: int = 0
    deadline_days: int
    valid_until: str


class ConversationContext(BaseModel):
    borrower_id: str
    loan_id: str
    current_stage: Stage = Stage.ASSESSMENT
    conversation_history: list[dict] = Field(default_factory=list)
    assessment_data: Optional[AssessmentData] = None
    resolution_offer: Optional[ResolutionOffer] = None
    assessment_outcome: Optional[str] = None
    resolution_outcome: Optional[str] = None
    final_notice_outcome: Optional[str] = None
    borrower_status: BorrowerStatus = BorrowerStatus.ENGAGED
    # Behaviour classification (set by ResolutionAgent tool call)
    borrower_behaviour: BorrowerBehaviour = BorrowerBehaviour.UNKNOWN
    borrower_behaviour_confidence: float = 0.0
    # Settlement documents (set by FinalNoticeAgent tool call)
    settlement_documents: list[dict] = Field(default_factory=list)
    # Token + cost tracking (serialised as plain dicts so Pydantic can handle them)
    token_usage: dict = Field(default_factory=dict)      # agent → {input, output}
    total_cost_usd: float = 0.0
    # Tracks whether each agent has already sent its opening disclosure
    # so Claude is told not to repeat it on subsequent turns
    assessment_opened: bool = False
    # Set to True after the hardship offer has been made to the borrower.
    # While True, the assessment agent waits for yes/no before collecting more facts.
    hardship_offer_pending: bool = False
    # Final notice confirmation flow controls:
    # - ask up to 3 times total
    # - attempt 2 handles renegotiation with a hard warning
    # - attempt 3 is final before escalation
    final_notice_confirmation_asked: bool = False
    final_notice_reask_used: bool = False
    final_notice_confirmation_attempts: int = 0
    # Question dedup state: FactKey.value → {"asked": bool, "answered": bool, "value": str|None}
    question_state: dict = Field(default_factory=dict)
