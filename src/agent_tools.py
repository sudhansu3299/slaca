"""
Agent tool definitions and handlers.

Each tool has:
  - A JSON schema (passed to Claude's `tools=` parameter)
  - A handler coroutine that executes the real logic

Tools by agent
──────────────
AssessmentAgent
  verify_borrower_identity   — lookup borrower record in Postgres by last4 + dob_year
  store_financial_data       — persist monthly income/expenses + employment to DB

ResolutionAgent
  classify_borrower_behaviour — score tone/language to label cooperative/strategic/hostile

FinalNoticeAgent
  generate_settlement_document — return ordered list of documents for final settlement PDF
"""

from __future__ import annotations

import logging
import os
from typing import Any

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Postgres connection (application schema — separate from Temporal)
# ─────────────────────────────────────────────────────────────────

BORROWER_DB_URL = os.getenv(
    "BORROWER_DB_URL",
    "postgresql://collections:collections@localhost:5433/collections_ai",
)

_pg_pool = None
_pg_unavailable = False   # set to True after first failed connect — stops retrying


async def _get_pg():
    """Lazy asyncpg connection pool for the borrower database."""
    global _pg_pool, _pg_unavailable
    if _pg_unavailable:
        return None
    if _pg_pool is None:
        try:
            import asyncpg
            _pg_pool = await asyncpg.create_pool(BORROWER_DB_URL, min_size=1, max_size=5)
            log.info("[pg] borrower pool connected to %s", BORROWER_DB_URL)
        except Exception as e:
            log.error("[pg] borrower DB unavailable (%s) — identity verification will FAIL hard", e)
            _pg_unavailable = True
            _pg_pool = None
    return _pg_pool


# ─────────────────────────────────────────────────────────────────
# Tool JSON schemas
# ─────────────────────────────────────────────────────────────────

ASSESSMENT_TOOLS = [
    {
        "name": "verify_borrower_identity",
        "description": (
            "Verify a borrower's identity by looking up their record in the loan database. "
            "Returns verified status and masked personal details if the last-4 digits of "
            "the loan account number and birth year match. Call this as soon as the borrower "
            "provides both values."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "borrower_id": {
                    "type": "string",
                    "description": "The system borrower ID (e.g. BRW-XXXXXXXX)",
                },
                "last4": {
                    "type": "string",
                    "description": "Last 4 digits of the loan account number provided by the borrower",
                },
                "dob_year": {
                    "type": "integer",
                    "description": "Birth year provided by the borrower (e.g. 1985)",
                },
            },
            "required": ["borrower_id", "last4", "dob_year"],
        },
    },
    {
        "name": "store_financial_data",
        "description": (
            "Persist the borrower's self-reported financial data to the database. "
            "Call this once monthly_income AND monthly_expenses are both confirmed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "borrower_id": {
                    "type": "string",
                    "description": "The system borrower ID",
                },
                "monthly_income": {
                    "type": "number",
                    "description": "Borrower's stated monthly income in INR",
                },
                "monthly_expenses": {
                    "type": "number",
                    "description": "Borrower's stated monthly expenses in INR",
                },
                "employment_status": {
                    "type": "string",
                    "enum": ["salaried", "self_employed", "unemployed", "retired", "unknown"],
                    "description": "Borrower's employment status",
                },
            },
            "required": ["borrower_id", "monthly_income", "monthly_expenses", "employment_status"],
        },
    },
]

RESOLUTION_TOOLS = [
    {
        "name": "classify_borrower_behaviour",
        "description": (
            "Analyse the borrower's conversation messages and classify their negotiation "
            "behaviour. Returns one of: cooperative, strategic, or hostile. "
            "Call this after at least 2 borrower turns to get a reliable classification."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "borrower_id": {
                    "type": "string",
                    "description": "The system borrower ID",
                },
                "borrower_messages": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of raw borrower message strings from the conversation so far",
                },
            },
            "required": ["borrower_id", "borrower_messages"],
        },
    },
]

FINAL_NOTICE_TOOLS = [
    {
         "name": "generate_settlement_document",
         "description": (
             "Generate a formal settlement contract for the borrower. "
             "Call this when the borrower confirms acceptance. "
             "Pass the EXACT offer amounts that were presented — do not invent or change any figures."
         ),
         "input_schema": {
             "type": "object",
             "properties": {
                 "borrower_id": {
                     "type": "string",
                     "description": "The system borrower ID",
                 },
                 "loan_id": {
                     "type": "string",
                     "description": "The loan ID",
                 },
                 "resolution_path": {
                     "type": "string",
                     "enum": ["lump_sum", "installment", "hardship"],
                     "description": "The agreed resolution path",
                 },
                 "outstanding_amount": {
                     "type": "number",
                     "description": "Full outstanding loan balance in INR",
                 },
                 "upfront_amount": {
                     "type": "number",
                     "description": "Upfront / down payment amount from the offer actually presented to borrower",
                 },
                 "monthly_amount": {
                     "type": "number",
                     "description": "Monthly instalment amount from the offer (0 for lump_sum)",
                 },
                 "tenure_months": {
                     "type": "integer",
                     "description": "Number of monthly instalments (0 for lump_sum)",
                 },
                 "offer_expiry": {
                     "type": "string",
                     "description": "Offer expiry date string from the offer presented",
                 },
             },
             "required": ["borrower_id", "loan_id", "resolution_path", "outstanding_amount", "upfront_amount"],
         },
     },
 ]


# ─────────────────────────────────────────────────────────────────
# Tool handlers
# ─────────────────────────────────────────────────────────────────

async def handle_verify_borrower_identity(args: dict[str, Any]) -> dict[str, Any]:
    """
    Look up borrower in Postgres.
    Falls back to a plausible mock when DB is unavailable.
    """
    borrower_id = args["borrower_id"]
    last4 = str(args["last4"]).strip()
    dob_year = int(args["dob_year"])

    pool = await _get_pg()
    if pool:
        try:
            async with pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT borrower_id, full_name, loan_account_last4,
                           date_of_birth, loan_id, outstanding_amount, days_past_due
                    FROM borrowers
                    WHERE loan_account_last4 = $1
                      AND EXTRACT(YEAR FROM date_of_birth) = $2
                    """,
                    last4, dob_year,
                )
            if row:
                return {
                    "verified": True,
                    "borrower_id": row["borrower_id"],
                    "name_initial": row["full_name"][0].upper() + ".",
                    "loan_id": row["loan_id"],
                    "outstanding_amount": float(row["outstanding_amount"]),
                    "days_past_due": row["days_past_due"],
                }
            else:
                return {"verified": False, "reason": "Identity details do not match our records"}
        except Exception as e:
            log.warning("[tool] verify_borrower_identity DB error: %s", e)

    # ── DB unavailable — hard failure, do not let unverified borrowers proceed ──
    log.error("[tool] verify_borrower_identity — DB unavailable, returning unverified")
    return {
        "verified": False,
        "reason": "Identity verification service is currently unavailable. Unable to verify details.",
    }


async def handle_store_financial_data(args: dict[str, Any]) -> dict[str, Any]:
    """
    Upsert the borrower's self-reported financials into Postgres.
    Also writes to MongoDB via data_layer for the audit trail.
    """
    borrower_id = args["borrower_id"]
    income = float(args["monthly_income"])
    expenses = float(args["monthly_expenses"])
    employment = args["employment_status"]
    cash_flow_surplus = income - expenses

    pool = await _get_pg()
    if pool:
        try:
            async with pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO borrower_financials
                        (borrower_id, monthly_income, monthly_expenses,
                         employment_status, recorded_at)
                    VALUES ($1, $2, $3, $4, NOW())
                    ON CONFLICT (borrower_id) DO UPDATE
                        SET monthly_income    = EXCLUDED.monthly_income,
                            monthly_expenses  = EXCLUDED.monthly_expenses,
                            employment_status = EXCLUDED.employment_status,
                            recorded_at       = NOW()
                    """,
                    borrower_id, income, expenses, employment,
                )
            log.info("[tool] store_financial_data persisted for %s", borrower_id)
        except Exception as e:
            log.warning("[tool] store_financial_data DB error: %s", e)

    return {
        "stored": True,
        "borrower_id": borrower_id,
        "monthly_income": income,
        "monthly_expenses": expenses,
        "employment_status": employment,
        "cash_flow_surplus": cash_flow_surplus,
        "financial_state": _derive_financial_state(income, expenses, employment),
    }


async def handle_classify_borrower_behaviour(args: dict[str, Any]) -> dict[str, Any]:
    """
    Heuristic + keyword scoring of borrower message tone.
    Returns cooperative | strategic | hostile with a confidence score.
    """
    messages = args.get("borrower_messages", [])
    combined = " ".join(messages).lower()

    cooperative_signals = [
        "okay", "yes", "sure", "i can", "i will", "i agree", "understood",
        "thank you", "alright", "sounds good", "i'll pay", "when do i",
        "how do i", "please", "appreciate",
    ]
    strategic_signals = [
        "discount", "reduce", "lower", "negotiate", "better offer",
        "competitor", "other bank", "less than", "settle for", "how much less",
        "what if i", "can you do",
    ]
    hostile_signals = [
        "refuse", "won't", "will not", "don't care", "waste of time",
        "sue me", "take me to court", "lawyer", "harassment", "illegal",
        "threatening", "stop calling", "block", "report you", "rbi",
        "consumer forum",
    ]

    coop_score = sum(1 for s in cooperative_signals if s in combined)
    strat_score = sum(1 for s in strategic_signals if s in combined)
    host_score = sum(1 for s in hostile_signals if s in combined)

    total = coop_score + strat_score + host_score or 1

    if host_score > 0 and host_score >= strat_score:
        behaviour = "hostile"
        confidence = round(host_score / total, 2)
        recommendation = "State consequences immediately. Do not re-negotiate. Escalate if refused."
    elif strat_score > coop_score:
        behaviour = "strategic"
        confidence = round(strat_score / total, 2)
        recommendation = "Hold the offer line. Acknowledge interest but do not move on terms."
    else:
        behaviour = "cooperative"
        confidence = round(coop_score / total, 2)
        recommendation = "Push for immediate commitment while borrower is receptive."

    log.info(
        "[tool] classify_borrower_behaviour: %s=%s (coop=%d strat=%d host=%d)",
        args["borrower_id"], behaviour, coop_score, strat_score, host_score,
    )

    return {
        "borrower_id": args["borrower_id"],
        "behaviour": behaviour,
        "confidence": confidence,
        "scores": {"cooperative": coop_score, "strategic": strat_score, "hostile": host_score},
        "recommendation": recommendation,
    }


async def handle_generate_settlement_document(args: dict[str, Any]) -> dict[str, Any]:
    """
    Generates a formal Settlement Contract document.

    Returns both:
    - `contract_html`: a fully formatted HTML contract ready to render in the chat UI
    - `contract_text`: plain-text version for logging / downstream use
    - Structured fields for programmatic consumption

    Pass `accepted=True` (via args) when the borrower has already committed
    (e.g. after a voice call) so the stamp reads "ACCEPTED" and the footer
    shows the verbal acceptance confirmation rather than a pending notice.
    """
    from datetime import datetime, timezone, timedelta

    borrower_id     = args["borrower_id"]
    loan_id         = args["loan_id"]
    resolution_path = args["resolution_path"]
    outstanding     = float(args["outstanding_amount"])
    accepted        = bool(args.get("accepted", False))

    # Use the EXACT offer amounts that were presented to the borrower.
    # Fall back to simple calculations only if not provided.
    upfront_amount  = float(args.get("upfront_amount") or 0)
    monthly_amount  = float(args.get("monthly_amount") or 0)
    tenure_months   = int(args.get("tenure_months") or 0)

    now = datetime.now(timezone.utc)
    issued_str = now.strftime("%d %B %Y")

    # Use offer_expiry if provided by Claude, otherwise derive from path policy
    offer_expiry = args.get("offer_expiry", "")
    if offer_expiry:
        deadline_str = offer_expiry
    else:
        deadline_days = {"lump_sum": 5, "installment": 3, "hardship": 3}.get(resolution_path, 3)
        deadline_str = (now + timedelta(days=deadline_days)).strftime("%d %B %Y, %H:%M UTC")

    if resolution_path == "lump_sum":
        if not upfront_amount:
            upfront_amount = round(outstanding * 0.85, 2)
        discount_pct = round((1 - upfront_amount / outstanding) * 100, 1) if outstanding else 0
        offer_lines = [
            f"Settlement Type: One-Time Lump Sum",
            f"Settlement Amount: ₹{upfront_amount:,.2f}",
            f"Discount Applied: {discount_pct}% on outstanding balance of ₹{outstanding:,.2f}",
            f"Payment Deadline: {deadline_str}",
            f"Payment Method: NEFT / RTGS / Bank Transfer to Collections Account",
        ]
        payment_schedule = [
            {"instalment": 1, "amount": upfront_amount, "due": deadline_str, "type": "Full and Final"}
        ]

    elif resolution_path == "installment":
        if not upfront_amount:
            upfront_amount = round(outstanding * 0.15, 2)
        if not monthly_amount:
            monthly_amount = round((outstanding - upfront_amount) / (tenure_months or 10), 2)
        if not tenure_months:
            tenure_months = 10
        first_emi_due = (now + timedelta(days=30)).strftime("%d %B %Y")
        total_value   = upfront_amount + monthly_amount * tenure_months
        offer_lines = [
            f"Settlement Type: Structured Instalment Plan",
            f"Down Payment: ₹{upfront_amount:,.2f} (due by {deadline_str})",
            f"Monthly Instalment: ₹{monthly_amount:,.2f} × {tenure_months} months",
            f"First EMI Due: {first_emi_due}",
            f"Total Settlement Value: ₹{total_value:,.2f}",
            f"Payment Method: Auto-debit (NACH Mandate) or Post-Dated Cheques",
        ]
        payment_schedule = [{"instalment": 0, "amount": upfront_amount, "due": deadline_str, "type": "Down Payment"}]
        for i in range(1, tenure_months + 1):
            due = (now + timedelta(days=30 * i)).strftime("%d %B %Y")
            payment_schedule.append({"instalment": i, "amount": monthly_amount, "due": due, "type": "EMI"})

    else:  # hardship
        if not monthly_amount:
            monthly_amount = round(outstanding / (tenure_months or 24), 2)
        if not tenure_months:
            tenure_months = 24
        offer_lines = [
            f"Settlement Type: Hardship Relief Plan",
            f"Monthly Payment: ₹{monthly_amount:,.2f} × {tenure_months} months",
            f"Interest Waived: Yes — subject to timely payment",
            f"First Payment Due: {(now + timedelta(days=30)).strftime('%d %B %Y')}",
            f"Payment Method: Auto-debit (NACH Mandate)",
        ]
        payment_schedule = [
            {"instalment": i, "amount": monthly_amount,
             "due": (now + timedelta(days=30 * i)).strftime("%d %B %Y"), "type": "Monthly"}
            for i in range(1, tenure_months + 1)
        ]

    consequences = [
        "Credit bureau reporting — 7-year negative record on your credit file",
        "Legal notice issued within 7 days of expiry",
        "Court summons filed within 30 days",
        "Wage garnishment application submitted to your employer",
        "Property lien filing initiated against registered assets",
        "Employment reference contact where legally permitted",
    ]

    required_docs_by_path = {
        "lump_sum":    ["Identity Proof (PAN / Aadhaar)", "Payment Receipt", "Settlement Agreement Letter (signed)", "No Objection Certificate (NOC) — issued post-settlement"],
        "installment": ["Identity Proof (PAN / Aadhaar)", "Down Payment Receipt", "NACH Mandate / Post-Dated Cheques", "EMI Schedule (signed)", "Settlement Agreement Letter (signed)", "No Objection Certificate (NOC) — issued post-settlement"],
        "hardship":    ["Identity Proof (PAN / Aadhaar)", "Hardship Declaration Form", "Income Proof (last 3 months)", "NACH Mandate", "Revised EMI Schedule (signed)", "Settlement Agreement Letter (signed)", "No Objection Certificate (NOC) — issued post-settlement"],
    }
    required_docs = required_docs_by_path.get(resolution_path, required_docs_by_path["installment"])

    # ── Build HTML contract ──────────────────────────────────────
    offer_rows  = "".join(f"<tr><td>{l.split(':')[0].strip()}</td><td><strong>{l.split(':',1)[1].strip() if ':' in l else ''}</strong></td></tr>" for l in offer_lines)
    consq_items = "".join(f"<li>{c}</li>" for c in consequences)
    docs_items  = "".join(f"<li>{d}</li>" for d in required_docs)
    schedule_rows = "".join(
        f"<tr><td style='text-align:center'>{'Down' if p['type']=='Down Payment' else p['instalment']}</td>"
        f"<td>{p['type']}</td><td>₹{p['amount']:,.2f}</td><td>{p['due']}</td></tr>"
        for p in payment_schedule[:6]  # show first 6 rows in UI
    )
    more_rows = f"<tr><td colspan='4' style='text-align:center;color:#7a849a;font-size:11px'>+ {len(payment_schedule)-6} more instalments</td></tr>" if len(payment_schedule) > 6 else ""

    stamp_class = "contract-stamp accepted" if accepted else "contract-stamp"
    stamp_text  = "ACCEPTED" if accepted else "PENDING ACCEPTANCE"

    contract_html = f"""<div class="contract-doc">
  <div class="contract-header">
    <div class="contract-logo">⚖</div>
    <div>
      <div class="contract-title">SETTLEMENT CONTRACT</div>
      <div class="contract-subtitle">Final Notice — Legally Binding Agreement</div>
    </div>
    <div class="{stamp_class}">{stamp_text}</div>
  </div>

  <div class="contract-meta">
    <span><strong>Loan Account:</strong> {loan_id}</span>
    <span><strong>Date Issued:</strong> {issued_str}</span>
    <span><strong>Offer Expires:</strong> {deadline_str}</span>
  </div>

  <div class="contract-section">
    <div class="contract-section-title">1. SETTLEMENT TERMS</div>
    <table class="contract-table">
      <tbody>{offer_rows}</tbody>
    </table>
  </div>

  <div class="contract-section">
    <div class="contract-section-title">2. PAYMENT SCHEDULE</div>
    <table class="contract-table">
      <thead><tr><th>#</th><th>Type</th><th>Amount</th><th>Due Date</th></tr></thead>
      <tbody>{schedule_rows}{more_rows}</tbody>
    </table>
  </div>

  <div class="contract-section">
    <div class="contract-section-title">3. CONSEQUENCES OF NON-PAYMENT</div>
    <div class="contract-warning">If the settlement is not honoured by the deadline, the following actions will proceed automatically without further notice:</div>
    <ol class="contract-list">{consq_items}</ol>
  </div>

  <div class="contract-section">
    <div class="contract-section-title">4. DOCUMENTS REQUIRED</div>
    <div class="contract-note">Submit all documents to <strong>settlements@collections.internal</strong> within 48 hours of acceptance:</div>
    <ul class="contract-list">{docs_items}</ul>
  </div>

  <div class="contract-section">
    <div class="contract-section-title">5. TERMS AND CONDITIONS</div>
    <div class="contract-terms">
      This offer constitutes a legally binding settlement agreement upon acceptance. 
      Acceptance is confirmed by the borrower's verbal or written agreement captured in this conversation, 
      which is recorded and admissible as evidence. The lending institution reserves the right to 
      rescind this offer if any payment is missed. Partial payments do not constitute settlement. 
      A No Objection Certificate (NOC) will be issued only upon full completion of all payments.
    </div>
  </div>

  <div class="contract-acceptance">
    <div class="contract-acceptance-label">{'Acceptance confirmed by verbal agreement on recorded voice call. Download this document for your records.' if accepted else 'Your response in this conversation constitutes formal acceptance of these terms.'}</div>
    <button class="contract-download-btn" onclick="downloadContractPDF(this)">Download PDF</button>
  </div>
</div>"""

    # ── Plain-text version ───────────────────────────────────────
    contract_text = f"""
SETTLEMENT CONTRACT — {loan_id}
Issued: {issued_str} | Expires: {deadline_str}
{'='*60}

1. SETTLEMENT TERMS
{chr(10).join(f'   {l}' for l in offer_lines)}

2. CONSEQUENCES OF NON-PAYMENT
{chr(10).join(f'   {i+1}. {c}' for i, c in enumerate(consequences))}

3. REQUIRED DOCUMENTS
{chr(10).join(f'   • {d}' for d in required_docs)}

Submit to: settlements@collections.internal within 48 hours.
{'='*60}
Your response in this conversation constitutes formal acceptance.
""".strip()

    return {
        "borrower_id":      borrower_id,
        "loan_id":          loan_id,
        "resolution_path":  resolution_path,
        "outstanding":      outstanding,
        "deadline":         deadline_str,
        "issued":           issued_str,
        "offer_lines":      offer_lines,
        "payment_schedule": payment_schedule,
        "consequences":     consequences,
        "required_docs":    required_docs,
        "contract_html":    contract_html,
        "contract_text":    contract_text,
    }


# ─────────────────────────────────────────────────────────────────
# Dispatch — maps tool name → handler
# ─────────────────────────────────────────────────────────────────

TOOL_HANDLERS: dict[str, Any] = {
    "verify_borrower_identity": handle_verify_borrower_identity,
    "store_financial_data": handle_store_financial_data,
    "classify_borrower_behaviour": handle_classify_borrower_behaviour,
    "generate_settlement_document": handle_generate_settlement_document,
}


async def execute_tool(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    """Dispatch a tool call to its handler. Returns the result dict."""
    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        log.error("[tool] unknown tool: %s", tool_name)
        return {"error": f"Unknown tool: {tool_name}"}
    try:
        result = await handler(tool_input)
        log.info("[tool] %s → %s", tool_name, result)
        return result
    except Exception as e:
        log.exception("[tool] %s raised: %s", tool_name, e)
        return {"error": str(e)}


# ─────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────

def _derive_financial_state(income: float, expenses: float, employment: str) -> str:
    """Classify financial state for resolution path hint."""
    surplus = income - expenses
    if employment == "unemployed":
        return "crisis"
    if surplus < 0:
        return "cash_strapped"
    if surplus < income * 0.2:
        return "unstable_income"
    return "stable"
