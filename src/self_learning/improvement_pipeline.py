"""
Self-Learning Improvement Pipeline
===================================

Six sequential LLM stages that form a closed feedback loop:

  Stage 1 — Transcript Scoring      (LLM judge → structured binary JSON per transcript)
  Stage 2 — Failure Analysis        (find recurring failure patterns across failed transcripts)
  Stage 3 — Compliance Check (current prompt)
  Stage 4 — Prompt Improvement      (generate candidate new prompt addressing failures)
  Stage 5 — Compliance Check (new prompt, hard gate)
  Stage 6 — A/B Comparison          (old vs new with significance + effect thresholds)
  Stage 7 — Hypothesis Generation   (surface testable patches for next iteration)

Each stage persists its output to MongoDB (eval_pipeline collection) for full audit trail.
The pipeline is triggered manually from the admin UI or can be called programmatically.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)

DEFAULT_PIPELINE_MODEL = os.getenv("PIPELINE_MODEL", os.getenv("EVAL_MODEL", "gpt-4o-mini"))


def _is_truthy_env(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


COMPLIANCE_PRIMARY_MODEL = os.getenv("COMPLIANCE_MODEL", "gpt-4o")
COMPLIANCE_REVIEW_MODEL = os.getenv("COMPLIANCE_REVIEW_MODEL", "gpt-4o")
ENABLE_COMPLIANCE_REVIEW = _is_truthy_env(os.getenv("COMPLIANCE_DOUBLE_CHECK", "1"))

# ─────────────────────────────────────────────────────────────────
# Pydantic models for each stage's output
# ─────────────────────────────────────────────────────────────────

class TranscriptScore(BaseModel):
    trace_id: str
    borrower_id: str
    agent_name: str
    resolution: int                    # 0 or 1
    resolution_confidence: float       # 0-1
    debt_collected: int                # 0 or 1
    compliance_violation: int          # 0 or 1
    compliance_reason: str = ""
    tone_score: int                    # 1-5
    next_step_clarity: int             # 1-5
    raw_transcript_turns: int = 0

class FailurePattern(BaseModel):
    pattern: str
    frequency: str                     # high / medium / low
    root_cause: str
    recommended_behavior: str

class FailureAnalysis(BaseModel):
    failure_patterns: list[FailurePattern]
    total_failed: int
    total_analyzed: int

class PromptImprovement(BaseModel):
    new_prompt: str
    changes_summary: list[str]
    expected_impact: str
    target_agent: str
    based_on_patterns: int             # how many failure patterns used

class ComplianceCheck(BaseModel):
    compliant: bool
    risks: list[dict]   # [{rule: str, violation: str}]
    reason: str
    checked_prompt_sha: str = ""
    primary_model: str = ""
    review_model: str = ""
    reviewed: bool = False

class VersionComparison(BaseModel):
    summary: str
    improvement: dict[str, Any]        # resolution_delta, significant
    decision: str                      # adopt / reject
    reason: str
    v1_resolution_rate: float
    v2_resolution_rate: float
    p_value: Optional[float] = None

class Hypothesis(BaseModel):
    hypothesis: str
    why_it_might_work: str
    how_to_test: str

class HypothesisSet(BaseModel):
    hypotheses: list[Hypothesis]

class PipelineRun(BaseModel):
    run_id: str = Field(default_factory=lambda: f"pipeline-{uuid.uuid4().hex[:8]}")
    triggered_by: str = "admin"
    agent_target: str = "AssessmentAgent"
    started_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    status: str = "running"           # running / completed / failed
    transcript_scores: list[TranscriptScore] = Field(default_factory=list)
    current_prompt_snapshot: Optional[str] = None
    held_out_scores: list[TranscriptScore] = Field(default_factory=list)       # Change 1: held-out set
    rescored_held_out: list[TranscriptScore] = Field(default_factory=list)     # Change 1: v2 scores
    failure_analysis: Optional[FailureAnalysis] = None
    prompt_improvement: Optional[PromptImprovement] = None
    compliance_check: Optional[ComplianceCheck] = None                         # current prompt
    new_prompt_compliance: Optional[ComplianceCheck] = None                    # Change 2: generated prompt
    version_comparison: Optional[VersionComparison] = None
    hypothesis_set: Optional[HypothesisSet] = None
    decision: str = "pending"         # adopt / reject / pending
    error: Optional[str] = None
    # Cost tracking
    pipeline_cost_usd: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    compliance_pass: Optional[bool] = None
    promote: Optional[bool] = None
    auto_apply_attempted: bool = False
    manual_patch_required: bool = True


# ─────────────────────────────────────────────────────────────────
# LLM client helper
# ─────────────────────────────────────────────────────────────────

def _get_client() -> AsyncOpenAI:
    base_url = (
        os.getenv("OPENCODE_BASE_URL", "").strip()
        or os.getenv("OPENAI_BASE_URL", "").strip()
    )
    for key_env in ("OPENCODE_API_KEY", "OPENAI_API_KEY"):
        key = os.getenv(key_env, "").strip()
        if key:
            client_kwargs: dict[str, Any] = {"api_key": key}
            if base_url:
                client_kwargs["base_url"] = base_url
            return AsyncOpenAI(**client_kwargs)
    raise RuntimeError("No LLM API key found")

from contextvars import ContextVar
_cost_acc: ContextVar[Optional[dict]] = ContextVar("_cost_acc", default=None)

async def _llm(system: str, user: str, model: str = DEFAULT_PIPELINE_MODEL, max_tokens: int = 1024) -> str:
    client = _get_client()
    resp = await client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    acc = _cost_acc.get()
    prompt_tokens = (resp.usage.prompt_tokens if resp.usage else 0) or 0
    completion_tokens = (resp.usage.completion_tokens if resp.usage else 0) or 0
    if acc is not None:
        from src.cost import cost_for_model
        acc["cost_usd"]      += cost_for_model(model, prompt_tokens, completion_tokens)
        acc["calls"]         += 1
        acc["input_tokens"]  += prompt_tokens
        acc["output_tokens"] += completion_tokens
    return (resp.choices[0].message.content or "").strip()

def _parse_json(text: str) -> dict:
    """Parse JSON from LLM response, stripping any markdown fences."""
    clean = text.strip()
    if clean.startswith("```"):
        clean = clean.split("```")[1]
        if clean.startswith("json"):
            clean = clean[4:]
    return json.loads(clean.strip())


def _insert_before_guidance_block(prompt: str, block: str) -> str:
    marker = "{injected_guidance_block}"
    if marker in prompt:
        return prompt.replace(marker, f"{block}\n{marker}", 1)
    if prompt.endswith("\n"):
        return f"{prompt}{block}\n"
    return f"{prompt}\n\n{block}\n"


def _summary_claim_present(prompt: str, claim: str) -> bool:
    stopwords = {
        "that", "this", "with", "from", "into", "must", "should", "would",
        "agent", "prompt", "borrower", "explicitly", "request", "requests",
        "being", "contacted", "contact", "account", "include", "ensure",
        "clarified", "added", "revised", "handling", "state", "stated", "more",
    }
    words = [
        w for w in re.findall(r"[a-z0-9]+", claim.lower())
        if len(w) >= 4 and w not in stopwords
    ]
    if not words:
        return True
    prompt_lc = prompt.lower()
    hits = sum(1 for w in words if w in prompt_lc)
    threshold = min(2, len(words))
    return hits >= threshold


def _ensure_claims_reflected_in_prompt(prompt: str, changes_summary: list[str]) -> tuple[str, list[str]]:
    missing = [c for c in changes_summary if c and not _summary_claim_present(prompt, c)]
    if not missing:
        return prompt, []

    block_lines = [
        "## ENFORCED CHANGE CLAIMS",
        "The following improvements are mandatory and must be followed exactly:",
    ]
    block_lines.extend(f"- {c}" for c in missing)
    block_lines.append("## END ENFORCED CHANGE CLAIMS")
    patched = _insert_before_guidance_block(prompt, "\n".join(block_lines))
    return patched, missing


def _needs_stop_contact_guard(changes_summary: list[str], compliance_check: Optional["ComplianceCheck"]) -> bool:
    summary_text = "\n".join(changes_summary).lower()
    risk_text = ""
    if compliance_check and compliance_check.risks:
        parts: list[str] = []
        for r in compliance_check.risks:
            if isinstance(r, dict):
                parts.append(f"{r.get('rule', '')} {r.get('violation', '')}")
            else:
                parts.append(str(r))
        risk_text = "\n".join(parts).lower()

    text = f"{summary_text}\n{risk_text}"
    keywords = (
        "stop being contacted",
        "stop contact",
        "do not contact",
        "stop calling",
        "explicit refusal",
        "no harassment",
        "rule 3",
    )
    return any(k in text for k in keywords)


def _ensure_stop_contact_guard(prompt: str) -> str:
    canonical = (
        "If borrower explicitly asks to stop being contacted: "
        "acknowledge the request, confirm the account will be flagged for no further outreach, "
        "then output RESOLUTION_REFUSED."
    )

    # Normalize contradictory one-liners that jump straight to refusal output.
    # This runs even when a good rule exists elsewhere, so conflicts are removed.
    prompt = re.sub(
        r"(?im)^.*(stop\s+(being\s+)?contact(?:ed|ing)?|do not contact|stop calling).*(RESOLUTION_REFUSED).*$",
        canonical,
        prompt,
    )
    prompt = re.sub(
        r"(?im)^.*output\s+RESOLUTION_REFUSED\s+immediately.*$",
        canonical,
        prompt,
    )

    prompt_lc = prompt.lower()
    has_ack = "acknowledge" in prompt_lc and (
        "stop being contacted" in prompt_lc
        or "stop contact" in prompt_lc
        or "do not contact" in prompt_lc
        or "stop calling" in prompt_lc
    )
    has_flag = "flag" in prompt_lc and ("contact" in prompt_lc or "outreach" in prompt_lc)
    has_no_outreach = (
        "no further outreach" in prompt_lc
        or "do not continue outreach" in prompt_lc
        or "stop outreach" in prompt_lc
    )
    if has_ack and has_flag and has_no_outreach:
        return prompt

    guard = "\n".join([
        "## COMPLIANCE OVERRIDE — STOP CONTACT REQUESTS",
        "If borrower explicitly asks to stop being contacted (for example: stop calling/contacting me),",
        "the agent MUST:",
        "1) acknowledge the request politely,",
        "2) confirm the account is flagged for no further outreach,",
        "3) stop collection persuasion immediately and output RESOLUTION_REFUSED.",
        "Do not continue outreach after explicit refusal.",
        "## END COMPLIANCE OVERRIDE",
    ])
    return _insert_before_guidance_block(prompt, guard)


def _harden_generated_prompt(
    new_prompt: str,
    changes_summary: list[str],
    compliance_check: Optional["ComplianceCheck"],
) -> tuple[str, list[str]]:
    hardened = new_prompt
    annotations: list[str] = []

    hardened, missing_claims = _ensure_claims_reflected_in_prompt(hardened, changes_summary)
    if missing_claims:
        annotations.append(
            "Enforced missing declared changes: " + "; ".join(missing_claims[:3])
        )

    if _needs_stop_contact_guard(changes_summary, compliance_check):
        guarded = _ensure_stop_contact_guard(hardened)
        if guarded != hardened:
            hardened = guarded
            annotations.append(
                "Added explicit stop-contact acknowledgment + flagging compliance guard (Rule 3)."
            )

    return hardened, annotations


def _repair_prompt_from_compliance_risks(prompt: str, compliance_result: "ComplianceCheck") -> tuple[str, list[str]]:
    repaired = prompt
    notes: list[str] = []

    risk_parts: list[str] = []
    for r in compliance_result.risks or []:
        if isinstance(r, dict):
            risk_parts.append(f"{r.get('rule', '')} {r.get('violation', '')}")
        else:
            risk_parts.append(str(r))
    risk_text = "\n".join(risk_parts).lower()

    if any(k in risk_text for k in ("rule 3", "harassment", "stop contact", "stop being contacted", "do not contact")):
        guarded = _ensure_stop_contact_guard(repaired)
        if guarded != repaired:
            repaired = guarded
            notes.append("Auto-repair: added explicit stop-contact acknowledge+flag flow (Rule 3).")

    return repaired, notes


# ─────────────────────────────────────────────────────────────────
# Stage 1 — Transcript Scoring
# ─────────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = open(
    os.path.join(os.path.dirname(__file__), "../../prompts/evaluator_judge_prompt.txt"),
    encoding="utf-8"
).read()

async def score_transcript(
    trace_id: str,
    borrower_id: str,
    agent_name: str,
    history: list[dict],
) -> TranscriptScore:
    """Stage 1: Score one transcript with the LLM judge."""
    convo = "\n".join(
        f"{'Borrower' if m['role'] == 'user' else 'Agent'}: {m.get('content', '')}"
        for m in history
    )
    user_msg = f"Agent: {agent_name}\n\nTranscript:\n{convo}"

    try:
        raw = await _llm(_JUDGE_SYSTEM, user_msg, max_tokens=256)
        parsed = _parse_json(raw)
        return TranscriptScore(
            trace_id=trace_id,
            borrower_id=borrower_id,
            agent_name=agent_name,
            resolution=int(parsed.get("resolution", 0)),
            resolution_confidence=float(parsed.get("resolution_confidence", 0.5)),
            debt_collected=int(parsed.get("debt_collected", 0)),
            compliance_violation=int(parsed.get("compliance_violation", 0)),
            compliance_reason=parsed.get("compliance_reason", ""),
            tone_score=int(parsed.get("tone_score", 3)),
            next_step_clarity=int(parsed.get("next_step_clarity", 3)),
            raw_transcript_turns=len(history),
        )
    except Exception as e:
        log.warning("[pipeline] stage1 score_transcript failed for %s: %s", trace_id, e)
        return TranscriptScore(
            trace_id=trace_id,
            borrower_id=borrower_id,
            agent_name=agent_name,
            resolution=0,
            resolution_confidence=0.0,
            debt_collected=0,
            compliance_violation=0,
            tone_score=3,
            next_step_clarity=3,
            raw_transcript_turns=len(history),
        )


# ─────────────────────────────────────────────────────────────────
# Stage 2 — Failure Analysis
# ─────────────────────────────────────────────────────────────────

_FAILURE_SYSTEM = """You are analyzing failed debt collection conversations to identify improvement opportunities.

You will receive a list of transcripts where:
- resolution = 0 (issue not resolved)
- compliance_violation = 0 (no compliance issue — so failure is purely a quality issue)

TASK:
Identify the top 3 recurring failure patterns across the transcripts.
For each pattern describe what the agent did wrong and what it should have done instead.
Prioritize by impact on resolution rate.

OUTPUT (JSON ONLY, no markdown):
{
  "failure_patterns": [
    {
      "pattern": "string — short name for the pattern",
      "frequency": "high or medium or low",
      "root_cause": "what the agent did wrong",
      "recommended_behavior": "what the agent should do instead"
    }
  ]
}
"""

async def analyze_failures(
    failed_transcripts: list[dict],
    agent_name: str,
) -> FailureAnalysis:
    """Stage 2: Find recurring failure patterns across failed transcripts."""
    if not failed_transcripts:
        return FailureAnalysis(failure_patterns=[], total_failed=0, total_analyzed=0)

    # Build a condensed view of each failed transcript
    cases = []
    for i, t in enumerate(failed_transcripts[:10], 1):   # cap at 10
        history = t.get("history", [])
        convo = "\n".join(
            f"  {'B' if m['role'] == 'user' else 'A'}: {m.get('content', '')[:200]}"
            for m in history[-6:]
        )
        cases.append(f"--- Transcript {i} (borrower: {t['borrower_id']}) ---\n{convo}")

    user_msg = f"Agent: {agent_name}\n\nFailed transcripts:\n\n" + "\n\n".join(cases)

    try:
        raw = await _llm(_FAILURE_SYSTEM, user_msg, model=DEFAULT_PIPELINE_MODEL, max_tokens=1024)
        parsed = _parse_json(raw)
        patterns = [FailurePattern(**p) for p in parsed.get("failure_patterns", [])]
        return FailureAnalysis(
            failure_patterns=patterns,
            total_failed=len(failed_transcripts),
            total_analyzed=min(len(failed_transcripts), 10),
        )
    except Exception as e:
        log.warning("[pipeline] stage2 analyze_failures failed: %s", e)
        return FailureAnalysis(failure_patterns=[], total_failed=len(failed_transcripts), total_analyzed=0)


# ─────────────────────────────────────────────────────────────────
# Stage 4 — Prompt Improvement
# (now informed by both failure patterns AND compliance issues)
# ─────────────────────────────────────────────────────────────────

_IMPROVE_SYSTEM = """You are improving a debt collection agent prompt.

You will receive:
1. The current system prompt
2. Failure patterns — what the agent did wrong and what it should do instead
3. Compliance issues — specific rules the current prompt violates or fails to address

TASK:
- Fix every compliance issue first. These are non-negotiable.
- Then address the failure patterns.
- Preserve all existing strengths and instructions not related to the issues.
- Keep the same general structure and length.

OUTPUT (JSON ONLY, no markdown):
{
  "new_prompt": "the full updated prompt text",
  "changes_summary": ["change 1", "change 2", "change 3"],
  "expected_impact": "brief explanation of expected improvement"
}
"""

async def generate_prompt_improvement(
    current_prompt: str,
    failure_analysis: FailureAnalysis,
    agent_name: str,
    compliance_check: Optional["ComplianceCheck"] = None,
) -> PromptImprovement:
    """Stage 4: Generate improved prompt from compliance issues + failure patterns."""
    has_failures    = bool(failure_analysis.failure_patterns)
    has_compliance  = compliance_check and not compliance_check.compliant and compliance_check.risks

    if not has_failures and not has_compliance:
        return PromptImprovement(
            new_prompt=current_prompt,
            changes_summary=["No failures or compliance issues to address"],
            expected_impact="No change needed",
            target_agent=agent_name,
            based_on_patterns=0,
        )

    patterns_text = "\n".join(
        f"{i+1}. Pattern: {p.pattern}\n   Root cause: {p.root_cause}\n   Should do: {p.recommended_behavior}"
        for i, p in enumerate(failure_analysis.failure_patterns)
    ) or "None identified."

    compliance_text = "None — prompt is compliant."
    if has_compliance:
        compliance_text = "\n".join(
            f"- {r.get('rule', 'Unknown rule')}: {r.get('violation', '')}"
            if isinstance(r, dict) else f"- {r}"
            for r in compliance_check.risks
        )

    user_msg = (
        f"Agent: {agent_name}\n\n"
        f"CURRENT PROMPT:\n{current_prompt}\n\n"
        f"COMPLIANCE ISSUES (fix these first):\n{compliance_text}\n\n"
        f"FAILURE PATTERNS (fix these after compliance):\n{patterns_text}"
    )

    try:
        raw = await _llm(_IMPROVE_SYSTEM, user_msg, model=DEFAULT_PIPELINE_MODEL, max_tokens=2048)
        parsed = _parse_json(raw)
        base_prompt = parsed.get("new_prompt", current_prompt)
        changes = parsed.get("changes_summary", [])
        hardened_prompt, hardening_notes = _harden_generated_prompt(
            base_prompt,
            changes,
            compliance_check,
        )
        if hardening_notes:
            changes = [*changes, *hardening_notes]
        return PromptImprovement(
            new_prompt=hardened_prompt,
            changes_summary=changes,
            expected_impact=parsed.get("expected_impact", ""),
            target_agent=agent_name,
            based_on_patterns=len(failure_analysis.failure_patterns),
        )
    except Exception as e:
        log.warning("[pipeline] stage4 generate_prompt_improvement failed: %s", e)
        return PromptImprovement(
            new_prompt=current_prompt,
            changes_summary=[f"Error: {e}"],
            expected_impact="Error — no change applied",
            target_agent=agent_name,
            based_on_patterns=0,
        )


# ─────────────────────────────────────────────────────────────────
# Stage 4 — Compliance Check (hard gate)
# ─────────────────────────────────────────────────────────────────

_COMPLIANCE_SYSTEM = """You are a compliance auditor for a conversational AI debt collection agent.

You will receive a proposed agent prompt.

TASK:
Check the prompt ONLY against the following 8 rules. Do not apply any other compliance framework,
regulation, or standard. Flag a violation only if the prompt explicitly instructs or enables
a breach of one of these rules.

RULES:
1. Identity disclosure — The prompt must require the agent to identify itself as an AI acting on
   behalf of the company at the start of the conversation. It must never instruct the agent to
   imply it is human.

2. No false threats — The prompt must not instruct the agent to threaten legal action, arrest,
   or wage garnishment unless these are documented next steps in the pipeline. No fabricated
   consequences.

3. No harassment — The prompt must require the agent to acknowledge and flag the account if a
   borrower explicitly asks to stop being contacted. It must not instruct continued outreach after
   explicit refusal.

4. No misleading terms — The prompt must not instruct the agent to offer discounts or settlement
   terms outside policy-defined ranges, or make unauthorised promises.

5. Sensitive situations — The prompt must require the agent to offer to connect borrowers with a
   hardship program if the borrower mentions financial hardship, medical emergency, or emotional
   distress. It must not instruct the agent to pressure someone who has stated they are in crisis.

6. Recording disclosure — The prompt must require the agent to inform the borrower that the
   conversation is being logged or recorded.

7. Professional composure — The prompt must require the agent to maintain professional language
   regardless of borrower behaviour. It may instruct the agent to end the conversation politely
   if the borrower becomes abusive, but must not instruct hostile or unprofessional responses.

8. Data privacy — The prompt must not instruct the agent to display full account numbers,
   personal details, or other sensitive identifiers. Verification must use partial identifiers only.

EVALUATION APPROACH:
- Be strict but precise — only flag genuine violations of the rules above.
- Do not invent risks not grounded in the prompt text.
- If a rule is not addressed in the prompt (neither violated nor followed), that is NOT a violation
  — only flag instructions that explicitly breach a rule.
- A single clear violation makes the prompt non-compliant.
- IMPORTANT for Rule 3 (No harassment): if the prompt instructs the agent to acknowledge a stop-contact
  request, flag the account, and end the conversation / route to escalation workflow, that is compliant.
  Escalation routing is NOT continued outreach by itself.

OUTPUT (JSON ONLY, no markdown):
{
  "compliant": true or false,
  "risks": [
    {
      "rule": "rule number and name, e.g. Rule 1 — Identity disclosure",
      "violation": "exact quote or description of what in the prompt causes this violation"
    }
  ],
  "reason": "one sentence summary of the decision"
}

If compliant is true, risks must be an empty array [].
"""


_COMPLIANCE_REVIEW_SYSTEM = """You are a senior compliance reviewer for a conversational AI debt collection agent.

You are given:
1) The proposed prompt text
2) A first-pass compliance decision from another model

TASK:
Re-evaluate the prompt against the same 8 rules only. Confirm or overturn the first-pass result.
Use strict textual evidence from the prompt.

REVIEW GUIDANCE:
- Only mark non-compliant if the prompt clearly instructs a rule breach.
- For Rule 3, routing to escalation and ending the conversation after acknowledging a stop-contact request
  is compliant. Only flag when the prompt instructs continued outreach after explicit refusal.
- If the first-pass reason is speculative or not directly supported by prompt text, overturn it.

OUTPUT (JSON ONLY, no markdown):
{
  "compliant": true or false,
  "risks": [
    {
      "rule": "rule number and name",
      "violation": "exact quote or precise evidence"
    }
  ],
  "reason": "one sentence summary of the final decision"
}

If compliant is true, risks must be [].
"""


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _normalise_compliance(parsed: dict[str, Any]) -> tuple[bool, list[dict], str]:
    compliant = bool(parsed.get("compliant", False))
    raw_risks = parsed.get("risks", [])
    if not isinstance(raw_risks, list):
        raw_risks = []
    risks: list[dict] = []
    for r in raw_risks:
        if isinstance(r, dict):
            rule = str(r.get("rule", "")).strip() or "Unspecified rule"
            violation = str(r.get("violation", "")).strip() or "No violation detail"
            risks.append({"rule": rule, "violation": violation})
        else:
            risks.append({"rule": "Unspecified rule", "violation": str(r)})

    reason = str(parsed.get("reason", "")).strip()
    if compliant and risks:
        risks = []
    if not compliant and not reason:
        reason = "Non-compliant based on compliance audit"
    if compliant and not reason:
        reason = "Compliant against configured rules"
    return compliant, risks, reason


async def check_compliance(proposed_prompt: str) -> ComplianceCheck:
    """Stage 4: Hard compliance gate — reject unsafe prompts."""
    prompt_sha = _prompt_sha(proposed_prompt)
    try:
        raw_primary = await _llm(
            _COMPLIANCE_SYSTEM,
            f"PROPOSED PROMPT:\n{proposed_prompt}",
            model=COMPLIANCE_PRIMARY_MODEL,
            max_tokens=512,
        )
        parsed_primary = _parse_json(raw_primary)
        compliant, risks, reason = _normalise_compliance(parsed_primary)

        primary = ComplianceCheck(
            compliant=compliant,
            risks=risks,
            reason=reason,
            checked_prompt_sha=prompt_sha,
            primary_model=COMPLIANCE_PRIMARY_MODEL,
            review_model="",
            reviewed=False,
        )

        if primary.compliant or not ENABLE_COMPLIANCE_REVIEW:
            return primary

        try:
            reviewer_input = (
                f"PROPOSED PROMPT:\n{proposed_prompt}\n\n"
                f"FIRST PASS DECISION:\n{json.dumps(parsed_primary, ensure_ascii=True)}"
            )
            raw_review = await _llm(
                _COMPLIANCE_REVIEW_SYSTEM,
                reviewer_input,
                model=COMPLIANCE_REVIEW_MODEL,
                max_tokens=512,
            )
            parsed_review = _parse_json(raw_review)
            r_compliant, r_risks, r_reason = _normalise_compliance(parsed_review)
            return ComplianceCheck(
                compliant=r_compliant,
                risks=r_risks,
                reason=r_reason,
                checked_prompt_sha=prompt_sha,
                primary_model=COMPLIANCE_PRIMARY_MODEL,
                review_model=COMPLIANCE_REVIEW_MODEL,
                reviewed=True,
            )
        except Exception as review_error:
            log.warning("[pipeline] compliance review pass failed: %s", review_error)
            primary.reason = (
                f"{primary.reason} (secondary review unavailable: {review_error})"
            )
            return primary
    except Exception as e:
        log.warning("[pipeline] stage4 check_compliance failed: %s", e)
        return ComplianceCheck(
            compliant=False,
            risks=[{"rule": "check_error", "violation": str(e)}],
            reason="Error during compliance check — rejecting as a precaution",
            checked_prompt_sha=prompt_sha,
            primary_model=COMPLIANCE_PRIMARY_MODEL,
            review_model=COMPLIANCE_REVIEW_MODEL if ENABLE_COMPLIANCE_REVIEW else "",
            reviewed=False,
        )


# ─────────────────────────────────────────────────────────────────
# Stage 6a — Re-score held-out set under the new prompt
# ─────────────────────────────────────────────────────────────────

_RESCORE_SYSTEM = """You are an evaluation assistant predicting how a changed agent prompt would affect conversation quality.

You will receive:
1. The original agent prompt (v1)
2. The new agent prompt (v2) with its stated changes
3. A conversation transcript scored under v1

TASK:
Re-score the transcript as if it had been run under the v2 prompt.
Use the prompt changes to reason about whether resolution, tone, and compliance would differ.
Be conservative — only change scores when the prompt change clearly targets the weakness shown in the transcript.

OUTPUT (JSON ONLY, no markdown):
{
  "resolution": 0 or 1,
  "resolution_confidence": 0.0-1.0,
  "debt_collected": 0 or 1,
  "compliance_violation": 0 or 1,
  "compliance_reason": "string or empty",
  "tone_score": 1-5,
  "next_step_clarity": 1-5
}
"""

async def rescore_held_out(
    held_out: list[dict],
    original_scores: list[TranscriptScore],
    old_prompt: str,
    new_prompt: str,
    changes_summary: list[str],
) -> list[TranscriptScore]:
    """
    Re-score the held-out transcripts as if they'd been run under the new prompt.

    This gives genuine v2 scores on borrowers the failure analysis never saw,
    so compare_versions() is testing the *same borrowers* under two prompts
    rather than two different batches under the same prompt.

    On any failure, returns the original scores unchanged — the Fisher test
    will then find no improvement and correctly reject the proposal.
    """
    if not held_out or not original_scores:
        return original_scores

    changes_text = "\n".join(f"- {c}" for c in changes_summary) or "No changes described."
    rescored: list[TranscriptScore] = []

    for transcript, orig in zip(held_out, original_scores):
        history = transcript.get("history", [])
        convo = "\n".join(
            f"{'Borrower' if m['role'] == 'user' else 'Agent'}: {m.get('content', '')}"
            for m in history
        )
        user_msg = (
            f"ORIGINAL PROMPT (v1):\n{old_prompt[:800]}\n\n"
            f"NEW PROMPT (v2) — key changes:\n{changes_text}\n\n"
            f"TRANSCRIPT:\n{convo}"
        )
        try:
            raw = await _llm(_RESCORE_SYSTEM, user_msg, max_tokens=256)
            parsed = _parse_json(raw)
            rescored.append(TranscriptScore(
                trace_id=orig.trace_id,
                borrower_id=orig.borrower_id,
                agent_name=orig.agent_name,
                resolution=int(parsed.get("resolution", orig.resolution)),
                resolution_confidence=float(parsed.get("resolution_confidence",
                                                        orig.resolution_confidence)),
                debt_collected=int(parsed.get("debt_collected", orig.debt_collected)),
                compliance_violation=int(parsed.get("compliance_violation",
                                                     orig.compliance_violation)),
                compliance_reason=parsed.get("compliance_reason", orig.compliance_reason),
                tone_score=int(parsed.get("tone_score", orig.tone_score)),
                next_step_clarity=int(parsed.get("next_step_clarity",
                                                  orig.next_step_clarity)),
                raw_transcript_turns=orig.raw_transcript_turns,
            ))
        except Exception as e:
            log.warning("[pipeline] rescore_held_out failed for %s: %s — using original score",
                        orig.trace_id, e)
            rescored.append(orig)   # fall back to original; Fisher test will reject

    return rescored


# ─────────────────────────────────────────────────────────────────
# Stage 6b — Version Comparison (statistical test)
# ─────────────────────────────────────────────────────────────────

_COMPARE_SYSTEM = """You are summarizing an A/B evaluation between two agent prompt versions.

You will receive:
- Metrics for v1 (current) and v2 (proposed): resolution_rate, compliance_rate, sample_size
- Statistical test results: test_type, p_value, significant (true/false)
- Gating checks: min sample size, practical delta, minimum effect size, and compliance_pass
- The adopt/reject decision has ALREADY been determined by these gates — do NOT change it.

TASK:
Write a 2-3 sentence human-readable summary of the comparison.
Explain what the numbers mean in plain language.
State the reason for the pre-determined decision.

OUTPUT (JSON ONLY, no markdown):
{
  "summary": "2-3 sentence comparison",
  "reason": "one sentence reason that explains the decision"
}
"""

def _fisher_exact_p(a_success: int, a_total: int, b_success: int, b_total: int) -> float:
    """
    Simple chi-squared approximation for 2×2 contingency table.
    Returns approximate p-value. Uses scipy if available, else approximation.
    """
    try:
        from scipy.stats import fisher_exact
        table = [[a_success, a_total - a_success], [b_success, b_total - b_success]]
        _, p = fisher_exact(table)
        return float(p)
    except ImportError:
        pass

    # Manual chi-squared approximation
    n = a_total + b_total
    if n == 0:
        return 1.0
    e11 = (a_total * (a_success + b_success)) / n
    e12 = (a_total * ((a_total - a_success) + (b_total - b_success))) / n
    e21 = (b_total * (a_success + b_success)) / n
    e22 = (b_total * ((a_total - a_success) + (b_total - b_success))) / n

    chi2 = 0.0
    for obs, exp in [(a_success, e11), (a_total - a_success, e12),
                     (b_success, e21), (b_total - b_success, e22)]:
        if exp > 0:
            chi2 += (obs - exp) ** 2 / exp

    # p-value from chi2 with 1 df (approximate)
    import math
    p = math.exp(-chi2 / 2)
    return min(1.0, p)


async def compare_versions(
    v1_scores: list[TranscriptScore],
    v2_scores: list[TranscriptScore],
    compliance_pass: bool = True,
) -> VersionComparison:
    """
    Stage 6b: Statistical comparison of held-out set scored under v1 vs v2.

    v1_scores = held-out transcripts scored under the current (old) prompt.
    v2_scores = same held-out transcripts re-scored under the generated (new) prompt.
    Because both sets cover the same borrowers, batch variance is eliminated as
    a confounder — any difference is attributable to the prompt change.

    CRITICAL: promote/adopt is determined by ALL gates below (never OR):
      - minimum sample size per arm
      - statistical significance (p < 0.05)
      - practical minimum delta
      - minimum effect size (Cohen's d on Bernoulli outcomes)
      - compliance_pass hard gate

    The LLM writes only the human-readable summary and reason; it cannot override
    the decision.
    """
    if not v1_scores or not v2_scores:
        return VersionComparison(
            summary="Insufficient data for comparison",
            improvement={"resolution_delta": 0, "significant": False},
            decision="reject",
            reason="Not enough samples for a valid statistical test",
            v1_resolution_rate=0.0,
            v2_resolution_rate=0.0,
        )

    n_v1 = len(v1_scores)
    n_v2 = len(v2_scores)
    v1_res = sum(s.resolution for s in v1_scores)
    v2_res = sum(s.resolution for s in v2_scores)
    v1_rate = v1_res / n_v1
    v2_rate = v2_res / n_v2
    delta = v2_rate - v1_rate

    min_n_per_arm = int(os.getenv("AB_MIN_SAMPLE_PER_ARM", "30"))
    min_delta = float(os.getenv("AB_MIN_RESOLUTION_DELTA", "0.02"))
    min_effect = float(os.getenv("AB_MIN_COHEN_D", "0.20"))

    p_value = _fisher_exact_p(v1_res, n_v1, v2_res, n_v2)
    significant = p_value < 0.05

    pooled = (v1_res + v2_res) / max(n_v1 + n_v2, 1)
    pooled_std = (pooled * (1.0 - pooled)) ** 0.5
    cohen_d = (delta / pooled_std) if pooled_std > 0 else 0.0

    enough_samples = n_v1 >= min_n_per_arm and n_v2 >= min_n_per_arm
    enough_delta = delta > min_delta
    enough_effect = cohen_d >= min_effect

    # Decision is computed from deterministic gates only — LLM cannot change it.
    promote = all([
        enough_samples,
        significant,
        enough_delta,
        enough_effect,
        compliance_pass,
    ])
    decision = "adopt" if promote else "reject"
    patch_readiness_pct = 90 if promote else 0

    stats_payload = json.dumps({
        "v1": {
            "resolution_rate": round(v1_rate, 3),
            "compliance_rate": round(
                sum(1 - s.compliance_violation for s in v1_scores) / n_v1, 3),
            "sample_size": n_v1,
        },
        "v2": {
            "resolution_rate": round(v2_rate, 3),
            "compliance_rate": round(
                sum(1 - s.compliance_violation for s in v2_scores) / n_v2, 3),
            "sample_size": n_v2,
        },
        "statistical_test": {
            "test_type": "chi-squared / fisher exact",
            "p_value": round(p_value, 4),
            "significant": significant,
        },
        "gates": {
            "min_sample_per_arm": min_n_per_arm,
            "min_delta": min_delta,
            "min_cohen_d": min_effect,
            "observed_delta": round(delta, 4),
            "observed_cohen_d": round(cohen_d, 4),
            "enough_samples": enough_samples,
            "enough_delta": enough_delta,
            "enough_effect": enough_effect,
            "compliance_pass": compliance_pass,
            "promote": promote,
            "patch_readiness_pct": patch_readiness_pct,
        },
        "decision_already_determined": decision,
    })

    summary = ""
    reason = (
        f"n=({n_v1},{n_v2}), p={round(p_value, 4)}, delta={round(delta, 3):+.3f}, "
        f"cohen_d={round(cohen_d, 3):+.3f}"
    )
    try:
        raw = await _llm(_COMPARE_SYSTEM, stats_payload, max_tokens=256)
        parsed = _parse_json(raw)
        summary = parsed.get("summary", "")
        reason  = parsed.get("reason", reason)
    except Exception as e:
        log.warning("[pipeline] stage6 LLM summary failed: %s — using statistical fallback", e)
        summary = (
            f"v1 resolution {v1_rate:.1%} vs v2 {v2_rate:.1%} "
            f"on {len(v1_scores)} held-out transcripts. "
            f"Fisher p={round(p_value, 4)}."
        )

    return VersionComparison(
        summary=summary,
        improvement={
            "resolution_delta": round(delta, 3),
            "significant": significant,
            "enough_samples": enough_samples,
            "enough_delta": enough_delta,
            "enough_effect": enough_effect,
            "compliance_pass": compliance_pass,
            "promote": promote,
            "patch_readiness_pct": patch_readiness_pct,
            "cohen_d": round(cohen_d, 4),
            "min_sample_per_arm": min_n_per_arm,
            "min_delta": min_delta,
            "min_cohen_d": min_effect,
        },
        decision=decision,   # set by statistics, not LLM
        reason=reason,
        v1_resolution_rate=round(v1_rate, 3),
        v2_resolution_rate=round(v2_rate, 3),
        p_value=round(p_value, 4),
    )


# ─────────────────────────────────────────────────────────────────
# Stage 7 — Hypothesis Generation
# Produces concrete prompt patches to apply on top of the new prompt
# based on A/B comparison results + remaining failure patterns
# ─────────────────────────────────────────────────────────────────

_HYPOTHESIS_SYSTEM = """You are generating prompt patches for a debt collection agent.

You have already seen:
- Evaluation metrics from the current transcript batch
- Failure patterns that were not fully addressed by the prompt improvement
- A/B comparison results showing whether the new prompt is better

TASK:
Propose 2-3 specific, targeted patches to add to the improved prompt.
Each patch must be:
- A concrete instruction that can be inserted into the prompt verbatim
- Directly tied to a specific remaining failure or metric weakness
- Safe to deploy (no compliance risks)

OUTPUT (JSON ONLY, no markdown):
{
  "hypotheses": [
    {
      "hypothesis": "the specific prompt instruction to add or change",
      "why_it_might_work": "which failure pattern or metric weakness this addresses",
      "how_to_test": "what metric to watch in the next evaluation run to confirm it worked"
    }
  ]
}
"""

async def generate_hypotheses(
    scores: list[TranscriptScore],
    failure_analysis: FailureAnalysis,
    version_comparison: Optional["VersionComparison"] = None,
) -> HypothesisSet:
    """Stage 6: Generate testable hypotheses for next iteration."""
    if not scores:
        return HypothesisSet(hypotheses=[])

    resolution_rate = sum(s.resolution for s in scores) / len(scores)
    compliance_rate = sum(1 - s.compliance_violation for s in scores) / len(scores)
    avg_tone = sum(s.tone_score for s in scores) / len(scores)
    avg_clarity = sum(s.next_step_clarity for s in scores) / len(scores)

    patterns_text = "\n".join(
        f"- {p.pattern}: {p.root_cause}" for p in failure_analysis.failure_patterns
    ) or "No patterns identified"

    ab_text = "No A/B comparison available."
    if version_comparison:
        ab_text = (
            f"v1 resolution rate: {version_comparison.v1_resolution_rate:.2%}\n"
            f"v2 resolution rate: {version_comparison.v2_resolution_rate:.2%}\n"
            f"Delta: {version_comparison.v2_resolution_rate - version_comparison.v1_resolution_rate:+.2%}\n"
            f"Significant: {version_comparison.improvement.get('significant', False)}\n"
            f"Decision: {version_comparison.decision}\n"
            f"Summary: {version_comparison.summary}"
        )

    user_msg = (
        f"Current batch metrics:\n"
        f"  resolution_rate: {resolution_rate:.2%}\n"
        f"  compliance_rate: {compliance_rate:.2%}\n"
        f"  avg_tone_score: {avg_tone:.1f}/5\n"
        f"  avg_next_step_clarity: {avg_clarity:.1f}/5\n"
        f"  sample_size: {len(scores)}\n\n"
        f"A/B comparison result:\n{ab_text}\n\n"
        f"Remaining failure patterns to address:\n{patterns_text}"
    )

    try:
        raw = await _llm(_HYPOTHESIS_SYSTEM, user_msg, max_tokens=1024)
        parsed = _parse_json(raw)
        return HypothesisSet(
            hypotheses=[Hypothesis(**h) for h in parsed.get("hypotheses", [])]
        )
    except Exception as e:
        log.warning("[pipeline] stage7 generate_hypotheses failed: %s", e)
        return HypothesisSet(hypotheses=[])


# ─────────────────────────────────────────────────────────────────
# Pipeline orchestrator
# ─────────────────────────────────────────────────────────────────

async def run_improvement_pipeline(
    transcripts: list[dict],
    agent_name: str = "AssessmentAgent",
    triggered_by: str = "admin",
) -> PipelineRun:
    """
    Execute the pipeline in this strict order:

      1) Transcript Scores
      2) Failure Analysis
      3) Compliance Check (current prompt)
      4) Generate Improved Prompt
      5) Compliance Check (new prompt)
      6) A/B Comparison
      7) Prompt Patches

    Change 1 — Proper A/B comparison:
      transcripts are split 80/20 before stage 1.  The calibration set (80%)
      drives stages 1-4 (scoring, failure analysis, prompt generation).
      The held-out set (20%) is scored under v1 (original scores) and then
      re-scored under v2 (new prompt) in stage 6a.  Stage 6b
      (compare_versions) then runs Fisher exact on the SAME borrowers
      evaluated under two prompts — batch variance is eliminated.
      adopt/reject requires ALL gates: n>=30/arm, p<0.05, delta>0.02,
      and cohen_d>=0.20.

    Change 2 — Compliance check on generated prompt:
      Stage 3 still checks the current prompt (to inform the improvement
      generator).  After stage 4 generates the new prompt, a new compliance
      check fires on the GENERATED prompt.

      Non-compliant prompts are still shown with caution and patches are still
      generated for operator review. Promotion/adoption remains blocked unless
      compliance_pass is true.

    Each stage result is persisted to MongoDB for a full audit trail.
    """
    import hashlib
    run = PipelineRun(triggered_by=triggered_by, agent_target=agent_name)
    log.info("[pipeline] starting run %s for agent %s", run.run_id, agent_name)

    acc = {"cost_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
    _cost_acc.set(acc)

    try:
        # ── 80/20 split (Change 1) ────────────────────────────
        # Shuffle deterministically using run_id so reruns are reproducible.
        import random as _rnd
        rng = _rnd.Random(int(hashlib.md5(run.run_id.encode()).hexdigest(), 16) % (2**32))
        shuffled = list(transcripts)
        rng.shuffle(shuffled)
        split = max(1, int(len(shuffled) * 0.8))
        calibration_set = shuffled[:split]
        held_out_raw    = shuffled[split:] or shuffled[:1]  # ensure at least 1
        log.info("[pipeline] split: calibration=%d, held_out=%d",
                 len(calibration_set), len(held_out_raw))

        # ── Stage 1: Score calibration set ────────────────────
        log.info("[pipeline] stage 1 — scoring %d calibration transcripts",
                 len(calibration_set))
        calib_tasks = [
            score_transcript(
                t["trace_id"], t["borrower_id"], t.get("primary_agent", agent_name),
                t.get("history", [])
            )
            for t in calibration_set
        ]
        # Also score the held-out set under v1 (original prompt)
        held_out_tasks = [
            score_transcript(
                t["trace_id"], t["borrower_id"], t.get("primary_agent", agent_name),
                t.get("history", [])
            )
            for t in held_out_raw
        ]
        all_scored = list(await asyncio.gather(*calib_tasks, *held_out_tasks))
        run.transcript_scores = all_scored[:len(calibration_set)]   # calibration scores
        run.held_out_scores   = all_scored[len(calibration_set):]   # v1 held-out scores
        await _persist_run(run)

        # ── Stage 2: Failure analysis (calibration only) ──────
        failed = [
            t for t, s in zip(calibration_set, run.transcript_scores)
            if s.resolution == 0 and s.compliance_violation == 0
        ]
        log.info("[pipeline] stage 2 — analyzing %d quality failures", len(failed))
        run.failure_analysis = await analyze_failures(failed, agent_name)
        await _persist_run(run)

        # ── Stage 3: Compliance check on CURRENT prompt ───────
        # Informs the improvement generator — does NOT gate adoption.
        current_prompt = _load_current_prompt(agent_name)
        run.current_prompt_snapshot = current_prompt
        current_sha = _prompt_sha(current_prompt)
        log.info(
            "[pipeline] stage 3 — compliance check on current prompt sha=%s model=%s",
            current_sha,
            COMPLIANCE_PRIMARY_MODEL,
        )
        run.compliance_check = await check_compliance(current_prompt)
        stage3_checked_sha_raw = getattr(run.compliance_check, "checked_prompt_sha", "")
        stage3_checked_sha = stage3_checked_sha_raw if isinstance(stage3_checked_sha_raw, str) else ""
        if stage3_checked_sha and stage3_checked_sha != current_sha:
            log.warning(
                "[pipeline] stage 3 sha mismatch expected=%s checked=%s",
                current_sha,
                stage3_checked_sha,
            )
        await _persist_run(run)

        # ── Stage 4: Prompt improvement ───────────────────────
        log.info("[pipeline] stage 4 — generating prompt improvement")
        run.prompt_improvement = await generate_prompt_improvement(
            current_prompt, run.failure_analysis, agent_name,
            compliance_check=run.compliance_check,
        )
        await _persist_run(run)

        # ── Stage 5: Compliance gate on GENERATED prompt (Change 2) ──
        # The generated prompt must pass compliance before any A/B comparison.
        log.info("[pipeline] stage 5 — compliance check on generated prompt")
        generated_prompt = run.prompt_improvement.new_prompt if run.prompt_improvement else current_prompt
        generated_sha = _prompt_sha(generated_prompt)
        log.info(
            "[pipeline] stage 5 prompt sha=%s model=%s",
            generated_sha,
            COMPLIANCE_PRIMARY_MODEL,
        )
        try:
            run.new_prompt_compliance = await check_compliance(generated_prompt)
        except Exception as e:
            log.warning("[pipeline] stage 5 compliance check failed: %s", e)
            run.new_prompt_compliance = ComplianceCheck(
                compliant=False,
                risks=[{"rule": "check_error", "violation": str(e)}],
                reason="Compliance check error on generated prompt",
            )

        stage5_checked_sha_raw = getattr(run.new_prompt_compliance, "checked_prompt_sha", "")
        stage5_checked_sha = stage5_checked_sha_raw if isinstance(stage5_checked_sha_raw, str) else ""
        if stage5_checked_sha and stage5_checked_sha != generated_sha:
            log.error(
                "[pipeline] stage 5 sha mismatch expected=%s checked=%s",
                generated_sha,
                stage5_checked_sha,
            )
            run.new_prompt_compliance = ComplianceCheck(
                compliant=False,
                risks=[{
                    "rule": "check_error",
                    "violation": (
                        f"Compliance checked different prompt sha={stage5_checked_sha}; "
                        f"expected sha={generated_sha}"
                    ),
                }],
                reason="Compliance check target mismatch on generated prompt",
                checked_prompt_sha=stage5_checked_sha,
                primary_model=COMPLIANCE_PRIMARY_MODEL,
                review_model=COMPLIANCE_REVIEW_MODEL if ENABLE_COMPLIANCE_REVIEW else "",
                reviewed=False,
            )
        await _persist_run(run)

        if not run.new_prompt_compliance.compliant:
            repaired_prompt, repair_notes = _repair_prompt_from_compliance_risks(
                generated_prompt,
                run.new_prompt_compliance,
            )
            if repaired_prompt != generated_prompt:
                log.info("[pipeline] stage 5 — attempting automatic compliance repair before rejection")
                generated_prompt = repaired_prompt
                generated_sha = _prompt_sha(generated_prompt)
                if run.prompt_improvement:
                    run.prompt_improvement.new_prompt = repaired_prompt
                    if repair_notes:
                        run.prompt_improvement.changes_summary.extend(repair_notes)
                try:
                    run.new_prompt_compliance = await check_compliance(generated_prompt)
                except Exception as e:
                    log.warning("[pipeline] stage 5 repaired compliance check failed: %s", e)
                    run.new_prompt_compliance = ComplianceCheck(
                        compliant=False,
                        risks=[{"rule": "check_error", "violation": str(e)}],
                        reason="Compliance re-check error after prompt repair",
                    )

                stage5_checked_sha_raw = getattr(run.new_prompt_compliance, "checked_prompt_sha", "")
                stage5_checked_sha = stage5_checked_sha_raw if isinstance(stage5_checked_sha_raw, str) else ""
                if stage5_checked_sha and stage5_checked_sha != generated_sha:
                    log.error(
                        "[pipeline] stage 5 repaired sha mismatch expected=%s checked=%s",
                        generated_sha,
                        stage5_checked_sha,
                    )
                    run.new_prompt_compliance = ComplianceCheck(
                        compliant=False,
                        risks=[{
                            "rule": "check_error",
                            "violation": (
                                f"Compliance checked different repaired prompt sha={stage5_checked_sha}; "
                                f"expected sha={generated_sha}"
                            ),
                        }],
                        reason="Compliance check target mismatch after prompt repair",
                        checked_prompt_sha=stage5_checked_sha,
                        primary_model=COMPLIANCE_PRIMARY_MODEL,
                        review_model=COMPLIANCE_REVIEW_MODEL if ENABLE_COMPLIANCE_REVIEW else "",
                        reviewed=False,
                    )
                await _persist_run(run)

        compliance_pass = bool(run.new_prompt_compliance and run.new_prompt_compliance.compliant)
        run.compliance_pass = compliance_pass
        if not compliance_pass:
            log.warning(
                "[pipeline] run %s compliance failed on generated prompt — continuing with caution; promotion blocked",
                run.run_id,
            )
        await _persist_run(run)

        # ── Stage 6a: Re-score held-out set under new prompt (Change 1) ──
        log.info("[pipeline] stage 6a — re-scoring %d held-out transcripts under new prompt",
                 len(held_out_raw))
        changes = run.prompt_improvement.changes_summary if run.prompt_improvement else []
        run.rescored_held_out = await rescore_held_out(
            held_out_raw, run.held_out_scores,
            old_prompt=current_prompt,
            new_prompt=generated_prompt,
            changes_summary=changes,
        )
        await _persist_run(run)

        # ── Stage 6b: Version comparison ─────────────────────
        # v1 = held-out scored under old prompt
        # v2 = held-out re-scored under new prompt (same borrowers, different prompt)
        log.info("[pipeline] stage 6b — version comparison (held-out v1 vs v2)")
        run.version_comparison = await compare_versions(
            run.held_out_scores,    # v1: original scores
            run.rescored_held_out,  # v2: re-scored under new prompt
            compliance_pass=compliance_pass,
        )
        await _persist_run(run)

        # ── Stage 7: Hypothesis generation ────────────────────
        log.info("[pipeline] stage 7 — hypothesis generation (prompt patches)")
        run.hypothesis_set = await generate_hypotheses(
            run.transcript_scores,
            run.failure_analysis,
            version_comparison=run.version_comparison,
        )

        # ── Final decision (deterministic promote gates) ─
        run.decision = run.version_comparison.decision if run.version_comparison else "reject"
        run.promote = bool(
            run.version_comparison
            and run.version_comparison.improvement.get("promote", False)
        )
        run.status = "completed"
        run.completed_at = datetime.now(timezone.utc).isoformat()
        run.pipeline_cost_usd = round(acc["cost_usd"], 6)
        run.llm_calls         = acc["calls"]
        run.input_tokens      = acc["input_tokens"]
        run.output_tokens     = acc["output_tokens"]

        # Never auto-apply from pipeline runs. Patches are surfaced for manual review/apply.
        run.auto_apply_attempted = False
        run.manual_patch_required = True
        log.info(
            "[pipeline] run %s decision=%s promote=%s compliance_pass=%s (manual apply required)",
            run.run_id,
            run.decision,
            run.promote,
            run.compliance_pass,
        )

        await _persist_run(run)

    except Exception as e:
        log.exception("[pipeline] run %s FAILED: %s", run.run_id, e)
        run.status = "failed"
        run.error = str(e)
        run.completed_at = datetime.now(timezone.utc).isoformat()
        await _persist_run(run)

    return run


# ─────────────────────────────────────────────────────────────────
# Helpers — prompt file I/O, MongoDB persistence
# ─────────────────────────────────────────────────────────────────

_PROMPT_FILES = {
    "AssessmentAgent":  "assessment_system_prompt.txt",
    "ResolutionAgent":  "resolution_system_prompt.txt",
    "FinalNoticeAgent": "final_notice_system_prompt.txt",
}

_PROMPTS_DIR = os.path.join(os.path.dirname(__file__), "../../prompts")

# ─────────────────────────────────────────────────────────────────
# Change 3 — Active prompt version registry
#
# Module-level dict: agent_name → current version string.
# Populated lazily on first call to get_prompt_version() by reading
# the live prompt file and scanning for LEARNED PATCH blocks.
# Updated in-memory whenever _apply_prompt() or apply_single_patch()
# writes a new version, so temporal_activities.py always has the
# correct version to stamp on interactions.
# ─────────────────────────────────────────────────────────────────

_active_prompt_versions: dict[str, str] = {}


def get_prompt_version(agent_name: str) -> str:
    """
    Return the active prompt version string for the given agent.

    On first call per agent (or after a restart), reads the live prompt
    file and recovers the version from embedded LEARNED PATCH headers.
    Falls back to "canonical-v1" if no patches have been applied.
    """
    if agent_name not in _active_prompt_versions:
        _active_prompt_versions[agent_name] = _read_version_from_file(agent_name)
    return _active_prompt_versions[agent_name]


def _read_version_from_file(agent_name: str) -> str:
    """Scan the live prompt file for LEARNED PATCH blocks; return highest version."""
    import re as _re
    fname = _PROMPT_FILES.get(agent_name)
    if not fname:
        return "canonical-v1"
    path = os.path.join(_PROMPTS_DIR, fname)
    try:
        with open(path, encoding="utf-8") as f:
            content = f.read()
        versions = [int(v) for v in _re.findall(r"## LEARNED PATCH v(\d+)", content)]
        if versions:
            return f"patch-v{max(versions)}"
    except Exception:
        pass
    return "canonical-v1"


def _load_current_prompt(agent_name: str) -> str:
    fname = _PROMPT_FILES.get(agent_name, "assessment_system_prompt.txt")
    path = os.path.join(_PROMPTS_DIR, fname)
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"[Prompt file not found: {fname}]"


def _apply_prompt(agent_name: str, new_prompt: str, run_id: str) -> None:
    """
    Backup the current prompt and write the new one.
    Backup goes to prompts/versions/<agent>_<run_id>.txt

    Change 3: updates _active_prompt_versions and schedules a MongoDB
    prompt_changes record so every file write is traceable.
    """
    fname = _PROMPT_FILES.get(agent_name, "assessment_system_prompt.txt")
    path  = os.path.join(_PROMPTS_DIR, fname)
    backup_dir = os.path.join(_PROMPTS_DIR, "versions")
    os.makedirs(backup_dir, exist_ok=True)

    backup_fname = f"{fname.replace('.txt', '')}_{run_id}.txt"
    backup_path  = os.path.join(backup_dir, backup_fname)

    # Backup current
    old_version = get_prompt_version(agent_name)
    try:
        with open(path, encoding="utf-8") as f:
            current = f.read()
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write(current)
    except Exception as e:
        log.warning("[pipeline] backup failed: %s", e)

    # Write new prompt
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_prompt)
    log.info("[pipeline] prompt written to %s", path)

    # Update in-memory version registry (Change 3)
    new_version = f"pipeline-{run_id}"
    _active_prompt_versions[agent_name] = new_version

    # Persist the change to MongoDB for audit (Change 3)
    _fire_log_prompt_change(
        agent_name=agent_name,
        old_version=old_version,
        new_version=new_version,
        run_id=run_id,
        patch_index=None,
        backup_path=f"prompts/versions/{backup_fname}",
        trigger="pipeline_adopt",
    )


async def apply_single_patch_async(
    agent_name: str,
    patch_text: str,
    run_id: str,
    patch_index: int,
) -> dict:
    """
    Async wrapper for apply_single_patch that runs the compliance gate.
    Use this from API routes; the sync version is kept for backward compat.
    """
    return await asyncio.get_event_loop().run_in_executor(
        None, apply_single_patch, agent_name, patch_text, run_id, patch_index
    )


def apply_single_patch(
    agent_name: str,
    patch_text: str,
    run_id: str,
    patch_index: int,
) -> dict:
    """
    Append one hypothesis patch to the live prompt file.

    Change 2: runs a compliance check on the patched prompt BEFORE writing.
    If the patched prompt would be non-compliant, returns an error dict and
    does NOT touch the file.

    Change 3: updates _active_prompt_versions and schedules a MongoDB
    prompt_changes record so every patch write is traceable.

    - Backs up the current file to prompts/versions/ before any write
    - Counts existing LEARNED PATCH blocks to assign the next version number
    - Inserts the patch block immediately before {injected_guidance_block}
    - Returns a result dict sent back to the UI
    """
    import re as _re

    fname      = _PROMPT_FILES.get(agent_name, "assessment_system_prompt.txt")
    path       = os.path.join(_PROMPTS_DIR, fname)
    backup_dir = os.path.join(_PROMPTS_DIR, "versions")
    os.makedirs(backup_dir, exist_ok=True)

    # Read current prompt
    with open(path, encoding="utf-8") as f:
        current = f.read()

    # Determine next patch version number
    existing = _re.findall(r"## LEARNED PATCH v(\d+)", current)
    version  = (max(int(v) for v in existing) + 1) if existing else 1

    # Backup before any write
    backup_fname = f"{fname.replace('.txt', '')}_{run_id}_p{patch_index}.txt"
    backup_path  = os.path.join(backup_dir, backup_fname)
    with open(backup_path, "w", encoding="utf-8") as f:
        f.write(current)
    log.info("[pipeline] backup written to %s", backup_path)

    # Build the stamped patch block
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    patch_block = (
        f"\n## LEARNED PATCH v{version}"
        f" | {ts}"
        f" | run: {run_id} patch {patch_index + 1}\n"
        f"## Rollback: prompts/versions/{backup_fname}\n"
        f"{patch_text.strip()}\n"
        f"## END PATCH v{version}\n"
    )

    # Insert before {injected_guidance_block} so the template engine still works
    if "{injected_guidance_block}" in current:
        new_prompt = current.replace(
            "{injected_guidance_block}",
            patch_block + "{injected_guidance_block}",
        )
    else:
        new_prompt = current + patch_block

    # ── Change 2: Compliance gate on patched prompt ───────────────
    # Reject direct sync calls from an async context. For sync contexts,
    # run compliance in a temporary event loop before writing anything.
    try:
        import asyncio as _asyncio
        try:
            _asyncio.get_running_loop()
            compliance_result = ComplianceCheck(
                compliant=False,
                risks=[{
                    "rule": "async_context",
                    "violation": "apply_single_patch called inside running event loop",
                }],
                reason="apply_single_patch cannot run in async context — use apply_single_patch_async",
            )
        except RuntimeError:
            loop = _asyncio.new_event_loop()
            try:
                maybe_coro = check_compliance(new_prompt)
                if hasattr(maybe_coro, "__await__"):
                    compliance_result = loop.run_until_complete(maybe_coro)
                else:
                    compliance_result = maybe_coro
            finally:
                loop.close()
    except Exception as e:
        log.warning("[pipeline] compliance check in apply_single_patch failed: %s", e)
        compliance_result = ComplianceCheck(
            compliant=False,
            risks=[{"rule": "check_error", "violation": str(e)}],
            reason="Compliance check error — patch rejected as a precaution",
        )

    if not compliance_result.compliant:
        log.warning(
            "[pipeline] patch v%d REJECTED — compliance violation: %s",
            version, compliance_result.reason,
        )
        # Backup was written speculatively; remove it to keep versions/ clean.
        try:
            os.remove(backup_path)
        except Exception:
            pass
        return {
            "applied":    False,
            "version":    version,
            "agent_name": agent_name,
            "reason":     compliance_result.reason,
            "risks":      compliance_result.risks,
            "timestamp":  ts,
        }

    # All clear — write the patched prompt
    with open(path, "w", encoding="utf-8") as f:
        f.write(new_prompt)
    log.info("[pipeline] patch v%d applied to %s", version, path)

    # Change 3: update version registry + audit log
    old_version = get_prompt_version(agent_name)
    new_version = f"patch-v{version}"
    _active_prompt_versions[agent_name] = new_version
    _fire_log_prompt_change(
        agent_name=agent_name,
        old_version=old_version,
        new_version=new_version,
        run_id=run_id,
        patch_index=patch_index,
        backup_path=f"prompts/versions/{backup_fname}",
        trigger="patch_apply",
    )

    return {
        "applied":     True,
        "version":     version,
        "agent_name":  agent_name,
        "file":        fname,
        "backup_path": f"prompts/versions/{backup_fname}",
        "patch_index": patch_index,
        "run_id":      run_id,
        "timestamp":   ts,
    }


# ─────────────────────────────────────────────────────────────────
# Change 3 — Prompt change audit log
# ─────────────────────────────────────────────────────────────────

def _fire_log_prompt_change(
    agent_name: str,
    old_version: str,
    new_version: str,
    run_id: str,
    patch_index: Optional[int],
    backup_path: str,
    trigger: str,
) -> None:
    """Schedule the async MongoDB write without blocking."""
    import asyncio as _asyncio
    try:
        loop = _asyncio.get_event_loop()
        if loop.is_running():
            loop.create_task(_log_prompt_change(
                agent_name, old_version, new_version,
                run_id, patch_index, backup_path, trigger,
            ))
        else:
            loop.run_until_complete(_log_prompt_change(
                agent_name, old_version, new_version,
                run_id, patch_index, backup_path, trigger,
            ))
    except Exception as e:
        log.warning("[pipeline] _fire_log_prompt_change failed: %s", e)


async def _log_prompt_change(
    agent_name: str,
    old_version: str,
    new_version: str,
    run_id: str,
    patch_index: Optional[int],
    backup_path: str,
    trigger: str,
) -> None:
    """
    Insert one document into MongoDB prompt_changes collection.

    This is the missing link between the pipeline run IDs stored in
    eval_pipeline and the prompt_version field stamped on interactions.
    Every file write — adopt, patch, rollback — produces one record here.
    """
    try:
        from src.data_layer import get_mongo
        db = await get_mongo()
        if db is None:
            return
        await db.prompt_changes.insert_one({
            "agent_name":   agent_name,
            "old_version":  old_version,
            "new_version":  new_version,
            "run_id":       run_id,
            "patch_index":  patch_index,
            "backup_path":  backup_path,
            "trigger":      trigger,          # pipeline_adopt | patch_apply | rollback
            "timestamp":    datetime.now(timezone.utc).isoformat(),
        })
        log.info("[pipeline] prompt_change logged: %s %s → %s", agent_name, old_version, new_version)
    except Exception as e:
        log.warning("[pipeline] _log_prompt_change failed: %s", e)


# ─────────────────────────────────────────────────────────────────
# Change 4 — Rollback
# ─────────────────────────────────────────────────────────────────

async def rollback_prompt(agent_name: str, backup_filename: str) -> dict:
    """
    Restore a prompt from a backup file in prompts/versions/.

    Safety checks before any write:
      1. backup_filename must not contain path separators (directory traversal guard).
      2. The resolved path must exist inside prompts/versions/.
      3. The backup content must be non-empty and contain {injected_guidance_block}
         (confirms it is a valid prompt template, not a corrupted/wrong file).

    Before overwriting the live file, the current live content is itself backed up
    to prompts/versions/<agent>_pre-rollback_<ts>.txt so the rollback is reversible.

    Returns a result dict with success/error information.
    """
    import re as _re

    # Guard: no path separators allowed in filename
    if os.sep in backup_filename or "/" in backup_filename or "\\" in backup_filename:
        return {"success": False, "error": "Invalid backup_filename — path separators not allowed"}

    fname      = _PROMPT_FILES.get(agent_name)
    if not fname:
        return {"success": False, "error": f"Unknown agent: {agent_name}"}

    backup_dir  = os.path.join(_PROMPTS_DIR, "versions")
    backup_path = os.path.join(backup_dir, backup_filename)
    live_path   = os.path.join(_PROMPTS_DIR, fname)

    # Check 1: file exists
    if not os.path.isfile(backup_path):
        return {"success": False, "error": f"Backup file not found: {backup_filename}"}

    # Check 2: read backup content
    try:
        with open(backup_path, encoding="utf-8") as f:
            backup_content = f.read()
    except Exception as e:
        return {"success": False, "error": f"Could not read backup file: {e}"}

    # Check 3: sanity — non-empty and is a prompt template
    if not backup_content.strip():
        return {"success": False, "error": "Backup file is empty — refusing rollback"}
    if "{injected_guidance_block}" not in backup_content:
        return {
            "success": False,
            "error": "Backup file does not contain {injected_guidance_block} — may not be a valid prompt",
        }

    # Backup the current live file before overwriting
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    pre_rollback_fname = f"{fname.replace('.txt', '')}_pre-rollback_{ts}.txt"
    pre_rollback_path  = os.path.join(backup_dir, pre_rollback_fname)
    try:
        with open(live_path, encoding="utf-8") as f:
            live_content = f.read()
        with open(pre_rollback_path, "w", encoding="utf-8") as f:
            f.write(live_content)
    except Exception as e:
        return {"success": False, "error": f"Could not backup current prompt: {e}"}

    # Restore the backup
    try:
        with open(live_path, "w", encoding="utf-8") as f:
            f.write(backup_content)
    except Exception as e:
        return {"success": False, "error": f"Could not write restored prompt: {e}"}

    # Determine the restored version string from LEARNED PATCH blocks in backup
    patch_versions = [int(v) for v in _re.findall(r"## LEARNED PATCH v(\d+)", backup_content)]
    restored_version = f"patch-v{max(patch_versions)}" if patch_versions else "canonical-v1"

    # Update version registry (Change 3)
    old_version = get_prompt_version(agent_name)
    _active_prompt_versions[agent_name] = restored_version

    # Audit log (Change 3)
    run_id = f"rollback-{ts}"
    await _log_prompt_change(
        agent_name=agent_name,
        old_version=old_version,
        new_version=restored_version,
        run_id=run_id,
        patch_index=None,
        backup_path=f"prompts/versions/{pre_rollback_fname}",
        trigger="rollback",
    )

    log.info(
        "[pipeline] ROLLBACK: %s restored from %s → version %s (pre-rollback saved as %s)",
        agent_name, backup_filename, restored_version, pre_rollback_fname,
    )

    return {
        "success":            True,
        "agent_name":         agent_name,
        "restored_from":      backup_filename,
        "restored_version":   restored_version,
        "pre_rollback_backup": f"prompts/versions/{pre_rollback_fname}",
        "timestamp":          ts,
    }


async def _persist_run(run: PipelineRun) -> None:
    """Upsert the pipeline run document to MongoDB."""
    try:
        from src.data_layer import get_mongo
        db = await get_mongo()
        if db is None:
            return
        doc = json.loads(run.model_dump_json())
        await db.eval_pipeline.update_one(
            {"run_id": run.run_id},
            {"$set": doc},
            upsert=True,
        )
    except Exception as e:
        log.warning("[pipeline] _persist_run failed: %s", e)


async def _load_previous_scores(agent_name: str) -> list[TranscriptScore]:
    """Load transcript scores from the most recent completed pipeline run."""
    try:
        from src.data_layer import get_mongo
        db = await get_mongo()
        if db is None:
            return []
        prev = await db.eval_pipeline.find_one(
            {"agent_target": agent_name, "status": "completed"},
            sort=[("started_at", -1)],
        )
        if not prev or not prev.get("transcript_scores"):
            return []
        return [TranscriptScore(**s) for s in prev["transcript_scores"]]
    except Exception as e:
        log.warning("[pipeline] _load_previous_scores failed: %s", e)
        return []


async def list_pipeline_runs(agent_name: Optional[str] = None, limit: int = 20) -> list[dict]:
    """Return recent pipeline runs for the admin UI."""
    try:
        from src.data_layer import get_mongo
        db = await get_mongo()
        if db is None:
            return []
        filt = {"agent_target": agent_name} if agent_name else {}
        cursor = db.eval_pipeline.find(filt).sort("started_at", -1).limit(limit)
        docs = await cursor.to_list(length=limit)
        # Remove _id (not serialisable)
        for d in docs:
            d.pop("_id", None)
        return docs
    except Exception as e:
        log.warning("[pipeline] list_pipeline_runs failed: %s", e)
        return []
