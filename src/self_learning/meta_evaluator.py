"""
L3 meta-evaluator pipeline.

Implements the first production L3 mechanism:
  scoring consistency audit on the L2 `resolution` label.

Flow:
  1. Load up to N recent transcripts for one agent or all agents.
  2. Reuse the latest cached L2 resolution label when available.
  3. Re-score the same transcript with an alternate, more anchored judge prompt.
  4. Compute observed agreement, expected agreement, and Cohen's kappa.
  5. Persist a versioned audit run to MongoDB and expose status for the admin UI.

Auto-triggering mirrors the improvement feeder: every 300 completed conversations,
run a background meta-evaluation across all agents.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.cost import cost_for_model
from src.evaluator import LLMJudge, _first_nonempty_env, _normalize_base_url
from src.prompts import evaluator_judge_prompt

log = logging.getLogger(__name__)


def _int_env(name: str, default: int) -> int:
    raw = os.getenv(name, str(default)).strip()
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


AUTO_TRIGGER_EVERY_N_ITERATIONS: int = max(1, _int_env("AUTO_META_EVAL_TRIGGER_EVERY", 300))
DEFAULT_SAMPLE_SIZE: int = max(5, _int_env("META_EVAL_SAMPLE_SIZE", 20))
TRANSCRIPT_FETCH_CAP: int = max(80, _int_env("META_EVAL_TRANSCRIPT_FETCH_CAP", 240))
ALL_AGENTS: list[str] = [
    "AssessmentAgent",
    "ResolutionAgent",
    "FinalNoticeAgent",
]

_AGENT_AUDIT_METRIC: dict[str, str] = {
    "AssessmentAgent": "assessment_completeness_identity",
    "ResolutionAgent": "resolution_commitment",
    "FinalNoticeAgent": "final_notice_clarity_deadline",
}

_AGENT_METRIC_SPEC: dict[str, str] = {
    "AssessmentAgent": (
        "assessment_completeness_identity:\n"
        "- Require identity verification as hard gate (last-4 + birth year or equivalent verified signal).\n"
        "- Require completeness over required fields (income, expense, employment, hardship/cash-flow markers).\n"
        "- Do not mark complete on partial fact collection."
    ),
    "ResolutionAgent": (
        "resolution_commitment:\n"
        "- resolution = 1 only with explicit borrower commitment + actionable timeline.\n"
        "- Reject vague intent or passive acknowledgment.\n"
        "- Do not infer commitment from agent-only statements."
    ),
    "FinalNoticeAgent": (
        "final_notice_clarity_deadline:\n"
        "- Require explicit consequence clarity (credit/legal/asset impact language).\n"
        "- Require concrete deadline (hard date or explicit expiry window).\n"
        "- Do not mark successful final notice if deadline/consequence is missing."
    ),
}

_COUNTER_KEY = "meta_eval:call_count"
_COUNTER_TTL = 7 * 24 * 3600
_RUNNING_LOCK_KEY = "meta_eval:running"
_RUNNING_LOCK_TTL = 30 * 60

_CURRENT_JUDGE_PROMPT = evaluator_judge_prompt()
_SHADOW_RESOLUTION_PROMPT = """
You are an independent audit judge checking whether an L2 evaluator's RESOLUTION label is reliable.

You are reviewing a debt-collection conversation transcript for one thing only:
whether the borrower actually reached a real resolution.

---

STRICT DECISION RULES (follow in order):

1. resolution = 1 ONLY if there is EXPLICIT, ACTIONABLE COMMITMENT from the borrower.
   This requires at least one of the following:
   - A clear payment statement with time: "I will pay tomorrow", "I'll clear it Friday"
   - Acceptance of a concrete plan: "Yes, set up the payment plan"
   - Confirmation of action taken: "I just paid", "Done", "Transaction completed"

2. resolution = 0 if ANY of the following occur:
   - Vague intent: "I'll try", "maybe", "I'll see"
   - Requests for time without commitment: "give me time", "later", "next week maybe"
   - Information seeking: "how do I pay?", "send link"
   - Passive agreement: "okay", "fine", "hmm"
   - Emotional or argumentative responses
   - Agent proposes plan but borrower does not explicitly accept

3. DO NOT infer commitment from tone, politeness, or agent statements.

4. If unsure, default to resolution = 0 (favor precision over recall).

---

EDGE CASE HANDLING:
- "I will try to pay tomorrow" → 0
- "Send me the link, I'll pay" → 0 (intent but no confirmed action)
- "Yes, I will pay tomorrow morning" → 1
- "Okay set it up" → 1 ONLY if clearly referring to a payment plan

---

Return JSON only with this exact schema:
{
  "resolution": 0 or 1,
  "resolution_confidence": 0.0-1.0,
  "debt_collected": 0 or 1,
  "compliance_violation": 0 or 1,
  "compliance_reason": "string",
  "tone_score": 1-5,
  "next_step_clarity": 1-5
}
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prompt_sha(prompt: str) -> str:
    return hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]


def _build_transcript_excerpt(history: list[dict], max_turns: int = 4) -> str:
    turns = history[-max_turns:]
    parts = []
    for turn in turns:
        speaker = "Borrower" if turn.get("role") == "user" else "Agent"
        parts.append(f"{speaker}: {str(turn.get('content', '')).strip()[:180]}")
    return " | ".join(parts)


def _conversation_text(history: list[dict]) -> str:
    return "\n".join(
        f"{'Borrower' if msg.get('role') == 'user' else 'Agent'}: {msg.get('content', '')}"
        for msg in history[-8:]
    )


def _parse_json(text: str) -> dict[str, Any]:
    clean = str(text or "").strip()
    if clean.startswith("```"):
        clean = clean.split("```", 2)[1]
        if clean.startswith("json"):
            clean = clean[4:]
    clean = clean.strip("` \n")
    return json.loads(clean)


def _extract_resolution(parsed: dict[str, Any]) -> int:
    try:
        return 1 if int(parsed.get("resolution", 0)) else 0
    except Exception:
        return 0


def _cohen_kappa(labels_a: list[int], labels_b: list[int]) -> tuple[float, float, float]:
    n = min(len(labels_a), len(labels_b))
    if n == 0:
        return 0.0, 0.0, 0.0

    a = labels_a[:n]
    b = labels_b[:n]
    observed = sum(1 for x, y in zip(a, b) if x == y) / n

    p_a_1 = sum(a) / n
    p_b_1 = sum(b) / n
    p_a_0 = 1.0 - p_a_1
    p_b_0 = 1.0 - p_b_1
    expected = (p_a_1 * p_b_1) + (p_a_0 * p_b_0)

    if abs(1.0 - expected) < 1e-9:
        kappa = 1.0 if abs(observed - 1.0) < 1e-9 else 0.0
    else:
        kappa = (observed - expected) / (1.0 - expected)

    return round(observed, 4), round(expected, 4), round(kappa, 4)


def _verdict_for_kappa(kappa: float) -> tuple[str, str, str]:
    if kappa < 0.40:
        return (
            "poor",
            "rubric is unreliable; prompt update recommended",
            "Tighten the L2 resolution rubric and add sharper acceptance anchors before using this signal for prompt changes.",
        )
    if kappa < 0.60:
        return (
            "moderate",
            "review ambiguous cases",
            "Review disagreement cases and re-anchor the resolution definition with positive and negative examples.",
        )
    return (
        "acceptable",
        "agreement acceptable",
        "Keep the current L2 resolution rubric; no change required unless future runs drift downward.",
    )


def _objective_recommendation(
    kappa: float,
    cases: list["MetaAuditCase"],
    default_recommendation: str,
) -> str:
    """
    Build an objective recommendation anchored to measured disagreement patterns
    between L2 labels and the L3 shadow prompt for the same agent.
    """
    n = len(cases)
    if n <= 0:
        return default_recommendation

    l2_pos = sum(1 for c in cases if c.primary_resolution == 1)
    l3_pos = sum(1 for c in cases if c.shadow_resolution == 1)
    fp = sum(1 for c in cases if c.primary_resolution == 1 and c.shadow_resolution == 0)
    fn = sum(1 for c in cases if c.primary_resolution == 0 and c.shadow_resolution == 1)
    disagreements = fp + fn
    disagreement_rate = disagreements / n

    if kappa < 0.40:
        direction = "over-calling resolution" if fp >= fn else "under-calling resolution"
        dominant = "false positives" if fp >= fn else "false negatives"
        return (
            f"L2 vs L3 agreement is weak (kappa={kappa:.3f}, disagreement={disagreements}/{n}, "
            f"L2 positives={l2_pos}/{n}, L3 positives={l3_pos}/{n}). "
            f"Disagreement is driven by {dominant} ({max(fp, fn)}/{disagreements}); "
            f"tighten the L2 resolution rubric for {direction} and re-audit after collecting >=20 same-agent samples."
        )
    if kappa < 0.60:
        dominant = "false positives" if fp >= fn else "false negatives"
        return (
            f"L2 vs L3 agreement is moderate (kappa={kappa:.3f}, disagreement={disagreements}/{n}, "
            f"L2 positives={l2_pos}/{n}, L3 positives={l3_pos}/{n}). "
            f"Prioritize the dominant mismatch type ({dominant}: {max(fp, fn)}/{disagreements}) "
            f"with explicit accept/reject anchors, then rerun the audit."
        )
    return (
        f"L2 vs L3 agreement is acceptable (kappa={kappa:.3f}, disagreement={disagreements}/{n}, "
        f"L2 positives={l2_pos}/{n}, L3 positives={l3_pos}/{n}). "
        f"Keep the current rubric and monitor for drift if disagreement rate exceeds {disagreement_rate:.1%}."
    )


def _extract_metric_spec_from_prompt(prompt: str, metric: str = "resolution") -> str:
    metric = metric.strip().lower()
    lines = prompt.splitlines()
    start = None
    for i, raw in enumerate(lines):
        line = raw.strip().lower()
        if line.startswith(f"{metric} ") or line.startswith(f"{metric}(") or line.startswith(f"{metric}:"):
            start = i
            break
    if start is None:
        return ""
    block: list[str] = []
    for raw in lines[start:start + 12]:
        if block and raw.strip().endswith(":") and not raw.strip().lower().startswith(metric):
            break
        block.append(raw.rstrip())
    return "\n".join(block).strip()


def _build_metric_tuning_proposals(
    kappa: float,
    cases: list["MetaAuditCase"],
    agent_name: str,
    metric: str = "resolution",
    weakest_metric: str = "",
) -> list[MetricTuningProposal]:
    n = len(cases)
    if n <= 0:
        return []

    fp = sum(1 for c in cases if c.primary_resolution == 1 and c.shadow_resolution == 0)
    fn = sum(1 for c in cases if c.primary_resolution == 0 and c.shadow_resolution == 1)
    disagreements = fp + fn
    if disagreements <= 0:
        return []

    disagreement_ratio = disagreements / n
    base_priority = int(round(min(100.0, max(0.0, (1.0 - max(kappa, -1.0)) * 55.0 + disagreement_ratio * 45.0))))
    proposals: list[MetricTuningProposal] = []

    target_metric = weakest_metric or metric

    if agent_name == "AssessmentAgent":
        if fp >= fn:
            proposals.append(
                MetricTuningProposal(
                    metric=target_metric,
                    action="tighten",
                    proposal="Require identity hard-gate + minimum required-field completeness before positive label.",
                    rationale=f"False positives dominate ({fp}/{disagreements}); L2 is over-accepting incomplete assessments.",
                    evidence_count=fp,
                    priority_score=max(0, min(100, base_priority)),
                )
            )
            proposals.append(
                MetricTuningProposal(
                    metric=target_metric,
                    action="remove",
                    proposal="Remove weak anchors that treat one-field confirmation as assessment complete.",
                    rationale="Single-field confirmations are inflating positive labels.",
                    evidence_count=fp,
                    priority_score=max(0, min(100, base_priority - 8)),
                )
            )
        else:
            proposals.append(
                MetricTuningProposal(
                    metric=target_metric,
                    action="add",
                    proposal="Add positive anchors for alternative but complete fact bundles that satisfy all required slots.",
                    rationale=f"False negatives dominate ({fn}/{disagreements}); L2 is missing valid completions.",
                    evidence_count=fn,
                    priority_score=max(0, min(100, base_priority)),
                )
            )
            proposals.append(
                MetricTuningProposal(
                    metric=target_metric,
                    action="tighten",
                    proposal="Clarify equivalence classes for required facts (e.g., salary/income synonyms).",
                    rationale="Semantic variants are being under-counted as complete.",
                    evidence_count=fn,
                    priority_score=max(0, min(100, base_priority - 8)),
                )
            )
        return proposals

    if agent_name == "FinalNoticeAgent":
        if fp >= fn:
            proposals.append(
                MetricTuningProposal(
                    metric=target_metric,
                    action="tighten",
                    proposal="Require both consequence clarity and explicit deadline before positive final-notice label.",
                    rationale=f"False positives dominate ({fp}/{disagreements}); L2 is over-calling adequate final notices.",
                    evidence_count=fp,
                    priority_score=max(0, min(100, base_priority)),
                )
            )
            proposals.append(
                MetricTuningProposal(
                    metric=target_metric,
                    action="remove",
                    proposal="Remove anchors that treat generic warning language as complete final notice.",
                    rationale="Generic warnings without deadline/consequence are driving mismatches.",
                    evidence_count=fp,
                    priority_score=max(0, min(100, base_priority - 8)),
                )
            )
        else:
            proposals.append(
                MetricTuningProposal(
                    metric=target_metric,
                    action="add",
                    proposal="Add positive anchors for clearly time-bounded notices that imply consequence escalation.",
                    rationale=f"False negatives dominate ({fn}/{disagreements}); L2 is missing valid final notices.",
                    evidence_count=fn,
                    priority_score=max(0, min(100, base_priority)),
                )
            )
            proposals.append(
                MetricTuningProposal(
                    metric=target_metric,
                    action="tighten",
                    proposal="Define accepted deadline expressions (date, hour window, explicit expiry phrasing).",
                    rationale="Deadline phrasing variance is causing inconsistent labels.",
                    evidence_count=fn,
                    priority_score=max(0, min(100, base_priority - 8)),
                )
            )
        return proposals

    # Default: ResolutionAgent
    if fp >= fn:
        proposals.append(
            MetricTuningProposal(
                metric=target_metric,
                action="tighten",
                proposal=(
                    "Require explicit borrower commitment with action + timeline "
                    "before setting resolution=1."
                ),
                rationale=(
                    f"False positives dominate ({fp}/{disagreements}) between L2 and L3 labels."
                ),
                evidence_count=fp,
                priority_score=max(0, min(100, base_priority)),
            )
        )
        proposals.append(
            MetricTuningProposal(
                metric=target_metric,
                action="remove",
                proposal=(
                    "Remove weak acceptance anchors that treat generic intent "
                    "phrases as resolved outcomes."
                ),
                rationale="Ambiguous language is inflating L2 positives.",
                evidence_count=fp,
                priority_score=max(0, min(100, base_priority - 8)),
            )
        )
    else:
        proposals.append(
            MetricTuningProposal(
                metric=target_metric,
                action="add",
                proposal=(
                    "Add a positive anchor for clear acceptance + actionable next step "
                    "even when canonical keywords are absent."
                ),
                rationale=(
                    f"False negatives dominate ({fn}/{disagreements}) between L2 and L3 labels."
                ),
                evidence_count=fn,
                priority_score=max(0, min(100, base_priority)),
            )
        )
        proposals.append(
            MetricTuningProposal(
                metric=target_metric,
                action="tighten",
                proposal=(
                    "Clarify paraphrase handling for commitment detection so semantic "
                    "acceptance maps to resolution=1."
                ),
                rationale="Current rubric misses non-template commitment language.",
                evidence_count=fn,
                priority_score=max(0, min(100, base_priority - 8)),
            )
        )
    return proposals


def _agent_metric_snapshot(agent_name: str, transcripts: list[dict], cases: list["MetaAuditCase"]) -> tuple[dict[str, float], str]:
    if not transcripts:
        return {}, ""

    if agent_name == "ResolutionAgent":
        n = max(1, len(transcripts))
        commitment = sum(1 for c in cases if c.primary_resolution == 1) / max(1, len(cases))
        anch_hits = 0
        obj_hits = 0
        cont_hits = 0
        for t in transcripts:
            hist = t.get("history") or []
            agent_text = " ".join(
                str(m.get("content", "")).lower()
                for m in hist
                if m.get("role") == "assistant" and str(m.get("stage", "")).lower() == "resolution"
            )
            if any(k in agent_text for k in ("upfront", "monthly", "installment", "deadline", "amount")):
                anch_hits += 1
            if any(k in agent_text for k in ("terms", "deadline", "confirm", "commit")):
                obj_hits += 1
            if any(k in agent_text for k in ("income", "employment", "cash flow", "assessment", "verified")):
                cont_hits += 1
        snap = {
            "commitment": round(commitment, 3),
            "anchoring": round(anch_hits / n, 3),
            "objection": round(obj_hits / n, 3),
            "continuity": round(cont_hits / n, 3),
        }
    elif agent_name == "AssessmentAgent":
        n = max(1, len(transcripts))
        completeness_sum = 0.0
        id_sum = 0.0
        turns_sum = 0.0
        tone_sum = 0.0
        for t in transcripts:
            hist = t.get("history") or []
            stage_msgs = [m for m in hist if str(m.get("stage", "")).lower() == "assessment"]
            user_text = " ".join(str(m.get("content", "")).lower() for m in stage_msgs if m.get("role") == "user")
            agent_text = " ".join(str(m.get("content", "")).lower() for m in stage_msgs if m.get("role") == "assistant")
            fields = [
                any(k in (user_text + " " + agent_text) for k in ("last 4", "last four", "account number")),
                any(k in (user_text + " " + agent_text) for k in ("birth year", "dob", "year of birth")),
                any(k in (user_text + " " + agent_text) for k in ("income", "monthly income")),
                any(k in (user_text + " " + agent_text) for k in ("expense", "expenses")),
                any(k in (user_text + " " + agent_text) for k in ("employment", "self employed", "employed")),
                any(k in (user_text + " " + agent_text) for k in ("cash flow", "hardship", "financial stress")),
            ]
            completeness_sum += (sum(1 for f in fields if f) / 6.0)
            id_sum += 1.0 if re.search(r"\b\d{4}\b", user_text) and re.search(r"\b(19|20)\d{2}\b", user_text) else 0.0
            turns_sum += max(1.0, float(len(stage_msgs)))
            tone_sum += 0.0 if any(k in agent_text for k in _EMPATHY_TERMS) else 1.0
        snap = {
            "completeness": round(completeness_sum / n, 3),
            "identity_verified": round(id_sum / n, 3),
            "turns": round(turns_sum / n, 3),
            "tone": round(tone_sum / n, 3),
        }
    else:
        n = max(1, len(transcripts))
        clarity_hits = 0
        deadline_hits = 0
        reask_hits = 0
        resolution = sum(1 for c in cases if c.primary_resolution == 1) / max(1, len(cases))
        for t in transcripts:
            hist = t.get("history") or []
            agent_text = " ".join(
                str(m.get("content", "")).lower()
                for m in hist
                if m.get("role") == "assistant" and str(m.get("stage", "")).lower() == "final_notice"
            )
            if any(k in agent_text for k in ("credit", "legal", "court", "asset", "seizure")):
                clarity_hits += 1
            if any(k in agent_text for k in ("deadline", "expires", "valid until", "hours")):
                deadline_hits += 1
            if any(k in agent_text for k in ("last 4", "birth year", "monthly income", "employment status")):
                reask_hits += 1
        snap = {
            "clarity": round(clarity_hits / n, 3),
            "deadline": round(deadline_hits / n, 3),
            "re_ask_penalty": round(min(1.0, reask_hits / n), 3),
            "resolution": round(resolution, 3),
        }

    weakest_metric = ""
    if snap:
        if "re_ask_penalty" in snap:
            weakest_metric = min(snap.keys(), key=lambda k: (1.0 - snap[k]) if k != "re_ask_penalty" else snap[k])
        else:
            weakest_metric = min(snap.keys(), key=lambda k: snap[k])
    return snap, weakest_metric


def _get_client() -> AsyncOpenAI:
    api_key = _first_nonempty_env("OPENCODE_API_KEY", "OPENAI_API_KEY")
    base_url = _normalize_base_url(_first_nonempty_env("OPENCODE_BASE_URL", "OPENAI_BASE_URL"))
    if not api_key:
        raise RuntimeError("Missing LLM API key. Set OPENCODE_API_KEY or OPENAI_API_KEY.")
    kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return AsyncOpenAI(**kwargs)


class MetaAuditCase(BaseModel):
    trace_id: str
    borrower_id: str
    agent_name: str
    primary_resolution: int
    shadow_resolution: int
    agreement: bool
    primary_source: str = "cached_l2"
    primary_raw: str = ""
    shadow_raw: str = ""
    transcript_excerpt: str = ""


class MetricTuningProposal(BaseModel):
    metric: str
    action: str  # add | remove | tighten
    proposal: str
    rationale: str
    evidence_count: int = 0
    priority_score: int = 0  # 0-100


class MetaAuditSummary(BaseModel):
    agent_name: str
    audited_metric: str = "resolution"
    sample_size_requested: int = DEFAULT_SAMPLE_SIZE
    sample_size_used: int = 0
    current_prompt_sha: str = Field(default_factory=lambda: _prompt_sha(_CURRENT_JUDGE_PROMPT))
    shadow_prompt_sha: str = Field(default_factory=lambda: _prompt_sha(_SHADOW_RESOLUTION_PROMPT))
    current_judge_model: str = LLMJudge.JUDGE_MODEL
    shadow_judge_model: str = LLMJudge.JUDGE_MODEL
    observed_agreement: float = 0.0
    expected_agreement: float = 0.0
    cohen_kappa: float = 0.0
    disagreement_count: int = 0
    verdict: str = "needs_data"
    flag: str = "not enough data"
    recommendation: str = "Gather more scored transcripts before auditing L2 consistency."
    l2_metric_spec: str = ""
    tuning_proposals: list[MetricTuningProposal] = Field(default_factory=list)
    agent_metric_snapshot: dict[str, float] = Field(default_factory=dict)
    weakest_metric: str = ""
    cases: list[MetaAuditCase] = Field(default_factory=list)


class MetaEvaluationRun(BaseModel):
    run_id: str = Field(default_factory=lambda: f"meta-{uuid.uuid4().hex[:8]}")
    triggered_by: str = "admin"
    target_agents: list[str] = Field(default_factory=lambda: list(ALL_AGENTS))
    audit_metric: str = "resolution"
    sample_size: int = DEFAULT_SAMPLE_SIZE
    started_at: str = Field(default_factory=_now)
    completed_at: Optional[str] = None
    status: str = "running"
    audits: list[MetaAuditSummary] = Field(default_factory=list)
    overall_verdict: str = "needs_data"
    notes: list[str] = Field(default_factory=list)
    error: Optional[str] = None
    meta_eval_cost_usd: float = 0.0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


async def _persist_run(run: MetaEvaluationRun) -> None:
    try:
        from src.data_layer import get_mongo

        db = await get_mongo()
        if db is None:
            return
        doc = json.loads(run.model_dump_json())
        await db.meta_eval_runs.update_one(
            {"run_id": run.run_id},
            {"$set": doc},
            upsert=True,
        )
    except Exception as e:
        log.warning("[meta_eval] persist failed: %s", e)


async def _latest_run_summary() -> Optional[dict[str, Any]]:
    from src.data_layer import get_mongo

    db = await get_mongo()
    if db is None:
        return None
    doc = await db.meta_eval_runs.find_one({}, {"_id": 0}, sort=[("started_at", -1)])
    return doc


async def list_runs(agent_name: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
    try:
        from src.data_layer import get_mongo

        db = await get_mongo()
        if db is None:
            return []
        filt: dict[str, Any] = {"target_agents": agent_name} if agent_name else {}
        cursor = db.meta_eval_runs.find(filt, {"_id": 0}).sort("started_at", -1).limit(limit)
        return await cursor.to_list(length=limit)
    except Exception as e:
        log.warning("[meta_eval] list_runs failed: %s", e)
        return []


async def get_run(run_id: str) -> Optional[dict[str, Any]]:
    try:
        from src.data_layer import get_mongo

        db = await get_mongo()
        if db is None:
            return None
        doc = await db.meta_eval_runs.find_one({"run_id": run_id}, {"_id": 0})
        return doc
    except Exception as e:
        log.warning("[meta_eval] get_run failed: %s", e)
        return None


async def _load_cached_resolution(trace_id: str) -> tuple[Optional[int], str]:
    from src.data_layer import get_mongo

    db = await get_mongo()
    if db is None:
        return None, ""

    doc = await db.eval_results.find_one(
        {"interaction_id": trace_id},
        {"dimension_scores": 1, "feedback": 1, "_id": 0},
        sort=[("timestamp", -1)],
    )
    if not doc:
        return None, ""

    dims = doc.get("dimension_scores") or {}
    if "resolution" in dims:
        try:
            return (1 if int(dims.get("resolution", 0)) else 0), str(doc.get("feedback", ""))
        except Exception:
            pass

    feedback = str(doc.get("feedback", ""))
    try:
        parsed = _parse_json(feedback)
        return _extract_resolution(parsed), feedback
    except Exception:
        return None, feedback


async def _run_judge_prompt(
    agent_name: str,
    history: list[dict],
    prompt: str,
    cost_acc: dict[str, float],
) -> tuple[dict[str, Any], str]:
    client = _get_client()
    response = await client.chat.completions.create(
        model=LLMJudge.JUDGE_MODEL,
        max_tokens=512,
        temperature=0.0,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Agent: {agent_name}\n\n{_conversation_text(history)}"},
        ],
    )

    prompt_tokens = (response.usage.prompt_tokens if response.usage else 0) or 0
    completion_tokens = (response.usage.completion_tokens if response.usage else 0) or 0
    cost_acc["cost_usd"] += cost_for_model(LLMJudge.JUDGE_MODEL, prompt_tokens, completion_tokens)
    cost_acc["calls"] += 1
    cost_acc["input_tokens"] += prompt_tokens
    cost_acc["output_tokens"] += completion_tokens

    text = (response.choices[0].message.content or "").strip()
    return _parse_json(text), text


async def _resolve_primary_label(
    transcript: dict[str, Any],
    cost_acc: dict[str, float],
) -> tuple[int, str, str]:
    cached_resolution, cached_raw = await _load_cached_resolution(transcript["trace_id"])
    if cached_resolution is not None:
        return cached_resolution, "cached_l2", cached_raw

    parsed, raw = await _run_judge_prompt(
        transcript["primary_agent"],
        transcript.get("history", []),
        _CURRENT_JUDGE_PROMPT,
        cost_acc,
    )
    return _extract_resolution(parsed), "live_l2_backfill", raw


async def _audit_one_transcript(
    transcript: dict[str, Any],
    cost_acc: dict[str, float],
) -> MetaAuditCase:
    primary_resolution, primary_source, primary_raw = await _resolve_primary_label(transcript, cost_acc)
    shadow_parsed, shadow_raw = await _run_judge_prompt(
        transcript["primary_agent"],
        transcript.get("history", []),
        _SHADOW_RESOLUTION_PROMPT,
        cost_acc,
    )
    shadow_resolution = _extract_resolution(shadow_parsed)
    return MetaAuditCase(
        trace_id=transcript["trace_id"],
        borrower_id=transcript["borrower_id"],
        agent_name=transcript["primary_agent"],
        primary_resolution=primary_resolution,
        shadow_resolution=shadow_resolution,
        agreement=primary_resolution == shadow_resolution,
        primary_source=primary_source,
        primary_raw=primary_raw,
        shadow_raw=shadow_raw,
        transcript_excerpt=_build_transcript_excerpt(transcript.get("history", [])),
    )


async def _fetch_agent_transcripts(agent_name: str, sample_size: int) -> list[dict[str, Any]]:
    from src.admin_api import _fetch_last_n_transcripts

    fetch_size = min(max(sample_size * 6, 60), TRANSCRIPT_FETCH_CAP)
    filtered: list[dict[str, Any]] = []

    while True:
        transcripts = await _fetch_last_n_transcripts(fetch_size)
        filtered = [t for t in transcripts if t.get("primary_agent") == agent_name][:sample_size]
        if len(filtered) >= sample_size or fetch_size >= TRANSCRIPT_FETCH_CAP:
            return filtered
        fetch_size = min(fetch_size * 2, TRANSCRIPT_FETCH_CAP)


async def _run_agent_audit(
    agent_name: str,
    sample_size: int,
    cost_acc: dict[str, float],
) -> MetaAuditSummary:
    transcripts = await _fetch_agent_transcripts(agent_name, sample_size)
    audited_metric = _AGENT_AUDIT_METRIC.get(agent_name, "resolution")
    summary = MetaAuditSummary(
        agent_name=agent_name,
        audited_metric=audited_metric,
        sample_size_requested=sample_size,
        sample_size_used=len(transcripts),
    )
    if not transcripts:
        return summary

    cases = await asyncio.gather(*[
        _audit_one_transcript(transcript, cost_acc)
        for transcript in transcripts
    ])

    primary_labels = [case.primary_resolution for case in cases]
    shadow_labels = [case.shadow_resolution for case in cases]
    observed, expected, kappa = _cohen_kappa(primary_labels, shadow_labels)
    verdict, flag, recommendation = _verdict_for_kappa(kappa)
    disagreements = [case for case in cases if not case.agreement]
    recommendation = _objective_recommendation(kappa, cases, recommendation)

    summary.sample_size_used = len(cases)
    summary.observed_agreement = observed
    summary.expected_agreement = expected
    summary.cohen_kappa = kappa
    summary.disagreement_count = len(disagreements)
    summary.verdict = verdict
    summary.flag = flag
    metric_snapshot, weakest_metric = _agent_metric_snapshot(agent_name, transcripts, cases)
    summary.agent_metric_snapshot = metric_snapshot
    summary.weakest_metric = weakest_metric
    proposals = _build_metric_tuning_proposals(
        kappa=kappa,
        cases=cases,
        agent_name=agent_name,
        metric=summary.audited_metric,
        weakest_metric=weakest_metric,
    )
    summary.l2_metric_spec = _extract_metric_spec_from_prompt(_CURRENT_JUDGE_PROMPT, "resolution") or _AGENT_METRIC_SPEC.get(agent_name, "")
    summary.tuning_proposals = proposals
    if proposals:
        top = proposals[0]
        recommendation = (
            f"{recommendation} Top metric action: {top.action} `{top.metric}` "
            f"(priority={top.priority_score}/100) — {top.proposal}"
        )
    if metric_snapshot:
        recommendation = (
            f"Metric snapshot: {json.dumps(metric_snapshot, ensure_ascii=True)}. "
            f"Weakest metric: {weakest_metric or 'n/a'}. "
            f"{recommendation}"
        )
    summary.recommendation = recommendation
    summary.cases = cases
    return summary


def _overall_verdict(audits: list[MetaAuditSummary]) -> str:
    verdicts = [audit.verdict for audit in audits]
    if not verdicts:
        return "needs_data"
    if any(v == "poor" for v in verdicts):
        return "poor"
    if any(v == "moderate" for v in verdicts):
        return "moderate"
    if all(v == "acceptable" for v in verdicts):
        return "acceptable"
    return "needs_data"


async def run_meta_evaluation_pipeline(
    run_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    triggered_by: str = "admin",
) -> MetaEvaluationRun:
    targets = [agent_name] if agent_name else list(ALL_AGENTS)
    run = MetaEvaluationRun(
        run_id=run_id or f"meta-{uuid.uuid4().hex[:8]}",
        triggered_by=triggered_by,
        target_agents=targets,
        sample_size=max(5, sample_size),
    )
    await _persist_run(run)

    cost_acc = {"cost_usd": 0.0, "calls": 0, "input_tokens": 0, "output_tokens": 0}
    try:
        run.notes.append("L3 audit mechanism: scoring consistency on L2 resolution labels.")
        for target in targets:
            audit = await _run_agent_audit(target, run.sample_size, cost_acc)
            run.audits.append(audit)
            await _persist_run(run)

        run.overall_verdict = _overall_verdict(run.audits)
        if run.overall_verdict == "poor":
            run.notes.append("At least one agent fell below kappa 0.40; L2 resolution rubric needs tightening.")
        elif run.overall_verdict == "moderate":
            run.notes.append("At least one agent fell between kappa 0.40 and 0.60; review disagreement cases.")
        elif run.overall_verdict == "acceptable":
            run.notes.append("All audited agents met kappa >= 0.60 on the shadow resolution prompt.")

        run.status = "completed"
        run.completed_at = _now()
        run.meta_eval_cost_usd = round(cost_acc["cost_usd"], 6)
        run.llm_calls = int(cost_acc["calls"])
        run.input_tokens = int(cost_acc["input_tokens"])
        run.output_tokens = int(cost_acc["output_tokens"])
        await _persist_run(run)
        return run
    except Exception as e:
        log.exception("[meta_eval] run %s failed: %s", run.run_id, e)
        run.status = "failed"
        run.error = str(e)
        run.completed_at = _now()
        run.meta_eval_cost_usd = round(cost_acc["cost_usd"], 6)
        run.llm_calls = int(cost_acc["calls"])
        run.input_tokens = int(cost_acc["input_tokens"])
        run.output_tokens = int(cost_acc["output_tokens"])
        await _persist_run(run)
        return run


async def _acquire_running_lock() -> bool:
    from src.data_layer import get_redis

    r = await get_redis()
    if r is None:
        return True
    try:
        return bool(await r.set(_RUNNING_LOCK_KEY, "1", nx=True, ex=_RUNNING_LOCK_TTL))
    except Exception:
        return True


async def _release_running_lock() -> None:
    from src.data_layer import get_redis

    r = await get_redis()
    if r is None:
        return
    try:
        await r.delete(_RUNNING_LOCK_KEY)
    except Exception:
        pass


async def _bg_run(run_id: str, agent_name: Optional[str], sample_size: int, triggered_by: str) -> None:
    lock_ok = await _acquire_running_lock()
    if not lock_ok:
        log.info("[meta_eval] skipping background start; another run is active")
        skipped = MetaEvaluationRun(
            run_id=run_id,
            triggered_by=triggered_by,
            target_agents=[agent_name] if agent_name else list(ALL_AGENTS),
            sample_size=max(5, sample_size),
            status="skipped",
            completed_at=_now(),
            notes=["Skipped because another meta-evaluator run was already active."],
        )
        await _persist_run(skipped)
        return
    try:
        await run_meta_evaluation_pipeline(
            run_id=run_id,
            agent_name=agent_name,
            sample_size=sample_size,
            triggered_by=triggered_by,
        )
    finally:
        await _release_running_lock()


async def trigger_now(
    agent_name: Optional[str] = None,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    triggered_by: str = "manual",
) -> dict[str, Any]:
    run_id = f"meta-{uuid.uuid4().hex[:8]}"
    asyncio.create_task(_bg_run(run_id, agent_name, max(5, sample_size), triggered_by))
    return {
        "run_id": run_id,
        "target_agents": [agent_name] if agent_name else list(ALL_AGENTS),
        "sample_size": max(5, sample_size),
        "triggered_by": triggered_by,
        "status": "started",
    }


async def on_conversation_complete() -> None:
    from src.data_layer import get_redis

    r = await get_redis()
    if r is None:
        log.warning("[meta_eval] Redis unavailable — skipping auto-trigger check")
        return

    try:
        count = await r.incr(_COUNTER_KEY)
        await r.expire(_COUNTER_KEY, _COUNTER_TTL)
        log.info(
            "[meta_eval] call_count=%d trigger_at=%d",
            count,
            AUTO_TRIGGER_EVERY_N_ITERATIONS,
        )
        if count < AUTO_TRIGGER_EVERY_N_ITERATIONS:
            return

        overflow = max(0, count - AUTO_TRIGGER_EVERY_N_ITERATIONS)
        if overflow:
            await r.setex(_COUNTER_KEY, _COUNTER_TTL, str(overflow))
        else:
            await r.delete(_COUNTER_KEY)

        log.info("[meta_eval] threshold reached — auto-triggering scoring consistency audit")
        asyncio.create_task(_bg_run(
            run_id=f"meta-{uuid.uuid4().hex[:8]}",
            agent_name=None,
            sample_size=DEFAULT_SAMPLE_SIZE,
            triggered_by="auto_300",
        ))
    except Exception as e:
        log.warning("[meta_eval] on_conversation_complete error: %s", e)


async def get_counter() -> int:
    from src.data_layer import get_redis

    r = await get_redis()
    if r is None:
        return -1
    try:
        raw = await r.get(_COUNTER_KEY)
        return int(raw) if raw else 0
    except Exception:
        return -1


async def get_status() -> dict[str, Any]:
    count = await get_counter()
    last_run = await _latest_run_summary()
    return {
        "call_count": count,
        "trigger_every_n": AUTO_TRIGGER_EVERY_N_ITERATIONS,
        "sample_size": DEFAULT_SAMPLE_SIZE,
        "calls_until_next": (
            max(0, AUTO_TRIGGER_EVERY_N_ITERATIONS - count)
            if count >= 0 else -1
        ),
        "last_run": last_run,
    }
