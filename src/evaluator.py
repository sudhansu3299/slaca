"""
Evaluation System (MS9)

Two evaluation methods:
1. Rule-based  — deterministic, zero LLM cost, fast
2. LLM-as-judge — uses a cheap/strong model for subjective assessment

Rule-based evaluates:
  - Repeated questions (QuestionTracker violations)
  - Tone compliance (forbidden phrases check)
  - Resolution push (did agent steer toward commitment?)

LLM-as-judge evaluates:
  - Overall quality of a full conversation
  - Tone naturalness
  - Logical flow

Both return a structured EvalReport.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional

from src.prompts import check_tone, evaluator_judge_prompt
from src.question_tracker import QuestionTracker, FactKey
from src.handoff import _deserialise_qt
from src.models import ConversationContext

def _first_nonempty_env(*names: str) -> str:
    import os
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return ""

def _normalize_base_url(url: str) -> str:
    if not url:
        return ""
    # Allow env values copied with inline comments, e.g. "host # note".
    url = url.split("#", 1)[0].strip()
    if not url:
        return ""
    if url.startswith(("http://", "https://")):
        return url
    return f"https://{url.lstrip('/')}"


# ──────────────────────────────────────────────────────────── #
# Evaluation result structures
# ──────────────────────────────────────────────────────────── #

@dataclass
class RuleCheckResult:
    check_name: str
    passed: bool
    score: float           # 0.0 – 1.0
    detail: str = ""


@dataclass
class EvalReport:
    agent_name: str
    overall_passed: bool
    overall_score: float   # 0.0 – 1.0
    checks: list[RuleCheckResult] = field(default_factory=list)
    llm_judgment: Optional[str] = None
    llm_score: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"=== Eval: {self.agent_name} ===",
            f"Overall: {'PASS' if self.overall_passed else 'FAIL'} ({self.overall_score:.2f})",
        ]
        for c in self.checks:
            status = "✓" if c.passed else "✗"
            lines.append(f"  {status} {c.check_name}: {c.score:.2f} — {c.detail}")
        if self.llm_judgment:
            lines.append(f"  LLM judge: {self.llm_judgment[:100]}")
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────── #
# Rule-based evaluator
# ──────────────────────────────────────────────────────────── #

class RuleBasedEvaluator:
    """
    Zero-cost deterministic evaluation.
    Runs checks in order; returns EvalReport.
    """

    PASS_THRESHOLD = 0.70   # overall score must exceed this to PASS

    def evaluate(
        self,
        agent_name: str,
        conversation_history: list[dict],
        context: Optional[ConversationContext] = None,
    ) -> EvalReport:
        checks = [
            self._check_no_repeat_questions(agent_name, conversation_history, context),
            self._check_tone_compliance(agent_name, conversation_history),
            self._check_resolution_push(agent_name, conversation_history),
            self._check_response_length(agent_name, conversation_history),
            self._check_outcome_marker(agent_name, conversation_history),
        ]

        total_weight = len(checks)
        overall_score = sum(c.score for c in checks) / total_weight if checks else 0.0
        overall_passed = overall_score >= self.PASS_THRESHOLD

        return EvalReport(
            agent_name=agent_name,
            overall_passed=overall_passed,
            overall_score=round(overall_score, 3),
            checks=checks,
        )

    # ──────────────────────────── Individual checks ─────────── #

    def _check_no_repeat_questions(
        self,
        agent_name: str,
        history: list[dict],
        context: Optional[ConversationContext],
    ) -> RuleCheckResult:
        if context is None:
            return RuleCheckResult("no_repeat_questions", True, 1.0, "no context provided")

        qt = _deserialise_qt(context)
        violations = []
        for msg in history:
            if msg.get("role") == "assistant" and msg.get("stage") == self._stage_for(agent_name):
                v = qt.validate_no_repeat(msg.get("content", ""))
                violations.extend(v)

        if not violations:
            return RuleCheckResult("no_repeat_questions", True, 1.0, "no repeats detected")

        score = max(0.0, 1.0 - len(set(violations)) * 0.3)
        return RuleCheckResult(
            "no_repeat_questions", False, score,   # any violation = fail
            f"repeated: {list(set(violations))}"
        )

    def _check_tone_compliance(
        self, agent_name: str, history: list[dict]
    ) -> RuleCheckResult:
        agent_msgs = [
            m.get("content", "")
            for m in history
            if m.get("role") == "assistant"
        ]
        if not agent_msgs:
            return RuleCheckResult("tone_compliance", True, 1.0, "no agent messages")

        all_violations: list[str] = []
        for msg in agent_msgs:
            all_violations.extend(check_tone(agent_name, msg))

        if not all_violations:
            return RuleCheckResult("tone_compliance", True, 1.0, "tone correct")

        score = max(0.0, 1.0 - len(all_violations) * 0.25)
        return RuleCheckResult(
            "tone_compliance", False, score,   # any violation = fail
            f"violations: {list(set(all_violations))[:3]}"
        )

    def _check_resolution_push(
        self, agent_name: str, history: list[dict]
    ) -> RuleCheckResult:
        """Agent should ask for commitment / state consequences / push resolution."""
        if agent_name == "AssessmentAgent":
            # Assessment just collects facts; push is not required
            return RuleCheckResult("resolution_push", True, 1.0, "N/A for assessment")

        push_keywords = {
            "ResolutionAgent": ["commit", "agree", "confirm", "deadline", "expires"],
            "FinalNoticeAgent": ["expire", "deadline", "legal", "credit", "court"],
        }
        keywords = push_keywords.get(agent_name, [])
        agent_msgs = " ".join(
            m.get("content", "").lower()
            for m in history if m.get("role") == "assistant"
        )
        hits = sum(1 for kw in keywords if kw in agent_msgs)
        score = min(1.0, hits / max(len(keywords) * 0.5, 1))
        return RuleCheckResult(
            "resolution_push", score >= 0.5, score,
            f"{hits}/{len(keywords)} push keywords found"
        )

    def _check_response_length(
        self, agent_name: str, history: list[dict]
    ) -> RuleCheckResult:
        """Voice agent responses must be short; chat agents can be longer."""
        agent_msgs = [
            m.get("content", "")
            for m in history if m.get("role") == "assistant"
        ]
        if not agent_msgs:
            return RuleCheckResult("response_length", True, 1.0, "no messages")

        avg_len = sum(len(m.split()) for m in agent_msgs) / len(agent_msgs)

        if agent_name == "ResolutionAgent":
            # Voice: should be ≤ 40 words per turn
            limit = 40
            passed = avg_len <= limit
            score = min(1.0, limit / max(avg_len, 1))
        else:
            # Chat: ≤ 120 words per turn
            limit = 120
            passed = avg_len <= limit
            score = min(1.0, limit / max(avg_len, 1))

        return RuleCheckResult(
            "response_length", passed, round(score, 2),
            f"avg {avg_len:.0f} words (limit {limit})"
        )

    def _check_outcome_marker(
        self, agent_name: str, history: list[dict]
    ) -> RuleCheckResult:
        """
        Did the agent produce a stage-completing turn?
        Markers like ASSESSMENT_COMPLETE are stripped from the stored `content`
        before being saved to history (to keep them out of the borrower-facing
        message), so we check the `advanced` flag that the pipeline/activities
        set on the history entry whenever `response.should_advance == True`.
        """
        agent_msgs = [m for m in history if m.get("role") == "assistant"]
        advanced = any(m.get("advanced", False) for m in agent_msgs)

        return RuleCheckResult(
            "outcome_marker",
            advanced,
            1.0 if advanced else 0.0,
            "stage completed" if advanced else "no stage completion detected",
        )

    def _stage_for(self, agent_name: str) -> str:
        return {
            "AssessmentAgent": "assessment",
            "ResolutionAgent": "resolution",
            "FinalNoticeAgent": "final_notice",
        }.get(agent_name, "")


# ──────────────────────────────────────────────────────────── #
# LLM-as-judge (stub — uses cheap model, async)
# ──────────────────────────────────────────────────────────── #

class LLMJudge:
    """
    LLM-as-judge for subjective conversation quality.
    Uses a cheaper mini model to keep cost low.
    Can be skipped in unit tests via SKIP_LLM_JUDGE env var.
    """

    JUDGE_MODEL = os.getenv("EVAL_MODEL", "gpt-4o-mini")  # cheap model for evaluation
    JUDGE_PROMPT = evaluator_judge_prompt()

    async def judge(
        self,
        agent_name: str,
        conversation_history: list[dict],
    ) -> tuple[float, str]:
        """Returns (score 0-1, reason string)."""
        import os
        if os.getenv("SKIP_LLM_JUDGE"):
            return 0.85, "skipped (SKIP_LLM_JUDGE set)"

        from openai import AsyncOpenAI
        api_key = _first_nonempty_env(
            "OPENCODE_API_KEY",
            "OPENAI_API_KEY",
        )
        base_url = _normalize_base_url(_first_nonempty_env(
            "OPENCODE_BASE_URL",
            "OPENAI_BASE_URL",
        ))
        # Some SDKs also read env vars directly; blank values can produce
        # malformed request URLs. Clear them to avoid accidental overrides.
        for env_name in ("OPENCODE_BASE_URL", "OPENAI_BASE_URL"):
            if not os.getenv(env_name, "").strip():
                os.environ.pop(env_name, None)
        if not api_key:
            raise RuntimeError(
                "Missing LLM API key. Set one of OPENCODE_API_KEY or OPENAI_API_KEY."
            )
        client_kwargs = {"api_key": api_key} if api_key else {}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = AsyncOpenAI(**client_kwargs)

        convo = "\n".join(
            f"{'Borrower' if m['role'] == 'user' else 'Agent'}: {m.get('content', '')}"
            for m in conversation_history[-8:]
        )

        response = await client.chat.completions.create(
            model=self.JUDGE_MODEL,
            max_tokens=512,
            temperature=0.0,
            messages=[
                {"role": "system", "content": self.JUDGE_PROMPT},
                {"role": "user", "content": f"Agent: {agent_name}\n\n{convo}"},
            ],
        )

        text = (response.choices[0].message.content or "").strip()

        # The judge prompt instructs JSON output — parse it first
        import json as _json
        score = 0.5
        reason = text
        try:
            # Strip markdown fences if present
            clean = text.strip("` \n")
            if clean.startswith("json"):
                clean = clean[4:].strip()
            parsed = _json.loads(clean)
            # Current judge schema:
            #   resolution            0|1
            #   resolution_confidence 0.0-1.0
            #   debt_collected        0|1
            #   compliance_violation  0|1  (0 = compliant = good)
            #   tone_score            1-5
            #   next_step_clarity     1-5
            #
            # Normalise all to 0-1, higher = better, then average.
            tone_norm        = (parsed.get("tone_score", 3) - 1) / 4          # 1-5 → 0-1
            clarity_norm     = (parsed.get("next_step_clarity", 3) - 1) / 4   # 1-5 → 0-1
            resolution_conf  = float(parsed.get("resolution_confidence", 0.5))
            compliance_norm  = 1.0 - int(parsed.get("compliance_violation", 0))  # 0=good→1.0
            debt_norm        = float(parsed.get("debt_collected", 0))

            score = (tone_norm + clarity_norm + resolution_conf + compliance_norm + debt_norm) / 5
            reason = text   # return full JSON string so admin_api.py can re-parse
        except (_json.JSONDecodeError, TypeError):
            pass

        return min(1.0, max(0.0, score)), reason
