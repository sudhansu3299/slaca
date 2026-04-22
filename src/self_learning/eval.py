"""
Self-learning evaluation loop.

Each agent has its own SelfLearningLoop that:
1. Scores an interaction against criteria
2. Records the result
3. After enough data, evolves thresholds (makes criteria harder or softer)
4. Surfaces the top N learnings as injected guidance for the next prompt

Design:
- Criteria are per-agent and version-stamped
- Thresholds evolve DOWN (easier) when a metric fails >5/10 times in a row
  (the agent can't meet the bar → lower it) and UP (harder) when passing >9/10
- Learnings are extracted as (trigger_phrase → outcome) pairs
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field
from src.models import Stage


# ------------------------------------------------------------------ #
# Data models
# ------------------------------------------------------------------ #

class EvaluationCriteria(BaseModel):
    metric: str
    threshold: float
    direction: str = "higher"   # "higher" = bigger is better; "lower" = smaller is better
    weight: float = 1.0


class EvaluationResult(BaseModel):
    criteria: list[EvaluationCriteria]
    scores: dict[str, float] = {}
    passed: bool = False
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = {}


class Learning(BaseModel):
    pattern: str
    outcome: str     # "success" | "failure"
    agent: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class EvaluationMethodology(BaseModel):
    agent_name: str
    version: str = "1.0.0"
    criteria: list[EvaluationCriteria] = Field(default_factory=list)
    historical_results: list[EvaluationResult] = Field(default_factory=list)
    last_updated: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_result(self, result: EvaluationResult) -> None:
        self.historical_results.append(result)
        self.last_updated = datetime.now(timezone.utc).isoformat()
        # Evolve every 10 results
        if len(self.historical_results) % 10 == 0:
            self._evolve()

    def _evolve(self) -> None:
        recent = self.historical_results[-10:]
        changed = False

        for criteria in self.criteria:
            passes = sum(
                1 for r in recent
                if self._passes(r.scores.get(criteria.metric, 0), criteria)
            )

            if passes <= 2:
                # Failing too often → relax threshold by 10%
                if criteria.direction == "higher":
                    criteria.threshold = round(criteria.threshold * 0.90, 4)
                else:
                    criteria.threshold = round(criteria.threshold * 1.10, 4)
                changed = True

            elif passes >= 9:
                # Passing almost always → tighten threshold by 5%
                if criteria.direction == "higher":
                    criteria.threshold = round(min(criteria.threshold * 1.05, 1.0), 4)
                else:
                    criteria.threshold = round(criteria.threshold * 0.95, 4)
                changed = True

        if changed:
            self.version = self._bump_version()

    def _passes(self, score: float, criteria: EvaluationCriteria) -> bool:
        if criteria.direction == "higher":
            return score >= criteria.threshold
        return score <= criteria.threshold

    def _bump_version(self) -> str:
        parts = self.version.split(".")
        parts[-1] = str(int(parts[-1]) + 1)
        return ".".join(parts)


# ------------------------------------------------------------------ #
# Self-learning loop
# ------------------------------------------------------------------ #

class SelfLearningLoop:
    """
    Per-agent self-learning loop.

    Usage:
        loop = SelfLearningLoop("AssessmentAgent")
        result = await loop.evaluate_interaction(history, outcome="success")
        injected_guidance = loop.get_injected_guidance(top_n=3)
    """

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.methodology = EvaluationMethodology(agent_name=agent_name)
        self.learnings: list[Learning] = []
        self._init_criteria()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    async def evaluate_interaction(
        self,
        conversation_history: list[dict],
        outcome: Optional[str] = None,
    ) -> EvaluationResult:
        scores: dict[str, float] = {}
        for criteria in self.methodology.criteria:
            scores[criteria.metric] = self._compute(criteria.metric, conversation_history, outcome)

        passed = all(
            self.methodology._passes(scores.get(c.metric, 0), c)
            for c in self.methodology.criteria
        )

        result = EvaluationResult(
            criteria=list(self.methodology.criteria),
            scores=scores,
            passed=passed,
        )
        self.methodology.add_result(result)

        if outcome:
            self._record_learnings(conversation_history, outcome)

        return result

    def get_injected_guidance(self, top_n: int = 3) -> str:
        """Return top-N successful patterns as a compact prompt injection."""
        successful = [l for l in self.learnings if l.outcome == "success"][-top_n:]
        if not successful:
            return ""
        lines = [f"[Learned patterns — use these when relevant:]"]
        for l in successful:
            lines.append(f"  • {l.pattern}")
        return "\n".join(lines)

    def methodology_summary(self) -> str:
        lines = [
            f"Eval Methodology v{self.methodology.version} ({self.agent_name})",
            f"Interactions evaluated: {len(self.methodology.historical_results)}",
        ]
        for c in self.methodology.criteria:
            lines.append(f"  {c.metric}: threshold={c.threshold} ({c.direction})")
        return "\n".join(lines)

    # ------------------------------------------------------------------ #
    # Criteria initialisation
    # ------------------------------------------------------------------ #

    def _init_criteria(self) -> None:
        if self.agent_name == "AssessmentAgent":
            self.methodology.criteria = [
                EvaluationCriteria(metric="identity_verified", threshold=1.0, direction="higher"),
                EvaluationCriteria(metric="financial_completeness", threshold=0.80, direction="higher"),
                EvaluationCriteria(metric="no_repeat_questions", threshold=1.0, direction="higher"),
                EvaluationCriteria(metric="turns_to_complete", threshold=8, direction="lower"),
            ]
        elif self.agent_name == "ResolutionAgent":
            self.methodology.criteria = [
                EvaluationCriteria(metric="offer_presented", threshold=1.0, direction="higher"),
                EvaluationCriteria(metric="commitment_rate", threshold=0.70, direction="higher"),
                EvaluationCriteria(metric="turns_to_outcome", threshold=5, direction="lower"),
            ]
        elif self.agent_name == "FinalNoticeAgent":
            self.methodology.criteria = [
                EvaluationCriteria(metric="consequences_stated", threshold=0.80, direction="higher"),
                EvaluationCriteria(metric="deadline_stated", threshold=1.0, direction="higher"),
                EvaluationCriteria(metric="resolution_rate", threshold=0.75, direction="higher"),
            ]

    # ------------------------------------------------------------------ #
    # Metric computation
    # ------------------------------------------------------------------ #

    def _compute(
        self,
        metric: str,
        history: list[dict],
        outcome: Optional[str],
    ) -> float:
        if metric == "identity_verified":
            # Check if any user message contained a 4-digit number (proxy for ID)
            import re
            user_msgs = [m for m in history if m.get("role") == "user"]
            for m in user_msgs:
                if re.search(r"\b\d{4}\b", m.get("content", "")):
                    return 1.0
            return 0.0

        elif metric == "financial_completeness":
            keywords = ["income", "expense", "employed", "asset", "liabilit", "cash flow"]
            content = " ".join(m.get("content", "").lower() for m in history)
            found = sum(1 for kw in keywords if kw in content)
            return round(found / len(keywords), 3)

        elif metric == "no_repeat_questions":
            # Simple heuristic: count duplicate question phrases in agent messages
            agent_msgs = [m.get("content", "").lower() for m in history if m.get("role") == "assistant"]
            if len(agent_msgs) < 2:
                return 1.0
            # Check if same 5-word sequence appears twice
            from collections import Counter
            phrases = []
            for msg in agent_msgs:
                words = msg.split()
                phrases += [" ".join(words[i:i+5]) for i in range(len(words) - 4)]
            counts = Counter(phrases)
            duplicates = sum(1 for c in counts.values() if c > 1)
            return 1.0 if duplicates == 0 else max(0.0, 1.0 - duplicates * 0.2)

        elif metric == "turns_to_complete":
            return float(len([m for m in history if m.get("role") == "assistant"]))

        elif metric == "turns_to_outcome":
            return float(len([m for m in history if m.get("role") == "assistant"]))

        elif metric == "offer_presented":
            keywords = ["offer", "payment", "installment", "lump", "discount", "monthly"]
            agent_msgs = " ".join(
                m.get("content", "").lower() for m in history if m.get("role") == "assistant"
            )
            return 1.0 if any(kw in agent_msgs for kw in keywords) else 0.0

        elif metric == "commitment_rate":
            return 1.0 if outcome in ("committed", "resolved") else 0.0

        elif metric == "consequences_stated":
            consequence_keywords = ["credit", "legal", "court", "garnish", "lien", "asset"]
            content = " ".join(m.get("content", "").lower() for m in history)
            found = sum(1 for kw in consequence_keywords if kw in content)
            return round(found / len(consequence_keywords), 3)

        elif metric == "deadline_stated":
            content = " ".join(m.get("content", "").lower() for m in history)
            return 1.0 if any(kw in content for kw in ["expire", "deadline", "by ", "days"]) else 0.0

        elif metric == "resolution_rate":
            return 1.0 if outcome in ("resolved", "committed") else 0.0

        return 0.5   # unknown metric → neutral

    # ------------------------------------------------------------------ #
    # Learning extraction
    # ------------------------------------------------------------------ #

    def _record_learnings(self, history: list[dict], outcome: str) -> None:
        agent_msgs = [m for m in history if m.get("role") == "assistant"]
        user_msgs = [m for m in history if m.get("role") == "user"]

        # Pattern: last agent message before a successful outcome
        if outcome in ("success", "committed", "resolved") and agent_msgs:
            last = agent_msgs[-1].get("content", "")[:120]
            self.learnings.append(Learning(
                pattern=f"Before success: '{last}'",
                outcome="success",
                agent=self.agent_name,
            ))

        # Pattern: user phrasing that led to success
        if outcome in ("success", "committed", "resolved") and user_msgs:
            last_user = user_msgs[-1].get("content", "")[:80]
            self.learnings.append(Learning(
                pattern=f"User phrasing before success: '{last_user}'",
                outcome="success",
                agent=self.agent_name,
            ))

        # Pattern: agent message before failure
        if outcome in ("refused", "escalated", "failed") and agent_msgs:
            last = agent_msgs[-1].get("content", "")[:120]
            self.learnings.append(Learning(
                pattern=f"Before failure: '{last}'",
                outcome="failure",
                agent=self.agent_name,
            ))

        # Keep last 100 learnings only (memory bound)
        self.learnings = self.learnings[-100:]
