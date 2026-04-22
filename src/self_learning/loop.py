"""
Self-Learning Loop (MS7)

Loop:
  1. Run simulation (via SimulationEngine in MS8)
  2. Evaluate outcomes against criteria
  3. Update agent prompts / strategies based on learnings
  4. Track metrics: recovery rate, commitment success, drop-off points, repetition errors

This module owns the "update prompts" step — the feedback from evaluation
is injected back into the next agent invocation via _injected_guidance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from src.self_learning.eval import SelfLearningLoop, EvaluationResult


# ──────────────────────────────────────────────────────────── #
# Metrics from milestone spec
# ──────────────────────────────────────────────────────────── #

@dataclass
class RunMetrics:
    """Metrics tracked per simulation run."""
    run_id: str
    outcome: str                    # resolved | escalated | unresolved
    recovery_rate: float = 0.0      # 1.0 if resolved
    commitment_success: float = 0.0 # 1.0 if committed in resolution
    drop_off_stage: Optional[str] = None   # stage where borrower disengaged
    repetition_errors: int = 0      # count of re-asked questions
    total_turns: int = 0
    cost_usd: float = 0.0


@dataclass
class StrategyUpdate:
    """A strategy change derived from evaluation."""
    agent_name: str
    metric: str
    old_threshold: float
    new_threshold: float
    trigger: str                    # "relaxed" | "tightened"
    guidance_injected: str


@dataclass
class LearningHistory:
    """Tracks all strategy updates across simulation runs."""
    updates: list[StrategyUpdate] = field(default_factory=list)
    run_metrics: list[RunMetrics] = field(default_factory=list)

    def add_run(self, metrics: RunMetrics) -> None:
        self.run_metrics.append(metrics)

    def add_update(self, update: StrategyUpdate) -> None:
        self.updates.append(update)

    @property
    def average_recovery_rate(self) -> float:
        if not self.run_metrics:
            return 0.0
        return sum(m.recovery_rate for m in self.run_metrics) / len(self.run_metrics)

    @property
    def average_commitment_rate(self) -> float:
        if not self.run_metrics:
            return 0.0
        return sum(m.commitment_success for m in self.run_metrics) / len(self.run_metrics)

    @property
    def total_repetition_errors(self) -> int:
        return sum(m.repetition_errors for m in self.run_metrics)

    @property
    def drop_off_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for m in self.run_metrics:
            if m.drop_off_stage:
                dist[m.drop_off_stage] = dist.get(m.drop_off_stage, 0) + 1
        return dist

    def summary(self) -> str:
        lines = [
            f"=== Learning History ({len(self.run_metrics)} runs) ===",
            f"Avg recovery rate:    {self.average_recovery_rate:.2%}",
            f"Avg commitment rate:  {self.average_commitment_rate:.2%}",
            f"Total repetition errors: {self.total_repetition_errors}",
            f"Drop-off distribution: {self.drop_off_distribution}",
            f"Strategy updates: {len(self.updates)}",
        ]
        return "\n".join(lines)


# ──────────────────────────────────────────────────────────── #
# Meta-Evaluation — evaluating the evaluator itself
# ──────────────────────────────────────────────────────────── #

@dataclass
class MetaEvalReport:
    """
    Meta-evaluation: judges whether the evaluation methodology itself is
    producing useful signal.

    Good methodology:
      - High variance across runs (distinguishes good from bad)
      - Thresholds correlate with outcome (passing → resolved)
      - Evolves over time (versions increment)
    """
    evaluator_name: str
    num_evaluations: int
    score_variance: float          # higher = better discrimination
    outcome_correlation: float     # 0-1; how well eval_passed predicts outcome
    versions_evolved: int          # versions bumped over the run
    verdict: str                    # "effective" | "ineffective" | "needs_data"


class MetaEvaluator:
    """
    Evaluates the SelfLearningLoop's evaluation quality.
    
    This is the "meta-evaluation" capability: the eval loop doesn't just
    score the agents, we also score the eval itself and bump the
    methodology version when needed.
    """

    MIN_EVALUATIONS = 5

    def assess(
        self,
        loop: "SelfLearningLoop",
        run_metrics: list[RunMetrics],
    ) -> MetaEvalReport:
        n = len(loop.methodology.historical_results)
        if n < self.MIN_EVALUATIONS:
            return MetaEvalReport(
                evaluator_name=loop.agent_name,
                num_evaluations=n,
                score_variance=0.0,
                outcome_correlation=0.0,
                versions_evolved=0,
                verdict="needs_data",
            )

        # 1. Score variance — spread in overall pass/fail across runs
        passed_flags = [1.0 if r.passed else 0.0 for r in loop.methodology.historical_results]
        variance = _variance(passed_flags)

        # 2. Outcome correlation — did eval_passed predict run resolution?
        outcome_correlation = _correlate_eval_with_outcome(
            loop.methodology.historical_results, run_metrics
        )

        # 3. Versions evolved
        version_parts = loop.methodology.version.split(".")
        versions_evolved = int(version_parts[-1]) - 0

        # Verdict
        if variance < 0.05:
            verdict = "ineffective"       # eval never distinguishes
        elif outcome_correlation > 0.5 and versions_evolved > 0:
            verdict = "effective"
        else:
            verdict = "ineffective"

        return MetaEvalReport(
            evaluator_name=loop.agent_name,
            num_evaluations=n,
            score_variance=round(variance, 4),
            outcome_correlation=round(outcome_correlation, 4),
            versions_evolved=versions_evolved,
            verdict=verdict,
        )


def _variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _correlate_eval_with_outcome(
    eval_results: list,
    run_metrics: list[RunMetrics],
) -> float:
    """
    Returns fraction of runs where eval.passed == (run.outcome == resolved).
    """
    if not eval_results or not run_metrics:
        return 0.0
    pairs = list(zip(eval_results, run_metrics))
    hits = sum(
        1 for e, m in pairs
        if (e.passed and m.recovery_rate > 0)
        or (not e.passed and m.recovery_rate == 0)
    )
    return hits / len(pairs) if pairs else 0.0


# ──────────────────────────────────────────────────────────── #
# Prompt Updater
# ──────────────────────────────────────────────────────────── #

class PromptUpdater:
    """
    Reads evaluation results and injects updated guidance into the agent.

    "Update prompts" = set agent._injected_guidance with top learned patterns.
    The agent's get_system_prompt() picks this up on the next invocation.
    """

    def __init__(self, history: LearningHistory):
        self.history = history

    def update_agent(self, agent, loop: SelfLearningLoop) -> Optional[StrategyUpdate]:
        """
        1. Check if any threshold changed (evolution happened)
        2. Build new guidance string from successful learnings
        3. Inject into agent._injected_guidance
        Returns StrategyUpdate if something changed, else None.
        """
        guidance = loop.get_injected_guidance(top_n=3)
        agent._injected_guidance = guidance

        # Check for threshold changes in the last evolution
        if len(loop.methodology.historical_results) < 10:
            return None

        recent = loop.methodology.historical_results[-10:]
        for criteria in loop.methodology.criteria:
            # Find if threshold changed vs initial
            fails = sum(
                1 for r in recent
                if not loop.methodology._passes(r.scores.get(criteria.metric, 0), criteria)
            )
            if fails >= 5:
                update = StrategyUpdate(
                    agent_name=agent.name,
                    metric=criteria.metric,
                    old_threshold=round(criteria.threshold / 0.90, 4),   # approximate original
                    new_threshold=criteria.threshold,
                    trigger="relaxed",
                    guidance_injected=guidance[:200],
                )
                self.history.add_update(update)
                return update
        return None


# ──────────────────────────────────────────────────────────── #
# Run metric builder
# ──────────────────────────────────────────────────────────── #

def build_run_metrics(
    run_id: str,
    outcome: str,
    conversation_history: list[dict],
    cost_usd: float,
    eval_results: list[EvaluationResult],
) -> RunMetrics:
    """Compute all metrics for a completed simulation run."""
    stages = [m.get("stage", "") for m in conversation_history]
    agent_turns = [m for m in conversation_history if m.get("role") == "assistant"]

    # Recovery rate
    recovery_rate = 1.0 if outcome in ("resolved", "committed") else 0.0

    # Commitment success (did resolution produce a commitment?)
    resolution_turns = [m for m in conversation_history if m.get("stage") == "resolution"]
    commitment_success = 1.0 if any(
        "commit" in m.get("content", "").lower() or
        "agree" in m.get("content", "").lower()
        for m in resolution_turns if m.get("role") == "user"
    ) else 0.0

    # Drop-off: stage where things stalled (last stage if not resolved)
    if outcome in ("resolved", "committed"):
        drop_off_stage = None
    else:
        seen = set(stages)
        drop_off_stage = "final_notice" if "final_notice" in seen else \
                        "resolution" if "resolution" in seen else "assessment"

    # Repetition errors: count of eval results where no_repeat_questions < 1.0
    repetition_errors = sum(
        1 for r in eval_results
        if r.scores.get("no_repeat_questions", 1.0) < 1.0
    )

    return RunMetrics(
        run_id=run_id,
        outcome=outcome,
        recovery_rate=recovery_rate,
        commitment_success=commitment_success,
        drop_off_stage=drop_off_stage,
        repetition_errors=repetition_errors,
        total_turns=len(agent_turns),
        cost_usd=cost_usd,
    )
