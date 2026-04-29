"""
Transcript Generator + Auto-Learning Loop
==========================================

Generates N synthetic transcripts distributed across all 4 borrower personas,
persists each one to MongoDB via log_interaction / log_outcome, and can
optionally run the 6-stage improvement pipeline in a loop.

Persona distribution (per batch of 8):
  2 × cooperative        → expects AGREEMENT
  2 × broke              → expects AGREEMENT (hardship path)
  2 × strategic_defaulter → expects LEGAL / AGREEMENT (lump-sum)
  2 × hostile            → expects LEGAL

The loop runs: generate batch → run pipeline → apply patches → generate next batch
showing auto-learning across negotiation and resolution agents.

Usage (programmatic):
    from src.transcript_generator import run_generation_loop
    await run_generation_loop(total=30, batch_size=8, agent_target="ResolutionAgent")
    await run_generation_loop(total=30, run_pipeline=False)  # generation only

Usage (CLI):
    python -m src.transcript_generator --total 30 --batch-size 8
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────

@dataclass
class GenerationConfig:
    total: int = 30
    batch_size: int = 8
    agent_target: str = "ResolutionAgent"   # agent to focus learning on
    run_pipeline_every_n: int = 30          # run improvement pipeline after every N transcripts
    max_turns_per_stage: int = 6
    personas: list[str] = field(default_factory=lambda: [
        "cooperative", "cooperative",
        "broke", "broke",
        "strategic_defaulter", "strategic_defaulter",
        "hostile", "hostile",
    ])


@dataclass
class GenerationProgress:
    job_id: str
    status: str = "running"           # running / completed / failed
    total: int = 0
    completed: int = 0
    failed: int = 0
    pipeline_runs: int = 0
    current_persona: str = ""
    current_batch: int = 0
    total_batches: int = 0
    log: list[str] = field(default_factory=list)
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    error: Optional[str] = None

    def emit(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.log.append(entry)
        log.info("[generator] %s", msg)


# Global registry so the API can poll progress
_jobs: dict[str, GenerationProgress] = {}


def get_job(job_id: str) -> Optional[GenerationProgress]:
    return _jobs.get(job_id)


def list_jobs() -> list[GenerationProgress]:
    return sorted(_jobs.values(), key=lambda j: j.started_at, reverse=True)


# ─────────────────────────────────────────────────────────────────
# Core — run one transcript through CollectionsPipeline
# ─────────────────────────────────────────────────────────────────

async def _run_one_transcript(
    persona_name: str,
    borrower_id: str,
    loan_id: str,
    outstanding_amount: float,
    days_past_due: int,
    max_turns: int,
    progress: GenerationProgress,
) -> dict:
    """
    Run one full 3-agent pipeline with a scripted borrower persona.
    Returns a result dict compatible with _fetch_last_n_transcripts.
    """
    from src.simulation import PersonaScript, PersonaType, SimulationEngine
    from src.pipeline import CollectionsPipeline
    from src.data_layer import log_interaction, log_outcome, upsert_borrower_case
    from src.models import Stage
    from src.self_learning.improvement_pipeline import get_prompt_version

    try:
        persona_type = PersonaType(persona_name.lower())
    except ValueError:
        persona_type = PersonaType.COOPERATIVE

    script = PersonaScript(persona_type)
    mock_responses = SimulationEngine.get_mock_llm_responses(persona_type)

    pipeline = CollectionsPipeline()

    # Patch the agents to use mock LLM responses instead of real API calls
    # This makes the generator fast and free
    _patch_agent_with_mock(pipeline.assessment_agent,  mock_responses.get("AssessmentAgent", []))
    _patch_agent_with_mock(pipeline.resolution_agent,  mock_responses.get("ResolutionAgent", []))
    _patch_agent_with_mock(pipeline.final_notice_agent, mock_responses.get("FinalNoticeAgent", []))

    # Pre-verify identity so the assessment agent's hard gate is satisfied
    # (in simulation there is no real Postgres lookup — mark it directly)
    _pre_verify_identity(pipeline.assessment_agent, persona_type)

    def input_provider(ctx, agent_name: str) -> Optional[str]:
        return script.respond(agent_name)

    progress.emit(f"  Running {persona_name} transcript {borrower_id}…")

    result = await pipeline.run(
        borrower_id=borrower_id,
        loan_id=loan_id,
        principal_amount=outstanding_amount * 1.15,
        outstanding_amount=outstanding_amount,
        days_past_due=days_past_due,
        input_provider=input_provider,
        max_turns_per_stage=max_turns,
    )

    # Persist to MongoDB
    trace_id = f"int-{uuid.uuid4().hex[:12]}"
    ts = datetime.now(timezone.utc).isoformat()

    # Write one interaction record per turn (all stages)
    stage_agent_map = {
        "assessment":    "AssessmentAgent",
        "resolution":    "ResolutionAgent",
        "final_notice":  "FinalNoticeAgent",
    }

    prev_stage = None
    for msg in result.conversation:
        if msg.get("role") != "assistant":
            continue
        stage = msg.get("stage", "assessment")
        agent_name = stage_agent_map.get(stage, "AssessmentAgent")

        # Find the corresponding user turn
        user_content = ""
        for um in result.conversation:
            if um.get("role") == "user" and um.get("stage") == stage and um.get("turn") == msg.get("turn"):
                user_content = um.get("content", "")
                break

        prompt_version = get_prompt_version(agent_name)

        asyncio.create_task(log_interaction(
            borrower_id=borrower_id,
            agent_name=agent_name,
            agent_version="v1.0",
            prompt_version=prompt_version,
            model="mock",
            model_params={"mock": True},
            input_text=user_content,
            output_text=msg.get("content", ""),
            structured_context={"persona": persona_name, "stage": stage},
            decision="advance" if msg.get("advanced") else "continue",
            trace_id=trace_id,
        ))

    # Write outcome
    outcome_label = _map_outcome(result.outcome)
    asyncio.create_task(log_outcome(
        borrower_id=borrower_id,
        outcome=outcome_label,
        agent_versions={"AssessmentAgent": "v1.0", "ResolutionAgent": "v1.0", "FinalNoticeAgent": "v1.0"},
        metadata={"persona": persona_name, "total_turns": result.total_turns, "generated": True},
    ))
    asyncio.create_task(upsert_borrower_case(
        borrower_id, result.final_stage.upper(), outcome_label
    ))

    progress.emit(
        f"    ✓ {persona_name} → outcome={outcome_label} "
        f"turns={result.total_turns} stage={result.final_stage}"
    )

    return {
        "borrower_id": borrower_id,
        "persona": persona_name,
        "outcome": outcome_label,
        "total_turns": result.total_turns,
        "final_stage": result.final_stage,
        "trace_id": trace_id,
        "conversation": result.conversation,
    }


def _map_outcome(outcome: str) -> str:
    """Map pipeline outcome strings to MongoDB outcome labels."""
    mapping = {
        "resolved":              "AGREEMENT",
        "committed":             "AGREEMENT",
        "escalated":             "LEGAL",
        "assessment_incomplete": "NO_DEAL",
        "resolution_no_outcome": "NO_DEAL",
        "unresolved":            "NO_DEAL",
        "max_turns_reached":     "NO_DEAL",
    }
    return mapping.get(outcome, "NO_DEAL")


def _pre_verify_identity(assessment_agent, persona_type) -> None:
    """
    Monkey-patch the assessment agent's process() so identity_verified=True
    is set on the context before the hard gate is evaluated.
    In simulation there is no Postgres — we bypass the check directly.
    """
    from src.simulation import PersonaType
    original_process = assessment_agent.process

    async def _patched_process(context, user_input):
        # Mark identity as verified for simulation runs
        if context.assessment_data:
            context.assessment_data.identity_verified = True
        return await original_process(context, user_input)

    assessment_agent.process = _patched_process


def _patch_agent_with_mock(agent, responses: list[str]) -> None:
    """Replace agent._call_claude with an iterator over scripted responses."""
    from src.token_budget import TokenUsage
    it = iter(responses)

    async def _mock_call(*args, **kwargs):
        text = next(it, "I understand. Please continue.")
        return text, TokenUsage(input_tokens=150, output_tokens=60)

    async def _mock_call_with_tools(*args, **kwargs):
        text = next(it, "I understand. Please continue.")
        return text, TokenUsage(input_tokens=150, output_tokens=60)

    agent._call_claude = _mock_call
    agent._call_claude_with_tools = _mock_call_with_tools


# ─────────────────────────────────────────────────────────────────
# Batch generator
# ─────────────────────────────────────────────────────────────────

async def _generate_batch(
    batch_num: int,
    personas: list[str],
    progress: GenerationProgress,
    base_configs: list[dict],
) -> list[dict]:
    """Generate one batch of transcripts (one per persona in the list)."""
    results = []
    for i, persona in enumerate(personas):
        cfg = base_configs[i % len(base_configs)]
        borrower_id = f"GEN-{persona[:4].upper()}-{uuid.uuid4().hex[:6].upper()}"
        loan_id     = f"LN-GEN-{uuid.uuid4().hex[:6].upper()}"
        progress.current_persona = persona

        try:
            r = await _run_one_transcript(
                persona_name=persona,
                borrower_id=borrower_id,
                loan_id=loan_id,
                outstanding_amount=cfg["outstanding_amount"],
                days_past_due=cfg["days_past_due"],
                max_turns=cfg["max_turns"],
                progress=progress,
            )
            results.append(r)
            progress.completed += 1
        except Exception as e:
            progress.failed += 1
            progress.emit(f"  ✗ {persona} {borrower_id} FAILED: {e}")
            log.exception("[generator] transcript failed: %s", e)

        # Small delay to avoid hammering logs
        await asyncio.sleep(0.05)

    return results


# ─────────────────────────────────────────────────────────────────
# Pipeline trigger
# ─────────────────────────────────────────────────────────────────

async def _run_pipeline_and_apply(
    agent_target: str,
    batch_results: list[dict],
    progress: GenerationProgress,
    batch_num: int,
) -> None:
    """
    Build transcript dicts compatible with admin_api._fetch_last_n_transcripts
    format, then run the improvement pipeline. If adopted, the prompt is updated
    automatically by the pipeline's _apply_prompt().
    """
    from src.self_learning.improvement_pipeline import run_improvement_pipeline
    from src.admin_api import _fetch_last_n_transcripts, _agent_to_stage

    progress.emit(f"  → Running improvement pipeline on {agent_target} (batch {batch_num})…")

    # Wait a moment for the async MongoDB writes to land
    await asyncio.sleep(1.5)

    # Fetch the latest transcripts from MongoDB (includes just-written ones)
    target_n = 30
    transcripts = await _fetch_last_n_transcripts(n=target_n)
    if not transcripts:
        progress.emit("  ⚠ No transcripts in MongoDB yet — skipping pipeline")
        return
    if len(transcripts) < target_n:
        progress.emit(f"  ⚠ Only {len(transcripts)} transcripts available (target {target_n}); running with available data")

    import os
    prev_exec_mode = os.getenv("REAL_V2_EXECUTION_MODE")
    prev_replay_mode = os.getenv("REPLAY_BORROWER_MODE")
    os.environ["REAL_V2_EXECUTION_MODE"] = "real"
    os.environ["REPLAY_BORROWER_MODE"] = "history"
    try:
        pipeline_run = await run_improvement_pipeline(
            transcripts=transcripts,
            agent_name=agent_target,
            triggered_by=f"generator_batch_{batch_num}",
        )
    finally:
        if prev_exec_mode is None:
            os.environ.pop("REAL_V2_EXECUTION_MODE", None)
        else:
            os.environ["REAL_V2_EXECUTION_MODE"] = prev_exec_mode
        if prev_replay_mode is None:
            os.environ.pop("REPLAY_BORROWER_MODE", None)
        else:
            os.environ["REPLAY_BORROWER_MODE"] = prev_replay_mode

    progress.pipeline_runs += 1
    decision = pipeline_run.decision or "pending"
    compliance = "✓ compliant" if (pipeline_run.compliance_check and pipeline_run.compliance_check.compliant) else "✗ non-compliant"

    patterns = len(pipeline_run.failure_analysis.failure_patterns) if pipeline_run.failure_analysis else 0
    hyps     = len(pipeline_run.hypothesis_set.hypotheses) if pipeline_run.hypothesis_set else 0

    progress.emit(
        f"  Pipeline run {pipeline_run.run_id}: "
        f"decision={decision.upper()} | {compliance} | "
        f"{patterns} failure patterns | {hyps} prompt patches"
    )

    if decision == "adopt":
        progress.emit(f"  ✅ Prompt for {agent_target} UPDATED — patches applied")
    else:
        progress.emit(f"  📋 Prompt unchanged (decision=reject) — patches logged for review")

    # Log hypothesis patches
    if pipeline_run.hypothesis_set:
        for i, h in enumerate(pipeline_run.hypothesis_set.hypotheses, 1):
            progress.emit(f"    Patch {i}: {h.hypothesis[:120]}")


# ─────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────

# Loan configs per persona to produce realistic variation
_PERSONA_LOAN_CONFIGS = {
    "cooperative":           {"outstanding_amount": 85_000,  "days_past_due": 90,  "max_turns": 6},
    "broke":                 {"outstanding_amount": 40_000,  "days_past_due": 60,  "max_turns": 6},
    "strategic_defaulter":   {"outstanding_amount": 120_000, "days_past_due": 180, "max_turns": 6},
    "hostile":               {"outstanding_amount": 95_000,  "days_past_due": 120, "max_turns": 6},
}

_DEFAULT_CONFIG = {"outstanding_amount": 85_000, "days_past_due": 90, "max_turns": 6}


async def run_generation_loop(
    total: int = 30,
    batch_size: int = 8,
    agent_target: str = "ResolutionAgent",
    run_pipeline_every_n: int = 30,
    run_pipeline: bool = True,
    job_id: Optional[str] = None,
) -> GenerationProgress:
    """
    Main entry point.

    Generates `total` transcripts in batches of `batch_size`.
    If run_pipeline=True, runs the improvement pipeline after every
    `run_pipeline_every_n` transcripts and once at the end.

    Returns the GenerationProgress object (also accessible via get_job(job_id)).
    """
    if job_id is None:
        job_id = f"gen-{uuid.uuid4().hex[:8]}"

    progress = GenerationProgress(
        job_id=job_id,
        total=total,
        total_batches=(total + batch_size - 1) // batch_size,
    )
    _jobs[job_id] = progress

    # Build the persona rotation — repeat the 4-persona cycle to fill `total`
    persona_cycle = [
        "cooperative", "broke", "strategic_defaulter", "hostile",
        "cooperative", "broke", "strategic_defaulter", "hostile",
    ]
    all_personas = []
    while len(all_personas) < total:
        all_personas.extend(persona_cycle)
    all_personas = all_personas[:total]

    # Pre-build loan configs per persona
    base_configs = [
        _PERSONA_LOAN_CONFIGS.get(p, _DEFAULT_CONFIG)
        for p in all_personas
    ]

    mode = "generate+pipeline" if run_pipeline else "generate-only"
    progress.emit(
        f"Starting generation: {total} transcripts across {progress.total_batches} batches"
        f" | mode: {mode}"
        f" | learning agent: {agent_target}"
        f" | pipeline every {run_pipeline_every_n} transcripts"
    )
    progress.emit(f"Persona distribution: {_count_personas(all_personas)}")

    transcripts_since_pipeline = 0
    batch_num = 0

    try:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            batch_personas = all_personas[batch_start:batch_end]
            batch_configs   = base_configs[batch_start:batch_end]
            batch_num += 1
            progress.current_batch = batch_num

            progress.emit(
                f"\n─── Batch {batch_num}/{progress.total_batches} "
                f"({batch_start+1}–{batch_end} of {total}) ───"
            )

            batch_results = await _generate_batch(
                batch_num=batch_num,
                personas=batch_personas,
                progress=progress,
                base_configs=batch_configs,
            )
            transcripts_since_pipeline += len(batch_results)

            # Summary for this batch
            outcomes = [r["outcome"] for r in batch_results]
            progress.emit(
                f"  Batch {batch_num} done: "
                f"AGREEMENT={outcomes.count('AGREEMENT')} "
                f"LEGAL={outcomes.count('LEGAL')} "
                f"NO_DEAL={outcomes.count('NO_DEAL')}"
            )

            # Run improvement pipeline every run_pipeline_every_n transcripts
            if run_pipeline and run_pipeline_every_n > 0 and transcripts_since_pipeline >= run_pipeline_every_n:
                await _run_pipeline_and_apply(
                    agent_target=agent_target,
                    batch_results=batch_results,
                    progress=progress,
                    batch_num=batch_num,
                )
                transcripts_since_pipeline = 0

        if run_pipeline:
            # Final pipeline run on all generated data
            progress.emit(f"\n─── Final pipeline run after all {total} transcripts ───")
            await _run_pipeline_and_apply(
                agent_target=agent_target,
                batch_results=[],
                progress=progress,
                batch_num=batch_num + 1,
            )

        progress.status = "completed"
        progress.completed_at = datetime.now(timezone.utc).isoformat()
        progress.emit(
            f"\n✅ Generation complete: {progress.completed} transcripts | "
            f"{progress.failed} failed | {progress.pipeline_runs} pipeline runs"
        )

    except Exception as e:
        progress.status = "failed"
        progress.error = str(e)
        progress.completed_at = datetime.now(timezone.utc).isoformat()
        progress.emit(f"✗ Generation loop failed: {e}")
        log.exception("[generator] run_generation_loop failed: %s", e)

    return progress


def _count_personas(personas: list[str]) -> str:
    from collections import Counter
    c = Counter(personas)
    return " | ".join(f"{k}={v}" for k, v in sorted(c.items()))


# ─────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    import os
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
    )

    parser = argparse.ArgumentParser(description="Generate synthetic transcripts + auto-learning loop")
    parser.add_argument("--total",      type=int, default=30,               help="Total transcripts to generate")
    parser.add_argument("--batch-size", type=int, default=8,                help="Batch size between pipeline runs")
    parser.add_argument("--agent",      type=str, default="ResolutionAgent",help="Agent to focus learning on")
    parser.add_argument("--pipeline-every", type=int, default=30,          help="Run pipeline every N transcripts")
    args = parser.parse_args()

    async def main():
        progress = await run_generation_loop(
            total=args.total,
            batch_size=args.batch_size,
            agent_target=args.agent,
            run_pipeline_every_n=args.pipeline_every,
        )
        print("\n".join(progress.log))
        print(f"\nFinal status: {progress.status}")
        print(f"Completed: {progress.completed}/{progress.total}")
        print(f"Pipeline runs: {progress.pipeline_runs}")

    asyncio.run(main())
