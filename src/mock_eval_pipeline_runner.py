"""
Complete mocked improvement pipeline runner.

Goals:
- Run the 7-stage improvement pipeline fully offline (mocked LLM + mocked v2 execution)
- Persist pipeline run data to MongoDB so Admin UI shows it like a real run
- Auto-apply prompt patch so a new prompt version is created on disk
- Optionally seed post-patch interactions so Agent Analytics can show version rows

Usage:
  .venv/bin/python -m src.mock_eval_pipeline_runner --agent ResolutionAgent --n 60 --auto-patch --seed-analytics 40
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from src.data_layer import get_mongo, log_interaction
from src.self_learning import improvement_pipeline as pip


@dataclass
class RunnerConfig:
    agent: str
    n: int
    auto_patch: bool
    seed_analytics: int
    ensure_v2: bool
    run_id: str


def _json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True)


def _det_hash(text: str) -> float:
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]
    return int(h, 16) / 0xFFFFFFFF


def _render_improved_prompt(user_msg: str) -> str:
    marker_start = "CURRENT PROMPT:\n"
    marker_end = "\n\nCOMPLIANCE ISSUES"
    if marker_start in user_msg and marker_end in user_msg:
        current = user_msg.split(marker_start, 1)[1].split(marker_end, 1)[0]
    else:
        current = user_msg
    patch = (
        "## MOCK IMPROVEMENT BLOCK\n"
        "- After every objection: restate approved terms + ask commitment in same turn.\n"
        "- Always surface deadline in first offer presentation.\n"
        "## END MOCK IMPROVEMENT BLOCK\n"
    )
    if "{injected_guidance_block}" in current:
        return current.replace("{injected_guidance_block}", patch + "{injected_guidance_block}", 1)
    return current + "\n" + patch


async def _mock_llm(system: str, user: str, model: str = "", max_tokens: int = 0, temperature: float = 0.0) -> str:
    sys_lc = system.lower()

    # Stage 2: failure analysis
    if "analyzing failed debt collection conversations" in sys_lc:
        return _json({
            "failure_patterns": [
                {
                    "pattern": "Objection loops without close",
                    "frequency": "high",
                    "root_cause": "Agent clarifies but misses explicit commitment ask",
                    "recommended_behavior": "Close every objection turn with direct commitment question",
                },
                {
                    "pattern": "Deadline mention is delayed",
                    "frequency": "medium",
                    "root_cause": "Expiry is surfaced too late",
                    "recommended_behavior": "State expiry in first offer line",
                },
            ]
        })

    # Stage 4: improved prompt generation
    if "you are improving a debt collection agent prompt" in sys_lc:
        return _json({
            "new_prompt": _render_improved_prompt(user),
            "changes_summary": [
                "Added objection->restate->commitment cadence",
                "Moved deadline emphasis earlier",
            ],
            "expected_impact": "Higher close rate with clearer negotiation control",
        })

    # Stage 5: compliance checks
    if "compliance auditor for a conversational ai debt collection agent" in sys_lc:
        return _json({"compliant": True, "risks": [], "reason": "Compliant"})
    if "senior compliance reviewer" in sys_lc:
        return _json({"compliant": True, "risks": [], "reason": "Reviewed and compliant"})

    # Stage 6b: comparison narrative
    if "summarizing an a/b evaluation between two agent prompt versions" in sys_lc:
        return _json({
            "summary": "Mock run shows consistent uplift on resolution and effect size.",
            "reason": "Gates pass with significant p-value and practical delta.",
        })

    # Stage 7: hypothesis generation
    if "you are generating prompt patches for a debt collection agent" in sys_lc:
        return _json({
            "hypotheses": [
                {
                    "hypothesis": "After each objection, restate approved terms and ask for commitment immediately.",
                    "why_it_might_work": "Reduces negotiation drift",
                    "how_to_test": "Track resolution rate and turns-to-commit",
                },
                {
                    "hypothesis": "Mention offer expiry in the first offer sentence for every resolution path.",
                    "why_it_might_work": "Improves urgency and conversion",
                    "how_to_test": "Compare acceptance under fixed sample windows",
                },
            ]
        })

    # Stage 6a rescore fallback or Stage 1 scoring shape
    if "predicting how a changed agent prompt would affect conversation quality" in sys_lc:
        # v2 counterfactual: stronger resolution rate (~0.62)
        p = _det_hash(user)
        resolved = 1 if p < 0.62 else 0
        return _json({
            "resolution": resolved,
            "resolution_confidence": 0.87 if resolved else 0.23,
            "debt_collected": resolved,
            "compliance_violation": 0,
            "compliance_reason": "",
            "tone_score": 4 if resolved else 3,
            "next_step_clarity": 4 if resolved else 3,
        })

    # Stage 1 judge: baseline resolution rate (~0.30)
    p = _det_hash(user)
    resolved = 1 if p < 0.30 else 0
    return _json({
        "resolution": resolved,
        "resolution_confidence": 0.82 if resolved else 0.18,
        "debt_collected": resolved,
        "compliance_violation": 0,
        "compliance_reason": "",
        "tone_score": 4 if resolved else 3,
        "next_step_clarity": 4 if resolved else 3,
    })


async def _mock_execute_real_v2_on_transcripts(
    transcripts: list[dict],
    v1_scores: list[pip.TranscriptScore],
    agent_name: str,
    prompt_template: str,
) -> list[pip.TranscriptScore]:
    # Produce deterministic uplift on the same borrowers, simulating "real v2 execution".
    out: list[pip.TranscriptScore] = []
    for t, s in zip(transcripts, v1_scores):
        key = f"{t.get('trace_id','')}-{agent_name}-v2"
        p = _det_hash(key)
        resolved = 1 if p < 0.62 else 0
        out.append(
            pip.TranscriptScore(
                trace_id=f"{s.trace_id}:v2_mockreal",
                borrower_id=s.borrower_id,
                agent_name=agent_name,
                resolution=resolved,
                resolution_confidence=0.88 if resolved else 0.22,
                debt_collected=resolved,
                compliance_violation=0,
                compliance_reason="",
                tone_score=4 if resolved else 3,
                next_step_clarity=4 if resolved else 3,
                raw_transcript_turns=s.raw_transcript_turns,
            )
        )
    return out


def _build_transcripts(agent: str, n: int) -> list[dict[str, Any]]:
    transcripts: list[dict[str, Any]] = []
    for i in range(n):
        accepted_line = "I accept and will proceed today." if i % 4 == 0 else "I cannot commit right now."
        history = [
            {"role": "user", "stage": "assessment", "content": "I received the notice."},
            {"role": "assistant", "stage": "assessment", "content": "Share last 4 and birth year."},
            {"role": "user", "stage": "assessment", "content": "7823 and 1989."},
            {"role": "assistant", "stage": "resolution", "content": "Offer is 25% upfront plus installments."},
            {"role": "user", "stage": "resolution", "content": accepted_line},
            {"role": "assistant", "stage": "final_notice", "content": "Final offer expires in 48 hours."},
        ]
        transcripts.append({
            "trace_id": f"MOCKPIPE-{agent[:3].upper()}-{i:04d}",
            "borrower_id": f"BRW-MOCKPIPE-{i:04d}",
            "primary_agent": agent,
            "history": history,
        })
    return transcripts


def _patch_num(version: str) -> int:
    if version.startswith("patch-v"):
        try:
            return int(version.split("patch-v", 1)[1])
        except Exception:
            return 0
    return 0


async def _ensure_v2_start(agent: str, run_id: str) -> None:
    cur = pip.get_prompt_version(agent)
    if _patch_num(cur) >= 1:
        return
    bootstrap_text = "Bootstrap patch: preserve baseline behavior while enabling versioned evolution tracking."
    await pip.apply_single_patch_async(agent, bootstrap_text, f"{run_id}-bootstrap", 0)


async def _seed_interactions(agent: str, prompt_version: str, n: int) -> None:
    if n <= 0:
        return
    now_tag = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    for i in range(n):
        resolved = (i % 3 == 0)
        if agent == "AssessmentAgent":
            out = (
                "Identity verified. Monthly income and expenses captured. Employment status noted. "
                "Cash flow difficulty classified. ASSESSMENT_COMPLETE:INSTALLMENT"
                if resolved else
                "Need missing details for expenses and employment before completion."
            )
        elif agent == "ResolutionAgent":
            out = (
                "Terms are approved with deadline 48 hours. Can you confirm you agree today? "
                "Confirmed. RESOLUTION_COMPLETE"
                if resolved else
                "These are the approved terms and deadline. Please confirm commitment."
            )
        else:
            out = (
                "Final offer expires in 48 hours. Credit and legal escalation may follow. "
                "Accepted. COLLECTIONS_COMPLETE"
                if resolved else
                "Final notice: deadline active. Credit, legal, and asset escalation steps apply."
            )
        await log_interaction(
            borrower_id=f"BRW-AN-{agent[:3]}-{now_tag}-{i:03d}",
            agent_name=agent,
            agent_version="v1.0",
            prompt_version=prompt_version,
            model="mock",
            model_params={"mock": True, "source": "mock_eval_pipeline_runner"},
            input_text="Mock borrower utterance",
            output_text=out,
            structured_context={"stage": agent.replace("Agent", "").lower()},
            decision="advance" if resolved else "continue",
            confidence=0.8 if resolved else 0.3,
            reasoning_summary="mock seeded interaction",
            trace_id=f"trace-{agent[:3]}-{prompt_version}-{now_tag}-{i:03d}",
        )


async def run(cfg: RunnerConfig) -> None:
    db = await get_mongo()
    if db is None:
        raise RuntimeError("MongoDB unavailable. Start MongoDB so UI can show run details.")

    # Relax gates slightly for faster demos while preserving logic.
    import os
    os.environ.setdefault("AB_MIN_SAMPLE_PER_ARM", "20")
    os.environ.setdefault("AB_MIN_RESOLUTION_DELTA", "0.02")
    os.environ.setdefault("AB_MIN_COHEN_D", "0.20")
    os.environ.setdefault("REAL_V2_EXECUTION_MODE", "simulator")
    os.environ.setdefault("REPLAY_BORROWER_MODE", "simulator")

    if cfg.ensure_v2:
        await _ensure_v2_start(cfg.agent, cfg.run_id)

    transcripts = _build_transcripts(cfg.agent, cfg.n)

    orig_llm = pip._llm
    orig_exec_real_v2 = pip.execute_real_v2_on_transcripts
    pip._llm = _mock_llm
    pip.execute_real_v2_on_transcripts = _mock_execute_real_v2_on_transcripts
    try:
        run_doc = await pip.run_improvement_pipeline(
            transcripts=transcripts,
            agent_name=cfg.agent,
            triggered_by="mock_eval_script",
            run_id=cfg.run_id,
        )

        print("\n=== Mock Pipeline Run ===")
        print(f"run_id: {run_doc.run_id}")
        print(f"status: {run_doc.status}")
        print(f"agent: {run_doc.agent_target}")
        print(f"decision: {run_doc.decision}")
        print(f"promote: {run_doc.promote}")
        print(f"v2_execution_mode: {getattr(run_doc, 'v2_execution_mode', 'unknown')}")
        print(f"v2_execution_note: {getattr(run_doc, 'v2_execution_note', '')}")
        if run_doc.version_comparison:
            vc = run_doc.version_comparison
            print(f"v1_rate: {vc.v1_resolution_rate}")
            print(f"v2_rate: {vc.v2_resolution_rate}")
            print(f"p_value: {vc.p_value}")
            print(f"reason: {vc.reason}")
        if run_doc.executed_v2_scores:
            resolved = sum(int(s.resolution) for s in run_doc.executed_v2_scores)
            print(f"executed_v2_resolved: {resolved}/{len(run_doc.executed_v2_scores)}")
        print(f"hypotheses: {len((run_doc.hypothesis_set.hypotheses if run_doc.hypothesis_set else []))}")

        if cfg.auto_patch:
            hyps = run_doc.hypothesis_set.hypotheses if run_doc.hypothesis_set else []
            if not hyps:
                raise RuntimeError("No hypothesis patches available to apply.")
            patch_result = await pip.apply_single_patch_async(
                cfg.agent,
                hyps[0].hypothesis,
                run_doc.run_id,
                0,
            )
            print("\n=== Auto Patch Result ===")
            print(json.dumps(patch_result, indent=2))
            if not patch_result.get("applied"):
                reason = patch_result.get("reason", "unknown")
                risks = patch_result.get("risks", [])
                raise RuntimeError(f"Patch rejected: {reason} | risks={risks}")
            new_ver = pip.get_prompt_version(cfg.agent)
            print(f"active_prompt_version: {new_ver}")

            if cfg.seed_analytics > 0:
                await _seed_interactions(cfg.agent, new_ver, cfg.seed_analytics)
                print(f"seeded_interactions: {cfg.seed_analytics} (prompt_version={new_ver})")
    finally:
        pip._llm = orig_llm
        pip.execute_real_v2_on_transcripts = orig_exec_real_v2

    print("\nUI endpoints:")
    print(f"- Pipeline run detail: /api/admin/pipeline/runs/{cfg.run_id}")
    print("- Pipeline list: /api/admin/pipeline/runs")
    print("- Analytics: /api/admin/analytics/agents")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fully mocked improvement pipeline with optional auto-patch")
    parser.add_argument("--agent", choices=["AssessmentAgent", "ResolutionAgent", "FinalNoticeAgent"], default="ResolutionAgent")
    parser.add_argument("--n", type=int, default=60, help="Number of synthetic transcripts for the mocked run")
    parser.add_argument("--auto-patch", action="store_true", help="Auto-apply first generated hypothesis patch")
    parser.add_argument("--seed-analytics", type=int, default=0, help="Seed N interactions on the new prompt version for analytics table")
    parser.add_argument("--ensure-v2", action="store_true", help="If current prompt has no patches, add a bootstrap patch first so next is v2")
    parser.add_argument("--run-id", default="", help="Optional explicit run_id (default: generated)")
    args = parser.parse_args()

    run_id = args.run_id.strip() or f"pipeline-mock-{args.agent[:3].lower()}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"

    cfg = RunnerConfig(
        agent=args.agent,
        n=max(20, int(args.n)),
        auto_patch=bool(args.auto_patch),
        seed_analytics=max(0, int(args.seed_analytics)),
        ensure_v2=bool(args.ensure_v2),
        run_id=run_id,
    )
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
