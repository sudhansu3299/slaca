"""
Offline demo runner for the self-learning improvement pipeline.

What it does:
1) Builds synthetic transcripts (no live calls, no voice provider APIs)
2) Mocks all improvement-pipeline LLM calls
3) Runs run_improvement_pipeline()
4) Optionally applies the first generated hypothesis patch
5) Prints the resulting prompt version and file path

Usage:
  python -m src.mock_pipeline_demo --agent ResolutionAgent --n 40 --apply-patch
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from dataclasses import dataclass
from typing import Any

from src.self_learning import improvement_pipeline as pip


@dataclass
class DemoConfig:
    agent_name: str = "ResolutionAgent"
    n_transcripts: int = 40
    apply_patch: bool = True


def _safe_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=True)


def _insert_before_guidance(prompt: str, text: str) -> str:
    marker = "{injected_guidance_block}"
    if marker in prompt:
        return prompt.replace(marker, f"{text}\n{marker}", 1)
    if prompt.endswith("\n"):
        return f"{prompt}{text}\n"
    return f"{prompt}\n{text}\n"


def _extract_current_prompt(user_msg: str) -> str:
    start = "CURRENT PROMPT:\n"
    stop = "\n\nCOMPLIANCE ISSUES"
    if start in user_msg and stop in user_msg:
        return user_msg.split(start, 1)[1].split(stop, 1)[0]
    return user_msg


def _make_mock_transcripts(n: int, agent_name: str) -> list[dict[str, Any]]:
    random.seed(7)
    out: list[dict[str, Any]] = []
    for i in range(n):
        persona = "cooperative" if i % 2 == 0 else "hostile"
        accepted = (i % 4 == 0) or (i % 7 == 0)
        borrower_line = (
            "I can accept and proceed today."
            if accepted
            else "I refuse these terms right now."
        )
        history = [
            {"role": "user", "stage": "assessment", "content": "I received the notice."},
            {"role": "assistant", "stage": "assessment", "content": "Provide last 4 and birth year."},
            {"role": "user", "stage": "assessment", "content": "7823 and 1988."},
            {"role": "assistant", "stage": "resolution", "content": "Offer is 30% upfront + installments."},
            {"role": "user", "stage": "resolution", "content": borrower_line},
            {"role": "assistant", "stage": "final_notice", "content": "Final offer expires in 48 hours."},
        ]
        out.append(
            {
                "trace_id": f"MOCK-{agent_name[:3].upper()}-{i:04d}",
                "borrower_id": f"BRW-MOCK-{i:04d}",
                "primary_agent": agent_name,
                "history": history,
                "persona": persona,
            }
        )
    return out


async def _mock_llm(system: str, user: str, model: str = "", max_tokens: int = 0, temperature: float = 0.0) -> str:
    sys_lc = system.lower()
    user_lc = user.lower()

    # Stage 1 judge + Stage 6a rescore shape
    if "resolution_confidence" in system and "next_step_clarity" in system:
        resolved = 1 if ("accept" in user_lc or "proceed today" in user_lc) else 0
        return _safe_json(
            {
                "resolution": resolved,
                "resolution_confidence": 0.88 if resolved else 0.24,
                "debt_collected": resolved,
                "compliance_violation": 0,
                "compliance_reason": "",
                "tone_score": 4 if resolved else 3,
                "next_step_clarity": 4 if resolved else 3,
            }
        )

    # Stage 2 failure analysis
    if "analyzing failed debt collection conversations" in sys_lc:
        return _safe_json(
            {
                "failure_patterns": [
                    {
                        "pattern": "Weak objection handling",
                        "frequency": "high",
                        "root_cause": "Agent restates terms without commitment ask",
                        "recommended_behavior": "Restate terms then ask explicit commitment each turn",
                    },
                    {
                        "pattern": "Deadline not emphasized",
                        "frequency": "medium",
                        "root_cause": "Expiry appears late in conversation",
                        "recommended_behavior": "State expiry with the first offer presentation",
                    },
                ]
            }
        )

    # Stage 4 prompt improvement
    if "you are improving a debt collection agent prompt" in sys_lc:
        current = _extract_current_prompt(user)
        improved = _insert_before_guidance(
            current,
            "## MOCK IMPROVEMENT\n- Always restate approved terms and ask for commitment in the same turn.\n## END MOCK IMPROVEMENT",
        )
        return _safe_json(
            {
                "new_prompt": improved,
                "changes_summary": [
                    "Added explicit objection-response + commitment cadence",
                    "Moved deadline emphasis earlier in negotiation flow",
                ],
                "expected_impact": "Higher conversion and clearer commitment transitions",
            }
        )

    # Stage 5 compliance (primary + review)
    if "compliance auditor for a conversational ai debt collection agent" in sys_lc:
        return _safe_json({"compliant": True, "risks": [], "reason": "Prompt is compliant"})
    if "senior compliance reviewer" in sys_lc:
        return _safe_json({"compliant": True, "risks": [], "reason": "Reviewed and compliant"})

    # Stage 6b narrative summary
    if "summarizing an a/b evaluation between two agent prompt versions" in sys_lc:
        return _safe_json(
            {
                "summary": "Mock A/B indicates measurable uplift in resolution with stable compliance.",
                "reason": "All statistical and practical gates passed in this offline simulation.",
            }
        )

    # Stage 7 hypothesis generation
    if "you are generating prompt patches for a debt collection agent" in sys_lc:
        return _safe_json(
            {
                "hypotheses": [
                    {
                        "hypothesis": "After each objection, restate exact approved terms and ask for commitment in one sentence.",
                        "why_it_might_work": "Reduces drift and improves closure rate",
                        "how_to_test": "Track commitment rate and turns-to-commit in the next run",
                    }
                ]
            }
        )

    # Fallback for any unexpected call shape
    return _safe_json({"summary": "ok", "reason": "mock fallback"})


async def run_demo(cfg: DemoConfig) -> None:
    # Keep A/B gates friendly for offline demo while still meaningful.
    os.environ.setdefault("AB_MIN_SAMPLE_PER_ARM", "10")
    os.environ.setdefault("AB_MIN_RESOLUTION_DELTA", "0.02")
    os.environ.setdefault("AB_MIN_COHEN_D", "0.20")

    transcripts = _make_mock_transcripts(cfg.n_transcripts, cfg.agent_name)

    original_llm = pip._llm
    pip._llm = _mock_llm
    try:
        run_id = "pipeline-mockdemo"
        run = await pip.run_improvement_pipeline(
            transcripts=transcripts,
            agent_name=cfg.agent_name,
            triggered_by="mock_demo",
            run_id=run_id,
        )

        print("\n=== Mock Pipeline Result ===")
        print(f"run_id: {run.run_id}")
        print(f"agent: {cfg.agent_name}")
        print(f"decision: {run.decision}")
        print(f"promote: {run.promote}")
        vc = run.version_comparison
        if vc:
            print(f"v1_rate: {vc.v1_resolution_rate}")
            print(f"v2_rate: {vc.v2_resolution_rate}")
            print(f"p_value: {vc.p_value}")
            print(f"reason: {vc.reason}")

        hypotheses = (run.hypothesis_set.hypotheses if run.hypothesis_set else [])
        print(f"hypothesis_count: {len(hypotheses)}")
        for i, h in enumerate(hypotheses, 1):
            print(f"  {i}. {h.hypothesis}")

        if cfg.apply_patch and hypotheses:
            result = await pip.apply_single_patch_async(
                agent_name=cfg.agent_name,
                patch_text=hypotheses[0].hypothesis,
                run_id=run.run_id,
                patch_index=0,
            )
            print("\n=== Patch Apply Result ===")
            print(json.dumps(result, indent=2))
            print(f"active_prompt_version: {pip.get_prompt_version(cfg.agent_name)}")
        elif cfg.apply_patch:
            print("\nNo hypothesis patches available to apply.")
    finally:
        pip._llm = original_llm


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline mocked improvement pipeline demo")
    parser.add_argument("--agent", default="ResolutionAgent", choices=["AssessmentAgent", "ResolutionAgent", "FinalNoticeAgent"])
    parser.add_argument("--n", type=int, default=40, help="Number of synthetic transcripts")
    parser.add_argument("--apply-patch", action="store_true", help="Apply first generated hypothesis patch")
    args = parser.parse_args()

    cfg = DemoConfig(
        agent_name=args.agent,
        n_transcripts=max(10, args.n),
        apply_patch=bool(args.apply_patch),
    )
    asyncio.run(run_demo(cfg))


if __name__ == "__main__":
    main()
