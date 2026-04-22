"""
Test Harness — generate & evaluate conversations locally (no Temporal needed).

Use this for fast iteration:
  python -m src.test_harness                              # 1 cooperative, mocked LLM
  python -m src.test_harness --persona hostile            # hostile, mocked
  python -m src.test_harness --all                        # all 4 personas, mocked
  python -m src.test_harness --all --runs 3               # 3 runs per persona
  python -m src.test_harness --persona cooperative --live         # real Claude
  python -m src.test_harness --persona cooperative --live --verbose  # full audit trace

Tweaking:
  --personas-file personas/my_script.json   # override borrower lines
  --prompt-overrides prompts/my_tweaks.json # inject extra prompt guidance
  --log-dir audit-logs                      # where to write .log / .jsonl (default)
  --no-log                                  # don't write files (stdout only)

Outputs:
  - Per-turn audit trace (when --verbose)
  - Per-stage eval report
  - Aggregate observability metrics (recovery / commitment / drop-off / repetition)
  - audit-logs/<run-id>.log   human-readable turn trace
  - audit-logs/<run-id>.jsonl structured JSONL (one event per line)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import asdict
from typing import Optional

# Load .env before any API client is constructed
from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("SKIP_LLM_JUDGE", "1")

from unittest.mock import patch

from pathlib import Path

from src.audit_logger import AuditLogger
from src.cost import AGENT_MODEL
from src.evaluator import RuleBasedEvaluator
from src.persona_override import OverrideablePersonaScript
from src.pipeline import CollectionsPipeline
from src.prompt_override import apply_prompt_overrides, load_prompt_overrides
from src.self_learning.loop import LearningHistory, build_run_metrics
from src.simulation import PersonaScript, PersonaType, SimulationEngine
from src.token_budget import TokenUsage


def mock_claude(responses: list[str]):
    it = iter(responses)
    async def _mock(*a, **kw):
        try:
            text = next(it)
        except StopIteration:
            text = "Please continue."
        return text, TokenUsage(input_tokens=300, output_tokens=60)
    return _mock


async def run_one(
    persona: PersonaType,
    run_idx: int = 0,
    live: bool = False,
    personas_file: Optional[Path] = None,
    prompt_overrides_file: Optional[Path] = None,
    log_dir: Optional[Path] = None,
    no_log: bool = False,
) -> dict:
    profile = next(p for p in SimulationEngine.make_profiles() if p.persona == persona)
    pipeline = CollectionsPipeline()

    # Apply prompt overrides (additive — each agent's _injected_guidance)
    if prompt_overrides_file:
        overrides = load_prompt_overrides(prompt_overrides_file)
        apply_prompt_overrides(pipeline, overrides)

    borrower = OverrideablePersonaScript(persona, personas_file)
    llm = SimulationEngine.get_mock_llm_responses(persona)
    evaluator = RuleBasedEvaluator()

    run_id = f"{profile.borrower_id}-r{run_idx}"
    audit = None if no_log else AuditLogger(run_id, log_dir)

    try:
        if audit:
            audit.log_event("run_config", {
                "persona": persona.value,
                "live": live,
                "personas_file": str(personas_file) if personas_file else None,
                "prompt_overrides_file": str(prompt_overrides_file) if prompt_overrides_file else None,
                "model": AGENT_MODEL,
                "profile": profile.borrower_id,
            })

        if live:
            # Real Claude — no patching
            result = await pipeline.run(
                borrower_id=run_id,
                loan_id=profile.loan_id,
                principal_amount=profile.principal_amount,
                outstanding_amount=profile.outstanding_amount,
                days_past_due=profile.days_past_due,
                input_provider=lambda ctx, agent: borrower.respond(agent),
                max_turns_per_stage=8,
            )
        else:
            with patch.object(pipeline.assessment_agent, "_call_claude",
                              side_effect=mock_claude(llm.get("AssessmentAgent", []))), \
                 patch.object(pipeline.resolution_agent, "_call_claude",
                              side_effect=mock_claude(llm.get("ResolutionAgent", []))), \
                 patch.object(pipeline.final_notice_agent, "_call_claude",
                              side_effect=mock_claude(llm.get("FinalNoticeAgent", []))):
                result = await pipeline.run(
                    borrower_id=run_id,
                    loan_id=profile.loan_id,
                    principal_amount=profile.principal_amount,
                    outstanding_amount=profile.outstanding_amount,
                    days_past_due=profile.days_past_due,
                    input_provider=lambda ctx, agent: borrower.respond(agent),
                    max_turns_per_stage=8,
                )

        # Evaluate each stage
        stage_reports = {}
        for agent_name, stage in [
            ("AssessmentAgent", "assessment"),
            ("ResolutionAgent", "resolution"),
            ("FinalNoticeAgent", "final_notice"),
        ]:
            stage_history = [m for m in result.conversation if m.get("stage") == stage]
            if stage_history:
                r = evaluator.evaluate(agent_name, stage_history)
                stage_reports[agent_name] = r

        # Write audit log (turns + stage summaries + final report)
        if audit:
            _write_audit_events(audit, result.conversation, stage_reports, pipeline.cost_tracker)

        out = {
            "persona": persona.value,
            "profile": profile.borrower_id,
            "run_id": run_id,
            "outcome": result.outcome,
            "final_stage": result.final_stage,
            "total_turns": result.total_turns,
            "handoff_tokens": result.handoff_tokens,
            "cost_usd": result.total_cost_usd,
            "conversation": result.conversation,
            "cost_tracker": pipeline.cost_tracker,
            "pipeline": pipeline,
            "audit_log_path": str(audit.log_path) if audit else None,
            "audit_jsonl_path": str(audit.jsonl_path) if audit else None,
            "eval": {a: {
                "overall_score": r.overall_score,
                "passed": r.overall_passed,
                "checks": [{"name": c.check_name, "passed": c.passed, "score": c.score,
                           "detail": c.detail}
                          for c in r.checks],
            } for a, r in stage_reports.items()},
        }

        if audit:
            audit.log_final_report({
                "outcome": out["outcome"],
                "final_stage": out["final_stage"],
                "total_turns": out["total_turns"],
                "handoff_tokens": out["handoff_tokens"],
                "cost_usd": out["cost_usd"],
                "eval": out["eval"],
            })

        return out
    finally:
        if audit:
            audit.close()


def _write_audit_events(audit, conversation, stage_reports, cost_tracker) -> None:
    """Mirror verbose stdout into the audit .log plus structured .jsonl events."""
    stages = ["assessment", "resolution", "final_notice"]
    stage_titles = {
        "assessment":   "STAGE 1: ASSESSMENT (chat)",
        "resolution":   "STAGE 2: RESOLUTION (voice)",
        "final_notice": "STAGE 3: FINAL NOTICE (chat)",
    }
    agent_map = {
        "assessment":   "AssessmentAgent",
        "resolution":   "ResolutionAgent",
        "final_notice": "FinalNoticeAgent",
    }

    audit.write_line("═" * 72)
    audit.write_line(f"RUN {audit.run_id}")
    audit.write_line("═" * 72)

    for stage in stages:
        stage_msgs = [m for m in conversation if m.get("stage") == stage]
        if not stage_msgs:
            continue
        audit.write_line(f"\n┌─ {stage_titles[stage]} " + "─" * 40)

        turn_num = 0
        i = 0
        while i < len(stage_msgs):
            if stage_msgs[i].get("role") == "user":
                turn_num += 1
                borrower = stage_msgs[i].get("content", "")
                agent_msg = ""
                advanced = False
                tokens = 0
                if i + 1 < len(stage_msgs) and stage_msgs[i + 1].get("role") == "assistant":
                    agent_msg = stage_msgs[i + 1].get("content", "")
                    advanced = stage_msgs[i + 1].get("advanced", False)
                    tokens = stage_msgs[i + 1].get("tokens", 0)
                    i += 2
                else:
                    i += 1
                marker = " ✓ ADVANCED" if advanced else ""
                audit.write_line(f"│ t{turn_num} │ Borrower: {borrower}")
                audit.write_line(f"│    │ Agent:    {agent_msg}")
                audit.write_line(f"│    │ out_tokens={tokens}{marker}")
                audit.log_turn(
                    stage=stage, turn=turn_num,
                    borrower=borrower, agent_msg=agent_msg,
                    advanced=advanced, tokens_out=tokens,
                )
            else:
                i += 1

        eval_r = stage_reports.get(agent_map[stage])
        if eval_r:
            audit.write_line(f"│ Stage eval: "
                            f"{'PASS' if eval_r.overall_passed else 'FAIL'} "
                            f"(score={eval_r.overall_score:.2f})")
            for c in eval_r.checks:
                mark = "✓" if c.passed else "✗"
                audit.write_line(f"│   {mark} {c.check_name:22} {c.score:.2f} — {c.detail}")
            audit.log_stage_summary(stage, {
                "passed": eval_r.overall_passed,
                "score": eval_r.overall_score,
                "checks": [{"name": c.check_name, "passed": c.passed,
                           "score": c.score, "detail": c.detail}
                          for c in eval_r.checks],
            })
        audit.write_line("└" + "─" * 71)

    audit.write_line("")
    audit.write_line("── Cost breakdown ──")
    for agent_name, usage in cost_tracker.per_agent.items():
        audit.write_line(
            f"  {agent_name}: {usage.input_tokens} in / "
            f"{usage.output_tokens} out = ${usage.cost_usd:.4f}"
        )
    audit.write_line(f"  TOTAL: ${cost_tracker.total_cost_usd:.4f}")


def print_verbose_trace(run: dict, live: bool) -> None:
    """Option A audit format — per-turn trace + per-stage summary."""
    print()
    print("═" * 72)
    print(f"{'LIVE' if live else 'MOCK'} RUN — {run['persona']} ({run['profile']})")
    print(f"model={AGENT_MODEL}")
    print("═" * 72)

    conv = run.get("conversation", [])
    stages = ["assessment", "resolution", "final_notice"]
    stage_titles = {
        "assessment":   "STAGE 1: ASSESSMENT (chat)",
        "resolution":   "STAGE 2: RESOLUTION (voice)",
        "final_notice": "STAGE 3: FINAL NOTICE (chat)",
    }

    for stage in stages:
        stage_msgs = [m for m in conv if m.get("stage") == stage]
        if not stage_msgs:
            continue
        print(f"\n┌─ {stage_titles[stage]} " + "─" * (50 - len(stage_titles[stage])))

        # Pair up user/assistant turns
        turn_num = 0
        i = 0
        while i < len(stage_msgs):
            if stage_msgs[i].get("role") == "user":
                turn_num += 1
                borrower = stage_msgs[i].get("content", "")
                agent_msg = ""
                advanced = False
                tokens = 0
                if i + 1 < len(stage_msgs) and stage_msgs[i + 1].get("role") == "assistant":
                    agent_msg = stage_msgs[i + 1].get("content", "")
                    advanced = stage_msgs[i + 1].get("advanced", False)
                    tokens = stage_msgs[i + 1].get("tokens", 0)
                    i += 2
                else:
                    i += 1
                marker = " ✓ ADVANCED" if advanced else ""
                print(f"│ t{turn_num} │ Borrower: {borrower}")
                print(f"│    │ Agent:    {agent_msg}")
                print(f"│    │ out_tokens={tokens}{marker}")
            else:
                i += 1

        # Stage eval summary
        agent_name = {
            "assessment":   "AssessmentAgent",
            "resolution":   "ResolutionAgent",
            "final_notice": "FinalNoticeAgent",
        }[stage]
        eval_data = run["eval"].get(agent_name)
        if eval_data:
            status = "PASS" if eval_data["passed"] else "FAIL"
            print(f"│")
            print(f"│ Stage eval: {status} (score={eval_data['overall_score']:.2f})")
            for c in eval_data["checks"]:
                mark = "✓" if c["passed"] else "✗"
                detail = f" — {c['detail']}" if c.get("detail") else ""
                print(f"│   {mark} {c['name']:22} {c['score']:.2f}{detail}")
        print(f"└" + "─" * 71)


def print_observability_report(runs: list[dict], history: LearningHistory) -> None:
    """Milestone-aligned observability metrics."""
    print()
    print("═" * 72)
    print("OBSERVABILITY METRICS")
    print("═" * 72)

    total_cost = 0.0
    for r in runs:
        total_cost += r.get("cost_usd", 0.0)

    # Pick a single-run focus if only one
    if len(runs) == 1:
        r = runs[0]
        print(f"Pipeline outcome         : {r['outcome']}")
        print(f"Final stage              : {r['final_stage']}")
        print(f"Total turns              : {r['total_turns']}")
        print(f"Handoff tokens           : {r['handoff_tokens']}  (per-stage; limit 500)")
        tracker = r.get("cost_tracker")
        if tracker:
            print(f"\n── Cost breakdown ─────────────────────────────────────────────────────")
            for agent_name, usage in tracker.per_agent.items():
                print(f"  {agent_name:22} : {usage.input_tokens:,} in / {usage.output_tokens:,} out = ${usage.cost_usd:.4f}")
            print(f"  {'TOTAL':22} : ${tracker.total_cost_usd:.4f} / $20.00 budget")
            print(f"  {'REMAINING':22} : ${tracker.budget_remaining():.4f}")

        print(f"\n── Rule-based evaluation ──────────────────────────────────────────────")
        for agent_name, eval_data in r["eval"].items():
            status = "PASS" if eval_data["passed"] else "FAIL"
            print(f"  {agent_name:22} : {status}  (score={eval_data['overall_score']:.2f})")
            for c in eval_data["checks"]:
                mark = "✓" if c["passed"] else "✗"
                print(f"    {mark} {c['name']:22} {c['score']:.2f}")

    print(f"\n── Self-learning metrics ─────────────────────────────────────────────")
    print(f"  Recovery rate          : {history.average_recovery_rate:.1%}")
    print(f"  Commitment success     : {history.average_commitment_rate:.1%}")
    print(f"  Drop-off distribution  : {history.drop_off_distribution or '{}'}")
    print(f"  Repetition errors      : {history.total_repetition_errors}")
    print(f"  Total runs             : {len(history.run_metrics)}")

    if len(runs) > 1:
        print(f"\n── Cost summary (all runs) ───────────────────────────────────────────")
        print(f"  Total: ${total_cost:.4f} / $20.00 budget   remaining: ${20 - total_cost:.4f}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", default="cooperative",
                        choices=[p.value for p in PersonaType])
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--runs", type=int, default=1,
                        help="Runs per persona (for aggregates)")
    parser.add_argument("--live", action="store_true",
                        help="Use real Claude API instead of mocks")
    parser.add_argument("--verbose", action="store_true",
                        help="Print each conversation turn")
    parser.add_argument("--json", action="store_true", help="Output JSON only")
    parser.add_argument("--personas-file", type=Path, default=None,
                        help="JSON file overriding borrower script per agent")
    parser.add_argument("--prompt-overrides", type=Path, default=None,
                        help="JSON file injecting extra guidance into agent prompts")
    parser.add_argument("--log-dir", type=Path, default=Path("audit-logs"),
                        help="Directory for audit .log and .jsonl files")
    parser.add_argument("--no-log", action="store_true",
                        help="Skip writing audit files (stdout only)")
    args = parser.parse_args()

    if args.live and not (
        os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("OPENCODE_API_KEY")
        or os.getenv("OPENAI_API_KEY")
    ):
        print(
            "ERROR: --live requires ANTHROPIC_API_KEY, OPENCODE_API_KEY, or OPENAI_API_KEY in env or .env",
            file=sys.stderr,
        )
        sys.exit(1)

    personas = list(PersonaType) if args.all else [PersonaType(args.persona)]
    history = LearningHistory()
    all_runs = []

    for persona in personas:
        for i in range(args.runs):
            r = await run_one(
                persona, i,
                live=args.live,
                personas_file=args.personas_file,
                prompt_overrides_file=args.prompt_overrides,
                log_dir=args.log_dir,
                no_log=args.no_log,
            )
            all_runs.append(r)

            metrics = build_run_metrics(
                run_id=f"{persona.value}-r{i}",
                outcome=r["outcome"],
                conversation_history=r.get("conversation", []),
                cost_usd=r["cost_usd"],
                eval_results=[],
            )
            history.add_run(metrics)

    if args.json:
        # Strip non-serialisable fields
        runs_serialisable = [
            {k: v for k, v in r.items() if k not in ("cost_tracker", "pipeline")}
            for r in all_runs
        ]
        print(json.dumps({
            "runs": runs_serialisable,
            "aggregate": {
                "total_runs": len(history.run_metrics),
                "avg_recovery_rate": history.average_recovery_rate,
                "avg_commitment_rate": history.average_commitment_rate,
                "total_repetition_errors": history.total_repetition_errors,
                "drop_off_distribution": history.drop_off_distribution,
            },
        }, indent=2, default=str))
        return

    # Verbose turn-by-turn
    if args.verbose:
        for r in all_runs:
            print_verbose_trace(r, live=args.live)

    # Compact multi-run summary (shown only when >1 run or not verbose)
    if not args.verbose or len(all_runs) > 1:
        print()
        print("═" * 72)
        print(f"Test Harness — {len(all_runs)} run(s) across {len(personas)} persona(s)")
        print("═" * 72)
        for r in all_runs:
            print(f"\n── {r['persona']} ({r['profile']})")
            print(f"   outcome={r['outcome']:12} stage={r['final_stage']:14} "
                  f"turns={r['total_turns']} cost=${r['cost_usd']:.4f}")
            print(f"   handoff_tokens={r['handoff_tokens']}")
            for agent, eval_data in r["eval"].items():
                status = "PASS" if eval_data["passed"] else "FAIL"
                print(f"   [{agent:19}] {status} (score={eval_data['overall_score']:.2f})")
                for c in eval_data["checks"]:
                    mark = "✓" if c["passed"] else "✗"
                    print(f"     {mark} {c['name']:22} {c['score']:.2f}")

    # Observability report
    print_observability_report(all_runs, history)

    # Audit log file paths
    log_paths = [r for r in all_runs if r.get("audit_log_path")]
    if log_paths:
        print()
        print("── Audit log files ──")
        for r in log_paths:
            print(f"  {r['persona']:22} {r['audit_log_path']}")
            print(f"  {'':22} {r['audit_jsonl_path']}")


if __name__ == "__main__":
    asyncio.run(main())
