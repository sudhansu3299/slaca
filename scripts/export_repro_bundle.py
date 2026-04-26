from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import argparse
import asyncio
import csv
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from typing import Any

from src.data_layer import get_mongo
from dotenv import load_dotenv


ENV_KEYS = [
    "MONGODB_URL",
    "MONGODB_DB",
    "OPENAI_BASE_URL",
    "OPENCODE_BASE_URL",
    "AGENT_MODEL",
    "EVAL_MODEL",
    "EVAL_TEMP",
    "PROMPT_GENERATION_MODEL",
    "PROMPT_GENERATION_TEMP",
    "REAL_V2_EXECUTION_MODE",
    "REPLAY_BORROWER_MODE",
]

load_dotenv()


def _flatten_scores(run_doc: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bucket in ("transcript_scores", "held_out_scores", "executed_v2_scores"):
        for s in run_doc.get(bucket, []) or []:
            rows.append(
                {
                    "bucket": bucket,
                    "trace_id": s.get("trace_id"),
                    "borrower_id": s.get("borrower_id"),
                    "agent_name": s.get("agent_name"),
                    "resolution": s.get("resolution"),
                    "resolution_confidence": s.get("resolution_confidence"),
                    "debt_collected": s.get("debt_collected"),
                    "compliance_violation": s.get("compliance_violation"),
                    "compliance_reason": s.get("compliance_reason"),
                    "tone_score": s.get("tone_score"),
                    "next_step_clarity": s.get("next_step_clarity"),
                    "raw_transcript_turns": s.get("raw_transcript_turns"),
                }
            )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def _seed_from_run_id(run_id: str) -> int:
    return int(hashlib.md5(run_id.encode("utf-8")).hexdigest(), 16) % (2**32)


async def _build_transcripts_from_interactions(db, trace_ids: list[str]) -> list[dict[str, Any]]:
    if not trace_ids:
        return []
    transcripts: list[dict[str, Any]] = []
    for trace_id in sorted(set(trace_ids)):
        turns = await db.interactions.find(
            {"trace_id": trace_id},
            {"_id": 0, "borrower_id": 1, "agent_name": 1, "input": 1, "output": 1, "timestamp": 1, "decision": 1},
        ).sort("timestamp", 1).to_list(length=1000)
        if not turns:
            continue
        borrower_id = str(turns[0].get("borrower_id", ""))
        history: list[dict[str, Any]] = []
        agent_counts: dict[str, int] = {}
        for t in turns:
            agent = str(t.get("agent_name", ""))
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
            history.append(
                {
                    "role": "user",
                    "content": t.get("input", ""),
                    "timestamp": t.get("timestamp", ""),
                }
            )
            history.append(
                {
                    "role": "assistant",
                    "content": t.get("output", ""),
                    "timestamp": t.get("timestamp", ""),
                    "advanced": t.get("decision") in ("advance", "committed", "resolved"),
                }
            )
        primary_agent = max(agent_counts, key=agent_counts.get) if agent_counts else ""
        transcripts.append(
            {
                "trace_id": trace_id,
                "borrower_id": borrower_id,
                "primary_agent": primary_agent,
                "history": history,
            }
        )
    return transcripts


async def _run(args: argparse.Namespace) -> None:
    db = await get_mongo()
    if db is None:
        raise RuntimeError("MongoDB unavailable")

    run_id = args.run_id.strip()
    run_doc = await db.eval_pipeline.find_one({"run_id": run_id}, {"_id": 0})
    if not run_doc:
        raise RuntimeError(f"run_id '{run_id}' not found in eval_pipeline")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_doc_path = out_dir / "run_doc.json"
    run_doc_path.write_text(json.dumps(run_doc, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = _flatten_scores(run_doc)
    (out_dir / "per_conversation_scores.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(out_dir / "per_conversation_scores.csv", rows)

    trace_ids = [str(r.get("trace_id") or "") for r in rows if r.get("trace_id")]
    transcripts = await _build_transcripts_from_interactions(db, trace_ids)
    (out_dir / "transcripts.json").write_text(
        json.dumps(transcripts, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if args.export_interactions:
        interactions = await db.interactions.find(
            {"trace_id": {"$in": sorted(set(trace_ids))}},
            {"_id": 0},
        ).sort("timestamp", 1).to_list(length=50000)
        (out_dir / "interactions.json").write_text(
            json.dumps(interactions, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    seed = _seed_from_run_id(run_id)
    env_snapshot = {k: os.getenv(k, "") for k in ENV_KEYS}
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            cwd=Path(__file__).resolve().parents[1],
        ).strip()
    except Exception:
        git_sha = ""
    repro_config = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": run_id,
        "agent_target": run_doc.get("agent_target"),
        "triggered_by": run_doc.get("triggered_by"),
        "git_commit": git_sha,
        "pipeline_split_seed": seed,
        "seed_formula": "int(md5(run_id).hexdigest(), 16) % (2**32)",
        "env": env_snapshot,
        "files": {
            "run_doc_json": str(run_doc_path),
            "scores_json": str(out_dir / "per_conversation_scores.json"),
            "scores_csv": str(out_dir / "per_conversation_scores.csv"),
            "transcripts_json": str(out_dir / "transcripts.json"),
        },
    }
    (out_dir / "repro_config.json").write_text(
        json.dumps(repro_config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    rerun_cmd = (
        f".venv/bin/python scripts/rerun_pipeline_from_snapshot.py "
        f"--agent-name {run_doc.get('agent_target')} "
        f"--run-id {run_id} "
        f"--transcripts-file {out_dir / 'transcripts.json'} "
        f"--allow-overwrite-run-id "
        f"--output-json {out_dir / 'rerun_result.json'}"
    )
    (out_dir / "RERUN_COMMAND.txt").write_text(rerun_cmd + "\n", encoding="utf-8")

    rerun_sh = out_dir / "RERUN.sh"
    rerun_sh.write_text(
        "#!/usr/bin/env bash\nset -euo pipefail\n"
        f"cd \"{Path(__file__).resolve().parents[1]}\"\n"
        + rerun_cmd
        + "\n",
        encoding="utf-8",
    )
    os.chmod(rerun_sh, 0o755)

    print(f"WROTE {out_dir}")
    print(f"RUN_ID {run_id}")
    print(f"AGENT {run_doc.get('agent_target')}")
    print(f"SEED {seed}")
    print("RERUN_COMMAND")
    print(rerun_cmd)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export reproducibility bundle for one eval_pipeline run.")
    parser.add_argument("--run-id", required=True, help="run_id from eval_pipeline")
    parser.add_argument("--output-dir", required=True, help="Output directory for bundle artifacts")
    parser.add_argument(
        "--export-interactions",
        action="store_true",
        help="Also export raw interaction rows for involved trace_ids",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
