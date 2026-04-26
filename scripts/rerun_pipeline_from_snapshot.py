from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import argparse
import asyncio
import json
from datetime import datetime, timezone

from src.data_layer import get_mongo
from src.self_learning.improvement_pipeline import run_improvement_pipeline


def _load_transcripts(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("transcripts file must be a JSON array")
    return data


async def _run(args: argparse.Namespace) -> None:
    transcripts = _load_transcripts(Path(args.transcripts_file))
    if not transcripts:
        raise ValueError("transcripts file contains no conversations")

    db = await get_mongo()
    if db is None:
        raise RuntimeError("MongoDB unavailable")

    run_id = args.run_id.strip()
    if not run_id:
        raise ValueError("run_id is required")

    exists = await db.eval_pipeline.find_one({"run_id": run_id}, {"_id": 1})
    if exists and not args.allow_overwrite_run_id:
        raise RuntimeError(
            f"run_id '{run_id}' already exists. "
            "Pass --allow-overwrite-run-id to overwrite."
        )

    run = await run_improvement_pipeline(
        transcripts=transcripts,
        agent_name=args.agent_name,
        triggered_by=args.triggered_by,
        run_id=run_id,
    )

    out = {
        "run_id": run.run_id,
        "agent_target": run.agent_target,
        "status": run.status,
        "decision": run.decision,
        "started_at": run.started_at,
        "completed_at": run.completed_at,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if args.output_json:
        Path(args.output_json).write_text(
            json.dumps(out, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    print(json.dumps(out, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rerun improvement pipeline from a transcript snapshot.",
    )
    parser.add_argument("--agent-name", required=True, help="AssessmentAgent | ResolutionAgent | FinalNoticeAgent")
    parser.add_argument("--run-id", required=True, help="Run ID to use (seed is derived from this)")
    parser.add_argument("--transcripts-file", required=True, help="Path to JSON transcripts snapshot")
    parser.add_argument(
        "--triggered-by",
        default="repro_bundle_rerun",
        help="Value stored in run.triggered_by",
    )
    parser.add_argument(
        "--allow-overwrite-run-id",
        action="store_true",
        help="Allow rerun to overwrite existing run_id document in eval_pipeline",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Optional path to write compact rerun metadata JSON",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
