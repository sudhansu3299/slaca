from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv

from src.data_layer import get_mongo

load_dotenv()


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_ts(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    return text


def _run_is_real(run: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    llm_calls = int(run.get("llm_calls") or 0)
    v2_mode = str(run.get("v2_execution_mode") or "").strip().lower()
    status = str(run.get("status") or "").strip().lower()

    if status != "completed":
        reasons.append(f"status={status or 'missing'}")
    if llm_calls <= 0:
        reasons.append("llm_calls<=0")
    if v2_mode != "real":
        reasons.append(f"v2_execution_mode={v2_mode or 'missing'}")

    return (len(reasons) == 0), reasons


def _run_summary(run: dict[str, Any]) -> dict[str, Any]:
    vc = run.get("version_comparison") or {}
    v1 = _safe_float(vc.get("v1_resolution_rate"), 0.0)
    v2 = _safe_float(vc.get("v2_resolution_rate"), 0.0)
    delta = round(v2 - v1, 4)
    return {
        "run_id": str(run.get("run_id") or ""),
        "agent_target": str(run.get("agent_target") or ""),
        "started_at": _parse_ts(run.get("started_at")),
        "completed_at": _parse_ts(run.get("completed_at")),
        "decision": str(run.get("decision") or ""),
        "llm_calls": int(run.get("llm_calls") or 0),
        "pipeline_cost_usd": round(_safe_float(run.get("pipeline_cost_usd"), 0.0), 6),
        "v2_execution_mode": str(run.get("v2_execution_mode") or ""),
        "v1_resolution_rate": round(v1, 4),
        "v2_resolution_rate": round(v2, 4),
        "resolution_delta": delta,
        "p_value": vc.get("p_value"),
    }


def _assess_agent_convergence(
    rows: list[dict[str, Any]],
    min_runs: int,
    window: int,
    min_avg_delta: float,
    stability_eps: float,
) -> dict[str, Any]:
    recent = rows[-window:] if len(rows) > window else rows[:]
    deltas = [float(r["resolution_delta"]) for r in recent]
    non_negative = all(d >= 0.0 for d in deltas)
    avg_delta = (sum(deltas) / len(deltas)) if deltas else 0.0
    stable = True
    if len(deltas) >= 2:
        for i in range(1, len(deltas)):
            if abs(deltas[i] - deltas[i - 1]) > stability_eps:
                stable = False
                break

    criteria = {
        "enough_runs": len(rows) >= min_runs,
        "non_negative_recent_deltas": non_negative,
        "recent_avg_delta_gte_threshold": avg_delta >= min_avg_delta,
        "delta_stability_within_eps": stable,
    }
    converged = all(criteria.values())
    return {
        "converged": converged,
        "criteria": criteria,
        "recent_window_size": len(recent),
        "recent_avg_delta": round(avg_delta, 4),
        "recent_deltas": [round(d, 4) for d in deltas],
    }


async def _run(args: argparse.Namespace) -> None:
    db = await get_mongo()
    if db is None:
        raise RuntimeError("MongoDB unavailable")

    query: dict[str, Any] = {"status": "completed"}
    if args.agent:
        query["agent_target"] = args.agent

    raw_runs = await db.eval_pipeline.find(query, {"_id": 0}).sort("started_at", 1).limit(args.limit).to_list(length=args.limit)

    real_rows_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    rejected_runs: list[dict[str, Any]] = []

    for run in raw_runs:
        summary = _run_summary(run)
        is_real, reasons = _run_is_real(run)
        summary["real_eligibility_reasons"] = reasons
        if is_real:
            real_rows_by_agent[summary["agent_target"]].append(summary)
        else:
            rejected_runs.append(summary)

    agents = sorted(real_rows_by_agent.keys())
    convergence: dict[str, Any] = {}
    for agent in agents:
        convergence[agent] = _assess_agent_convergence(
            rows=real_rows_by_agent[agent],
            min_runs=args.min_runs_per_agent,
            window=args.stability_window,
            min_avg_delta=args.min_avg_delta,
            stability_eps=args.stability_epsilon,
        )

    overall_pass = bool(convergence) and all(v.get("converged") for v in convergence.values())
    report = {
        "generated_at_utc": _iso_now(),
        "filters": {
            "agent": args.agent or "",
            "limit": args.limit,
            "required_status": "completed",
            "required_v2_execution_mode": "real",
            "required_llm_calls_gt": 0,
        },
        "thresholds": {
            "min_runs_per_agent": args.min_runs_per_agent,
            "stability_window": args.stability_window,
            "min_avg_delta": args.min_avg_delta,
            "stability_epsilon": args.stability_epsilon,
        },
        "counts": {
            "runs_scanned": len(raw_runs),
            "eligible_real_runs": sum(len(v) for v in real_rows_by_agent.values()),
            "excluded_runs": len(rejected_runs),
            "agents_with_real_runs": len(agents),
        },
        "overall_converged": overall_pass,
        "agents": convergence,
        "real_runs_by_agent": real_rows_by_agent,
        "excluded_runs": rejected_runs,
    }

    out_path = Path(args.output_json).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"WROTE {out_path}")
    print(f"RUNS_SCANNED {report['counts']['runs_scanned']}")
    print(f"ELIGIBLE_REAL_RUNS {report['counts']['eligible_real_runs']}")
    print(f"OVERALL_CONVERGED {report['overall_converged']}")
    if not report["overall_converged"]:
        print("CONVERGENCE_EVIDENCE_NOT_ESTABLISHED")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build convergence evidence report from live LLM eval_pipeline runs.",
    )
    parser.add_argument("--agent", default="", help="Optional agent filter")
    parser.add_argument("--limit", type=int, default=300, help="Max completed runs to scan")
    parser.add_argument("--min-runs-per-agent", type=int, default=3, help="Minimum live runs required per agent")
    parser.add_argument("--stability-window", type=int, default=3, help="Recent run window for stability checks")
    parser.add_argument("--min-avg-delta", type=float, default=0.02, help="Minimum average delta in recent window")
    parser.add_argument("--stability-epsilon", type=float, default=0.05, help="Allowed delta change between consecutive runs")
    parser.add_argument(
        "--output-json",
        default="artifacts/live_convergence_report.json",
        help="Output JSON report path",
    )
    args = parser.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
