"""
Shared resolution metrics for pipeline runs.

Used by admin API (analytics convergence, test trend) and offline reports
(evolution_report) so displayed rates match the dashboard.
"""
from __future__ import annotations

from typing import Any, Optional


def normalize_prompt_version(version: str) -> str:
    """
    Collapse accidental double-prefixing from older pipeline writes
    (e.g. pipeline-pipeline-abc123 → pipeline-abc123).
    """
    raw = (version or "").strip()
    if not raw:
        return ""
    v = raw
    while v.startswith("pipeline-pipeline-"):
        v = v[len("pipeline-") :]
    return v


def trace_root(value: Any) -> str:
    return str(value or "").split(":", 1)[0]


def resolution_rate_from_scores(scores: Any) -> Optional[float]:
    """
    Resolution rate from a list of TranscriptScore-like dicts.
    Matches admin convergence + test-resolution-trend logic.
    """
    if not isinstance(scores, list) or not scores:
        return None
    try:
        valid = [s for s in scores if isinstance(s, dict) and "resolution" in s]
        if not valid:
            return None
        resolved = sum(1 for s in valid if int(s.get("resolution") or 0) == 1)
        return float(resolved / len(valid))
    except Exception:
        return None


def is_mock_pipeline_run(run: dict) -> bool:
    rid = str(run.get("run_id", "")).lower()
    trig = str(run.get("triggered_by", "")).lower()
    return rid.startswith("mock-") or ("mock" in rid) or ("mock" in trig)


def analytics_train_v1_v2_rates(run: dict) -> tuple[Optional[float], Optional[float]]:
    """
    v1/v2 rates used in Agent Analytics convergence (same as _build_convergence_payload).
    v2 prefers executed_v2_scores; v1 prefers transcript_scores + held_out_scores, else VC.
    """
    vc = run.get("version_comparison") or {}
    executed_scores = run.get("executed_v2_scores") or []
    v2_rate = resolution_rate_from_scores(executed_scores)
    if v2_rate is None:
        v2_raw = vc.get("v2_resolution_rate")
        if isinstance(v2_raw, (int, float)):
            v2_rate = float(v2_raw)
        else:
            v2_rate = None

    v1_scores = (run.get("transcript_scores") or []) + (run.get("held_out_scores") or [])
    v1_rate = resolution_rate_from_scores(v1_scores)
    if v1_rate is None:
        v1_raw = vc.get("v1_resolution_rate")
        if isinstance(v1_raw, (int, float)):
            v1_rate = float(v1_raw)
        else:
            v1_rate = None

    return v1_rate, v2_rate


def version_comparison_train_rates(run: dict) -> tuple[Optional[float], Optional[float]]:
    """Train-split rates from Fisher A/B object only (Pipeline tab score trend table)."""
    vc = run.get("version_comparison") or {}
    v1 = vc.get("v1_resolution_rate")
    v2 = vc.get("v2_resolution_rate")
    o1 = float(v1) if isinstance(v1, (int, float)) else None
    o2 = float(v2) if isinstance(v2, (int, float)) else None
    return o1, o2


def test_holdout_resolution_point(run: dict) -> Optional[dict[str, Any]]:
    """
    One point compatible with GET /pipeline/test-resolution-trend.
    Returns None if holdout rates cannot be derived.
    """
    if str(run.get("status") or "").lower() != "completed":
        return None

    test_before = run.get("test_v1_resolution_rate")
    test_after = run.get("test_v2_resolution_rate")
    test_delta = run.get("test_resolution_delta")

    if isinstance(test_before, (int, float)) and isinstance(test_after, (int, float)):
        if not isinstance(test_delta, (int, float)):
            test_delta = float(test_after) - float(test_before)
    elif not isinstance(test_before, (int, float)) or not isinstance(test_after, (int, float)):
        held_out = run.get("held_out_scores") or []
        executed = run.get("executed_v2_scores") or []
        held_out_ids = {trace_root(t.get("trace_id")) for t in held_out if isinstance(t, dict)}
        executed_test = [
            s for s in executed
            if isinstance(s, dict) and trace_root(s.get("trace_id")) in held_out_ids
        ]
        r_before = resolution_rate_from_scores(held_out)
        r_after = resolution_rate_from_scores(executed_test) if executed_test else resolution_rate_from_scores(executed)
        if r_before is None or r_after is None:
            return None
        test_before = r_before
        test_after = r_after
        test_delta = float(r_after - r_before)

    return {
        "run_id": run.get("run_id"),
        "agent_target": run.get("agent_target"),
        "started_at": run.get("started_at"),
        "decision": run.get("decision"),
        "test_v1_resolution_rate": round(float(test_before), 4),
        "test_v2_resolution_rate": round(float(test_after), 4),
        "test_resolution_delta": round(float(test_delta), 4),
    }


def build_convergence_payload(
    runs: list[dict[str, Any]],
    prompt_changes: list[dict[str, Any]],
    plateau_window: int = 3,
    plateau_delta_eps: float = 0.015,
    plateau_band_eps: float = 0.03,
    *,
    max_display_points: Optional[int] = 10,
) -> dict[str, Any]:
    """
    Same payload as Agent Analytics convergence charts (delta vs version / run).

    ``points`` are completed runs in chronological order with score-derived
    v1/v2 rates; ``version`` on each point comes from ``prompt_changes`` when
    the run_id matches an adoption/patch audit row.

    ``max_display_points`` (default 10) matches the Agent Analytics dashboard
    window; pass ``None`` to include every eligible run (e.g. evolution HTML).
    """
    version_by_run: dict[str, str] = {}
    for change in prompt_changes:
        rid = str(change.get("run_id") or "").strip()
        new_ver = normalize_prompt_version(str(change.get("new_version") or "").strip())
        if rid and new_ver:
            version_by_run[rid] = new_ver

    eligible: list[dict[str, Any]] = []

    for run in sorted(runs, key=lambda r: str(r.get("started_at") or "")):
        if str(run.get("status") or "").lower() != "completed":
            continue
        exec_mode = str(run.get("v2_execution_mode") or "").lower() or "unknown"

        v1_rate, v2_rate = analytics_train_v1_v2_rates(run)
        if v2_rate is None:
            continue
        if v1_rate is None:
            v1_rate = 0.0
        rid = str(run.get("run_id") or "")
        eligible.append({
            "run_id": rid,
            "started_at": run.get("started_at"),
            "version": version_by_run.get(rid, rid),
            "v2_resolution_rate": float(v2_rate),
            "v2_execution_mode": exec_mode,
            "v1_resolution_rate": float(v1_rate),
            "resolution_delta": float(v2_rate) - float(v1_rate),
            "decision": str(run.get("decision") or "pending"),
        })

    if not eligible:
        return {"status": "insufficient_data", "points": []}

    baseline_v1_rate = float(eligible[0].get("v1_resolution_rate", 0.0))

    if max_display_points is None or max_display_points <= 0:
        display_points = list(eligible)
    else:
        display_points = (
            eligible[-max_display_points:]
            if len(eligible) > max_display_points
            else list(eligible)
        )

    rates = [float(p["v2_resolution_rate"]) for p in display_points]
    for p in display_points:
        p["delta_from_v1"] = float(p["v2_resolution_rate"]) - baseline_v1_rate
    trend_non_decreasing = all(rates[i] >= rates[i - 1] - 1e-9 for i in range(1, len(rates)))
    stability_band = (max(rates) - min(rates)) if len(rates) > 1 else 0.0

    if len(eligible) >= plateau_window:
        tail = display_points[-plateau_window:] if len(display_points) >= plateau_window else display_points
        tail_rates = [float(p["v2_resolution_rate"]) for p in tail]
        plateau = (max(tail_rates) - min(tail_rates) <= plateau_band_eps)
    else:
        plateau = False

    return {
        "status": "ok",
        "points": display_points,
        "baseline_v1_rate": round(baseline_v1_rate, 4),
        "trend_non_decreasing": trend_non_decreasing,
        "stability_band": round(stability_band, 4),
        "plateau": plateau,
        "plateau_window": plateau_window,
        "plateau_delta_eps": plateau_delta_eps,
        "plateau_band_eps": plateau_band_eps,
    }
