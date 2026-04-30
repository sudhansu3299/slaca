#!/usr/bin/env python3
"""
Build a self-learning evolution report: JSON + HTML with charts.

Includes ``convergence`` payload (delta vs prompt version / run_id) matching
Agent Analytics charts; evolution uses all eligible runs (dashboard uses last 10).

Train resolution uses the same score-derived rules as Agent Analytics convergence
(executed_v2_scores for v2; transcript_scores + held_out_scores for v1; falls back
to version_comparison when lists are missing). Test/holdout uses the same logic
as GET /api/admin/pipeline/test-resolution-trend. Optional pipeline_ab_* fields
match the Pipeline tab "score trend" table (version_comparison only).

Data sources:
  - mongo (default): eval_pipeline collection, completed + running optional
  - artifacts: glob of run_doc.json (e.g. artifacts/repro-*/run_doc.json)

Example:
  .venv/bin/python scripts/evolution_report.py --limit 400
  .venv/bin/python scripts/evolution_report.py --source artifacts --glob 'artifacts/repro-*/run_doc.json'
"""
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


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _extract_point(run: dict[str, Any]) -> dict[str, Any]:
    from src.pipeline_report_metrics import (
        analytics_train_v1_v2_rates,
        test_holdout_resolution_point,
        version_comparison_train_rates,
    )

    vc = run.get("version_comparison") or {}
    status = str(run.get("status") or "")
    completed = status.lower() == "completed"

    a_v1, a_v2 = analytics_train_v1_v2_rates(run) if completed else (None, None)
    train_delta = None
    if a_v1 is not None and a_v2 is not None:
        train_delta = round(float(a_v2) - float(a_v1), 4)

    pvc_v1, pvc_v2 = version_comparison_train_rates(run)
    pvc_delta = None
    if pvc_v1 is not None and pvc_v2 is not None:
        pvc_delta = round(float(pvc_v2) - float(pvc_v1), 4)

    test_pt = test_holdout_resolution_point(run) if completed else None
    test_v1 = test_v2 = test_delta = None
    if test_pt:
        test_v1 = float(test_pt["test_v1_resolution_rate"])
        test_v2 = float(test_pt["test_v2_resolution_rate"])
        test_delta = float(test_pt["test_resolution_delta"])

    p_val = vc.get("p_value")
    p_float = _safe_float(p_val) if p_val is not None else None

    decision = str(run.get("decision") or vc.get("decision") or "pending").lower()
    cost_raw = _safe_float(run.get("pipeline_cost_usd"))

    ts = run.get("transcript_scores") or []
    hs = run.get("held_out_scores") or []
    transcript_n = len(ts) + len(hs) if isinstance(ts, list) and isinstance(hs, list) else 0

    return {
        "run_id": str(run.get("run_id") or ""),
        "started_at": str(run.get("started_at") or ""),
        "completed_at": str(run.get("completed_at") or ""),
        "status": status,
        "decision": decision,
        "triggered_by": str(run.get("triggered_by") or ""),
        "train_v1_pct": round(a_v1 * 100, 2) if a_v1 is not None else None,
        "train_v2_pct": round(a_v2 * 100, 2) if a_v2 is not None else None,
        "train_delta_pct": round(train_delta * 100, 2) if train_delta is not None else None,
        "pipeline_ab_v1_pct": round(pvc_v1 * 100, 2) if pvc_v1 is not None else None,
        "pipeline_ab_v2_pct": round(pvc_v2 * 100, 2) if pvc_v2 is not None else None,
        "pipeline_ab_delta_pct": round(pvc_delta * 100, 2) if pvc_delta is not None else None,
        "test_v1_pct": round(test_v1 * 100, 2) if test_v1 is not None else None,
        "test_v2_pct": round(test_v2 * 100, 2) if test_v2 is not None else None,
        "test_delta_pct": round(test_delta * 100, 2) if test_delta is not None else None,
        "p_value": p_float,
        "v2_execution_mode": str(run.get("v2_execution_mode") or ""),
        "llm_calls": int(run.get("llm_calls") or 0),
        "pipeline_cost_usd": round(cost_raw if cost_raw is not None else 0.0, 6),
        "train_n": int(run.get("train_transcript_count") or 0) or transcript_n,
        "test_n": int(run.get("test_transcript_count") or 0),
        "transcript_count": transcript_n,
    }


def _load_runs_from_artifacts(glob_pattern: str) -> list[dict[str, Any]]:
    root = ROOT_DIR
    paths = sorted(root.glob(glob_pattern))
    runs: list[dict[str, Any]] = []
    for p in paths:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data.get("run_id"):
                runs.append(data)
        except (OSError, json.JSONDecodeError):
            continue
    runs.sort(key=lambda r: str(r.get("started_at") or r.get("completed_at") or ""))
    return runs


async def _load_prompt_changes_by_agent() -> dict[str, list[dict[str, Any]]]:
    from dotenv import load_dotenv

    load_dotenv()
    from src.data_layer import get_mongo

    db = await get_mongo()
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if db is None:
        return out
    docs = await db.prompt_changes.find({}, {"_id": 0}).sort("timestamp", 1).to_list(8000)
    for doc in docs:
        ag = str(doc.get("agent_name") or "").strip()
        if ag:
            out[ag].append(doc)
    return out


async def _load_runs_from_mongo(
    *,
    agent: str,
    limit: int,
    include_running: bool,
) -> list[dict[str, Any]]:
    from dotenv import load_dotenv

    load_dotenv()
    from src.data_layer import get_mongo

    db = await get_mongo()
    if db is None:
        raise RuntimeError("MongoDB unavailable (set MONGODB_URL or use --source artifacts)")

    filt: dict[str, Any] = {}
    if agent:
        filt["agent_target"] = agent
    if not include_running:
        filt["status"] = "completed"

    cursor = db.eval_pipeline.find(filt, {"_id": 0}).sort("started_at", 1).limit(limit)
    return await cursor.to_list(length=limit)


def _build_series(
    runs: list[dict[str, Any]],
    agent: str,
    *,
    include_mock: bool,
    prompt_changes_by_agent: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    from src.pipeline_report_metrics import build_convergence_payload, is_mock_pipeline_run

    by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        if not include_mock and is_mock_pipeline_run(run):
            continue
        ag = str(run.get("agent_target") or "unknown")
        if agent and ag != agent:
            continue
        by_agent[ag].append(run)

    agents_out: dict[str, Any] = {}
    for ag, rows in sorted(by_agent.items()):
        rows_sorted = sorted(
            rows,
            key=lambda r: str(r.get("started_at") or r.get("completed_at") or ""),
        )
        points = []
        for i, run in enumerate(rows_sorted, start=1):
            pt = _extract_point(run)
            pt["iteration"] = i
            points.append(pt)
        conv = build_convergence_payload(
            rows_sorted,
            prompt_changes_by_agent.get(ag, []),
            max_display_points=None,
        )
        agents_out[ag] = {
            "run_count": len(points),
            "points": points,
            "convergence": conv,
        }
    return agents_out


def _render_html(report: dict[str, Any]) -> str:
    json_blob = json.dumps(report, ensure_ascii=False).replace("</", "<\\/")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Evolution report — self-learning pipeline</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
  <style>
    :root {{
      --bg: #0c0f14;
      --panel: #151a22;
      --text: #e8eaed;
      --muted: #8b939e;
      --accent: #3dd6c3;
      --v1: #6b9fff;
      --v2: #7ee787;
      --reject: #f85149;
      --adopt: #3fb950;
      --pending: #d29922;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; font-family: ui-sans-serif, system-ui, sans-serif;
      background: var(--bg); color: var(--text); line-height: 1.5;
    }}
    header {{
      padding: 1.5rem 2rem; border-bottom: 1px solid #252b36;
      background: linear-gradient(180deg, #121820 0%, var(--bg) 100%);
    }}
    header h1 {{ margin: 0 0 0.35rem; font-size: 1.35rem; font-weight: 600; }}
    header p {{ margin: 0; color: var(--muted); font-size: 0.9rem; }}
    main {{ padding: 1.5rem 2rem 3rem; max-width: 1200px; margin: 0 auto; }}
    section.agent {{
      background: var(--panel); border-radius: 12px; padding: 1.25rem 1.5rem;
      margin-bottom: 2rem; border: 1px solid #252b36;
    }}
    section.agent h2 {{
      margin: 0 0 1rem; font-size: 1.1rem; font-weight: 600;
      color: var(--accent);
    }}
    .grid {{
      display: grid; gap: 1.25rem;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    }}
    .chart-wrap {{
      background: #0e1218; border-radius: 8px; padding: 0.75rem;
      border: 1px solid #1e2430;
    }}
    .chart-wrap h3 {{
      margin: 0 0 0.5rem; font-size: 0.8rem; font-weight: 600;
      color: var(--muted); text-transform: uppercase; letter-spacing: 0.04em;
    }}
    canvas {{ max-height: 280px; }}
    .meta {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.75rem; }}
  </style>
</head>
<body>
  <header>
    <h1>Evolution report</h1>
    <p><strong>Convergence</strong> charts match Agent Analytics (adopted Δ vs version, v2 rate + rolling avg). Train % uses the same score-derived rules; <code>pipeline_ab_*</code> in JSON matches the Pipeline A/B table; test % matches test-resolution-trend.</p>
  </header>
  <main id="main"></main>
  <script type="application/json" id="evolution-data">{json_blob}</script>
  <script>
    const report = JSON.parse(document.getElementById('evolution-data').textContent);
    const main = document.getElementById('main');

    const chartDefaults = {{
      responsive: true,
      maintainAspectRatio: false,
      plugins: {{
        legend: {{ labels: {{ color: '#b4bcc8' }} }},
      }},
      scales: {{
        x: {{
          ticks: {{ color: '#8b939e', maxRotation: 45 }},
          grid: {{ color: 'rgba(255,255,255,0.06)' }},
        }},
        y: {{
          ticks: {{ color: '#8b939e' }},
          grid: {{ color: 'rgba(255,255,255,0.06)' }},
        }},
      }},
    }};

    function decisionColor(d) {{
      if (d === 'adopt') return '#3fb950';
      if (d === 'reject') return '#f85149';
      return '#8b939e';
    }}

    function mkCanvas(grid, title) {{
      const w = document.createElement('div');
      w.className = 'chart-wrap';
      w.innerHTML = `<h3>${{title}}</h3><canvas></canvas>`;
      grid.appendChild(w);
      return w.querySelector('canvas');
    }}

    for (const [agent, block] of Object.entries(report.agents || {{}})) {{
      const sec = document.createElement('section');
      sec.className = 'agent';
      sec.innerHTML = `<h2>${{agent}}</h2><div class="grid" data-agent="${{agent}}"></div>
        <div class="meta">${{block.run_count}} runs · source: ${{report.source}}</div>`;
      main.appendChild(sec);
      const grid = sec.querySelector('.grid');

      const conv = block.convergence || {{}};
      const convPts = conv.points || [];
      const adoptPts = convPts.filter(p => String(p.decision||'').toLowerCase().startsWith('adopt'));
      const verLabels = adoptPts.map(p => {{
        const v = p.version || p.run_id || '';
        const s = String(v);
        return s.length > 14 ? s.slice(0,14)+'…' : s;
      }});
      const deltaPct = adoptPts.map(p => (Number(p.resolution_delta)||0) * 100);
      const deltaRolling = adoptPts.map((_, i) => {{
        const s = Math.max(0, i - 2);
        const slice = deltaPct.slice(s, i + 1);
        return slice.reduce((a,b)=>a+b,0) / slice.length;
      }});
      let yMin = -40, yMax = 40;
      if (deltaPct.length) {{
        const m = Math.max(20, ...deltaPct.map(Math.abs), ...deltaRolling.map(Math.abs));
        yMin = -Math.min(100, m);
        yMax = Math.min(100, m);
        if (yMax - yMin < 20) {{ yMin = -30; yMax = 30; }}
      }}
      if (adoptPts.length) {{
        new Chart(mkCanvas(grid, 'Adopted Δ vs version (Agent Analytics)'), {{
          type: 'line',
          data: {{
            labels: verLabels,
            datasets: [
              {{ label: 'Δ v2−v1 (pp)', data: deltaPct, borderColor: '#58a6ff', tension: 0.15, spanGaps: true, pointRadius: 4 }},
              {{ label: '3-run rolling avg', data: deltaRolling, borderColor: '#bc8cff', borderDash: [4,3], tension: 0.15, spanGaps: true, pointRadius: 2 }},
            ],
          }},
          options: {{
            ...chartDefaults,
            plugins: {{
              ...chartDefaults.plugins,
              subtitle: {{
                display: true,
                text: 'Adopted points: ' + adoptPts.length +
                  ' · plateau: ' + (conv.plateau ? 'yes' : 'no') +
                  ' · stability band: ' + (typeof conv.stability_band === 'number' ? (conv.stability_band*100).toFixed(1)+'%' : '—'),
                color: '#8b939e',
                font: {{ size: 11 }},
              }},
            }},
            scales: {{
              ...chartDefaults.scales,
              y: {{
                ...chartDefaults.scales.y,
                min: yMin,
                max: yMax,
                title: {{ display: true, text: 'Δ percentage points (v2−v1, same run)', color: '#8b939e' }},
              }},
            }},
          }},
        }});
      }} else {{
        const emptyC = document.createElement('div');
        emptyC.className = 'chart-wrap';
        emptyC.innerHTML = '<h3>Adopted Δ vs version (Agent Analytics)</h3>'
          + '<p style="color:#8b939e;font-size:0.85rem;padding:0.25rem 0;">No adopted runs in convergence series.</p>';
        grid.appendChild(emptyC);
      }}

      const ratePts = convPts.slice(-5);
      if (ratePts.length && conv.status === 'ok') {{
        const rl = ratePts.map(p => {{
          const v = p.version || p.run_id || '';
          const s = String(v);
          return s.length > 14 ? s.slice(0,14)+'…' : s;
        }});
        const v2r = ratePts.map(p => (Number(p.v2_resolution_rate)||0) * 100);
        const roll3 = ratePts.map((_, i) => {{
          const s = Math.max(0, i - 2);
          const sl = v2r.slice(s, i + 1);
          return sl.reduce((a,b)=>a+b,0) / sl.length;
        }});
        new Chart(mkCanvas(grid, 'V2 train resolution % (last ≤5 in convergence window)'), {{
          type: 'line',
          data: {{
            labels: rl,
            datasets: [
              {{ label: 'V2 rate', data: v2r, borderColor: '#7ee787', tension: 0.15, spanGaps: true }},
              {{ label: 'Rolling (3)', data: roll3, borderColor: '#bc8cff', borderDash: [4,3], tension: 0.15 }},
            ],
          }},
          options: {{
            ...chartDefaults,
            scales: {{
              ...chartDefaults.scales,
              y: {{ ...chartDefaults.scales.y, min: 0, max: 100, title: {{ display: true, text: '%', color: '#8b939e' }} }},
            }},
          }},
        }});
      }}

      const pts = block.points || [];
      const labels = pts.map(p => '#' + p.iteration);

      const trainV1 = pts.map(p => p.train_v1_pct);
      const trainV2 = pts.map(p => p.train_v2_pct);
      const testV1 = pts.map(p => p.test_v1_pct);
      const testV2 = pts.map(p => p.test_v2_pct);

      new Chart(mkCanvas(grid, 'Train resolution % (Agent Analytics rule)'), {{
        type: 'line',
        data: {{
          labels,
          datasets: [
            {{ label: 'V1 (train)', data: trainV1, borderColor: '#6b9fff', tension: 0.2, spanGaps: true }},
            {{ label: 'V2 (train)', data: trainV2, borderColor: '#7ee787', tension: 0.2, spanGaps: true }},
          ],
        }},
        options: {{
          ...chartDefaults,
          scales: {{
            ...chartDefaults.scales,
            y: {{ ...chartDefaults.scales.y, min: 0, max: 100, title: {{ display: true, text: '%', color: '#8b939e' }} }},
          }},
        }},
      }});

      new Chart(mkCanvas(grid, 'Holdout resolution % (test-resolution-trend rule)'), {{
        type: 'line',
        data: {{
          labels,
          datasets: [
            {{ label: 'V1 (test)', data: testV1, borderColor: '#a371f7', tension: 0.2, spanGaps: true }},
            {{ label: 'V2 (test)', data: testV2, borderColor: '#ff9b4a', tension: 0.2, spanGaps: true }},
          ],
        }},
        options: {{
          ...chartDefaults,
          scales: {{
            ...chartDefaults.scales,
            y: {{ ...chartDefaults.scales.y, min: 0, max: 100, title: {{ display: true, text: '%', color: '#8b939e' }} }},
          }},
        }},
      }});

      const trainD = pts.map(p => p.train_delta_pct);
      const testD = pts.map(p => p.test_delta_pct);
      new Chart(mkCanvas(grid, 'Resolution delta (V2 − V1, points)'), {{
        type: 'bar',
        data: {{
          labels,
          datasets: [
            {{ label: 'Train Δ', data: trainD, backgroundColor: 'rgba(61,214,195,0.55)' }},
            {{ label: 'Test Δ', data: testD, backgroundColor: 'rgba(107,159,255,0.5)' }},
          ],
        }},
        options: {{
          ...chartDefaults,
          scales: {{
            ...chartDefaults.scales,
            y: {{ ...chartDefaults.scales.y, title: {{ display: true, text: 'percentage points', color: '#8b939e' }} }},
          }},
        }},
      }});

      const pvals = pts.map(p => (p.p_value == null ? null : Math.min(p.p_value, 1)));
      new Chart(mkCanvas(grid, 'Fisher p-value (train comparison)'), {{
        type: 'line',
        data: {{
          labels,
          datasets: [
            {{ label: 'p-value', data: pvals, borderColor: '#ff7b72', tension: 0.15, spanGaps: true }},
          ],
        }},
        options: {{
          ...chartDefaults,
          plugins: {{
            legend: chartDefaults.plugins.legend,
            annotation: false,
          }},
          scales: {{
            ...chartDefaults.scales,
            y: {{
              ...chartDefaults.scales.y,
              min: 0,
              max: 1,
              title: {{ display: true, text: 'p', color: '#8b939e' }},
            }},
          }},
        }},
      }});

      const cost = pts.map(p => p.pipeline_cost_usd);
      new Chart(mkCanvas(grid, 'Pipeline cost (USD)'), {{
        type: 'line',
        data: {{
          labels,
          datasets: [
            {{ label: 'Cost USD', data: cost, borderColor: '#d2a8ff', tension: 0.2, fill: 'origin',
              backgroundColor: 'rgba(210,168,255,0.12)' }},
          ],
        }},
        options: {{ ...chartDefaults }},
      }});

      const decColors = pts.map(p => decisionColor(p.decision));
      new Chart(mkCanvas(grid, 'Decision (by run)'), {{
        type: 'bar',
        data: {{
          labels,
          datasets: [
            {{
              label: 'Decision',
              data: pts.map(() => 1),
              backgroundColor: decColors,
              barPercentage: 0.9,
              categoryPercentage: 0.95,
            }},
          ],
        }},
        options: {{
          ...chartDefaults,
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              callbacks: {{
                label(ctx) {{
                  const p = pts[ctx.dataIndex];
                  return [
                    'Decision: ' + p.decision.toUpperCase(),
                    'V2 train: ' + (p.train_v2_pct ?? '—') + '%',
                    'V2 test: ' + (p.test_v2_pct ?? '—') + '%',
                  ];
                }},
              }},
            }},
          }},
          scales: {{
            ...chartDefaults.scales,
            y: {{
              ...chartDefaults.scales.y,
              min: 0,
              max: 1.15,
              ticks: {{ stepSize: 1, callback: () => '' }},
              title: {{ display: true, text: 'Each bar = one run (color = decision)', color: '#8b939e' }},
            }},
          }},
        }},
      }});
    }}

    if (!Object.keys(report.agents || {{}}).length) {{
      main.innerHTML = '<p style="color:#8b939e">No runs matched filters.</p>';
    }}
  </script>
</body>
</html>
"""


async def _async_main(args: argparse.Namespace) -> Path:
    prompt_by_agent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if args.source == "artifacts":
        runs = _load_runs_from_artifacts(args.glob)
    else:
        runs = await _load_runs_from_mongo(
            agent=args.agent,
            limit=args.limit,
            include_running=args.include_running,
        )
        prompt_by_agent = await _load_prompt_changes_by_agent()

    agents = _build_series(
        runs,
        args.agent,
        include_mock=args.include_mock,
        prompt_changes_by_agent=prompt_by_agent,
    )
    report = {
        "generated_at_utc": _iso_now(),
        "source": args.source,
        "metric_alignment": (
            "convergence: same build as Agent Analytics (Δ=v2−v1 vs prompt version from "
            "prompt_changes); train charts: score lists + VC fallback; pipeline_ab_*: "
            "Pipeline score-trend (VC only); test_*: test-resolution-trend API"
        ),
        "filters": {
            "agent": args.agent or "",
            "limit": args.limit,
            "include_running": args.include_running,
            "include_mock": args.include_mock,
            "glob": args.glob if args.source == "artifacts" else "",
        },
        "run_count": len(runs),
        "agents": agents,
    }

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evolution_report.json"
    html_path = out_dir / "evolution_report.html"

    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_render_html(report), encoding="utf-8")
    print(f"Wrote {json_path}")
    print(f"Wrote {html_path}")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evolution report with charts (JSON + HTML).")
    parser.add_argument("--source", choices=("mongo", "artifacts"), default="mongo")
    parser.add_argument("--glob", default="artifacts/repro-*/run_doc.json", help="Glob for --source artifacts (relative to repo root)")
    parser.add_argument("--agent", default="", help="Filter to one agent")
    parser.add_argument("--limit", type=int, default=500, help="Max runs (mongo only)")
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Include non-completed runs (mongo only)",
    )
    parser.add_argument(
        "--include-mock",
        action="store_true",
        help="Include mock / synthetic pipeline runs (default: exclude, matching Agent Analytics)",
    )
    parser.add_argument("--output-dir", default="artifacts", help="Directory for evolution_report.json/html")
    args = parser.parse_args()
    try:
        asyncio.run(_async_main(args))
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
