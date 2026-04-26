# LLM Loop Cost Breakdown — 2026-04-26

Generated at: `2026-04-26T09:07:00+00:00`

This report is updated to match the latest values shown in the Admin UI (`/api/admin/stats/cost-breakdown`).

## Selected Latest Loop (UI-backed)
- Run ID: `pipeline-85ac64f3`
- Agent: `FinalNoticeAgent`
- Started at: `2026-04-26T09:02:39.492812+00:00`
- Triggered by: `admin_ui`
- Status / decision: `completed` / `adopt`
- Loop runs in scope: `1`
- LLM calls in scope: `68`
- Total whole-loop cost: **`$2.2398`**
- Budget usage: `11.2%` of `$20.00` (`$17.7602` remaining)

## Component Cost Breakdown (Whole Loop)

| Component | Cost (USD) |
|---|---:|
| Simulate borrower conversations | 2.1920 |
| LLM-as-judge scoring | 0.0153 |
| Prompt generation (proposer) | 0.0102 |
| Improvement pipeline failure analysis | 0.0060 |
| Compliance adversarial checks | 0.0051 |
| A/B comparison + hypothesis synthesis | 0.0060 |
| L3 meta-evaluator scoring consistency audit | 0.0052 |
| **Total** | **2.2398** |

## Source Artifacts
- Raw machine-readable breakdown: `architecture/LLM_LOOP_COST_BREAKDOWN_2026-04-26.json`
- Data source: running admin endpoint `GET /api/admin/stats/cost-breakdown`
- Split heuristic source: `src/admin_api.py` (`/stats/cost-breakdown`)
