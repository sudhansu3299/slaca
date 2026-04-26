# MetaEvaluator Flaw Evidence

Date: 2026-04-25
System: L3 MetaEvaluator (resolution audit)
Source: local admin API (`/api/admin/meta-evaluator/runs` and `/api/admin/meta-evaluator/runs/{run_id}`)

## Executive Summary
MetaEvaluator detected a critical flaw in L2 resolution scoring reliability for `ResolutionAgent`.
The flaw is evidenced by severe disagreement between primary and shadow judges and a `poor` overall verdict.

## Primary Evidence Run
- Run ID: `meta-e26505fe`
- Status: `completed`
- Started: `2026-04-25T16:55:44.349685+00:00`
- Completed: `2026-04-25T16:55:57.622492+00:00`
- Overall verdict: `poor`
- Meta note: `At least one agent fell below kappa 0.40; L2 resolution rubric needs tightening.`

### ResolutionAgent Audit Outcome (from same run)
- Agent: `ResolutionAgent`
- Verdict: `poor`
- Flag: `rubric is unreliable; prompt update recommended`
- Sample size used: `60`
- Disagreement count: `50`
- Observed agreement: `0.1667`
- Expected agreement: `0.1667`
- Cohen's kappa: `0.0`
- Recommendation: `Tighten the L2 resolution rubric and add sharper acceptance anchors before using this signal for prompt changes.`

## Concrete Disagreement Examples
All examples below are from run `meta-e26505fe`, `ResolutionAgent`, with `agreement=false`.

### Example 1
- Trace ID: `trace-v2fix-0027`
- Primary label: `resolution=1`
- Shadow label: `resolution=0`
- Shadow reason: `No explicit commitment or confirmation of a payment arrangement from the borrower.`
- Transcript excerpt: `Borrower: Mock seeded input for analytics row correction. | Agent: Voice call outcome: committed. Turns: 10.`

### Example 2
- Trace ID: `trace-v2fix-0055`
- Primary label: `resolution=1`
- Shadow label: `resolution=0`
- Shadow reason: `No explicit commitment or confirmation of a payment arrangement from the borrower.`
- Transcript excerpt: `Borrower: Mock seeded input for analytics row correction. | Agent: Voice call outcome: committed. Turns: 10.`

### Example 3
- Trace ID: `trace-v2fix-0039`
- Primary label: `resolution=1`
- Shadow label: `resolution=0`
- Shadow reason: `No explicit commitment or confirmation of a payment arrangement from the borrower.`
- Transcript excerpt: `Borrower: Mock seeded input for analytics row correction. | Agent: Voice call outcome: committed. Turns: 10.`

## Supporting Evidence Runs
Additional runs show recurring issues:

- `meta-c8e0a909`
  - `ResolutionAgent` verdict: `poor`
  - Sample size used: `20`
  - Disagreement count: `19`
  - Observed agreement: `0.05`
  - Flag: `rubric is unreliable; prompt update recommended`

- `meta-e26505fe`
  - `FinalNoticeAgent` verdict: `needs_data`
  - Flag: `not enough data`
  - Sample size used: `0`

## Why This Is Strong Evidence
- High-disagreement, low-kappa metrics are direct indicators that L2 scoring is unstable for `ResolutionAgent`.
- The same run includes explicit recommendations from MetaEvaluator to tighten rubric anchors before using this signal for prompt updates.
- Repetition across runs supports that the issue is systematic, not random noise.

## Suggested Follow-up (Operational)
- Freeze prompt promotion for `ResolutionAgent` until rubric calibration improves.
- Add explicit commitment criteria examples to the resolution rubric.
- Re-run MetaEvaluator after rubric update and require kappa >= 0.40 before unfreezing promotions.
