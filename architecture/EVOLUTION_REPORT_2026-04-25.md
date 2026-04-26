# Evolution Report (Corrected, Evidence-Strict) — 2026-04-25

## Integrity Note

You were right to challenge the prior version.

This corrected report only classifies claims as:
- **Observed in repo data** (file-backed)
- **Mock/demo/test evidence** (not production truth)
- **Not available in current checkout**

No mock/showcase result is presented below as real production performance.

---

## 1) Prompt Evolution by Agent

### AssessmentAgent

Status: **Observed in repo data**

Prompt snapshots exist:
- `prompts/versions/assessment_system_prompt_pipeline-bed0ec93_p0.txt`
- `prompts/versions/assessment_system_prompt_pipeline-bed0ec93_p1.txt`
- `prompts/versions/assessment_system_prompt_pipeline-9e9caabf_p1.txt`
- `prompts/versions/assessment_system_prompt_pipeline-9e9caabf_p2.txt`

These files contain explicit `LEARNED PATCH v1..v3` blocks and rollback pointers.

### ResolutionAgent

Status: **Observed in repo data, but mostly mock-run lineage**

Prompt snapshots exist:
- `prompts/versions/resolution_system_prompt_pipeline-mockdemo_p0.txt`
- `prompts/versions/resolution_system_prompt_pipeline-mock-res-20260425135338_p0.txt`
- `prompts/versions/resolution_system_prompt_pipeline-mock-res-20260425135806_p0.txt`

These are clearly mock/demo-labeled in filenames and run IDs (`pipeline-mock*`).

### FinalNoticeAgent

Status: **Not available in current checkout**

Observed:
- Live prompt exists: `prompts/final_notice_system_prompt.txt`
- No `prompts/versions/final_notice_*` archived snapshots found

Conclusion:
- Cannot provide true version-by-version FinalNotice prompt evolution from current files alone.

---

## 2) Quantitative Adoption/Rejection Evidence

### A) Production-grade A/B outcomes

Status: **Not available in current checkout as verified production records**

Reason:
- `architecture/v2_ab_evidence.json` is labeled `"artifact": "v2_ab_showcase"` (showcase artifact)
- Repo includes explicit mock pipeline runners (`src/mock_pipeline_demo.py`, `src/mock_eval_pipeline_runner.py`)

Therefore:
- A/B adoption percentages from `v2_ab_evidence.json` are **demo/showcase evidence**, not guaranteed real live-loop metrics.

### B) Deterministic gate logic (real system behavior)

Status: **Observed in repo code/tests**

Adoption/rejection logic is deterministic and enforced in:
- `src/self_learning/improvement_pipeline.py`
- `tests/test_self_learning_requirements.py`

Verified rejection scenarios in tests:
- identical v1/v2 performance => reject
- tiny effect size despite significance => reject
- LLM output cannot override deterministic reject

---

## 3) Metrics Across Prompt Versions

### What is truly available now

1. **Mock/demo metrics**:
   - `architecture/v2_ab_evidence.json` (showcase only)
2. **Run-level simulation logs**:
   - `audit-logs/*.jsonl` (many runs are `SIM-*`, i.e., synthetic/harness style)
3. **No verified production v1/v2 table for all 3 agents in this checkout**

Conclusion:
- A production-grade, per-agent v1/v2 metric table cannot be truthfully finalized from the current repo snapshot alone.

---

## 4) Regressions and Handling

### Observed regression (real, current)

Status: **Observed in repo execution on 2026-04-25**

Regression encountered:
- 13 test failures in prompt/milestone tests (`test_ms4_prompts.py`, `test_milestone4.py`)
- Cause: prompt template drift removed required canonical markers/persona blocks

Fix applied:
- `prompts/assessment_system_prompt.txt`
- `prompts/resolution_system_prompt.txt`

Validation after fix:
- `371 passed, 1 skipped`

### System-level regression protection (implemented)

Status: **Observed in code**

`src/self_learning/regression_monitor.py`:
- window-based post-adoption monitoring
- threshold-based regression detection
- auto-rollback path via `rollback_prompt(...)`
- persistence into `regression_events`

---

## 5) Meta-Evaluation Catch Case

### True status in this checkout

1. **Mechanism exists and is implemented**:
   - `src/self_learning/meta_evaluator.py`
   - Cohen's kappa + shadow-judge consistency checks
2. **Unit coverage exists**:
   - `tests/test_meta_evaluation.py`
3. **Concrete live-case record proving a real catch is not present in checked files**

Therefore:
- We can claim **capability and tested logic**.
- We cannot claim a **real production catch incident** without querying live `meta_eval_runs`.

---

## 6) Total LLM API Spend Across Learning Loop

### Truthful status

Status: **Not fully available as real total in current checkout**

Why:
- Full live aggregation requires Mongo records (`eval_pipeline`, `meta_eval_runs`, interaction scope) at runtime.
- Current repo snapshot contains code and some logs, but not a complete authoritative “entire loop spend ledger” for all runs.

What is available:
- Cost model + budget framework:
  - `src/cost.py`
  - `src/admin_api.py` (`/stats/cost-breakdown`)
- Per-run cost fields in some audit logs (mixed simulated/live harness contexts)

So:
- Any single total in this report without live DB extraction would be estimated, not final truth.

---

## 7) What We Can Defend in Interview Right Now

1. Pipeline architecture and runtime system are real and runnable.
2. Prompt versioning + rollback mechanics are real and test-covered.
3. Learning-loop gating logic is deterministic and test-covered.
4. Regression monitor and meta-evaluator are implemented and callable.
5. Previous “v1/v2 adoption metrics” should be treated as showcase/demo unless re-derived from live DB.

---

## 8) Required to Produce a Fully Authoritative Evolution Report

To convert this into a strict production-truth report, we need one live extraction pass from running services:

1. Pull latest `eval_pipeline` runs (real, not mock-triggered)
2. Pull matching `prompt_changes` by agent/version
3. Pull `regression_events`
4. Pull `meta_eval_runs`
5. Pull `/stats/cost-breakdown` from the live admin API

Once that is exported, we can produce an evidence-locked final report with no synthetic placeholders.
