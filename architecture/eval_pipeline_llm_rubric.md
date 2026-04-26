# Self-learning eval pipeline: LLM model rubric

This document describes **which model is used for which task** in the self-learning improvement / eval pipeline, and the **temperature** environment variables that tune those calls.

**Source of truth:** `src/self_learning/improvement_pipeline.py` (and agent runtime: `AGENT_MODEL` in `src/cost.py` / `src/agents/base.py` for real v2 replay). The diagram in `prompts/eval_pipeline.txt` may not match every default; prefer this file and the code.

## Model assignment by stage

| Stage | Task | Model env | Default |
| --- | --- | --- | --- |
| 1 | Transcript scoring (LLM judge → structured JSON) | `EVAL_MODEL` | `gpt-4o-mini` |
| 2 | Failure analysis (patterns across failed transcripts) | `FAILURE_ANALYSIS_MODEL` | `gpt-4o-mini` |
| 3 | Compliance check — **current** prompt (first pass) | `COMPLIANCE_MODEL` | `gpt-4o` |
| 3b | Compliance **review** (second pass, only if first pass is non-compliant and `COMPLIANCE_DOUBLE_CHECK` is enabled) | `COMPLIANCE_REVIEW_MODEL` | `gpt-4.1-nano` |
| 4 | Prompt improvement (new prompt + change summary) | `PROMPT_GENERATION_MODEL` or `PIPELINE_MODEL` | `gpt-4o` |
| 5 | Compliance check — **new** prompt (same `check_compliance` path; hard gate) | `COMPLIANCE_MODEL` (+ optional `COMPLIANCE_REVIEW_MODEL` as in 3b) | `gpt-4o` |
| 6a | v2 “execution” | See below (not a single model) | — |
| 6a | Counterfactual rescore of held-out transcripts (when that path runs) | `EVAL_MODEL` | `gpt-4o-mini` |
| 6b | A/B comparison — **narrative** `summary` / `reason` (stats are code, not LLM) | `EVAL_MODEL` | `gpt-4o-mini` |
| 7 | Hypothesis generation (patch ideas for next iteration) | `PROMPT_GENERATION_MODEL` or `PIPELINE_MODEL` | `gpt-4o` |

### Stage 6a: v2 execution (three modes)

- **Real replay:** `CollectionsPipeline` runs with the proposed v2 prompt. The **target agent** uses the normal production stack — configure via **`AGENT_MODEL`** (default in cost layer: `gpt-4o`). Each finished transcript is **scored** with stage 1 (`EVAL_MODEL` / `EVAL_TEMP`).
- **Simulator mode:** agent LLM calls are **mocked** (no live model for the agent).
- **Fallback:** if real/simulator execution does not win, the pipeline can fall back to **counterfactual rescoring** of held-out data using **`EVAL_MODEL`**.

Fisher p-value, Cohen’s \(d\), and adopt/reject **gates** in stage 6b are **deterministic**; the LLM only summarizes.

## Temperature environment variables

| Env var | Default | Used with | Applied to |
| --- | --- | --- | --- |
| `EVAL_TEMP` | `0.0` | `EVAL_MODEL` | Transcript judge, held-out rescore, A/B narrative |
| `FAILURE_ANALYSIS_TEMP` | `0.2` | `FAILURE_ANALYSIS_MODEL` | Failure analysis |
| `PROMPT_GENERATION_TEMP` | `0.7` | `PROMPT_GENERATION_MODEL` | Prompt improvement, hypothesis generation |

Primary and review **compliance** LLM calls use the shared `_llm` helper with **default temperature `0.0`** (not `PROMPT_GENERATION_TEMP`).

**Related:** `COMPLIANCE_DOUBLE_CHECK` (default on) controls whether a failing first-pass compliance result triggers the second model (`COMPLIANCE_REVIEW_MODEL`).
