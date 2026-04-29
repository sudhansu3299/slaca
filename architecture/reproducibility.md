# Reproducibility Guide

This document defines how to reproduce evolution/evaluation results in this repository.

## Scope

Reproducibility here means:

1. Same run identifier (`run_id`)
2. Same transcript cohort (`transcripts.json`)
3. Same pipeline seed derivation
4. Same key runtime config snapshot
5. Full per-conversation raw score artifacts

## Seed and Config

Pipeline split seed is deterministic from run id:

- `pipeline_split_seed = int(md5(run_id).hexdigest(), 16) % (2**32)`

Config snapshot is stored in:

- `artifacts/repro-<RUN_ID>/repro_config.json`

This file includes:

- `run_id`
- `agent_target`
- `triggered_by`
- `git_commit`
- `pipeline_split_seed`
- selected env keys used by the pipeline

## Required Raw Data Artifacts

For each reproducible run bundle:

- `run_doc.json` (full `eval_pipeline` run record)
- `per_conversation_scores.json`
- `per_conversation_scores.csv`
- `transcripts.json`
- `interactions.json` (optional but recommended for auditability)

## Generate Reproducibility Bundle

```bash
.venv/bin/python scripts/export_repro_bundle.py \
  --run-id <RUN_ID> \
  --output-dir artifacts/repro-<RUN_ID> \
  --export-interactions
```

## Single Command Rerun (End-to-End)

```bash
.venv/bin/python scripts/rerun_pipeline_from_snapshot.py \
  --agent-name <AssessmentAgent|ResolutionAgent|FinalNoticeAgent> \
  --run-id <RUN_ID> \
  --transcripts-file artifacts/repro-<RUN_ID>/transcripts.json \
  --allow-overwrite-run-id \
  --output-json artifacts/repro-<RUN_ID>/rerun_result.json
```

## Verified Commands (Latest Run Per Agent)

The following three bundles were exported and rerun successfully on this machine.

### 1) AssessmentAgent (latest completed run)

- `run_id`: `mock-run-29a922`
- `pipeline_split_seed`: `368933967`
- `git_commit`: `3b51a604f17d8f01d57e9e8f300316abd000892d`

Export command:

```bash
.venv/bin/python scripts/export_repro_bundle.py \
  --run-id mock-run-29a922 \
  --output-dir artifacts/repro-mock-run-29a922 \
  --export-interactions
```

Single-command rerun:

```bash
bash artifacts/repro-mock-run-29a922/RERUN.sh
```

Raw files:

- `artifacts/repro-mock-run-29a922/per_conversation_scores.json`
- `artifacts/repro-mock-run-29a922/per_conversation_scores.csv`

### 2) ResolutionAgent (latest completed run)

- `run_id`: `pipeline-55414e4c`
- `pipeline_split_seed`: `2226789600`
- `git_commit`: `3b51a604f17d8f01d57e9e8f300316abd000892d`

Export command:

```bash
.venv/bin/python scripts/export_repro_bundle.py \
  --run-id pipeline-55414e4c \
  --output-dir artifacts/repro-pipeline-55414e4c \
  --export-interactions
```

Single-command rerun:

```bash
bash artifacts/repro-pipeline-55414e4c/RERUN.sh
```

Raw files:

- `artifacts/repro-pipeline-55414e4c/per_conversation_scores.json`
- `artifacts/repro-pipeline-55414e4c/per_conversation_scores.csv`

### 3) FinalNoticeAgent (latest completed run)

- `run_id`: `pipeline-a66a2020`
- `pipeline_split_seed`: `2330947426`
- `git_commit`: `3b51a604f17d8f01d57e9e8f300316abd000892d`

Export command:

```bash
.venv/bin/python scripts/export_repro_bundle.py \
  --run-id pipeline-a66a2020 \
  --output-dir artifacts/repro-pipeline-a66a2020 \
  --export-interactions
```

Single-command rerun:

```bash
bash artifacts/repro-pipeline-a66a2020/RERUN.sh
```

Raw files:

- `artifacts/repro-pipeline-a66a2020/per_conversation_scores.json`
- `artifacts/repro-pipeline-a66a2020/per_conversation_scores.csv`

## Verification Status

Verified on 2026-04-29 by running:

```bash
bash artifacts/repro-mock-run-29a922/RERUN.sh
bash artifacts/repro-pipeline-55414e4c/RERUN.sh
bash artifacts/repro-pipeline-a66a2020/RERUN.sh
```

All three rerun commands completed and wrote `rerun_result.json` in their bundle directories.

## Current Report Example

The current evolution report references:

- run id: `pipeline-a66a2020`
- bundle dir: `artifacts/repro-pipeline-a66a2020`
- seed: `2330947426`
- git commit: `e867ded00b0468ed544c400abb46bf01ade5c468`

Exact rerun command:

```bash
.venv/bin/python scripts/rerun_pipeline_from_snapshot.py --agent-name FinalNoticeAgent --run-id pipeline-a66a2020 --transcripts-file /Users/sudhansu/project-slaca/artifacts/repro-pipeline-a66a2020/transcripts.json --allow-overwrite-run-id --output-json /Users/sudhansu/project-slaca/artifacts/repro-pipeline-a66a2020/rerun_result.json
```

## Notes and Limitations

- If LLM APIs are unavailable or model/provider behavior changes, outputs can differ in model-generated stages even with identical seed and transcript input.
- The bundle still provides full traceability of what was run and what was scored.
- If no API key is present at rerun time, model-backed stages degrade to fallback behavior and logs will show `No LLM API key found`.
- For assignment-grade "real-LLM convergence evidence", include at least one run where `llm_calls > 0` and `v2_execution_mode == real` in `run_doc.json`.
