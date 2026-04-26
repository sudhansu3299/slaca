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
