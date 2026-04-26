# Evolution Report — 2026-04-26 (DB + UI Real Data)

Generated at: `2026-04-26T08:37:52.947318+00:00`

This report is updated from live MongoDB records and the Admin UI analytics payload (`/analytics/agents`).

## Run Inventory
- Total `eval_pipeline` runs: **30**
- Real runs (mock-tagged excluded): **23**
- Excluded mock-tagged runs: **7**
- Latest real run: **pipeline-a66a2020** (FinalNoticeAgent, 2026-04-26T08:29:23.407946+00:00)

## Latest Real Run Per Agent

| Agent | Run ID | Started At | Decision | Compliance Pass | V2 Mode | v1 n | v2 n | v1 rate | v2 rate | p-value |
|------|--------|------------|----------|-----------------|--------|------|------|---------|---------|--------:|
| AssessmentAgent | pipeline-66b0ff28 | 2026-04-22T15:07:11.467772+00:00 | reject | n/a | n/a | 1 | 0 | 0.6 | 0.0 | 0.5169 |
| ResolutionAgent | pipeline-55414e4c | 2026-04-26T04:03:50.687981+00:00 | adopt | True | simulator | 30 | 30 | 0.133 | 1.0 | 0.0 |
| FinalNoticeAgent | pipeline-a66a2020 | 2026-04-26T08:29:23.407946+00:00 | adopt | False | simulator | 60 | 60 | 0.0 | 1.0 | 0.0 |

## UI Metrics Snapshot (from `/analytics/agents`)

### Agent 1 — Assessment (AssessmentAgent)
- Cards source version: `improved-v2`
- Info completeness: `0.079` (fields extracted / 6)
- Identity verified: `0.08` (hard gate)
- Turns to complete: `1.0` (lower is better, target <=5)
- Tone (clinical): `1.0` (no empathy leaks)

| Version Label | Version | N | Mean | SD | 95% CI | P vs Prev | Cohen's d | Decision | Primary change |
|---|---|---:|---:|---:|---|---:|---:|---|---|
| v1 baseline | canonical-v1 | 776 | 0.446 | 0.087 | [0.439, 0.452] | n/a | n/a | Baseline | Original prompt |
| v2 | improved-v2 | 150 | 0.397 | 0.076 | [0.385, 0.41] | 0.0 | -0.563 | Updated | Prompt update |
| v3 | mock-v2 | 1 | 0.0 | 0.0 | [0.0, 0.0] | 0.5169 | n/a | Updated | Prompt update |

### Agent 2 — Resolution (ResolutionAgent)
- Cards source version: `patch-v1`
- Commitment rate: `0.5` (full / partial / no deal)
- Offer anchoring: `1.0` (stayed in policy range)
- Objection handling: `1.0` (restated terms, no comfort)
- Context continuity: `0.0` (references prior facts)

| Version Label | Version | N | Mean | SD | 95% CI | P vs Prev | Cohen's d | Decision | Primary change |
|---|---|---:|---:|---:|---|---:|---:|---|---|
| v1 baseline | canonical-v1 | 142 | 0.5 | 0.158 | [0.474, 0.526] | n/a | n/a | Baseline | Original prompt |
| v2 | patch-v2 | 120 | 0.483 | 0.126 | [0.461, 0.506] | 0.3325 | -0.118 | Observed | Prompt update |
| v3 | patch-v3 | 40 | 0.625 | 0.0 | [0.625, 0.625] | 0.0 | 1.295 | Adopted | Manual patch applied |
| v4 | patch-v4 | 40 | 0.625 | 0.0 | [0.625, 0.625] | n/a | n/a | Adopted | Manual patch applied |
| v5 | patch-v1 | 10 | 0.625 | 0.0 | [0.625, 0.625] | n/a | n/a | Adopted | Manual patch applied |
| v6 | pipeline-43020190 | 30 | 1.0 | 0.0 | [1.0, 1.0] | 0.0 | 1.749 | Adopted | Added identity disclosure at the start of the conversation. |
| v7 | pipeline-6fd4096b | 20 | 1.0 | 0.0 | [1.0, 1.0] | 0.0 | 1.633 | Adopted | Added guidance to actively listen and ask clarifying questions to address borrower-specific issues. |
| v8 | pipeline-55414e4c | 30 | 1.0 | 0.0 | [1.0, 1.0] | 0.0 | 1.749 | Adopted | Added instruction to actively listen and ask clarifying questions to better address borrower's specific issues. |

### Agent 3 — Final Notice (FinalNoticeAgent)
- Cards source version: `canonical-v1`
- Consequence clarity: `0.221` (credit + legal + asset)
- Deadline specificity: `0.408` (hard date or explicit deadline)
- No re-ask penalty: `0.0` (deducted for repeated A1 questions)
- Resolution rate: `0.02` (borrower committed at this stage)

| Version Label | Version | N | Mean | SD | 95% CI | P vs Prev | Cohen's d | Decision | Primary change |
|---|---|---:|---:|---:|---|---:|---:|---|---|
| v1 baseline | canonical-v1 | 152 | 0.215 | 0.217 | [0.18, 0.249] | n/a | n/a | Baseline | Original prompt |
| v2 | pipeline-a66a2020 | 60 | 1.0 | 0.0 | [1.0, 1.0] | 0.0 | 2.0 | Adopted | No prompt change summary (upstream LLM unavailable) |

## FinalNoticeAgent — v2 from Latest Run #30

Run: `pipeline-a66a2020` (started `2026-04-26T08:29:23.407946+00:00`, decision `adopt`). Stage-6 used simulator mode; judge stages may be degraded when no LLM API key is configured.

| Version | N | Resolution rate | p-value | Cohen's d | Decision |
|---|---:|---:|---:|---:|---|
| v1 baseline | 60 | 0.0 | n/a | n/a | baseline |
| v2 | 60 | 1.0 | 0.0 | 2.0 | adopt |

## Artifact References
- `architecture/db-evolution-2026-04-26/eval_pipeline_runs_real_only.json`
- `architecture/db-evolution-2026-04-26/summary_real_only.json`
- `architecture/db-evolution-2026-04-26/*_REAL_ONLY.json` per-conversation score exports

## Reproducibility

This report includes a reproducibility bundle for run `pipeline-a66a2020`.

- Bundle directory: `artifacts/repro-pipeline-a66a2020`
- Config + seed file: `artifacts/repro-pipeline-a66a2020/repro_config.json`
  - `pipeline_split_seed`: `2330947426`
  - seed formula: `int(md5(run_id).hexdigest(), 16) % (2**32)`
  - run id: `pipeline-a66a2020`
- End-to-end rerun command (single command):
  - `.venv/bin/python scripts/rerun_pipeline_from_snapshot.py --agent-name FinalNoticeAgent --run-id pipeline-a66a2020 --transcripts-file /Users/sudhansu/project-slaca/artifacts/repro-pipeline-a66a2020/transcripts.json --allow-overwrite-run-id --output-json /Users/sudhansu/project-slaca/artifacts/repro-pipeline-a66a2020/rerun_result.json`

Raw reproducibility data files:
- Per-conversation scores JSON: `artifacts/repro-pipeline-a66a2020/per_conversation_scores.json`
- Per-conversation scores CSV: `artifacts/repro-pipeline-a66a2020/per_conversation_scores.csv`
- Transcript snapshot used for rerun: `artifacts/repro-pipeline-a66a2020/transcripts.json`
- Raw interactions for involved trace IDs: `artifacts/repro-pipeline-a66a2020/interactions.json`
- Full pipeline run document: `artifacts/repro-pipeline-a66a2020/run_doc.json`
