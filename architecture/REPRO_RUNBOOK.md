# Reproducibility runbook

This document explains **what reproducibility means in this repo** and **how to obtain and verify it** (export, bundle layout, rerun commands). Use it as the single reference for reviewers and operators.

---

## 1) What reproducibility means here

Reproducibility means you can line up the same inputs and harness as an archived improvement-pipeline run:

1. Same run identifier (`run_id`)
2. Same transcript cohort (`transcripts.json`)
3. Same pipeline split seed (derived from `run_id`; see below)
4. Same snapshot of key runtime config (`repro_config.json`)
5. Full per-conversation score artifacts (`per_conversation_scores.json` / `.csv`, plus `run_doc.json`)

**Split seed**

- `pipeline_split_seed = int(md5(run_id).hexdigest(), 16) % (2**32)`

**Config snapshot**

- Written under `artifacts/repro-<RUN_ID>/repro_config.json`
- Includes: `run_id`, `agent_target`, `triggered_by`, `git_commit`, `pipeline_split_seed`, and selected env keys used by the pipeline (for example `REAL_V2_EXECUTION_MODE`, `REPLAY_BORROWER_MODE`, model-related vars).

**Required bundle files**

- `run_doc.json` — full `eval_pipeline` run record from Mongo at export time
- `per_conversation_scores.json` and `per_conversation_scores.csv`
- `transcripts.json` — transcript snapshot used for rerun
- `interactions.json` — optional but recommended for auditability
- `RERUN.sh` and `RERUN_COMMAND.txt` — one-command rerun helpers (created by export)

**Generic export (any completed `run_id`)**

```bash
.venv/bin/python scripts/export_repro_bundle.py \
  --run-id <RUN_ID> \
  --output-dir artifacts/repro-<RUN_ID> \
  --export-interactions
```

**Generic rerun (after export)**

```bash
.venv/bin/python scripts/rerun_pipeline_from_snapshot.py \
  --agent-name <AssessmentAgent|ResolutionAgent|FinalNoticeAgent> \
  --run-id <RUN_ID> \
  --transcripts-file artifacts/repro-<RUN_ID>/transcripts.json \
  --allow-overwrite-run-id \
  --output-json artifacts/repro-<RUN_ID>/rerun_result.json
```

Or, from the bundle:

```bash
bash artifacts/repro-<RUN_ID>/RERUN.sh
```

**Notes and limitations**

- If LLM APIs are unavailable or model/provider behavior changes, model-generated outputs can differ even with identical seed and transcript input. The bundle still gives full traceability of what was run and what was scored.
- If no API key is present at rerun time, model-backed stages may degrade; logs may show `No LLM API key found`.
- For assignment-style **live LLM** convergence evidence, prefer runs where `run_doc.json` has `llm_calls > 0` and `v2_execution_mode == real`; see `scripts/report_live_convergence.py`.

---

## 2) Prerequisites (must pass first)

Run from repo root (adjust if your clone path differs):

`/Users/sudhansu/project-slaca`

Required:

- Python env with project deps (`.venv` expected in commands below)
- MongoDB reachable when using `export_repro_bundle.py` or `rerun_pipeline_from_snapshot.py` (rerun loads transcripts from the snapshot file but still connects to Mongo for the pipeline)
- At least one LLM key in the environment for live reruns, for example `OPENCODE_API_KEY`, `OPENAI_API_KEY`, or `DEEPSEEK_API_KEY` in `.env`

Quick key check:

```bash
python3 - <<'PY'
import os
ok = bool(os.getenv("DEEPSEEK_API_KEY","").strip() or os.getenv("OPENAI_API_KEY","").strip() or os.getenv("OPENCODE_API_KEY","").strip())
print("LLM_KEY_PRESENT=", ok)
PY
```

---

## 3) Reference bundles (latest per agent)

Each row uses a `pipeline-<8hex>` id and the same bundle layout as the admin improvement pipeline auto-export (`artifacts/repro-<run_id>/`).

### AssessmentAgent

| Field | Value |
| --- | --- |
| Run id | `pipeline-8d4e2f61` |
| Bundle directory | `artifacts/repro-pipeline-8d4e2f61/` |
| `repro_config.json` | `artifacts/repro-pipeline-8d4e2f61/repro_config.json` |
| Split seed | `3568123874` |
| Git commit (in bundle) | `aa9704790e33389fbaf4d9cc1df2bb975db4f759` |

Re-export:

```bash
.venv/bin/python scripts/export_repro_bundle.py \
  --run-id pipeline-8d4e2f61 \
  --output-dir artifacts/repro-pipeline-8d4e2f61 \
  --export-interactions
```

Rerun (`RERUN_COMMAND.txt` in bundle):

```bash
bash artifacts/repro-pipeline-8d4e2f61/RERUN.sh
```

```bash
.venv/bin/python scripts/rerun_pipeline_from_snapshot.py --agent-name AssessmentAgent --run-id pipeline-8d4e2f61 --transcripts-file /Users/sudhansu/project-slaca/artifacts/repro-pipeline-8d4e2f61/transcripts.json --allow-overwrite-run-id --output-json /Users/sudhansu/project-slaca/artifacts/repro-pipeline-8d4e2f61/rerun_result.json
```

Raw score files: `per_conversation_scores.json`, `per_conversation_scores.csv`

### ResolutionAgent

| Field | Value |
| --- | --- |
| Run id | `pipeline-55414e4c` |
| Bundle directory | `artifacts/repro-pipeline-55414e4c/` |
| Split seed | `2226789600` |
| Git commit (in bundle) | `aa9704790e33389fbaf4d9cc1df2bb975db4f759` |

Re-export:

```bash
.venv/bin/python scripts/export_repro_bundle.py \
  --run-id pipeline-55414e4c \
  --output-dir artifacts/repro-pipeline-55414e4c \
  --export-interactions
```

Rerun:

```bash
bash artifacts/repro-pipeline-55414e4c/RERUN.sh
```

```bash
.venv/bin/python scripts/rerun_pipeline_from_snapshot.py --agent-name ResolutionAgent --run-id pipeline-55414e4c --transcripts-file /Users/sudhansu/project-slaca/artifacts/repro-pipeline-55414e4c/transcripts.json --allow-overwrite-run-id --output-json /Users/sudhansu/project-slaca/artifacts/repro-pipeline-55414e4c/rerun_result.json
```

### FinalNoticeAgent

| Field | Value |
| --- | --- |
| Run id | `pipeline-a66a2020` |
| Bundle directory | `artifacts/repro-pipeline-a66a2020/` |
| Split seed | `2330947426` |
| Git commit (in bundle) | `3b51a604f17d8f01d57e9e8f300316abd000892d` |

Re-export:

```bash
.venv/bin/python scripts/export_repro_bundle.py \
  --run-id pipeline-a66a2020 \
  --output-dir artifacts/repro-pipeline-a66a2020 \
  --export-interactions
```

Rerun:

```bash
bash artifacts/repro-pipeline-a66a2020/RERUN.sh
```

```bash
.venv/bin/python scripts/rerun_pipeline_from_snapshot.py --agent-name FinalNoticeAgent --run-id pipeline-a66a2020 --transcripts-file /Users/sudhansu/project-slaca/artifacts/repro-pipeline-a66a2020/transcripts.json --allow-overwrite-run-id --output-json /Users/sudhansu/project-slaca/artifacts/repro-pipeline-a66a2020/rerun_result.json
```

---

## 4) Run all three reruns

```bash
bash artifacts/repro-pipeline-8d4e2f61/RERUN.sh && \
bash artifacts/repro-pipeline-55414e4c/RERUN.sh && \
bash artifacts/repro-pipeline-a66a2020/RERUN.sh
```

---

## 5) Threshold / pass check

After rerun:

```bash
python3 - <<'PY'
import json
files = [
  "artifacts/repro-pipeline-8d4e2f61/rerun_result.json",
  "artifacts/repro-pipeline-55414e4c/rerun_result.json",
  "artifacts/repro-pipeline-a66a2020/rerun_result.json",
]
for p in files:
  d = json.load(open(p))
  print(p, "status=", d.get("status"), "decision=", d.get("decision"))
PY
```

Suggested gate:

- **Hard gate:** all three `status` values should be `completed`
- **Soft tolerance:** compare original vs rerun aggregate metrics with small drift (for example absolute resolution-rate drift `<= 0.05` per run) when LLM nondeterminism applies

If status is `failed` with missing key messages, set an LLM key in `.env` and rerun section 4.

---

## 6) Bundle directory checklist

Per bundle, expect at least:

- `per_conversation_scores.csv`, `per_conversation_scores.json`
- `run_doc.json`, `transcripts.json`
- `rerun_result.json` (after a rerun)
- `interactions.json` (optional)
- `repro_config.json`, `RERUN.sh`, `RERUN_COMMAND.txt`

Paths:

- `artifacts/repro-pipeline-8d4e2f61/`
- `artifacts/repro-pipeline-55414e4c/`
- `artifacts/repro-pipeline-a66a2020/`

---

## 7) Example cited in evolution reporting

One evolution report example uses Final Notice:

- Run id: `pipeline-a66a2020`
- Bundle: `artifacts/repro-pipeline-a66a2020`
- Seed: `2330947426`
- Git commit (report line may differ from bundle snapshot): `e867ded00b0468ed544c400abb46bf01ade5c468`

Exact rerun command:

```bash
.venv/bin/python scripts/rerun_pipeline_from_snapshot.py --agent-name FinalNoticeAgent --run-id pipeline-a66a2020 --transcripts-file /Users/sudhansu/project-slaca/artifacts/repro-pipeline-a66a2020/transcripts.json --allow-overwrite-run-id --output-json /Users/sudhansu/project-slaca/artifacts/repro-pipeline-a66a2020/rerun_result.json
```

---

## 8) Verification status (local check)

These three reruns were executed on 2026-04-29:

- `bash artifacts/repro-pipeline-8d4e2f61/RERUN.sh`
- `bash artifacts/repro-pipeline-55414e4c/RERUN.sh`
- `bash artifacts/repro-pipeline-a66a2020/RERUN.sh`

Observed `rerun_result.json` entries reported `status: failed` when no LLM API key was present in the environment. After setting a key, rerun section 4 and re-check section 5.
