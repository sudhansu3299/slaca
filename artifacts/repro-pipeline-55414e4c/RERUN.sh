#!/usr/bin/env bash
set -euo pipefail
cd "/Users/sudhansu/project-slaca"
.venv/bin/python scripts/rerun_pipeline_from_snapshot.py --agent-name ResolutionAgent --run-id pipeline-55414e4c --transcripts-file /Users/sudhansu/project-slaca/artifacts/repro-pipeline-55414e4c/transcripts.json --allow-overwrite-run-id --output-json /Users/sudhansu/project-slaca/artifacts/repro-pipeline-55414e4c/rerun_result.json
