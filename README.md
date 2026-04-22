# AI Collections System

Self-learning 3-agent debt collections pipeline orchestrated by Temporal.

## Deliverables

| Deliverable | Location |
|---|---|
| Temporal workflow orchestrating 3-agent pipeline | `src/temporal_workflow.py` |
| 2× Chat agents + 1× Voice agent | `src/agents/` |
| Cross-modal handoff with context summarization | `src/handoff.py`, `src/memory.py` |
| Test harness for generating + evaluating conversations | `src/test_harness.py` |
| Self-learning loop with meta-evaluation | `src/self_learning/` |
| Docker Compose setup | `docker-compose.yml`, `Dockerfile` |

## Quick Start (fresh machine, ≤5 min)

```bash
# 1. Clone
git clone <repo> project-slaca && cd project-slaca

# 2. Boot entire stack (Temporal + worker + harness)
docker compose up --build
```

That's it. The stack:

1. **PostgreSQL** (Temporal backing store)
2. **Temporal server** (`localhost:7233`)
3. **Temporal UI** (`http://localhost:8233`)
4. **Worker** — listens on task queue `collections-queue`
5. **Harness** — starts 4 synthetic borrower workflows (all personas), prints outcomes, exits

All LLM calls are **mocked** by default (`USE_LLM_MOCK=1`) so no API keys needed for the smoke test.

## Local development (no Docker)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Terminal 1: start Temporal locally (needs temporal CLI)
temporal server start-dev

# Terminal 2: start worker
python -m src.temporal_activities

# Terminal 3: kick off a workflow
python -m src.starter --persona cooperative
# or run all 4 personas:
python -m src.starter --batch
```

## Test harness (no Temporal needed)

Fast iteration, fully local:

```bash
python -m src.test_harness                  # 1 cooperative run, mocked LLM
python -m src.test_harness --all            # all 4 personas, mocked
python -m src.test_harness --all --runs 3   # 12 runs total
python -m src.test_harness --all --json     # machine-readable output
python -m src.test_harness --persona cooperative --live            # real Claude Opus 4.5
python -m src.test_harness --persona cooperative --live --verbose  # turn-by-turn trace
```

### Audit logs (written to `./audit-logs/` by default)

Every harness run writes two files:

| File | Purpose |
|---|---|
| `<run-id>.log`   | Human-readable turn trace + stage eval + cost summary |
| `<run-id>.jsonl` | One JSON event per line: `run_started`, `run_config`, `turn`, `stage_summary`, `final_report`, `run_finished` |

The paths are printed at the end of each run. Override location with `--log-dir some/path` or suppress with `--no-log`.

Programmatic analysis example:

```bash
cat audit-logs/SIM-COOP-abc123-r0.jsonl | jq 'select(.event=="turn") | {stage, turn, agent, advanced}'
```

### Tweaking personas (borrower scripts)

Override what the borrower says per stage with a JSON file:

```json
// personas/my_borrower.json
{
  "AssessmentAgent":  ["Yes got the notice.", "Last 4 are 4455, born 1992.", ...],
  "ResolutionAgent":  ["What can you offer?", "I accept."],
  "FinalNoticeAgent": ["Yes I formally accept."]
}
```

Run with it:

```bash
python -m src.test_harness --persona cooperative --live \
    --personas-file personas/my_borrower.json
```

A sample is checked in at `personas/example_cooperative.json`.

### Tweaking prompts (agent guidance)

Inject extra instructions into any agent's system prompt without editing `src/prompts.py`:

```json
// prompts/my_tweaks.json
{
  "AssessmentAgent":  "Keep each response under 12 words.",
  "ResolutionAgent":  "Lead with total savings vs legal route.",
  "FinalNoticeAgent": "Use ALL CAPS for the deadline line only."
}
```

Apply it:

```bash
python -m src.test_harness --persona cooperative --live \
    --prompt-overrides prompts/my_tweaks.json
```

A sample is checked in at `prompts/example_tweaks.json`.

### Combining all knobs

```bash
python -m src.test_harness \
    --persona cooperative --live --verbose \
    --personas-file personas/my_borrower.json \
    --prompt-overrides prompts/my_tweaks.json \
    --log-dir audit-logs/experiment-42
```

## Run tests

```bash
.venv/bin/python -m pytest tests/ -q
```

## Architecture

```
Borrower ──▶ Temporal Workflow (one per borrower)
                 │
                 ├─ Activity: run_assessment_stage     [chat, Agent 1]
                 │    ├─ cold, clinical fact-gathering
                 │    └─ produces HandoffSummary (≤500 tok)
                 │
                 ├─ Activity: run_resolution_stage     [voice, Agent 2]
                 │    ├─ reads handoff → voice metadata
                 │    ├─ transactional dealmaking
                 │    └─ produces HandoffSummary + offer
                 │
                 └─ Activity: run_final_notice_stage   [chat, Agent 3]
                      ├─ consequences + deadline
                      └─ routes to resolved | escalated
```

**Truth** lives in `ConversationContext` (passed through the workflow).  
**Temporal** owns state persistence, retries, and timeouts.  
**Agents** are stateless processors — receive context, return `AgentResponse`.

## Cost tracking

```python
from src.cost import TieredCostTracker
tracker = TieredCostTracker()
# ... runs ...
print(tracker.full_report())
```

Budget: **$20 total** across all production + simulation + evaluation calls.

## Self-learning loop

`src/self_learning/loop.py` — runs evaluation → methodology evolves → prompts update.

Meta-evaluator (`MetaEvaluator`) judges whether the eval itself is producing useful signal (variance, outcome correlation, version evolution).

## Personas (test harness)

| Persona | Expected path | Expected outcome |
|---|---|---|
| Cooperative | INSTALLMENT | resolved |
| Hostile | LEGAL | escalated |
| Broke | HARDSHIP | resolved |
| Strategic Defaulter | LUMP_SUM | escalated |
