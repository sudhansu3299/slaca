# LLM loop cost — methodology and current defaults

**Last aligned with repo sources:** 2026-04-30 (`src/cost.py`, `src/token_budget.py`, `src/admin_api.py`).

## Where the numbers appear

- **Admin UI:** “Cost per Self-Learning Loop” uses the same payload as **`GET /api/admin/stats/cost-breakdown`**.
- **Scope:** Latest improvement loop — most recent `eval_pipeline` run, plus sibling runs in the same `triggered_by` cohort within **`LATEST_LOOP_WINDOW_MINUTES`** (default **20**), up to **one run per agent** (max three agents).

## Run budget cap

- **`TOTAL_COST_BUDGET_USD = 20.0`** in `src/token_budget.py` — hard cap used in convergence of cost reporting vs “budget” and `CostTracker.check_budget`.

## Default model pricing (USD per 1M tokens)

These are the committed values in **`src/cost.py`** → `MODEL_PRICING`. Override with env vars **`AGENT_MODEL`**, **`SIMULATION_MODEL`**, **`EVAL_MODEL`** (defaults below).

| Key / model id        | Input $/M | Output $/M | Default role |
|-----------------------|----------:|-----------:|--------------|
| `gpt-4o`              | 2.50      | 10.00      | `AGENT_MODEL` — agent conversation estimate in cost breakdown |
| `gpt-4o-mini`         | 0.15      | 0.60       | `SIMULATION_MODEL`, `EVAL_MODEL`; pipeline judge / failure / compliance buckets |
| `gpt-4.1-mini`        | 0.40      | 1.60       | Optional tier |
| `claude-opus-4-5`     | 15.00     | 75.00      | Same as `OPUS_*` in `src/token_budget.py` |
| `claude-haiku-4-5`    | 1.00      | 5.00       | |
| `claude-sonnet-4-5`   | 3.00      | 15.00      | |

**Defaults:** `AGENT_MODEL=gpt-4o`, `SIMULATION_MODEL=gpt-4o-mini`, `EVAL_MODEL=gpt-4o-mini`.

## How the breakdown rows are built

Implementation: **`get_cost_breakdown`** in `src/admin_api.py`.

1. **Simulate borrower conversations** — Estimated cost for interaction turns in scope: **`gpt-4o`** with fixed averages **1800** input tokens and **350** output tokens per counted interaction (`count` = matching `interactions` documents).

2. **When `pipeline_cost_usd` is tracked** (`> 0` on loop runs), pipeline spend is split into labeled buckets (same total as summed pipeline cost):
   - **LLM-as-judge scoring** — **36%** of pipeline cost, model **`gpt-4o-mini`**, count = scored transcript + held-out rows.
   - **Prompt generation (proposer)** — **24%**, **`gpt-4o`**, count = pipeline runs in loop.
   - **Improvement pipeline failure analysis** — **14%**, **`gpt-4o-mini`**, count = runs.
   - **Compliance adversarial checks** — **12%**, **`gpt-4o-mini`**, count = `max(n_runs * 2, 0)`.
   - **A/B comparison + hypothesis synthesis** — **Remainder** after the four buckets above, **`gpt-4o-mini`**.

3. **L3 meta-evaluator** — Latest `meta_eval_runs` document: **`meta_eval_cost_usd`** (and case counts for display). Model label in API: **`gpt-4o-mini`**.

4. **If pipeline cost is not tracked** — Fully **estimated** rows using per-operation average token assumptions (see code paths with `source: "estimated"`).

5. **Demo guard** — If **all** rows are estimated and total would exceed **$14.90**, totals are scaled down for dashboard demos (see `admin_api.py`).

## Machine-readable snapshot

Example structure and one illustrative totals row set (your DB will differ): **`architecture/LLM_LOOP_COST_BREAKDOWN.json`**.

To refresh from a running stack:

```bash
curl -sS "http://localhost:8000/api/admin/stats/cost-breakdown" | jq .
```

## Illustrative example (same order of magnitude as a full three-agent loop)

| Component | Cost (USD) | Notes |
|-----------|-----------:|--------|
| Simulate borrower conversations | ~2.19 | ~274 turns × estimated `gpt-4o` per-turn cost |
| LLM-as-judge scoring | ~0.015 | Share of tracked `pipeline_cost_usd` |
| Prompt generation (proposer) | ~0.010 | Share of tracked pipeline cost |
| Improvement pipeline failure analysis | ~0.006 | Share |
| Compliance adversarial checks | ~0.005 | Share |
| A/B comparison + hypothesis synthesis | ~0.006 | Remainder bucket |
| L3 meta-evaluator scoring consistency audit | ~0.005 | From `meta_eval_runs.meta_eval_cost_usd` |
| **Total (example)** | **~2.24** | **~11.2%** of **$20** budget |

Replace these figures with your live **`cost-breakdown`** response when reporting results.
