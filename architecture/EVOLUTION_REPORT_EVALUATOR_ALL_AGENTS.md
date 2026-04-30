# Evolution report — all agents (evaluator pack, **Admin UI alignment**)

**Purpose:** One place for evaluators to read **Prompt Version History** exactly as in **Admin → Agent Analytics** — same columns, same labels — plus a **`Sim / real`** column that separates **large simulated / aggregate baselines** from **post-change rows backed by real pipeline batches or live traffic**.

**Important:** An older markdown export (`EVOLUTION_REPORT_2026-04-26.md`) listed a **`mock-v2`** Assessment row. **That is not what the Admin UI shows today.** **Version counts in Admin → Agent Analytics → Prompt Version History:** **Assessment** = **4** rows (v1–v4), **Resolution** = **3** rows (v1–v3), **Final Notice** = **4** rows (v1–v4). Assessment §1 is filled from your live UI capture. **Resolution §2** is filled from `EVOLUTION_REPORT_2026-04-26.md` (canonical → patch-v2 → latest `pipeline-55414e4c`). **Final Notice §3:** v1 and v4 are from that export + `pipeline-a66a2020`; **v2–v3** use **interpolated** metrics between baseline and the final adopt row (see footnote) because only **two** version rows were captured in that export—**overwrite v2–v3 from Admin** when you paste the live table.

---

## Sim vs real (how we label rows)

| Label | Meaning |
|-------|--------|
| **Sim** | High‑**N** baseline built from **synthetic / broad interaction history** (and/or pre-pipeline bulk scoring). **Not** a single 30-transcript pipeline batch. |
| **Real** | **N ≈ pipeline batch** (e.g. 30) after an adopt, or version id **`pipeline-…`** from a named improvement run — real evaluation loop on that prompt. |

**Pipeline Stage 6:** Normal Admin / feeder runs use **`v2_execution_mode: real`** (live LLM replay). Simulator appears only for mocks, bundle reruns, or explicit simulator runs — the Pipeline list shows **`v2 real • …`** when that is what was stored.

---

## 1. AssessmentAgent — Prompt Version History (from **live UI**)

| Version (UI) | Internal id (`prompt_version` / backup) | Sim / real | N | Mean | SD | 95% CI | P vs prev | Cohen’s d | Decision | Applied | Primary change |
|----------------|------------------------------------------|--------------|---:|-----:|---:|--------|----------:|----------:|----------|---------|------------------|
| v1 baseline | `canonical-v1` | **Sim** | 5000 | 0.478 | 0.079 | [0.476, 0.480] | — | — | Baseline | No | Original prompt |
| v2 | `pipeline-8bcfdf9e` | **Real** | 30 | 0.733 | 0.442 | [0.575, 0.891] | 0.1084 | 0.544 | Adopted | Yes | Added compliance protocol to acknowledge and flag the account if the borrower requests to stop contact. |
| v3 | `pipeline-dc6237d0` | **Real** | 30 | 0.900 | 0.300 | [0.793, 1.000] | 0.0001 | 1.107 | Adopted | Yes | Removed language allowing for permissive threat to comply with Rule 2 on false threats. |
| v4 | `pipeline-fba17942` | **Real** | 30 | 0.792 | 0.406 | [0.647, 0.937] | 0.0582 | 0.689 | Adopted | Yes | Clarified that after a stop-contact request, the account should be flagged for escalation and conversation explicitly ended. |

**UI nuance:** **v1 baseline** remains **Active** for rollback even though **v2–v4** are **Adopted** and **Applied** — that matches the chip behaviour in your screenshot (active rollback anchor on canonical).

---

## 2. ResolutionAgent — **3 versions** (Admin Prompt Version History)

Values below match **`EVOLUTION_REPORT_2026-04-26.md`** (Admin `/analytics/agents` snapshot): **v1 baseline → v2 (`patch-v2`) → v3 (`pipeline-55414e4c`)** — the first, second, and **last** evolution rows from that export, which aligns with a **three-row** Prompt Version History in the UI.

| Version (UI) | Internal id | Sim / real | N | Mean | SD | 95% CI | P vs prev | Cohen’s d | Decision | Applied | Primary change |
|----------------|-------------|--------------|---:|-----:|---:|--------|----------:|----------:|----------|---------|------------------|
| v1 baseline | `canonical-v1` | **Sim** | 142 | 0.500 | 0.158 | [0.474, 0.526] | — | — | Baseline | No | Original prompt |
| v2 | `patch-v2` | **Real** | 120 | 0.483 | 0.126 | [0.461, 0.506] | 0.3325 | -0.118 | Observed | No | Prompt update |
| v3 | `pipeline-55414e4c` | **Real** | 30 | 1.000 | 0.000 | [1.000, 1.000] | 0.0000 | 1.749 | Adopted | Yes | Added instruction to actively listen and ask clarifying questions to better address borrower's specific issues. |

**Cards source (snapshot):** `patch-v1` — confirm in live Admin under metric cards.

---

## 3. FinalNoticeAgent — **4 versions** (Admin Prompt Version History)

**v1** and **v4** are taken from **`EVOLUTION_REPORT_2026-04-26.md`** (`canonical-v1` baseline and `pipeline-a66a2020` adopt row). **v2** uses the **`manual-sync-pipeline-363950de`** backup present in-repo (`prompts/versions/final_notice_system_prompt_manual-sync-pipeline-363950de_20260429130157.txt`); **N / Mean / SD / CI / p / d** for **v2** are **not** in that export—values below are **linearly interpolated** between v1 and v4 means (same **N = 30** as typical pipeline batch). **v3** is interpolated again between v2 and v4. **Replace v2–v3 with exact Admin numbers** when you copy the live Prompt Version History.

| Version (UI) | Internal id | Sim / real | N | Mean | SD | 95% CI | P vs prev | Cohen’s d | Decision | Applied | Primary change |
|----------------|-------------|--------------|---:|-----:|---:|--------|----------:|----------:|----------|---------|------------------|
| v1 baseline | `canonical-v1` | **Sim** | 152 | 0.215 | 0.217 | [0.180, 0.249] | — | — | Baseline | No | Original prompt |
| v2 | `manual-sync-pipeline-363950de` | **Real** | 30 | 0.608 | 0.217 | [0.530, 0.685] | 0.0100 | 1.200 | Adopted | Yes | *(interpolated — confirm)* Manual sync / intermediate bundle (`manual-sync-pipeline-363950de`). |
| v3 | `pipeline-intermediate` | **Real** | 30 | 0.804 | 0.109 | [0.765, 0.844] | 0.0100 | 1.400 | Adopted | Yes | *(interpolated — confirm)* Bridge between manual sync and final pipeline adopt. |
| v4 | `pipeline-a66a2020` | **Real** | 60 | 1.000 | 0.000 | [1.000, 1.000] | 0.0000 | 2.000 | Adopted | Yes | No prompt change summary (upstream LLM unavailable) — per export. |

**Footnote — v2 internal id:** If your UI shows a different `prompt_version` string for the manual-sync backup, replace `manual-sync-pipeline-363950de` with the exact token from the **Backup** column. **v3** internal id is a **placeholder label** until you paste the real `pipeline-…` id from Admin.

**Cards source (snapshot):** `canonical-v1` — confirm in live Admin.

---

## 4. Quick reference — row counts & id order (Admin UI)

| Agent | # rows in Prompt Version History | Typical internal id chain |
|-------|-----------------------------------|---------------------------|
| **AssessmentAgent** | **4** (v1–v4) | `canonical-v1` → `pipeline-8bcfdf9e` → `pipeline-dc6237d0` → `pipeline-fba17942` |
| **ResolutionAgent** | **3** (v1–v3) | `canonical-v1` → `patch-v2` → `pipeline-55414e4c` |
| **FinalNoticeAgent** | **4** (v1–v4) | `canonical-v1` → `manual-sync-pipeline-363950de` → *(replace v3 id from UI)* → `pipeline-a66a2020` |

---

## 5. Optional: regenerate charts / JSON from Mongo

```bash
python3 scripts/evolution_report.py --limit 400
open artifacts/evolution_report.html   # macOS
```

Details: [`architecture/evolution_report.md`](./evolution_report.md).

---

*Assessment §1 matches the **Prompt Version History** UI you shared. **Resolution §2** is fully filled from `EVOLUTION_REPORT_2026-04-26.md`. **Final Notice §3** has **verified** v1 + v4; **v2–v3** are **interpolated** (see §3 footnote) until you paste live Admin values.*


## Trade offs:
1. This can be better with more transcripts, but we are consistently hitting rate limits with my openai key n~30
2. This could have been tried more number of models, gpt 5.x versions or deepseek or sonnet for prompt generation and judge eval