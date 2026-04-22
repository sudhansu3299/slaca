"""
Audit logger — writes run output to disk for later inspection.

Two files per run, both in ./audit-logs/:
  <run-id>.log    — human-readable per-turn trace (same as --verbose stdout)
  <run-id>.jsonl  — structured one-line-per-event record for programmatic analysis

Both files are written atomically. The stdout output is unchanged.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional


DEFAULT_LOG_DIR = Path("audit-logs")


class AuditLogger:
    """
    Tee-style logger: every call writes to both .log and .jsonl for this run.
    """

    def __init__(self, run_id: str, log_dir: Optional[Path] = None):
        base = log_dir or Path(os.getenv("AUDIT_LOG_DIR", str(DEFAULT_LOG_DIR)))
        base.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.log_path = base / f"{run_id}.log"
        self.jsonl_path = base / f"{run_id}.jsonl"
        self._log_fh = open(self.log_path, "w", encoding="utf-8")
        self._jsonl_fh = open(self.jsonl_path, "w", encoding="utf-8")
        self.log_event("run_started", {"run_id": run_id})

    # ── plain text line ─────────────────────────────────────── #
    def write_line(self, line: str) -> None:
        self._log_fh.write(line.rstrip("\n") + "\n")
        self._log_fh.flush()

    # ── structured event ───────────────────────────────────── #
    def log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "event": event_type,
            **payload,
        }
        self._jsonl_fh.write(json.dumps(record, default=str) + "\n")
        self._jsonl_fh.flush()

    # ── turn-level event ────────────────────────────────────── #
    def log_turn(
        self,
        stage: str,
        turn: int,
        borrower: str,
        agent_msg: str,
        advanced: bool,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost_usd: float = 0.0,
    ) -> None:
        self.log_event("turn", {
            "stage": stage,
            "turn": turn,
            "borrower": borrower,
            "agent": agent_msg,
            "advanced": advanced,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "cost_usd": round(cost_usd, 6),
        })

    # ── stage summary ───────────────────────────────────────── #
    def log_stage_summary(self, stage: str, eval_data: dict) -> None:
        self.log_event("stage_summary", {"stage": stage, "eval": eval_data})

    # ── final report ────────────────────────────────────────── #
    def log_final_report(self, report: dict) -> None:
        self.log_event("final_report", report)

    # ── cleanup ─────────────────────────────────────────────── #
    def close(self) -> None:
        self.log_event("run_finished", {"run_id": self.run_id})
        try:
            self._log_fh.close()
        except Exception:
            pass
        try:
            self._jsonl_fh.close()
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
