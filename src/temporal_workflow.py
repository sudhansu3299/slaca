"""
Temporal workflow — one per borrower.

Matches the architecture diagram exactly:

  ┌───────────────────────────────────────────────────────────┐
  │  Borrower enters post-default pipeline                    │
  └───────────────────────────┬───────────────────────────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Assessment Agent  │◄──── Retry? max 3 attempts
                    │  (Chat)            │         yes ↑  no response
                    └─────────┬──────────┘         │       │
                    situation │                     │    exhausted
                    assessed  │                     │       │
                    ┌─────────▼──────────┐          │       │
                    │  Resolution Agent  │          │       │
                    │  (Voice Call)      │          │       │
                    └────┬──────────┬────┘          │       │
                deal     │    no    │               │       │
                agreed   │    deal  │               │       │
                    ┌────▼─┐  ┌────▼────────────┐   │       │
                    │ EXIT │  │  Final Notice   │   │       │
                    │ Log  │  │  Agent (Chat)   │   │       │
                    │agree │  └───┬─────────┬───┘   │       │
                    └──────┘  res.│    no   │res.   │       │
                                  │         │       │       │
                              ┌───▼──┐  ┌──▼───────▼───────▼──┐
                              │ EXIT │  │  EXIT: Flag legal/   │
                              │ Log  │  │  write-off           │
                              │ res. │  └─────────────────────┘
                              └──────┘

Retry logic (Assessment only — matches diagram):
  - If no response received (timeout/no input), retry the assessment
  - Max 3 attempts total before exhausted → legal/write-off exit

Temporal handles: state persistence, retries, timeouts.
All LLM calls and I/O are in activities (determinism rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports_passed_through():
    from src.models import Stage
    # These modules use threading, httpx, asyncpg, or other non-deterministic
    # primitives that Temporal's workflow sandbox cannot safely import.
    # They are only used in activities (not in workflow code itself) but the
    # sandbox's import tracker sees them transitively — marking them as
    # pass-through keeps the sandbox happy.
    import openai
    import httpx
    import sniffio
    try:
        import asyncpg
    except ImportError:
        pass


# ──────────────────────────── Data types ──────────────────── #

@dataclass
class CollectionsInput:
    borrower_id: str
    loan_id: str
    phone_number: str
    principal_amount: float
    outstanding_amount: float
    days_past_due: int
    workflow_id: str = ""           # full Temporal workflow ID — used as message bus key
    borrower_name: str = "Borrower" # display name for UI
    persona: str = "cooperative"    # for simulation; ignored when real chat is used
    use_real_chat: bool = False      # True → chat activities wait for WebSocket signals


@dataclass
class CollectionsOutput:
    borrower_id: str
    loan_id: str
    final_stage: str
    outcome: str                    # resolved | escalated | exhausted | no_deal
    total_turns: int
    total_tokens_in: int
    total_tokens_out: int
    handoff_tokens: list[int] = field(default_factory=list)
    assessment_attempts: int = 1


# ──────────────────────────── Retry / timeout policy ────────── #

# Per-activity retry: 3 attempts, exponential backoff.
# Applies to transient errors (network, API blips).
DEFAULT_RETRY = RetryPolicy(
    initial_interval=timedelta(seconds=2),
    maximum_interval=timedelta(seconds=30),
    maximum_attempts=3,
    backoff_coefficient=2.0,
    non_retryable_error_types=["ValueError"],
)

# Voice call can take up to 15 min
VOICE_TIMEOUT = timedelta(minutes=5)

# Chat stages: 10 min per stage (borrower may be slow)
CHAT_TIMEOUT = timedelta(minutes=10)

# Brief gap between stages (natural pacing)
STAGE_GAP = timedelta(seconds=3)

# Side-effect activities (DB writes etc.) — fast
SIDE_EFFECT_TIMEOUT = timedelta(seconds=30)

# Assessment retry: how many times we re-engage an unresponsive borrower
MAX_ASSESSMENT_ATTEMPTS = 3


# ──────────────────────────── Workflow ─────────────────────── #

@workflow.defn
class CollectionsWorkflow:
    """
    One workflow instance per borrower.
    Implements the architecture diagram transitions exactly.
    """

    def __init__(self):
        # Signal queue — chat messages sent by the UI land here
        self._message_queue: list[str] = []

    @workflow.signal
    def user_message(self, msg: str) -> None:
        """WebSocket UI → workflow: borrower typed a message."""
        self._message_queue.append(msg)

    @workflow.query
    def get_stage(self) -> str:
        """Let the UI poll current stage."""
        return self._current_stage if hasattr(self, "_current_stage") else "starting"

    @workflow.run
    async def run(self, inp: CollectionsInput) -> CollectionsOutput:
        workflow.logger.info(
            f"[{inp.borrower_id}] Collections workflow started"
        )
        self._current_stage = Stage.ASSESSMENT.value

        handoff_tokens: list[int] = []
        total_in = total_out = 0
        assessment_attempts = 0

        # ── Assessment retry loop (matches "Retry? max 3 attempts" diamond) ──
        a_result = None
        while assessment_attempts < MAX_ASSESSMENT_ATTEMPTS:
            assessment_attempts += 1
            workflow.logger.info(
                f"[{inp.borrower_id}] Assessment attempt {assessment_attempts}/{MAX_ASSESSMENT_ATTEMPTS}"
            )

            a_result = await workflow.execute_activity(
                "run_assessment_stage",
                {
                    "input_data": inp,
                    "attempt": assessment_attempts,
                },
                start_to_close_timeout=CHAT_TIMEOUT,
                retry_policy=DEFAULT_RETRY,
            )
            total_in += a_result.get("tokens_in", 0)
            total_out += a_result.get("tokens_out", 0)

            if a_result["outcome"] == "success":
                break   # situation assessed → move to Resolution

            # "no response" → retry up to max attempts
            if a_result["outcome"] == "no_response" and assessment_attempts < MAX_ASSESSMENT_ATTEMPTS:
                workflow.logger.info(
                    f"[{inp.borrower_id}] No response, retrying assessment "
                    f"({assessment_attempts}/{MAX_ASSESSMENT_ATTEMPTS})"
                )
                await workflow.sleep(STAGE_GAP)
                continue

            # Any other failure or exhausted retries → legal/write-off
            break

        if not a_result or a_result["outcome"] != "success":
            # exhausted path → EXIT: Flag for legal/write-off
            await _log_legal_writeoff(inp, "assessment_exhausted")
            return CollectionsOutput(
                borrower_id=inp.borrower_id, loan_id=inp.loan_id,
                final_stage=Stage.ASSESSMENT.value,
                outcome="exhausted",
                total_turns=len(a_result.get("conversation", [])) if a_result else 0,
                total_tokens_in=total_in, total_tokens_out=total_out,
                assessment_attempts=assessment_attempts,
            )

        if a_result.get("handoff_block"):
            handoff_tokens.append(len(a_result["handoff_block"]) // 4)

        await workflow.sleep(STAGE_GAP)

        # ── Resolution Agent (Voice Call) ─────────────────────────────────
        self._current_stage = Stage.RESOLUTION.value

        r_result = await workflow.execute_activity(
            "run_resolution_stage",
            {
                "input": inp,
                "prior": a_result,
            },
            start_to_close_timeout=VOICE_TIMEOUT,
            retry_policy=DEFAULT_RETRY,
        )
        total_in += r_result.get("tokens_in", 0)
        total_out += r_result.get("tokens_out", 0)

        if r_result.get("handoff_block"):
            handoff_tokens.append(len(r_result["handoff_block"]) // 4)

        # ── Outcome-based transition (diagram: deal agreed vs no deal) ────

        if r_result["outcome"] == "committed":
            # deal agreed → EXIT: Log agreement
            await workflow.execute_activity(
                "log_agreement",
                {
                    "borrower_id": inp.borrower_id,
                    "loan_id": inp.loan_id,
                    "workflow_id": inp.workflow_id,
                    "call_id": r_result.get("call_id", ""),
                    "offer": r_result.get("offer"),
                    "conversation": r_result.get("conversation", []),
                },
                start_to_close_timeout=SIDE_EFFECT_TIMEOUT,
                retry_policy=DEFAULT_RETRY,
            )
            total_turns = (
                len(a_result.get("conversation", []))
                + len(r_result.get("conversation", []))
            )
            return CollectionsOutput(
                borrower_id=inp.borrower_id, loan_id=inp.loan_id,
                final_stage=Stage.COMPLETE.value,
                outcome="resolved",
                total_turns=total_turns,
                total_tokens_in=total_in, total_tokens_out=total_out,
                handoff_tokens=handoff_tokens,
                assessment_attempts=assessment_attempts,
            )

        # no deal (refused / no_outcome) → Final Notice Agent

        await workflow.sleep(STAGE_GAP)

        # ── Final Notice Agent (Chat) ──────────────────────────────────────
        self._current_stage = Stage.FINAL_NOTICE.value

        f_result = await workflow.execute_activity(
            "run_final_notice_stage",
            {
                "input": inp,
                "prior_assessment": a_result,
                "prior_resolution": r_result,
            },
            start_to_close_timeout=CHAT_TIMEOUT,
            retry_policy=DEFAULT_RETRY,
        )
        total_in += f_result.get("tokens_in", 0)
        total_out += f_result.get("tokens_out", 0)

        total_turns = (
            len(a_result.get("conversation", []))
            + len(r_result.get("conversation", []))
            + len(f_result.get("conversation", []))
        )

        if f_result["outcome"] == "resolved":
            # resolved → EXIT: Log resolution
            await workflow.execute_activity(
                "log_resolution",
                {
                    "borrower_id": inp.borrower_id,
                    "loan_id": inp.loan_id,
                    "workflow_id": inp.workflow_id,
                    "call_id": r_result.get("call_id", ""),
                    "call_context": r_result.get("call_context"),
                    "offer": r_result.get("offer"),
                    "conversation": f_result.get("conversation", []),
                },
                start_to_close_timeout=SIDE_EFFECT_TIMEOUT,
                retry_policy=DEFAULT_RETRY,
            )
            return CollectionsOutput(
                borrower_id=inp.borrower_id, loan_id=inp.loan_id,
                final_stage=Stage.COMPLETE.value,
                outcome="resolved",
                total_turns=total_turns,
                total_tokens_in=total_in, total_tokens_out=total_out,
                handoff_tokens=handoff_tokens,
                assessment_attempts=assessment_attempts,
            )
        else:
            # no resolution → EXIT: Flag for legal/write-off
            await _log_legal_writeoff(inp, "final_notice_refused")
            return CollectionsOutput(
                borrower_id=inp.borrower_id, loan_id=inp.loan_id,
                final_stage=Stage.ESCALATED.value,
                outcome="escalated",
                total_turns=total_turns,
                total_tokens_in=total_in, total_tokens_out=total_out,
                handoff_tokens=handoff_tokens,
                assessment_attempts=assessment_attempts,
            )


async def _log_legal_writeoff(inp: CollectionsInput, reason: str) -> None:
    """Fire-and-forget side effect for legal/write-off exits."""
    await workflow.execute_activity(
        "create_legal_referral",
        {
            "borrower_id": inp.borrower_id,
            "loan_id": inp.loan_id,
            "reason": reason,
        },
        start_to_close_timeout=timedelta(seconds=30),
        retry_policy=RetryPolicy(maximum_attempts=3),
    )
