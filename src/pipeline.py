"""
End-to-end pipeline runner.

Orchestrates the three agents with handoffs, self-learning,
and cost tracking — without Temporal for local simulation.

For production: use temporal_workflow.py instead.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Awaitable, Callable, Optional

from src.agents.assessment import AssessmentAgent
from src.agents.resolution import ResolutionAgent
from src.agents.final_notice import FinalNoticeAgent
from src.handoff import HandoffBuilder
from src.models import (
    ConversationContext, Stage, AssessmentData, ResolutionPath
)
from src.self_learning.eval import SelfLearningLoop
from src.token_budget import CostTracker


@dataclass
class PipelineResult:
    borrower_id: str
    loan_id: str
    final_stage: str
    outcome: str
    total_turns: int
    total_cost_usd: float
    handoff_tokens: list[int] = field(default_factory=list)
    eval_summaries: list[str] = field(default_factory=list)
    conversation: list[dict] = field(default_factory=list)


class CollectionsPipeline:
    """
    Runs the full 3-agent pipeline with:
    - HandoffBuilder between stages
    - SelfLearningLoop per agent
    - Shared CostTracker
    """

    def __init__(self):
        self.cost_tracker = CostTracker()
        self.assessment_agent = AssessmentAgent(cost_tracker=self.cost_tracker)
        self.resolution_agent = ResolutionAgent(cost_tracker=self.cost_tracker)
        self.final_notice_agent = FinalNoticeAgent(cost_tracker=self.cost_tracker)

        self.assessment_loop = SelfLearningLoop("AssessmentAgent")
        self.resolution_loop = SelfLearningLoop("ResolutionAgent")
        self.final_notice_loop = SelfLearningLoop("FinalNoticeAgent")

    async def run(
        self,
        borrower_id: str,
        loan_id: str,
        principal_amount: float,
        outstanding_amount: float,
        days_past_due: int,
        input_provider,   # Callable[[ConversationContext, str], str]
        max_turns_per_stage: int = 8,
        event_cb: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> PipelineResult:
        ctx = ConversationContext(borrower_id=borrower_id, loan_id=loan_id)
        ctx.assessment_data = AssessmentData(
            borrower_id=borrower_id,
            loan_id=loan_id,
            principal_amount=principal_amount,
            outstanding_amount=outstanding_amount,
            days_past_due=days_past_due,
        )

        result = PipelineResult(
            borrower_id=borrower_id,
            loan_id=loan_id,
            final_stage="assessment",
            outcome="unresolved",
            total_turns=0,
            total_cost_usd=0.0,
        )

        # ── Stage 1: Assessment ──────────────────────────────────────── #
        outcome = await self._run_stage(
            ctx=ctx,
            agent=self.assessment_agent,
            loop=self.assessment_loop,
            stage=Stage.ASSESSMENT,
            input_provider=input_provider,
            max_turns=max_turns_per_stage,
            result=result,
            event_cb=event_cb,
        )

        if outcome not in ("advance", "max_turns_reached", "no_input"):
            result.final_stage = Stage.ASSESSMENT.value
            result.outcome = "assessment_incomplete"
            result.total_cost_usd = self.cost_tracker.total_cost_usd
            result.conversation = ctx.conversation_history
            self._schedule_learning_followups(
                resolved=False,
                agents_seen=self._agents_seen_from_conversation(result.conversation),
            )
            return result

        # Handoff: Assessment → Resolution
        summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
        HandoffBuilder.apply_to_context(summary, ctx)
        result.handoff_tokens.append(summary.estimated_tokens())
        ctx.current_stage = Stage.RESOLUTION

        # ── Stage 2: Resolution ──────────────────────────────────────── #
        outcome = await self._run_stage(
            ctx=ctx,
            agent=self.resolution_agent,
            loop=self.resolution_loop,
            stage=Stage.RESOLUTION,
            input_provider=input_provider,
            max_turns=max_turns_per_stage,
            result=result,
            event_cb=event_cb,
        )

        if outcome not in ("advance", "refused", "max_turns_reached", "no_input"):
            result.final_stage = Stage.RESOLUTION.value
            result.outcome = "resolution_no_outcome"
            result.total_cost_usd = self.cost_tracker.total_cost_usd
            result.conversation = ctx.conversation_history
            self._schedule_learning_followups(
                resolved=False,
                agents_seen=self._agents_seen_from_conversation(result.conversation),
            )
            return result

        # Handoff: Resolution → Final Notice
        summary = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
        HandoffBuilder.apply_to_context(summary, ctx)
        result.handoff_tokens.append(summary.estimated_tokens())
        ctx.current_stage = Stage.FINAL_NOTICE

        # ── Stage 3: Final Notice ────────────────────────────────────── #
        outcome = await self._run_stage(
            ctx=ctx,
            agent=self.final_notice_agent,
            loop=self.final_notice_loop,
            stage=Stage.FINAL_NOTICE,
            input_provider=input_provider,
            max_turns=max_turns_per_stage,
            result=result,
            event_cb=event_cb,
        )

        result.final_stage = ctx.current_stage.value
        result.outcome = ctx.final_notice_outcome or ctx.resolution_outcome or "unresolved"
        result.total_cost_usd = self.cost_tracker.total_cost_usd
        result.conversation = ctx.conversation_history
        resolved = result.outcome in ("AGREEMENT", "resolved", "committed", "advance")
        self._schedule_learning_followups(
            resolved=resolved,
            agents_seen=self._agents_seen_from_conversation(result.conversation),
        )

        return result

    @staticmethod
    def _agents_seen_from_conversation(conversation: list[dict]) -> list[str]:
        stage_to_agent = {
            Stage.ASSESSMENT.value: "AssessmentAgent",
            Stage.RESOLUTION.value: "ResolutionAgent",
            Stage.FINAL_NOTICE.value: "FinalNoticeAgent",
        }
        seen: set[str] = set()
        for msg in conversation:
            if msg.get("role") != "assistant":
                continue
            agent_name = stage_to_agent.get(msg.get("stage", ""))
            if agent_name:
                seen.add(agent_name)
        return sorted(seen)

    def _schedule_learning_followups(self, resolved: bool, agents_seen: list[str]) -> None:
        """Fan out completion signals to feeder + regression monitor."""
        try:
            from src.self_learning.feeder import on_conversation_complete

            asyncio.create_task(on_conversation_complete(agents_seen=agents_seen))
        except Exception:
            pass

        try:
            from src.self_learning.meta_evaluator import on_conversation_complete as on_meta_complete

            asyncio.create_task(on_meta_complete())
        except Exception:
            pass

        try:
            from src.self_learning.regression_monitor import record_outcome

            for agent_name in ("AssessmentAgent", "ResolutionAgent", "FinalNoticeAgent"):
                asyncio.create_task(record_outcome(agent_name, resolved))
        except Exception:
            pass

    async def _run_stage(
        self,
        ctx: ConversationContext,
        agent,
        loop: SelfLearningLoop,
        stage: Stage,
        input_provider,
        max_turns: int,
        result: PipelineResult,
        event_cb: Optional[Callable[[str], Awaitable[None]]] = None,
    ) -> str:
        ctx.current_stage = stage

        for turn in range(max_turns):
            self.cost_tracker.check_budget()

            user_input = input_provider(ctx, agent.name)
            if user_input is None:
                if event_cb is not None:
                    await event_cb(f"{stage.value} t{turn} borrower(no_input)")
                return "no_input"
            if event_cb is not None:
                await event_cb(f"{stage.value} t{turn} borrower={user_input}")

            ctx.conversation_history.append({
                "role": "user",
                "content": user_input,
                "stage": stage.value,
                "turn": turn,
            })

            response = await agent.process(ctx, user_input)
            if event_cb is not None:
                await event_cb(
                    f"{stage.value} t{turn} {agent.name} advance={response.should_advance} "
                    f"reply={response.message}"
                )

            ctx.conversation_history.append({
                "role": "assistant",
                "content": response.message,
                "stage": stage.value,
                "turn": turn,
                "tokens": response.tokens_used.output_tokens if response.tokens_used else 0,
                "advanced": response.should_advance,
            })

            result.total_turns += 1

            # Apply context updates
            for k, v in response.context_update.items():
                setattr(ctx, k, v)

            # Evaluate
            eval_result = await loop.evaluate_interaction(
                ctx.conversation_history,
                outcome="success" if response.should_advance else None,
            )
            result.eval_summaries.append(
                f"{stage.value} t{turn}: passed={eval_result.passed} "
                f"scores={eval_result.scores}"
            )

            if response.should_advance:
                if event_cb is not None:
                    await event_cb(f"{stage.value} t{turn} outcome=advance")
                return "advance"

        if event_cb is not None:
            await event_cb(f"{stage.value} outcome=max_turns_reached")
        return "max_turns_reached"

    def cost_report(self) -> str:
        return self.cost_tracker.report()
