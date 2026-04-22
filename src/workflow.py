import asyncio
from datetime import datetime
from typing import Optional

from temporalio import workflow
from temporalio.common import RetryPolicy

with workflow.unsafe.imports():
    from src.agents.assessment import AssessmentAgent
    from src.agents.resolution import ResolutionAgent, VoiceProvider
    from src.agents.final_notice import FinalNoticeAgent
    from src.models import (
        ConversationContext, Stage, AssessmentData,
        ResolutionOffer, ResolutionPath, FinancialSituation
    )
    from src.self_learning.eval import SelfLearningLoop


class CollectionsWorkflow:
    """Orchestrates handoffs between 3 AI agents for debt collections."""

    def __init__(self):
        self.assessment_agent = AssessmentAgent()
        self.resolution_agent = ResolutionAgent()
        self.final_notice_agent = FinalNoticeAgent()
        self.voice_provider = VoiceProvider()
        self.context = ConversationContext(
            borrower_id="",
            loan_id=""
        )

    async def run(self, borrower_id: str, loan_id: str, phone_number: str,
                initial_data: dict) -> dict:
        """
        Main entry point - executes the full collections pipeline.
        
        Returns final state of the collections attempt.
        """
        self.context.borrower_id = borrower_id
        self.context.loan_id = loan_id

        self.context.assessment_data = AssessmentData(
            borrower_id=borrower_id,
            loan_id=loan_id,
            principal_amount=initial_data.get("principal_amount", 0),
            outstanding_amount=initial_data.get("outstanding_amount", 0),
            days_past_due=initial_data.get("days_past_due", 0)
        )

        outcome = await self._run_assessment_phase()

        if outcome != "success":
            return self._get_final_state()

        outcome = await self._run_resolution_phase(phone_number)

        if outcome not in ["committed", "resolved"]:
            return self._get_final_state()

        outcome = await self._run_final_notice_phase()

        return self._get_final_state()

    async def _run_assessment_phase(self) -> str:
        """Phase 1: Chat-based Assessment Agent."""
        self.context.current_stage = Stage.ASSESSMENT
        learning_loop = SelfLearningLoop("AssessmentAgent")

        max_turns = 10
        for turn in range(max_turns):
            user_input = await self._wait_for_input()
            self.context.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
                "stage": Stage.ASSESSMENT.value
            })

            response = await self.assessment_agent.process(self.context, user_input)

            self.context.conversation_history.append({
                "role": "assistant",
                "content": response.message,
                "timestamp": datetime.now().isoformat(),
                "stage": Stage.ASSESSMENT.value
            })

            await self._send_message(response.message)

            await learning_loop.evaluate_interaction(
                self.context.conversation_history,
                "success" if response.should_advance else None
            )

            if response.should_advance:
                self.context.current_stage = Stage.RESOLUTION
                return "success"

        return "incomplete"

    async def _run_resolution_phase(self, phone_number: str) -> str:
        """Phase 2: Voice-based Resolution Agent."""
        self.context.current_stage = Stage.RESOLUTION
        learning_loop = SelfLearningLoop("ResolutionAgent")

        call_id = await self.voice_provider.initiate_call(
            phone_number=phone_number,
            agent_id="resolution_agent",
            context=self.context.model_dump()
        )

        await self._handle_voice_call(call_id, learning_loop)

        if self.context.resolution_outcome == "committed":
            self.context.current_stage = Stage.FINAL_NOTICE
            return "committed"

        return "no_commitment"

    async def _handle_voice_call(self, call_id: str, learning_loop: SelfLearningLoop):
        """Handle the voice call interaction loop."""
        await self._send_message(
            f"Calling your registered phone number. Please answer to discuss resolution options."
        )

        max_turns = 8
        for turn in range(max_turns):
            user_input = await self._wait_for_input()

            if user_input.lower() in ["hang up", "end call", "bye"]:
                break

            self.context.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
                "stage": Stage.RESOLUTION.value,
                "call_id": call_id
            })

            response = await self.resolution_agent.process(self.context, user_input)

            await self._send_message(response.message)

            self.context.conversation_history.append({
                "role": "assistant",
                "content": response.message,
                "timestamp": datetime.now().isoformat(),
                "stage": Stage.RESOLUTION.value,
                "call_id": call_id
            })

            await learning_loop.evaluate_interaction(
                self.context.conversation_history,
                "committed" if response.should_advance else None
            )

            if response.should_advance:
                self.context.resolution_offer = ResolutionOffer(
                    path=self.context.assessment_data.resolution_path,
                    deadline_days=5,
                    valid_until=datetime.now().isoformat()
                )
                break

    async def _run_final_notice_phase(self) -> str:
        """Phase 3: Chat-based Final Notice Agent."""
        self.context.current_stage = Stage.FINAL_NOTICE
        learning_loop = SelfLearningLoop("FinalNoticeAgent")

        max_turns = 5
        for turn in range(max_turns):
            user_input = await self._wait_for_input()
            self.context.conversation_history.append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat(),
                "stage": Stage.FINAL_NOTICE.value
            })

            response = await self.final_notice_agent.process(self.context, user_input)

            await self._send_message(response.message)

            self.context.conversation_history.append({
                "role": "assistant",
                "content": response.message,
                "timestamp": datetime.now().isoformat(),
                "stage": Stage.FINAL_NOTICE.value
            })

            await learning_loop.evaluate_interaction(
                self.context.conversation_history,
                "resolved" if response.should_advance else "escalated"
            )

            if response.should_advance:
                self.context.current_stage = Stage.COMPLETE
                return self.context.final_notice_outcome or "complete"

        return "escalated"

    def _get_final_state(self) -> dict:
        return {
            "borrower_id": self.context.borrower_id,
            "loan_id": self.context.loan_id,
            "final_stage": self.context.current_stage.value,
            "outcome": self.context.final_notice_outcome or self.context.resolution_outcome or "unresolved",
            "conversation_length": len(self.context.conversation_history),
            "assessment_data": self.context.assessment_data.model_dump() if self.context.assessment_data else None,
            "resolution_offer": self.context.resolution_offer.model_dump() if self.context.resolution_offer else None
        }

    async def _wait_for_input(self) -> str:
        """Wait for user input - implement based on your chat provider."""
        raise NotImplementedError("Implement based on your chat provider (Twilio, WhatsApp, etc.)")

    async def _send_message(self, message: str) -> None:
        """Send message to user - implement based on your chat provider."""
        raise NotImplementedError("Implement based on your chat provider")


async def run_collections(borrower_id: str, loan_id: str, phone_number: str,
                       principal_amount: float, outstanding_amount: float,
                       days_past_due: int) -> dict:
    """Entry point to run collections workflow."""
    workflow = CollectionsWorkflow()
    
    return await workflow.run(
        borrower_id=borrower_id,
        loan_id=loan_id,
        phone_number=phone_number,
        initial_data={
            "principal_amount": principal_amount,
            "outstanding_amount": outstanding_amount,
            "days_past_due": days_past_due
        }
    )