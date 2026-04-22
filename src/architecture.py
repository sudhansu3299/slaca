"""
System Architecture Definition
================================

Truth lives in: ConversationContext (workflow state) + MongoDB (persistent store)
Temporal holds: workflow execution state, retry logic, timeouts
Agents hold: nothing — they are stateless processors

Data Flow:
  Borrower → Chat (WhatsApp/SMS)
       ↓
  [Temporal Workflow: CollectionsWorkflow]
       ↓
  Activity: AssessmentActivity
       → Agent1.process(context, user_input) → AgentResponse
       → HandoffBuilder.build() → HandoffSummary (≤500 tokens)
       → store_context(context, db)
       ↓
  Activity: ResolutionActivity (Voice)
       → VoiceProvider.initiate_call(phone, context_summary)
       → Agent2.process(context, transcript) → AgentResponse
       → HandoffBuilder.build() → HandoffSummary
       → store_context(context, db)
       ↓
  Activity: FinalNoticeActivity
       → Agent3.process(context, user_input) → AgentResponse
       → update_loan_status(outcome)
       ↓
  EXIT: log_outcome(db)

Storage layers:
  - Temporal workflow state    → execution continuity, retries
  - ConversationContext (RAM)  → single-run truth during workflow
  - MongoDB documents          → persistent borrower record
  - SelfLearningStore (file)   → agent learnings & eval history

Agent boundaries:
  Agent1 (AssessmentAgent)  : identity verification + financial data collection
  Agent2 (ResolutionAgent)  : offer generation + commitment over voice
  Agent3 (FinalNoticeAgent) : consequence delivery + final resolution/escalation

The "truth" question:
  Primary truth = ConversationContext passed through Temporal workflow.
  Secondary truth = MongoDB for cross-run queries, dashboards, audit.
  Agents never own state — they receive context, return AgentResponse.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class StorageLayer(str, Enum):
    TEMPORAL = "temporal_workflow_state"
    CONTEXT = "conversation_context_ram"
    MONGODB = "mongodb_persistent"
    LEARNING_STORE = "self_learning_file_store"


class AgentBoundary(str, Enum):
    ASSESSMENT = "chat_identity_and_financial_data"
    RESOLUTION = "voice_offer_and_commitment"
    FINAL_NOTICE = "chat_consequence_and_resolution"


@dataclass
class AgentSpec:
    name: str
    modality: str           # "chat" | "voice"
    responsibility: str
    inputs: list[str]
    outputs: list[str]
    max_output_tokens: int
    handoff_to: Optional[str]


@dataclass
class SystemArchitecture:
    """
    Machine-readable system architecture.
    Used for validation and documentation generation.
    """

    agents: list[AgentSpec] = field(default_factory=list)
    storage_layers: list[StorageLayer] = field(default_factory=list)
    truth_owner: StorageLayer = StorageLayer.CONTEXT

    def validate_borrower_trace(self) -> list[str]:
        """
        Verify a borrower can be traced from start → end without ambiguity.
        Returns list of trace steps.
        """
        trace = []
        current = "borrower_enters"

        for agent in self.agents:
            trace.append(
                f"{current} → [{agent.name}:{agent.modality}] "
                f"→ inputs={agent.inputs} → outputs={agent.outputs}"
            )
            current = agent.handoff_to or "pipeline_end"

        trace.append(f"{current} → EXIT: outcome logged to MongoDB")
        return trace


def build_architecture() -> SystemArchitecture:
    arch = SystemArchitecture(
        storage_layers=[
            StorageLayer.TEMPORAL,
            StorageLayer.CONTEXT,
            StorageLayer.MONGODB,
            StorageLayer.LEARNING_STORE,
        ],
        truth_owner=StorageLayer.CONTEXT,
    )

    arch.agents = [
        AgentSpec(
            name="AssessmentAgent",
            modality="chat",
            responsibility="Verify identity, collect financial facts, determine resolution path",
            inputs=["borrower_id", "loan_id", "outstanding_amount", "days_past_due"],
            outputs=["identity_verified", "financial_situation", "resolution_path"],
            max_output_tokens=2000,
            handoff_to="ResolutionAgent",
        ),
        AgentSpec(
            name="ResolutionAgent",
            modality="voice",
            responsibility="Present policy-bound settlement offer, handle objections, get commitment",
            inputs=["handoff_summary", "resolution_path", "financial_situation"],
            outputs=["resolution_offer", "commitment_status"],
            max_output_tokens=300,   # voice = short turns
            handoff_to="FinalNoticeAgent",
        ),
        AgentSpec(
            name="FinalNoticeAgent",
            modality="chat",
            responsibility="State consequences, deliver final offer with deadline, route to complete/escalate",
            inputs=["handoff_summary", "resolution_offer", "commitment_status"],
            outputs=["final_outcome"],  # resolved | escalated
            max_output_tokens=2000,
            handoff_to=None,
        ),
    ]

    return arch


ARCHITECTURE = build_architecture()
