import os
import asyncio
import pytest
from unittest.mock import AsyncMock, patch

os.environ.setdefault("SKIP_LLM_JUDGE", "1")


@pytest.mark.asyncio
async def test_chat_enabled_after_final_notice_opened_before_recap():
    """
    UI behavior requirement:
    In `static/chat.html`, the chat input becomes enabled only after the system
    event `final_notice_opened` is handled. Therefore the websocket must send:
      final_notice_opened → (recap agent message) → (agreement question agent message)
    """
    # Ensure we're exercising the real-chat branch (ordering is only there).
    os.environ["USE_REAL_CHAT"] = "1"
    os.environ["USE_LLM_MOCK"] = "1"

    from src.temporal_activities import run_final_notice_stage
    from src.models import ResolutionPath
    from src.token_budget import TokenUsage

    events: list[tuple[str, str]] = []
    session_state = {
        "final_notice_recap_sent": False,
        "final_notice_opened_sent": False,
        "final_notice_opening_msg_sent": False,
    }

    async def _session_get(_borrower_id: str):
        return dict(session_state)

    async def _session_set(_borrower_id: str, data: dict):
        session_state.clear()
        session_state.update(data)

    async def _session_update(_borrower_id: str, **kwargs):
        session_state.update(kwargs)

    async def _push_stage_event(_workflow_id: str, ev: str, _content: str):
        events.append(("stage", ev))

    async def _push_agent_message(_workflow_id: str, content: str, _stage: str):
        # Reduce content to a recognizable key for assertions.
        if content.startswith("Call recap before we continue in chat:"):
            events.append(("agent", "recap"))
        elif "Do you agree to this" in content:
            events.append(("agent", "agreement_question"))
        else:
            events.append(("agent", "other"))

    async def _pop_message(_workflow_id: str, timeout: float = 300.0):
        # Provide explicit acceptance so FinalNoticeAgent generates the contract.
        await asyncio.sleep(0)
        return "Yes"

    async def _noop(*args, **kwargs):
        return None

    # Prevent any real LLM calls for the synthetic stage_open message.
    async def _mock_call_claude_with_tools(*args, **kwargs):
        return "Opening stub. COLLECTIONS_COMPLETE", TokenUsage(input_tokens=10, output_tokens=5)

    payload = {
        "input": {
            "borrower_id": "BRW-TEST-1",
            "loan_id": "LN-TEST-1",
            "workflow_id": "WF-TEST-1",
            "principal_amount": 100000,
            "outstanding_amount": 85000,
            "days_past_due": 90,
        },
        "prior_assessment": {
            "conversation": [],
            "question_state": {},
            "assessment_path": "installment",
        },
        "prior_resolution": {
            "outcome": "committed",
            "call_context": {
                "provider": "mock",
                "call_id": "mock_call_1",
                "phone_number": "+919999999999",
                "transcript_excerpt": "Borrower said: I can pay if terms are okay.",
            },
            "offer": {
                "path": ResolutionPath.INSTALLMENT.value,
                "discount_percentage": 0.0,
                "upfront_required": 42500,
                "monthly_payment": 22650,
                "tenure_months": 6,
                "deadline_days": 3,
                "valid_until": "2026-05-02T00:00:00+00:00",
            },
            "conversation": [
                {"role": "user", "content": "I can pay if terms are okay.", "stage": "resolution", "timestamp": ""},
                {"role": "assistant", "content": "Offer discussed.", "stage": "resolution", "timestamp": "", "advanced": False},
            ],
            "question_state": {},
        },
    }

    with patch("src.temporal_activities.session_get", new=_session_get), \
         patch("src.temporal_activities.session_set", new=_session_set), \
         patch("src.temporal_activities.session_update", new=_session_update), \
         patch("src.temporal_activities.push_stage_event", new=_push_stage_event), \
         patch("src.temporal_activities.push_agent_message", new=_push_agent_message), \
         patch("src.temporal_activities.pop_message", new=_pop_message), \
         patch("src.temporal_activities.upsert_borrower_case", new=_noop), \
         patch("src.temporal_activities.log_interaction", new=_noop), \
         patch("src.temporal_activities.log_outcome", new=_noop), \
         patch("src.temporal_activities.SelfLearningLoop.evaluate_interaction", new=AsyncMock(return_value=None)), \
         patch("src.agents.final_notice.FinalNoticeAgent._call_claude_with_tools", new=_mock_call_claude_with_tools):

        await run_final_notice_stage(payload)

    # The UI enables chat on `final_notice_opened`, so it must be before recap.
    idx_opened = next(i for i, (t, v) in enumerate(events) if t == "stage" and v == "final_notice_opened")
    idx_recap = next(i for i, (t, v) in enumerate(events) if t == "agent" and v == "recap")
    assert idx_opened < idx_recap, f"Expected final_notice_opened before recap. events={events}"

    # And it should also be before the agreement question.
    idx_agree = next(i for i, (t, v) in enumerate(events) if t == "agent" and v == "agreement_question")
    assert idx_opened < idx_agree, f"Expected final_notice_opened before agreement question. events={events}"

