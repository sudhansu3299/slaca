"""
Temporal activities.

Two modes per chat stage (Assessment / Final Notice):

  USE_REAL_CHAT=0  →  PersonaScript provides borrower turns (simulation/testing)
  USE_REAL_CHAT=1  →  Activity polls a Redis-backed message queue that the
                       WebSocket server writes to when the borrower types

Voice stage always uses Vapi (VOICE_PROVIDER=vapi) or mock.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from temporalio import activity
from temporalio.client import Client
from temporalio.worker import Worker

from src.agents.assessment import AssessmentAgent
from src.agents.resolution import ResolutionAgent
from src.agents.final_notice import FinalNoticeAgent
from src.handoff import HandoffBuilder
from src.models import (
    AssessmentData, ConversationContext, ResolutionOffer, ResolutionPath, Stage,
)
from src.self_learning.eval import SelfLearningLoop
from src.simulation import PersonaScript, PersonaType, SimulationEngine
from src.token_budget import CostTracker, TokenUsage
from src.voice import VoiceProvider, VoiceProviderName, CallConfig, build_voice_metadata


TASK_QUEUE = os.getenv("TASK_QUEUE", "collections-queue")

# ── Message bus (shared with chat_server) ────────────────────────── #
# Import once at module level so the same in-process queues are used.
from src.message_bus import (
    deliver_borrower_message as push_message,
    wait_for_borrower_message as pop_message,
    push_agent_message,
    push_stage_event,
)


# ── Voice provider factory ────────────────────────────────────────── #

def _build_voice_provider() -> VoiceProvider:
    provider_name = os.getenv("VOICE_PROVIDER", "mock").lower()

    if provider_name == "retell":
        api_key = os.getenv("RETELL_API_KEY", "")
        agent_id = os.getenv("RETELL_AGENT_ID", "")
        caller = os.getenv("CALLER_NUMBER", "")
        if not all([api_key, agent_id, caller]):
            missing = [k for k, v in [("RETELL_API_KEY", api_key),
                                       ("RETELL_AGENT_ID", agent_id),
                                       ("CALLER_NUMBER", caller)] if not v]
            raise RuntimeError(f"VOICE_PROVIDER=retell missing: {missing}")
        return VoiceProvider(CallConfig(provider=VoiceProviderName.RETELL,
                                        agent_id=agent_id, caller_number=caller,
                                        api_key=api_key))

    if provider_name == "vapi":
        return VoiceProvider(CallConfig(
            provider=VoiceProviderName.VAPI,
            agent_id=os.getenv("VAPI_ASSISTANT_ID", ""),
            caller_number=os.getenv("CALLER_NUMBER", ""),
            api_key=os.getenv("VAPI_API_KEY", ""),
        ))

    return VoiceProvider()  # MOCK


# ── Common helpers ────────────────────────────────────────────────── #

def _borrower_persona(data: Optional[dict] = None) -> PersonaType:
    if data and data.get("persona"):
        p = data["persona"].lower()
    else:
        p = os.getenv("SIM_PERSONA", "cooperative").lower()
    try:
        return PersonaType(p)
    except ValueError:
        return PersonaType.COOPERATIVE


def _make_context(input_data: dict, prior: Optional[dict] = None) -> ConversationContext:
    ctx = ConversationContext(
        borrower_id=input_data["borrower_id"],
        loan_id=input_data["loan_id"],
    )
    ctx.assessment_data = AssessmentData(
        borrower_id=input_data["borrower_id"],
        loan_id=input_data["loan_id"],
        principal_amount=input_data["principal_amount"],
        outstanding_amount=input_data["outstanding_amount"],
        days_past_due=input_data["days_past_due"],
    )
    if prior:
        if prior.get("question_state"):
            ctx.question_state = prior["question_state"]
        if prior.get("assessment_path"):
            try:
                ctx.assessment_data.resolution_path = ResolutionPath(prior["assessment_path"])
                ctx.assessment_data.identity_verified = True
            except ValueError:
                pass
        if prior.get("conversation"):
            ctx.conversation_history = list(prior["conversation"])
    return ctx


def _coerce_input(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if hasattr(raw, "__dataclass_fields__"):
        return asdict(raw)
    return raw


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Chat input provider ───────────────────────────────────────────── #

async def _get_user_input(
    workflow_id: str,
    borrower: PersonaScript,
    agent_name: str,
    use_real_chat: bool,
    timeout: float = 300.0,
) -> Optional[str]:
    """
    Get the next borrower message.
    - Real chat:  wait for WebSocket push (timeout → None → no_response)
    - Simulation: get next scripted line
    """
    if use_real_chat:
        return await pop_message(workflow_id, timeout=timeout)
    return borrower.respond(agent_name)


# ──────────────────────────────────────────────────────────── #
# Activity: run_assessment_stage
# ──────────────────────────────────────────────────────────── #

@activity.defn(name="run_assessment_stage")
async def run_assessment_stage(payload: Any) -> dict:
    # Temporal may deliver the dataclass directly or as a dict.
    # Workflow calls: execute_activity("run_assessment_stage", {"input_data": inp, "attempt": 1})
    if hasattr(payload, "__dataclass_fields__"):
        # Dataclass delivered directly (older Temporal behaviour)
        p = {"input_data": asdict(payload), "attempt": 1}
    elif isinstance(payload, dict):
        p = payload
    else:
        p = {"input_data": payload, "attempt": 1}

    data = _coerce_input(p.get("input_data", p))
    attempt = p.get("attempt", 1)

    use_mock = os.getenv("USE_LLM_MOCK", "1") == "1"
    use_real_chat = os.getenv("USE_REAL_CHAT", "0") == "1"
    workflow_id = data.get("workflow_id") or data.get("borrower_id", "unknown")

    persona = _borrower_persona(data)
    borrower = PersonaScript(persona)
    llm_mock = SimulationEngine.get_mock_llm_responses(persona).get("AssessmentAgent", [])

    cost = CostTracker()
    agent = AssessmentAgent(cost_tracker=cost)
    ctx = _make_context(data)
    ctx.current_stage = Stage.ASSESSMENT
    loop = SelfLearningLoop("AssessmentAgent")

    llm_iter = iter(llm_mock)
    outcome = "incomplete"

    # On first turn, emit the opening question immediately (real chat mode)
    if use_real_chat:
        opening = "Last 4 digits of your loan account and your birth year."
        # Push the opening question to the WebSocket
        from src.message_bus import push_agent_message
        await push_agent_message(workflow_id, opening, Stage.ASSESSMENT.value)

    for turn in range(12):
        user_input = await _get_user_input(
            workflow_id, borrower, "AssessmentAgent", use_real_chat, timeout=300.0
        )

        if user_input is None:
            outcome = "no_response"
            break

        ctx.conversation_history.append({
            "role": "user", "content": user_input,
            "stage": Stage.ASSESSMENT.value, "timestamp": _ts(),
        })

        if use_mock:
            text = next(llm_iter, "Please continue.")
            _orig = agent._call_claude
            async def _mock(*a, **kw):
                return text, TokenUsage(input_tokens=250, output_tokens=60)
            agent._call_claude = _mock
            try:
                resp = await agent.process(ctx, user_input)
            finally:
                agent._call_claude = _orig
        else:
            resp = await agent.process(ctx, user_input)

        ctx.conversation_history.append({
            "role": "assistant", "content": resp.message,
            "stage": Stage.ASSESSMENT.value, "timestamp": _ts(),
            "advanced": resp.should_advance,
        })

        # Push agent reply to WebSocket
        if use_real_chat:
            from src.message_bus import push_agent_message
            await push_agent_message(workflow_id, resp.message, Stage.ASSESSMENT.value)

        await loop.evaluate_interaction(
            ctx.conversation_history,
            "success" if resp.should_advance else None,
        )

        if resp.should_advance:
            outcome = "success"
            break

        # Force-advance if all required facts are already collected
        if turn >= 3 and agent.all_required_facts_present(ctx):
            activity.logger.info(
                f"[assessment] All facts present at turn {turn + 1}, forcing advance"
            )
            outcome = "success"
            if ctx.assessment_data and not ctx.assessment_data.resolution_path:
                qt = agent._get_question_tracker(ctx)
                answered = qt.get_answered()
                income = float(answered.get("monthly_income", "0") or "0")
                employment = answered.get("employment_status", "")
                cash_flow = answered.get("cash_flow_issue", "")
                if employment == "unemployed" or cash_flow in ("yes", "severe"):
                    ctx.assessment_data.resolution_path = ResolutionPath.HARDSHIP
                elif income > 40000:
                    ctx.assessment_data.resolution_path = ResolutionPath.INSTALLMENT
                else:
                    ctx.assessment_data.resolution_path = ResolutionPath.HARDSHIP
            break

    # Notify UI that assessment is complete
    if use_real_chat and outcome == "success":
        from src.message_bus import push_agent_message, push_stage_event
        await push_stage_event(workflow_id, "assessment_complete",
                               "Assessment complete. You will receive a call shortly.")

    summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
    handoff_block = summary.to_prompt_block()
    assessment_path = (ctx.assessment_data.resolution_path.value
                       if ctx.assessment_data and ctx.assessment_data.resolution_path else None)

    return {
        "outcome": outcome,
        "conversation": ctx.conversation_history,
        "handoff_block": handoff_block,
        "question_state": ctx.question_state,
        "assessment_path": assessment_path,
        "tokens_in": cost.total_tokens.input_tokens,
        "tokens_out": cost.total_tokens.output_tokens,
    }


# ──────────────────────────────────────────────────────────── #
# Activity: run_resolution_stage  (voice)
# ──────────────────────────────────────────────────────────── #

@activity.defn(name="run_resolution_stage")
async def run_resolution_stage(payload: Any) -> dict:
    data = _coerce_input(payload["input"] if isinstance(payload, dict) else payload.input)
    prior = payload["prior"] if isinstance(payload, dict) else payload.prior

    use_mock = os.getenv("USE_LLM_MOCK", "1") == "1"
    use_real_voice = os.getenv("VOICE_PROVIDER", "mock").lower() != "mock"
    workflow_id = data.get("workflow_id") or data.get("borrower_id", "unknown")

    persona = _borrower_persona(data)
    borrower = PersonaScript(persona)
    llm_mock = SimulationEngine.get_mock_llm_responses(persona).get("ResolutionAgent", [])

    cost = CostTracker()
    agent = ResolutionAgent(cost_tracker=cost)
    ctx = _make_context(data, prior)
    ctx.current_stage = Stage.RESOLUTION

    summary = HandoffBuilder.build(ctx, Stage.ASSESSMENT, Stage.RESOLUTION)
    voice = _build_voice_provider()
    phone_number = data.get("phone_number", os.getenv("BORROWER_PHONE", ""))

    call = None
    call_context: dict[str, Any] = {
        "provider": os.getenv("VOICE_PROVIDER", "mock").lower(),
        "call_id": "",
        "phone_number": phone_number,
        "transcript_excerpt": "",
        "transcript_chars": 0,
    }
    try:
        call = await voice.initiate_call(phone_number=phone_number, handoff_summary=summary)
        call_context["call_id"] = call.call_id
        call_context["provider"] = str(call.provider.value if hasattr(call.provider, "value") else call.provider)
        activity.logger.info(
            f"[voice] 📞 call_id={call.call_id} to={phone_number}"
        )
        print(f"[voice] 📞 Calling {phone_number} (call_id={call.call_id})")
    except Exception as e:
        activity.logger.warning(f"[voice] call init failed: {e}. Using script fallback.")

    loop = SelfLearningLoop("ResolutionAgent")
    llm_iter = iter(llm_mock)
    outcome = "no_outcome"
    prior_len = len(ctx.conversation_history)

    if use_real_voice and call and not use_mock:
        voice_result = await _run_real_voice_call(call, voice, agent, ctx, loop)
        outcome = voice_result["outcome"]
        transcript = voice_result.get("transcript", "")
        call_context["transcript_chars"] = len(transcript)
        call_context["transcript_excerpt"] = transcript[:1000]
    else:
        for turn in range(6):
            user_input = borrower.respond("ResolutionAgent")
            ctx.conversation_history.append({
                "role": "user", "content": user_input,
                "stage": Stage.RESOLUTION.value, "timestamp": _ts(),
            })

            if use_mock:
                text = next(llm_iter, "Please continue.")
                _orig = agent._call_claude
                async def _mock(*a, **kw):
                    return text, TokenUsage(input_tokens=200, output_tokens=50)
                agent._call_claude = _mock
                try:
                    resp = await agent.process(ctx, user_input)
                finally:
                    agent._call_claude = _orig
            else:
                resp = await agent.process(ctx, user_input)

            ctx.conversation_history.append({
                "role": "assistant", "content": resp.message,
                "stage": Stage.RESOLUTION.value, "timestamp": _ts(),
                "advanced": resp.should_advance,
            })

            await loop.evaluate_interaction(
                ctx.conversation_history,
                "committed" if resp.should_advance else None,
            )

            if resp.should_advance:
                outcome = "refused" if ctx.resolution_outcome == "refused" else "committed"
                break

    if call:
        await voice.end_call(call)

    summary2 = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
    handoff_block = summary2.to_prompt_block()
    offer_dict = ctx.resolution_offer.model_dump() if ctx.resolution_offer else None

    return {
        "outcome": outcome,
        "call_id": call.call_id if call else "",
        "call_context": call_context,
        "conversation": ctx.conversation_history[prior_len:],
        "handoff_block": handoff_block,
        "question_state": ctx.question_state,
        "offer": offer_dict,
        "tokens_in": cost.total_tokens.input_tokens,
        "tokens_out": cost.total_tokens.output_tokens,
    }


async def _run_real_voice_call(call, voice, agent, ctx, loop,
                                poll_interval=5, max_wait=300) -> dict:
    print(f"[voice] ⏳ Waiting for call {call.call_id} to complete...")
    elapsed = 0
    while elapsed < max_wait:
        await asyncio.sleep(poll_interval)
        elapsed += poll_interval
        transcript = await voice.get_transcript(call)
        if transcript:
            print(f"[voice] ✅ Call complete. Transcript: {len(transcript)} chars")
            break
        if elapsed % 30 == 0:
            print(f"[voice] ⏳ Still waiting... ({elapsed}s)")
    else:
        print("[voice] ⚠️  Timed out")
        return {"outcome": "no_outcome", "transcript": ""}

    turns = _parse_transcript(transcript)
    outcome = "no_outcome"
    for speaker, text in turns:
        if speaker.lower() in ("user", "borrower", "customer"):
            ctx.conversation_history.append({
                "role": "user", "content": text,
                "stage": Stage.RESOLUTION.value, "timestamp": _ts(),
                "source": "voice_transcript",
            })
            resp = await agent.process(ctx, text)
            ctx.conversation_history.append({
                "role": "assistant", "content": resp.message,
                "stage": Stage.RESOLUTION.value, "timestamp": _ts(),
                "advanced": resp.should_advance,
            })
            await loop.evaluate_interaction(ctx.conversation_history,
                                            "committed" if resp.should_advance else None)
            if resp.should_advance:
                outcome = "refused" if ctx.resolution_outcome == "refused" else "committed"
                break
    return {"outcome": outcome, "transcript": transcript}


def _parse_transcript(transcript: str) -> list[tuple[str, str]]:
    try:
        data = json.loads(transcript)
        if isinstance(data, list):
            return [(t.get("role", ""), t.get("content", "")) for t in data]
    except (json.JSONDecodeError, TypeError):
        pass
    turns = []
    for line in transcript.splitlines():
        line = line.strip()
        if line and ":" in line:
            speaker, _, text = line.partition(":")
            turns.append((speaker.strip(), text.strip()))
    return turns


# ──────────────────────────────────────────────────────────── #
# Activity: run_final_notice_stage
# ──────────────────────────────────────────────────────────── #

@activity.defn(name="run_final_notice_stage")
async def run_final_notice_stage(payload: Any) -> dict:
    data = _coerce_input(payload["input"] if isinstance(payload, dict) else payload.input)
    prior_assessment = payload["prior_assessment"]
    prior_resolution = payload["prior_resolution"]

    use_mock = os.getenv("USE_LLM_MOCK", "1") == "1"
    use_real_chat = os.getenv("USE_REAL_CHAT", "0") == "1"
    workflow_id = data.get("workflow_id") or data.get("borrower_id", "unknown")

    persona = _borrower_persona(data)
    borrower = PersonaScript(persona)
    llm_mock = SimulationEngine.get_mock_llm_responses(persona).get("FinalNoticeAgent", [])

    cost = CostTracker()
    agent = FinalNoticeAgent(cost_tracker=cost)

    full_prior = dict(prior_assessment)
    call_context = prior_resolution.get("call_context") or {}
    full_prior["conversation"] = (
        prior_assessment.get("conversation", [])
        + prior_resolution.get("conversation", [])
    )
    full_prior["question_state"] = (prior_resolution.get("question_state")
                                    or prior_assessment.get("question_state"))

    ctx = _make_context(data, full_prior)
    ctx.current_stage = Stage.FINAL_NOTICE

    if prior_resolution.get("offer"):
        ctx.resolution_offer = ResolutionOffer(**prior_resolution["offer"])

    if call_context:
        call_handoff_lines = [
            "[Voice Call Context]",
            f"Provider: {call_context.get('provider', 'unknown')}",
            f"Call ID: {call_context.get('call_id', 'unknown')}",
            f"Phone: {call_context.get('phone_number', 'unknown')}",
            f"Resolution Outcome: {prior_resolution.get('outcome', 'unknown')}",
        ]
        transcript_excerpt = str(call_context.get("transcript_excerpt", "")).strip()
        if transcript_excerpt:
            call_handoff_lines.append("Transcript Excerpt:")
            call_handoff_lines.append(transcript_excerpt)
        call_handoff = "\n".join(call_handoff_lines)
        ctx.conversation_history.append({
            "role": "assistant",
            "content": call_handoff,
            "stage": Stage.RESOLUTION.value,
            "timestamp": _ts(),
            "source": "voice_call_handoff",
        })

    loop = SelfLearningLoop("FinalNoticeAgent")
    llm_iter = iter(llm_mock)
    outcome = "escalated"
    prior_len = len(ctx.conversation_history)

    # Emit the final notice opening immediately (agent generates it from handoff context)
    if use_real_chat:
        # Trigger the agent with a synthetic "stage_open" so it produces the notice
        opening_resp = await agent.process(ctx, "[STAGE_OPEN: Generate the final notice now.]")
        opening_msg = opening_resp.message
        from src.message_bus import push_agent_message, push_stage_event
        # Tell the UI we're now in final_notice (re-enables the input)
        await push_stage_event(workflow_id, "final_notice_opened",
                               "The resolution call has concluded. Your final notice is below.")
        await push_agent_message(workflow_id, opening_msg, Stage.FINAL_NOTICE.value)

    for turn in range(5):
        user_input = await _get_user_input(
            workflow_id, borrower, "FinalNoticeAgent", use_real_chat, timeout=300.0
        )

        if user_input is None:
            outcome = "escalated"
            break

        ctx.conversation_history.append({
            "role": "user", "content": user_input,
            "stage": Stage.FINAL_NOTICE.value, "timestamp": _ts(),
        })

        if use_mock:
            text = next(llm_iter, "Please continue.")
            _orig = agent._call_claude
            async def _mock(*a, **kw):
                return text, TokenUsage(input_tokens=300, output_tokens=70)
            agent._call_claude = _mock
            try:
                resp = await agent.process(ctx, user_input)
            finally:
                agent._call_claude = _orig
        else:
            resp = await agent.process(ctx, user_input)

        ctx.conversation_history.append({
            "role": "assistant", "content": resp.message,
            "stage": Stage.FINAL_NOTICE.value, "timestamp": _ts(),
            "advanced": resp.should_advance,
        })

        if use_real_chat:
            from src.message_bus import push_agent_message
            await push_agent_message(workflow_id, resp.message, Stage.FINAL_NOTICE.value)

        await loop.evaluate_interaction(
            ctx.conversation_history,
            "resolved" if resp.should_advance and ctx.final_notice_outcome == "resolved"
            else "escalated",
        )

        if resp.should_advance:
            outcome = ctx.final_notice_outcome or "resolved"
            break

    if use_real_chat:
        from src.message_bus import push_stage_event
        await push_stage_event(workflow_id, f"final_notice_{outcome}",
                               "Case resolved." if outcome == "resolved"
                               else "This matter has been escalated to legal.")

    return {
        "outcome": outcome,
        "conversation": ctx.conversation_history[prior_len:],
        "tokens_in": cost.total_tokens.input_tokens,
        "tokens_out": cost.total_tokens.output_tokens,
    }


# ──────────────────────────────────────────────────────────── #
# Side-effect activities (EXIT nodes in the diagram)
# ──────────────────────────────────────────────────────────── #

@activity.defn(name="log_agreement")
async def log_agreement(payload: dict) -> dict:
    """EXIT: Log agreement (deal agreed after voice call)."""
    def _safe_part(value: str) -> str:
        keep = []
        for ch in value:
            if ch.isalnum() or ch in ("-", "_", "."):
                keep.append(ch)
            else:
                keep.append("_")
        return "".join(keep).strip("_") or "unknown"

    call_id = str(payload.get("call_id") or "").strip()
    if not call_id:
        borrower = _safe_part(str(payload.get("borrower_id", "unknown")))
        call_id = f"missing_call_id_{borrower}"

    root = Path(os.getenv("CALL_LOG_DIR", "call_logs"))
    log_dir = root / _safe_part(call_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_record = {
        "event": "agreement_logged",
        "timestamp": _ts(),
        "borrower_id": payload.get("borrower_id"),
        "loan_id": payload.get("loan_id"),
        "workflow_id": payload.get("workflow_id"),
        "call_id": call_id,
        "offer": payload.get("offer"),
        "conversation": payload.get("conversation", []),
    }
    agreement_file = log_dir / "agreement.json"
    agreement_file.write_text(
        json.dumps(log_record, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    activity.logger.info(
        f"[EXIT:log_agreement] borrower={payload['borrower_id']} "
        f"call_id={call_id} offer={payload.get('offer')} "
        f"log_file={agreement_file}"
    )
    print(
        f"✅ [EXIT] Agreement logged for borrower {payload['borrower_id']} "
        f"at {agreement_file}"
    )
    return {"ok": True, "event": "agreement_logged", **{k: payload[k]
            for k in ("borrower_id", "loan_id") if k in payload},
            "call_id": call_id, "log_dir": str(log_dir), "log_file": str(agreement_file)}


@activity.defn(name="log_resolution")
async def log_resolution(payload: dict) -> dict:
    """EXIT: Log resolution (resolved via Final Notice chat)."""
    activity.logger.info(
        f"[EXIT:log_resolution] borrower={payload['borrower_id']}"
    )
    print(f"✅ [EXIT] Resolution logged for borrower {payload['borrower_id']}")
    return {"ok": True, "event": "resolution_logged", **{k: payload[k]
            for k in ("borrower_id", "loan_id") if k in payload}}


@activity.defn(name="create_legal_referral")
async def create_legal_referral(payload: dict) -> dict:
    """EXIT: Flag for legal/write-off."""
    activity.logger.info(
        f"[EXIT:legal_referral] borrower={payload['borrower_id']} reason={payload['reason']}"
    )
    print(f"⚠️  [EXIT] Legal referral created for borrower {payload['borrower_id']}: {payload['reason']}")
    return {"ok": True, "referral_id": f"REF-{payload['borrower_id']}"}


@activity.defn(name="update_loan_status")
async def update_loan_status(payload: dict) -> dict:
    activity.logger.info(
        f"[update_loan_status] borrower={payload['borrower_id']} status={payload['status']}"
    )
    return {"ok": True, **payload}


@activity.defn(name="report_to_credit_bureau")
async def report_to_credit_bureau(payload: dict) -> dict:
    activity.logger.info(f"[credit_bureau] borrower={payload['borrower_id']}")
    return {"ok": True}


# ──────────────────────────────────────────────────────────── #
# Worker bootstrap
# ──────────────────────────────────────────────────────────── #

ALL_ACTIVITIES = [
    run_assessment_stage,
    run_resolution_stage,
    run_final_notice_stage,
    log_agreement,
    log_resolution,
    create_legal_referral,
    update_loan_status,
    report_to_credit_bureau,
]


async def start_worker(task_queue: str = TASK_QUEUE) -> None:
    from src.temporal_workflow import CollectionsWorkflow

    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    client = await Client.connect(address)

    worker = Worker(
        client,
        task_queue=task_queue,
        workflows=[CollectionsWorkflow],
        activities=ALL_ACTIVITIES,
    )
    print(f"[worker] listening on task queue '{task_queue}' (Temporal={address})")
    await worker.run()


if __name__ == "__main__":
    asyncio.run(start_worker())
