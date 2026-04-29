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
from src.token_budget import CostTracker, TokenUsage, estimate_tokens
from src.voice import VoiceProvider, VoiceProviderName, CallConfig, build_voice_metadata

# ── Data layer (Redis + MongoDB) ──────────────────────────────────── #
from src.data_layer import (
    session_set, session_get, session_update,
    summary_append, get_summary_with_fallback,
    retry_count, retry_increment, retry_reset,
    lock_acquire, lock_release,
    log_interaction, log_outcome, upsert_borrower_case,
)


TASK_QUEUE = os.getenv("TASK_QUEUE", "collections-queue")

# ── Message bus (shared with chat_server) ────────────────────────── #
from src.message_bus import (
    deliver_borrower_message as push_message,
    wait_for_borrower_message as pop_message,
    push_agent_message,
    push_stage_event,
)
from src.self_learning.improvement_pipeline import get_prompt_version


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


def _schedule_learning_followups(resolved: bool) -> None:
    """Fan out completion signals to feeder + regression monitor."""
    try:
        from src.self_learning.feeder import on_conversation_complete

        asyncio.create_task(on_conversation_complete())
    except Exception as e:
        activity.logger.debug("[self_learning] feeder notify skipped: %s", e)

    try:
        from src.self_learning.meta_evaluator import on_conversation_complete as on_meta_complete

        asyncio.create_task(on_meta_complete())
    except Exception as e:
        activity.logger.debug("[self_learning] meta-evaluator notify skipped: %s", e)

    try:
        from src.self_learning.regression_monitor import record_outcome

        for agent_name in ("AssessmentAgent", "ResolutionAgent", "FinalNoticeAgent"):
            asyncio.create_task(record_outcome(agent_name, resolved))
    except Exception as e:
        activity.logger.debug("[self_learning] regression notify skipped: %s", e)


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
    borrower_id = data.get("borrower_id", workflow_id)

    # ── Redis: set session state ───────────────────────────── #
    await session_set(borrower_id, {
        "stage": "ASSESSMENT",
        "status": "ACTIVE",
        "attempts": attempt,
        "last_agent": "AssessmentAgent",
        "final_notice_recap_sent": False,
        "final_notice_opened_sent": False,
        "final_notice_opening_msg_sent": False,
    })
    await upsert_borrower_case(borrower_id, "ASSESSMENT", "ACTIVE", attempt)

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

    # On first turn, emit the opening disclosure immediately (real chat mode)
    if use_real_chat:
        opening = (
            "This is an automated AI collections agent acting on behalf of Riverline. "
            "This conversation is being logged.\n\n"
            "Please provide the last 4 digits of your loan account number and your birth year."
        )
        await push_agent_message(workflow_id, opening, Stage.ASSESSMENT.value)
        ctx.assessment_opened = True

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
            _orig = agent._call_claude_with_tools
            async def _mock(*a, **kw):
                return text, TokenUsage(input_tokens=250, output_tokens=60)
            agent._call_claude_with_tools = _mock
            try:
                resp = await agent.process(ctx, user_input)
            finally:
                agent._call_claude_with_tools = _orig
        else:
            resp = await agent.process(ctx, user_input)

        ctx.conversation_history.append({
            "role": "assistant", "content": resp.message,
            "stage": Stage.ASSESSMENT.value, "timestamp": _ts(),
            "advanced": resp.should_advance,
        })

        # Push agent reply to WebSocket
        if use_real_chat:
            await push_agent_message(workflow_id, resp.message, Stage.ASSESSMENT.value)

        # ── MongoDB: log this turn (async, non-blocking) ─── #
        _tin = resp.tokens_used.input_tokens if resp.tokens_used else 0
        _tout = resp.tokens_used.output_tokens if resp.tokens_used else 0
        asyncio.create_task(log_interaction(
            borrower_id=borrower_id,
            agent_name="AssessmentAgent",
            agent_version="v1.0",
            prompt_version=get_prompt_version("AssessmentAgent"),
            model=agent.model,
            model_params={"temperature": 0.1, "max_tokens": 2000},
            input_text=user_input,
            output_text=resp.message,
            input_summary=user_input[:200],
            structured_context={
                "intent": "assessment",
                "stage": Stage.ASSESSMENT.value,
                "advanced": resp.should_advance,
            },
            decision="advance" if resp.should_advance else "continue",
            trace_id=workflow_id,
            input_tokens=_tin,
            output_tokens=_tout,
        ))

        # ── Redis: update summary ─────────────────────────── #
        asyncio.create_task(summary_append(borrower_id, {
            "agent": "AssessmentAgent",
            "agent_version": "v1.0",
            "summary": resp.message[:200],
            "structured_context": {"intent": "assessment", "turn": turn},
            "timestamp": _ts(),
        }))

        await loop.evaluate_interaction(
            ctx.conversation_history,
            "success" if resp.should_advance else None,
        )

        if resp.should_advance:
            outcome = "success"
            break

        # Force-advance only if all facts are present AND identity is verified by DB
        if turn >= 3 and agent.all_required_facts_present(ctx) and (
            ctx.assessment_data is not None and ctx.assessment_data.identity_verified
        ):
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
        await push_stage_event(workflow_id, "assessment_complete",
                               "Assessment complete. You will receive a call shortly.")

    # ── Redis: update session after stage ──────────────────── #
    final_status = "ACTIVE" if outcome == "success" else "WRITE_OFF"
    await session_update(borrower_id, stage="ASSESSMENT", status=final_status)
    await upsert_borrower_case(borrower_id, "ASSESSMENT", final_status, attempt)

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
    borrower_id = data.get("borrower_id", workflow_id)

    # ── Redis: update session to RESOLUTION ───────────────── #
    await session_update(borrower_id, stage="RESOLUTION", last_agent="ResolutionAgent")
    await upsert_borrower_case(borrower_id, "RESOLUTION", "ACTIVE")

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
    assessment_handoff_tokens = estimate_tokens(summary.to_prompt_block())

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

    # Always generate the offer upfront so it is available for final notice
    # regardless of whether the resolution path is voice, mock, or real chat.
    if not ctx.resolution_offer:
        ctx.resolution_offer = agent.generate_offer(ctx)

    if use_real_voice and call and not use_mock:
        voice_result = await _run_real_voice_call(call, voice, agent, ctx, loop)
        outcome = voice_result["outcome"]
        transcript = voice_result.get("transcript", "")
        call_context["transcript_chars"] = len(transcript)
        call_context["transcript_excerpt"] = transcript[:1000]
        for rec in voice_result.get("turn_logs", []):
            asyncio.create_task(log_interaction(
                borrower_id=borrower_id,
                agent_name="ResolutionAgent",
                agent_version="v1.0",
                prompt_version=get_prompt_version("ResolutionAgent"),
                model="vapi/" + os.getenv("VAPI_ASSISTANT_ID", "mock"),
                model_params={"modality": "voice", "source": "voice_turn"},
                input_text=rec.get("input", ""),
                output_text=rec.get("output", ""),
                structured_context={
                    "intent": "resolution_turn",
                    "stage": Stage.RESOLUTION.value,
                    "call_id": call_context.get("call_id", ""),
                    "offer": (ctx.resolution_offer.model_dump() if ctx.resolution_offer else None),
                },
                decision=rec.get("decision", "continue"),
                confidence=1.0 if rec.get("decision") in ("committed", "refused") else 0.0,
                trace_id=workflow_id,
                input_tokens=int(rec.get("input_tokens", 0) or 0),
                output_tokens=int(rec.get("output_tokens", 0) or 0),
            ))
    elif use_real_voice and not call:
        # The real call failed to initiate (Vapi/Retell error).
        # Do NOT fall through to the PersonaScript loop — that would produce a
        # fake "committed" outcome from a mock persona and cause the workflow to
        # exit at resolution without ever showing the final notice or contract.
        # Instead, report no_outcome so the workflow routes to Final Notice.
        activity.logger.warning(
            "[voice] call initiation failed — reporting no_outcome so workflow "
            "proceeds to final_notice stage"
        )
        outcome = "no_outcome"
    else:
        for turn in range(6):
            user_input = borrower.respond("ResolutionAgent")
            ctx.conversation_history.append({
                "role": "user", "content": user_input,
                "stage": Stage.RESOLUTION.value, "timestamp": _ts(),
            })

            if use_mock:
                text = next(llm_iter, "Please continue.")
                _orig = agent._call_claude_with_tools
                async def _mock(*a, **kw):
                    return text, TokenUsage(input_tokens=200, output_tokens=50)
                agent._call_claude_with_tools = _mock
                try:
                    resp = await agent.process(ctx, user_input)
                finally:
                    agent._call_claude_with_tools = _orig
            else:
                resp = await agent.process(ctx, user_input)

            ctx.conversation_history.append({
                "role": "assistant", "content": resp.message,
                "stage": Stage.RESOLUTION.value, "timestamp": _ts(),
                "advanced": resp.should_advance,
            })

            _tin = resp.tokens_used.input_tokens if resp.tokens_used else 0
            _tout = resp.tokens_used.output_tokens if resp.tokens_used else 0
            asyncio.create_task(log_interaction(
                borrower_id=borrower_id,
                agent_name="ResolutionAgent",
                agent_version="v1.0",
                prompt_version=get_prompt_version("ResolutionAgent"),
                model=agent.model,
                model_params={"temperature": 0.08, "max_tokens": 2000, "modality": "voice"},
                input_text=user_input,
                output_text=resp.message,
                structured_context={
                    "intent": "resolution_turn",
                    "stage": Stage.RESOLUTION.value,
                    "call_id": call_context.get("call_id", ""),
                    "offer": (ctx.resolution_offer.model_dump() if ctx.resolution_offer else None),
                },
                decision="committed" if resp.should_advance and ctx.resolution_outcome != "refused"
                         else ("refused" if resp.should_advance else "continue"),
                confidence=1.0 if resp.should_advance else 0.0,
                trace_id=workflow_id,
                input_tokens=_tin,
                output_tokens=_tout,
            ))

            await loop.evaluate_interaction(
                ctx.conversation_history,
                "committed" if resp.should_advance else None,
            )

            if resp.should_advance:
                outcome = "refused" if ctx.resolution_outcome == "refused" else "committed"
                # End the call immediately — don't leave it open after a terminal outcome
                if call:
                    await voice.end_call(call)
                    call = None   # mark as ended so the outer block doesn't double-call
                    print(f"[voice] 📵 Call terminated after terminal outcome: {outcome}")
                break

    if call:
        await voice.end_call(call)

    # ── Redis + MongoDB: log resolution outcome ────────────── #
    resolution_status = "ACTIVE"  # will be updated by final_notice
    await session_update(borrower_id, stage="RESOLUTION", status=resolution_status,
                         last_agent="ResolutionAgent")

    # Log resolution interaction
    offer_dict = ctx.resolution_offer.model_dump() if ctx.resolution_offer else None
    voice_turns = ctx.conversation_history[prior_len:]
    voice_summary = f"Voice call outcome: {outcome}. Turns: {len(voice_turns)}."
    asyncio.create_task(log_interaction(
        borrower_id=borrower_id,
        agent_name="ResolutionAgent",
        agent_version="v1.0",
        prompt_version=get_prompt_version("ResolutionAgent"),
        model="vapi/" + os.getenv("VAPI_ASSISTANT_ID", "mock"),
        model_params={"modality": "voice"},
        input_text=str(call_context.get("transcript_excerpt", "")[:500]),
        output_text=voice_summary,
        input_summary=voice_summary,
        structured_context={
            "intent": outcome,
            "stage": Stage.RESOLUTION.value,
            "call_id": call_context.get("call_id", ""),
            "offer": offer_dict,
            "received_handoff_tokens": int(assessment_handoff_tokens),
            "handoff_from": "AssessmentAgent",
        },
        decision=outcome,
        confidence=1.0 if outcome == "committed" else 0.0,
        trace_id=workflow_id,
        input_tokens=0,
        output_tokens=0,
    ))

    summary2 = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
    handoff_block = summary2.to_prompt_block()

    return {
        "outcome": outcome,
        "call_id": call.call_id if call else "",
        "call_context": call_context,
        "conversation": voice_turns,
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
        print("[voice] ⚠️  Timed out — ending call")
        await voice.end_call(call)
        return {"outcome": "no_outcome", "transcript": "", "turn_logs": []}

    turns = _parse_transcript(transcript)
    outcome = "no_outcome"
    turn_logs: list[dict[str, Any]] = []
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
            turn_logs.append({
                "input": text,
                "output": resp.message,
                "input_tokens": (resp.tokens_used.input_tokens if resp.tokens_used else 0),
                "output_tokens": (resp.tokens_used.output_tokens if resp.tokens_used else 0),
                "decision": ("committed" if resp.should_advance and ctx.resolution_outcome != "refused"
                             else ("refused" if resp.should_advance else "continue")),
            })
            await loop.evaluate_interaction(ctx.conversation_history,
                                            "committed" if resp.should_advance else None)
            if resp.should_advance:
                outcome = "refused" if ctx.resolution_outcome == "refused" else "committed"
                # Terminal state reached — end the call immediately
                await voice.end_call(call)
                print(f"[voice] 📵 Call terminated after terminal outcome: {outcome}")
                break
    return {"outcome": outcome, "transcript": transcript, "turn_logs": turn_logs}


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


def _format_voice_to_chat_recap(summary: HandoffSummary, resolution_outcome: str) -> str:
    """Human-readable recap surfaced in chat after the voice call."""
    offer_parts: list[str] = []
    if summary.offer_upfront:
        offer_parts.append(f"upfront ₹{summary.offer_upfront:,.0f}")
    if summary.offer_monthly:
        months = summary.offer_tenure_months or 0
        offer_parts.append(f"monthly ₹{summary.offer_monthly:,.0f} for {months} months")
    offer_text = ", ".join(offer_parts) if offer_parts else "offer details unavailable"
    deadline_text = summary.offer_valid_until or "deadline previously shared on call"

    objections = summary.objections[-3:] if summary.objections else []
    objections_text = "; ".join(objections) if objections else "no major objections recorded"
    position = summary.intent or resolution_outcome or "unknown"

    return (
        "Call recap before we continue in chat:\n"
        f"- Offer discussed: {offer_text}\n"
        f"- Deadline: {deadline_text}\n"
        f"- Objections raised on call: {objections_text}\n"
        f"- Your stated position: {position}"
    )


def _build_legal_notice_contract_html(
    borrower_id: str,
    loan_id: str,
    resolution_outcome: str,
    offer: Optional[ResolutionOffer],
) -> str:
    """Fallback legal notice document shown when final notice is not accepted."""
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    issued = now.strftime("%d %B %Y")
    offer_line = "No settlement terms accepted by borrower."
    if offer:
        if offer.monthly_payment:
            offer_line = (
                f"Prior offer discussed: ₹{offer.upfront_required:,.0f} down + "
                f"₹{offer.monthly_payment:,.0f}/month × {offer.tenure_months} months."
            )
        else:
            offer_line = f"Prior offer discussed: ₹{offer.upfront_required:,.0f} one-time settlement."

    return f"""<div class="contract-doc">
  <div class="contract-header">
    <div class="contract-logo">⚖</div>
    <div>
      <div class="contract-title">LEGAL ESCALATION NOTICE</div>
      <div class="contract-subtitle">Final Notice Outcome — Case Escalated</div>
    </div>
    <div class="contract-stamp">ESCALATED</div>
  </div>

  <div class="contract-meta">
    <span><strong>Borrower ID:</strong> {borrower_id}</span>
    <span><strong>Loan Account:</strong> {loan_id}</span>
    <span><strong>Date Issued:</strong> {issued}</span>
  </div>

  <div class="contract-section">
    <div class="contract-section-title">1. CASE SUMMARY</div>
    <div class="contract-terms">
      Resolution outcome recorded: {resolution_outcome}.<br/>
      {offer_line}
    </div>
  </div>

  <div class="contract-section">
    <div class="contract-section-title">2. NEXT LEGAL ACTIONS</div>
    <ol class="contract-list">
      <li>Credit bureau reporting is initiated.</li>
      <li>Formal legal notice is issued within 7 days.</li>
      <li>Recovery proceedings may be filed before the competent court.</li>
      <li>Further collection actions continue as permitted by law.</li>
    </ol>
  </div>

  <div class="contract-section">
    <div class="contract-section-title">3. REINSTATEMENT OPTION</div>
    <div class="contract-note">
      If you choose to settle after escalation, contact the collections desk immediately
      to request a revised settlement review.
    </div>
  </div>

  <div class="contract-acceptance">
    <div class="contract-acceptance-label">Download this legal notice PDF for your records.</div>
    <button class="contract-download-btn" onclick="downloadContractPDF(this)">Download PDF</button>
  </div>
</div>"""


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
    borrower_id = data.get("borrower_id", workflow_id)

    # ── Redis: update session to FINAL_NOTICE ─────────────── #
    await session_update(borrower_id, stage="FINAL_NOTICE", last_agent="FinalNoticeAgent")
    await upsert_borrower_case(borrower_id, "FINAL_NOTICE", "ACTIVE")

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

    summary_voice_to_chat = HandoffBuilder.build(ctx, Stage.RESOLUTION, Stage.FINAL_NOTICE)
    resolution_handoff_tokens = estimate_tokens(summary_voice_to_chat.to_prompt_block())
    recap_message = _format_voice_to_chat_recap(
        summary_voice_to_chat,
        str(prior_resolution.get("outcome", "unknown")),
    )

    loop = SelfLearningLoop("FinalNoticeAgent")
    llm_iter = iter(llm_mock)
    outcome = "escalated"
    prior_len = len(ctx.conversation_history)
    _contract_html = ""   # populated when borrower accepts and tool runs

    # Emit the final notice opening immediately (agent generates it from handoff context)
    if use_real_chat:
        sess = await session_get(borrower_id) or {}
        recap_sent = bool(sess.get("final_notice_recap_sent"))
        opened_event_sent = bool(sess.get("final_notice_opened_sent"))
        opening_msg_sent = bool(sess.get("final_notice_opening_msg_sent"))

        # 1) Enable chat as soon as the voice call is finished.
        # This must happen before we push recap/question messages.
        if not opened_event_sent:
            await push_stage_event(
                workflow_id,
                "final_notice_opened",
                "The resolution call has concluded. Your final notice is below.",
            )
            await session_update(borrower_id, final_notice_opened_sent=True)

        # 2) Show the call recap once.
        if not recap_sent:
            await push_agent_message(
                workflow_id,
                recap_message,
                Stage.FINAL_NOTICE.value,
            )
            ctx.conversation_history.append({
                "role": "assistant",
                "content": recap_message,
                "stage": Stage.FINAL_NOTICE.value,
                "timestamp": _ts(),
                "source": "voice_to_chat_recap",
            })
            await session_update(borrower_id, final_notice_recap_sent=True)

        # 3) Ask for explicit agreement once.
        # If the UI already received the opening question (e.g. reconnect/retry),
        # don't re-generate or re-send it.
        if not opening_msg_sent:
            opening_resp = await agent.process(ctx, "[STAGE_OPEN: Generate the final notice now.]")
            opening_msg = opening_resp.message
            asyncio.create_task(log_interaction(
                borrower_id=borrower_id,
                agent_name="FinalNoticeAgent",
                agent_version="v1.0",
                prompt_version=get_prompt_version("FinalNoticeAgent"),
                model=agent.model,
                model_params={"temperature": 0.05, "max_tokens": 2000, "kind": "stage_open"},
                input_text="[STAGE_OPEN: Generate the final notice now.]",
                output_text=opening_msg,
                structured_context={
                    "intent": "final_notice_opening",
                    "stage": Stage.FINAL_NOTICE.value,
                    "prior_resolution": prior_resolution.get("outcome"),
                },
                decision="continue",
                confidence=0.0,
                trace_id=workflow_id,
                input_tokens=(opening_resp.tokens_used.input_tokens if opening_resp.tokens_used else 0),
                output_tokens=(opening_resp.tokens_used.output_tokens if opening_resp.tokens_used else 0),
            ))
            await push_agent_message(workflow_id, opening_msg, Stage.FINAL_NOTICE.value)
            await session_update(borrower_id, final_notice_opening_msg_sent=True)

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
            # FinalNoticeAgent uses _call_claude_with_tools, not _call_claude,
            # so we must patch the right method to intercept mock responses.
            _orig = agent._call_claude_with_tools
            async def _mock(*a, **kw):
                return text, TokenUsage(input_tokens=300, output_tokens=70)
            agent._call_claude_with_tools = _mock
            try:
                resp = await agent.process(ctx, user_input)
            finally:
                agent._call_claude_with_tools = _orig
        else:
            resp = await agent.process(ctx, user_input)

        ctx.conversation_history.append({
            "role": "assistant", "content": resp.message,
            "stage": Stage.FINAL_NOTICE.value, "timestamp": _ts(),
            "advanced": resp.should_advance,
        })
        _tin = resp.tokens_used.input_tokens if resp.tokens_used else 0
        _tout = resp.tokens_used.output_tokens if resp.tokens_used else 0
        asyncio.create_task(log_interaction(
            borrower_id=borrower_id,
            agent_name="FinalNoticeAgent",
            agent_version="v1.0",
            prompt_version=get_prompt_version("FinalNoticeAgent"),
            model=agent.model,
            model_params={"temperature": 0.05, "max_tokens": 2000},
            input_text=user_input,
            output_text=resp.message,
            structured_context={
                "intent": "final_notice_turn",
                "stage": Stage.FINAL_NOTICE.value,
                "prior_resolution": prior_resolution.get("outcome"),
            },
            decision=("resolved" if resp.should_advance and ctx.final_notice_outcome == "resolved"
                      else ("escalated" if resp.should_advance else "continue")),
            confidence=1.0 if resp.should_advance else 0.0,
            trace_id=workflow_id,
            input_tokens=_tin,
            output_tokens=_tout,
        ))

        # Stash contract_html before advancing so we can push it after the
        # resolution status event (ensures correct render order in the UI)
        _contract_html = (resp.metadata.get("contract_html", "") if resp.metadata else "")

        if use_real_chat:
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
        await push_stage_event(workflow_id, f"final_notice_{outcome}",
                               "Case resolved." if outcome == "resolved"
                               else "This matter has been escalated to legal.")
        if outcome != "resolved" and not _contract_html:
            _contract_html = _build_legal_notice_contract_html(
                borrower_id=borrower_id,
                loan_id=data.get("loan_id", ""),
                resolution_outcome=str(prior_resolution.get("outcome", "unknown")),
                offer=ctx.resolution_offer,
            )
        # Push contract AFTER the resolution status so it renders last
        if _contract_html:
            await push_stage_event(workflow_id, "contract", _contract_html)

    # ── Redis + MongoDB: final status ─────────────────────── #
    final_status = "RESOLVED" if outcome == "resolved" else "LEGAL"
    await session_update(borrower_id, stage="FINAL_NOTICE", status=final_status)
    await upsert_borrower_case(borrower_id, "FINAL_NOTICE", final_status)

    # Log outcome to MongoDB
    mongo_outcome = "AGREEMENT" if outcome == "resolved" else "LEGAL"
    asyncio.create_task(log_outcome(
        borrower_id=borrower_id,
        outcome=mongo_outcome,
        agent_versions={
            "assessment": "v1.0",
            "resolution": "v1.0",
            "final_notice": "v1.0",
        },
        metadata={"final_notice_outcome": outcome},
    ))

    # Log final notice interaction
    fn_turns = ctx.conversation_history[prior_len:]
    asyncio.create_task(log_interaction(
        borrower_id=borrower_id,
        agent_name="FinalNoticeAgent",
        agent_version="v1.0",
        prompt_version=get_prompt_version("FinalNoticeAgent"),
        model=agent.model,
        model_params={"temperature": 0.05, "max_tokens": 2000},
        input_text=f"Final notice stage. Prior resolution: {prior_resolution.get('outcome')}",
        output_text=f"Outcome: {outcome}. Turns: {len(fn_turns)}.",
        structured_context={
            "intent": outcome,
            "stage": Stage.FINAL_NOTICE.value,
            "prior_resolution": prior_resolution.get("outcome"),
            "received_handoff_tokens": int(resolution_handoff_tokens),
            "handoff_from": "ResolutionAgent",
        },
        decision=outcome,
        confidence=1.0,
        trace_id=workflow_id,
        input_tokens=0,
        output_tokens=0,
    ))

    return {
        "outcome": outcome,
        "conversation": fn_turns,
        "tokens_in": cost.total_tokens.input_tokens,
        "tokens_out": cost.total_tokens.output_tokens,
    }


# ──────────────────────────────────────────────────────────── #
# Side-effect activities (EXIT nodes in the diagram)
# ──────────────────────────────────────────────────────────── #

@activity.defn(name="log_agreement")
async def log_agreement(payload: dict) -> dict:
    """
    EXIT: Log agreement (deal agreed after voice call).

    Also generates the settlement contract and pushes it to the chat UI
    so the borrower sees the formal document immediately after the call —
    the same experience they would get through the final notice stage.
    """
    def _safe_part(value: str) -> str:
        keep = []
        for ch in value:
            if ch.isalnum() or ch in ("-", "_", "."):
                keep.append(ch)
            else:
                keep.append("_")
        return "".join(keep).strip("_") or "unknown"

    borrower_id = payload.get("borrower_id", "unknown")
    loan_id     = payload.get("loan_id", "unknown")
    workflow_id = payload.get("workflow_id", borrower_id)
    offer       = payload.get("offer") or {}

    call_id = str(payload.get("call_id") or "").strip()
    if not call_id:
        call_id = f"missing_call_id_{_safe_part(str(borrower_id))}"

    root = Path(os.getenv("CALL_LOG_DIR", "call_logs"))
    log_dir = root / _safe_part(call_id)
    log_dir.mkdir(parents=True, exist_ok=True)

    log_record = {
        "event": "agreement_logged",
        "timestamp": _ts(),
        "borrower_id": borrower_id,
        "loan_id": loan_id,
        "workflow_id": workflow_id,
        "call_id": call_id,
        "offer": offer,
        "conversation": payload.get("conversation", []),
    }
    agreement_file = log_dir / "agreement.json"
    agreement_file.write_text(
        json.dumps(log_record, ensure_ascii=True, indent=2),
        encoding="utf-8",
    )

    activity.logger.info(
        f"[EXIT:log_agreement] borrower={borrower_id} "
        f"call_id={call_id} offer={offer} "
        f"log_file={agreement_file}"
    )
    print(
        f"✅ [EXIT] Agreement logged for borrower {borrower_id} "
        f"at {agreement_file}"
    )

    # ── Generate and push the settlement contract to the chat UI ──
    # The borrower committed on the voice call. They need to see the formal
    # contract document in the chat window immediately so they have the
    # payment details, required documents, and deadline in writing.
    use_real_chat = os.getenv("USE_REAL_CHAT", "0") == "1"
    if use_real_chat:
        try:
            from src.agent_tools import handle_generate_settlement_document
            from datetime import datetime, timezone

            resolution_path = offer.get("path", "lump_sum")
            outstanding     = float(offer.get("outstanding_amount",
                                              offer.get("upfront_required", 0)) or 0)
            upfront         = float(offer.get("upfront_required", 0) or 0)
            monthly         = float(offer.get("monthly_payment", 0) or 0)
            tenure          = int(offer.get("tenure_months", 0) or 0)
            valid_until     = offer.get("valid_until", "")

            # Format expiry for the document
            try:
                expiry_dt  = datetime.fromisoformat(valid_until)
                offer_expiry = expiry_dt.strftime("%d %B %Y, %H:%M UTC")
            except Exception:
                offer_expiry = valid_until or ""

            # outstanding_amount is not in offer dict by default — derive it
            # from upfront + remaining instalments when not explicitly set
            if not outstanding:
                outstanding = upfront + monthly * tenure if monthly else upfront

            contract_result = await handle_generate_settlement_document({
                "borrower_id":        borrower_id,
                "loan_id":            loan_id,
                "resolution_path":    resolution_path,
                "outstanding_amount": outstanding,
                "upfront_amount":     upfront,
                "monthly_amount":     monthly,
                "tenure_months":      tenure,
                "offer_expiry":       offer_expiry,
                "accepted":           True,   # borrower already committed on the call
            })

            contract_html = contract_result.get("contract_html", "")
            if contract_html:
                # Tell the UI the call is done and the case is resolved
                await push_stage_event(
                    workflow_id, "final_notice_opened",
                    "The voice call has concluded. Your settlement contract is below."
                )
                await push_stage_event(workflow_id, "final_notice_resolved", "Case resolved.")
                # Push the contract document — renders as the styled card in the chat
                await push_stage_event(workflow_id, "contract", contract_html)
                activity.logger.info(
                    "[EXIT:log_agreement] contract pushed to chat for %s", borrower_id
                )
            else:
                activity.logger.warning(
                    "[EXIT:log_agreement] contract HTML empty for %s", borrower_id
                )
        except Exception as e:
            activity.logger.warning(
                "[EXIT:log_agreement] contract generation/push failed for %s: %s",
                borrower_id, e,
            )

    _schedule_learning_followups(resolved=True)

    return {
        "ok": True,
        "event": "agreement_logged",
        **{k: payload[k] for k in ("borrower_id", "loan_id") if k in payload},
        "call_id": call_id,
        "log_dir": str(log_dir),
        "log_file": str(agreement_file),
    }


@activity.defn(name="log_resolution")
async def log_resolution(payload: dict) -> dict:
    """EXIT: Log resolution (resolved via Final Notice chat)."""
    activity.logger.info(
        f"[EXIT:log_resolution] borrower={payload['borrower_id']}"
    )
    print(f"✅ [EXIT] Resolution logged for borrower {payload['borrower_id']}")
    _schedule_learning_followups(resolved=True)
    return {"ok": True, "event": "resolution_logged", **{k: payload[k]
            for k in ("borrower_id", "loan_id") if k in payload}}


@activity.defn(name="create_legal_referral")
async def create_legal_referral(payload: dict) -> dict:
    """EXIT: Flag for legal/write-off."""
    activity.logger.info(
        f"[EXIT:legal_referral] borrower={payload['borrower_id']} reason={payload['reason']}"
    )
    print(f"⚠️  [EXIT] Legal referral created for borrower {payload['borrower_id']}: {payload['reason']}")
    _schedule_learning_followups(resolved=False)
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
