"""
Voice Integration Layer (MS5)

Wraps Retell / Vapi / Bland with:
1. Structured context injection — the call starts mid-context
2. Transcript polling to feed into ResolutionAgent
3. Call status tracking

The key challenge (milestone spec): inject structured context into the
voice system so the agent responds to prior chat data naturally.

Implementation:
  - HandoffSummary is serialised to a compact JSON dict
  - That dict is passed as `metadata` in the call initiation payload
  - The voice provider makes it available to the LLM via dynamic variables
  - Our ResolutionAgent system prompt contains the handoff block verbatim
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.handoff import HandoffSummary
from src.prompts import vapi_first_message_prompt, vapi_system_override_prompt

log = logging.getLogger(__name__)


class VoiceProviderName(str, Enum):
    RETELL = "retell"
    VAPI = "vapi"
    BLAND = "bland"
    MOCK = "mock"       # for testing


@dataclass
class CallConfig:
    provider: VoiceProviderName
    agent_id: str
    caller_number: str
    api_key: str = ""


@dataclass
class CallRecord:
    call_id: str
    provider: VoiceProviderName
    phone_number: str
    status: str = "initiated"       # initiated | in_progress | completed | failed
    transcript: str = ""
    duration_seconds: int = 0
    metadata: dict = field(default_factory=dict)


def build_voice_metadata(summary: HandoffSummary) -> dict:
    """
    Serialise HandoffSummary into the metadata dict injected into the voice call.

    Voice providers surface this dict as dynamic variables in the LLM prompt,
    so the agent "knows" what happened in the chat before the call.
    """
    return {
        # Identity — so the voice agent never re-asks
        "identity_verified": summary.identity_verified,
        "borrower_id": summary.borrower_id,
        "loan_id": summary.loan_id,
        # Financial context
        "outstanding_amount": summary.outstanding_amount,
        "days_past_due": summary.days_past_due,
        "monthly_income": summary.monthly_income or "",
        "employment_status": summary.employment_status or "",
        "financial_state": summary.financial_state or "unknown",
        # Intent signal from chat
        "intent": summary.intent or "unknown",
        "borrower_attitude": summary.borrower_attitude or "neutral",
        # Resolution path and computed offer amounts
        "resolution_path": summary.resolution_path or "",
        "offer_upfront": summary.offer_upfront or summary.outstanding_amount,
        "offer_monthly": summary.offer_monthly or "",
        "offer_discount_pct": summary.offer_discount_pct or 0,
        "offer_tenure_months": getattr(summary, "offer_tenure_months", 0) or 0,
        # Objections from chat (comma-separated for voice metadata)
        "prior_objections": "; ".join(summary.objections[:3]),
        # Handoff prompt block (used by Retell's dynamic_data injection)
        "handoff_prompt": summary.to_prompt_block(),
    }


class VoiceProvider:
    """
    Pluggable voice provider.
    Defaults to MOCK for testing; switch via config.
    """

    def __init__(self, config: Optional[CallConfig] = None):
        self.config = config or CallConfig(
            provider=VoiceProviderName.MOCK,
            agent_id="mock_agent",
            caller_number="+1000000000",
        )

    async def initiate_call(
        self,
        phone_number: str,
        handoff_summary: HandoffSummary,
    ) -> CallRecord:
        """
        Initiate an outbound voice call with structured context injected.
        Returns a CallRecord for tracking.
        """
        metadata = build_voice_metadata(handoff_summary)

        if self.config.provider == VoiceProviderName.MOCK:
            return CallRecord(
                call_id=f"mock_{phone_number.replace('+', '')}",
                provider=VoiceProviderName.MOCK,
                phone_number=phone_number,
                status="in_progress",
                metadata=metadata,
            )

        elif self.config.provider == VoiceProviderName.RETELL:
            return await self._retell_call(phone_number, metadata)

        elif self.config.provider == VoiceProviderName.VAPI:
            return await self._vapi_call(phone_number, metadata)

        elif self.config.provider == VoiceProviderName.BLAND:
            return await self._bland_call(phone_number, metadata)

        raise ValueError(f"Unknown provider: {self.config.provider}")

    async def get_transcript(self, call_record: CallRecord) -> str:
        """Poll for latest transcript from the provider."""
        if call_record.provider == VoiceProviderName.MOCK:
            return call_record.transcript   # set externally in tests

        elif call_record.provider == VoiceProviderName.RETELL:
            return await self._retell_transcript(call_record.call_id)

        elif call_record.provider == VoiceProviderName.VAPI:
            return await self._vapi_transcript(call_record.call_id)

        elif call_record.provider == VoiceProviderName.BLAND:
            return await self._bland_transcript(call_record.call_id)

        return ""

    async def end_call(self, call_record: CallRecord) -> None:
        """
        Terminate a live call via the provider API.
        Called immediately after a terminal outcome (committed / refused / escalated)
        so the call hangs up rather than waiting for the borrower to disconnect.
        """
        call_record.status = "completed"

        if call_record.provider == VoiceProviderName.VAPI and call_record.call_id:
            await self._vapi_end_call(call_record.call_id)
        elif call_record.provider == VoiceProviderName.RETELL and call_record.call_id:
            await self._retell_end_call(call_record.call_id)

    async def _vapi_end_call(self, call_id: str) -> None:
        """DELETE /call/{id} terminates the Vapi call immediately."""
        import httpx, os
        try:
            async with httpx.AsyncClient() as client:
                r = await client.delete(
                    f"https://api.vapi.ai/call/{call_id}",
                    headers={"Authorization": f"Bearer {os.getenv('VAPI_API_KEY', '')}"},
                    timeout=10,
                )
                log.info("[voice] Vapi call %s terminated (status %d)", call_id, r.status_code)
        except Exception as e:
            log.warning("[voice] _vapi_end_call failed for %s: %s", call_id, e)

    async def _retell_end_call(self, call_id: str) -> None:
        """POST /v2/call/{id}/stop terminates a Retell call."""
        import httpx, os
        try:
            async with httpx.AsyncClient() as client:
                r = await client.post(
                    f"https://api.retellai.com/v2/call/{call_id}/stop",
                    headers={"Authorization": f"Bearer {os.getenv('RETELL_API_KEY', '')}"},
                    timeout=10,
                )
                log.info("[voice] Retell call %s terminated (status %d)", call_id, r.status_code)
        except Exception as e:
            log.warning("[voice] _retell_end_call failed for %s: %s", call_id, e)

    # ──────────────────────────────── Provider implementations ─────── #

    async def _retell_call(self, phone_number: str, metadata: dict) -> CallRecord:
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.retellai.com/v2/call",
                json={
                    "from_number": self.config.caller_number,
                    "to_number": phone_number,
                    "agent_id": self.config.agent_id,
                    # Retell: metadata is surfaced as dynamic_data in prompt
                    "metadata": metadata,
                    # Inject handoff as the conversation opening context
                    "conversation_config_override": {
                        "reminder_message": metadata.get("handoff_prompt", "")
                    },
                },
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=15,
            )
            data = r.json()
            return CallRecord(
                call_id=data.get("call_id", ""),
                provider=VoiceProviderName.RETELL,
                phone_number=phone_number,
                status="initiated",
                metadata=metadata,
            )

    async def _retell_transcript(self, call_id: str) -> str:
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"https://api.retellai.com/v2/call/{call_id}",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=10,
            )
            data = r.json()
            if data.get("status") == "ended":
                return data.get("transcript", "")
            return ""

    async def _vapi_call(self, phone_number: str, metadata: dict) -> CallRecord:
        import httpx, os
        # Build a compact opening line so the agent starts mid-context
        outstanding      = metadata.get("outstanding_amount", "")
        path             = metadata.get("resolution_path", "")
        income           = metadata.get("monthly_income", "")
        employment       = metadata.get("employment_status", "")
        offer_upfront    = metadata.get("offer_upfront", outstanding)
        offer_monthly    = metadata.get("offer_monthly", "")
        offer_tenure     = metadata.get("offer_tenure_months", "")
        first_msg = vapi_first_message_prompt(
            outstanding_amount=str(outstanding),
            employment_status=str(employment),
            monthly_income=str(income),
            resolution_path=str(path),
            offer_upfront=str(offer_upfront),
            offer_monthly=str(offer_monthly),
            offer_tenure_months=str(offer_tenure),
        )
        # System prompt override injects the full handoff block
        system_override = vapi_system_override_prompt(
            handoff_prompt=str(metadata.get("handoff_prompt", ""))
        )
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.vapi.ai/call/phone",
                json={
                    "phoneNumberId": os.getenv("VAPI_PHONE_NUMBER_ID", ""),
                    "customer": {"number": phone_number},
                    "assistantId": self.config.agent_id,   # camelCase — Vapi requirement
                    "assistantOverrides": {
                        "firstMessage": first_msg,
                        "model": {
                        "provider": "openai",
                        "model": os.getenv("AGENT_MODEL", "gpt-4o"),
                        "messages": [
                            {"role": "system", "content": system_override}
                        ]
    }
                    },
                    "metadata": metadata,
                },
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=15,
            )
            # Guard: parse JSON safely — an empty or malformed body should raise
            # a clean RuntimeError so the caller can decide how to handle it.
            if not r.content:
                raise RuntimeError(f"Vapi call failed {r.status_code}: empty response body")
            try:
                data = r.json()
            except Exception as exc:
                raise RuntimeError(f"Vapi call failed {r.status_code}: non-JSON body — {exc}") from exc
            if r.status_code not in (200, 201):
                raise RuntimeError(f"Vapi call failed {r.status_code}: {data}")
            return CallRecord(
                call_id=data.get("id", ""),
                provider=VoiceProviderName.VAPI,
                phone_number=phone_number,
                status="initiated",
                metadata=metadata,
            )

    async def _vapi_transcript(self, call_id: str) -> str:
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"https://api.vapi.ai/call/{call_id}",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=10,
            )
            # Guard: Vapi sometimes returns an empty body (e.g. during call setup
            # or on transient 5xx). Treat that as "still in progress" rather than crash.
            if not r.content:
                log.warning("[voice] _vapi_transcript: empty response body for call %s (status %d)",
                            call_id, r.status_code)
                return ""
            try:
                data = r.json()
            except Exception as exc:
                log.warning("[voice] _vapi_transcript: non-JSON response for call %s: %s",
                            call_id, exc)
                return ""
            status = data.get("status", "")
            if status not in ("ended", "completed"):
                return ""   # still in progress — caller will retry

            # Vapi returns messages[] with role/content.
            # Also check top-level 'transcript' string for older API versions.
            transcript_str = data.get("transcript", "")
            if transcript_str:
                return transcript_str

            msgs = data.get("messages", [])
            # Filter to user (borrower) turns only, join as readable transcript
            user_lines = [
                m.get("content", m.get("message", ""))
                for m in msgs
                if m.get("role") in ("user", "customer")
            ]
            return "\n".join(l for l in user_lines if l)

    async def _bland_call(self, phone_number: str, metadata: dict) -> CallRecord:
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.post(
                "https://api.bland.ai/call",
                json={
                    "phone_number": phone_number,
                    "agent_id": self.config.agent_id,
                    "metadata": metadata,
                    # Bland: inject opening context via request_data
                    "request_data": metadata,
                },
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=15,
            )
            data = r.json()
            return CallRecord(
                call_id=data.get("call_id", ""),
                provider=VoiceProviderName.BLAND,
                phone_number=phone_number,
                status="initiated",
                metadata=metadata,
            )

    async def _bland_transcript(self, call_id: str) -> str:
        import httpx
        async with httpx.AsyncClient() as client:
            r = await client.get(
                f"https://api.bland.ai/call/{call_id}/transcript",
                headers={"Authorization": f"Bearer {self.config.api_key}"},
                timeout=10,
            )
            data = r.json()
            return data.get("transcript", "")
