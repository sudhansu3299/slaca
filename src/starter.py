"""
Temporal workflow starter.

Usage:
  # Mock (no real API calls — default)
  python -m src.starter --persona cooperative

  # Live (real Claude + real Retell call to your phone)
  python -m src.starter --persona cooperative --live --phone +917008098779

  # All 4 personas, mocked
  python -m src.starter --batch

  # All 4 personas, live
  python -m src.starter --batch --live --phone +917008098779
"""

from __future__ import annotations

import argparse
import asyncio
import os
import time
import uuid

from dotenv import load_dotenv
load_dotenv()

from temporalio.client import Client
from temporalio.service import RPCError

from src.simulation import PersonaType, SimulationEngine
from src.temporal_workflow import CollectionsInput, CollectionsWorkflow


DEFAULT_TASK_QUEUE = "collections-queue"
DEFAULT_LIVE_TASK_QUEUE = "collections-queue-live"


async def wait_for_worker(
    client: Client,
    task_queue: str = DEFAULT_TASK_QUEUE,
    timeout_seconds: int = 120,
) -> bool:
    """Poll Temporal until a worker is registered on the given task queue."""
    import temporalio.api.workflowservice.v1 as wsv1

    deadline = time.time() + timeout_seconds
    attempt = 0
    while time.time() < deadline:
        attempt += 1
        try:
            req = wsv1.DescribeTaskQueueRequest(
                namespace="default",
                task_queue={"name": task_queue, "kind": 1},
            )
            resp = await client.workflow_service.describe_task_queue(req)
            pollers = list(resp.pollers or [])
            if pollers:
                print(f"[starter] ✅ Worker detected on '{task_queue}' "
                      f"({len(pollers)} poller(s))")
                return True
        except RPCError as e:
            if attempt == 1:
                print(f"[starter] task queue not yet registered, polling... ({e})")
        except Exception as e:
            if attempt == 1:
                print(f"[starter] worker poll error (will retry): {e}")
        await asyncio.sleep(3)

    print(f"[starter] ⚠️  No worker on '{task_queue}' after {timeout_seconds}s — proceeding anyway")
    return False


async def start_one(
    client: Client,
    persona: PersonaType,
    phone_number: str,
    task_queue: str,
    live: bool = False,
) -> dict:
    profile = next(
        p for p in SimulationEngine.make_profiles()
        if p.persona == persona
    )

    # Use the real phone number if live, otherwise the placeholder in profile
    actual_phone = phone_number if live else profile.phone_number

    wf_id = f"collections-{profile.borrower_id}-{uuid.uuid4().hex[:6]}"

    handle = await client.start_workflow(
        CollectionsWorkflow.run,
        CollectionsInput(
            borrower_id=profile.borrower_id,
            loan_id=profile.loan_id,
            phone_number=actual_phone,
            principal_amount=profile.principal_amount,
            outstanding_amount=profile.outstanding_amount,
            days_past_due=profile.days_past_due,
            persona=persona.value,         # ← propagated to worker
        ),
        id=wf_id,
        task_queue=task_queue,
    )

    mode = "🔴 LIVE" if live else "🟡 MOCK"
    print(f"\n[starter] {mode} workflow started: {wf_id}")
    print(f"[starter]   persona={persona.value}  phone={actual_phone}")
    print(f"[starter]   loan=₹{profile.outstanding_amount:,.0f}  DPD={profile.days_past_due}")
    if live:
        print(f"[starter]   📞 Watch for a call to {actual_phone}")

    result = await handle.result()

    print(f"\n[starter] ✅ Workflow complete: {wf_id}")
    print(f"[starter]   outcome={result.outcome}  stage={result.final_stage}")
    print(f"[starter]   turns={result.total_turns}  "
          f"tokens={result.total_tokens_in}in/{result.total_tokens_out}out")
    print(f"[starter]   handoff_tokens={result.handoff_tokens}")

    return {
        "workflow_id": wf_id,
        "persona": persona.value,
        "outcome": result.outcome,
        "final_stage": result.final_stage,
        "total_turns": result.total_turns,
    }


async def main() -> None:
    parser = argparse.ArgumentParser(description="Kick off a collections workflow via Temporal")
    parser.add_argument("--persona", default="cooperative",
                        choices=[p.value for p in PersonaType],
                        help="Borrower persona to simulate")
    parser.add_argument("--phone", default=None,
                        help="Phone number to call (E.164 e.g. +917008098779). "
                             "Required with --live.")
    parser.add_argument("--live", action="store_true",
                        help="Use real Claude + real voice call (requires provider-specific voice keys in .env)")
    parser.add_argument("--batch", action="store_true",
                        help="Run all 4 personas sequentially")
    parser.add_argument("--wait-timeout", type=int, default=120,
                        help="Seconds to wait for worker registration")
    args = parser.parse_args()

    # Resolve phone number: --phone > BORROWER_PHONE env > fail if live
    phone = (
        args.phone
        or os.getenv("BORROWER_PHONE", "")
        or "+919999999999"   # placeholder used for mock runs
    )
    if args.live:
        provider_name = os.getenv("VOICE_PROVIDER", "mock").lower()
        if provider_name == "retell":
            required = ["RETELL_API_KEY", "RETELL_AGENT_ID", "CALLER_NUMBER"]
        elif provider_name == "vapi":
            required = ["VAPI_API_KEY", "VAPI_ASSISTANT_ID", "VAPI_PHONE_NUMBER_ID", "CALLER_NUMBER"]
        else:
            print(f"⚠️  --live requires VOICE_PROVIDER to be 'retell' or 'vapi' (got '{provider_name}')")
            print("   Set VOICE_PROVIDER and provider credentials in .env.")
            raise SystemExit(1)

        missing = [name for name in required if not os.getenv(name, "")]
        if missing:
            print(
                f"⚠️  --live with VOICE_PROVIDER={provider_name} missing env vars: "
                f"{', '.join(missing)}"
            )
            print("   See .env for setup instructions.")
            raise SystemExit(1)

    address = os.getenv("TEMPORAL_ADDRESS", "localhost:7233")
    mock_queue = os.getenv("TASK_QUEUE", DEFAULT_TASK_QUEUE)
    live_queue = os.getenv("TASK_QUEUE_LIVE", DEFAULT_LIVE_TASK_QUEUE)
    task_queue = live_queue if args.live else mock_queue
    print(f"[starter] Connecting to Temporal at {address}...")

    client = None
    for attempt in range(30):
        try:
            client = await Client.connect(address)
            break
        except Exception as e:
            if attempt == 0:
                print(f"[starter] Waiting for Temporal: {e}")
            await asyncio.sleep(2)
    if client is None:
        raise RuntimeError(f"Could not connect to Temporal at {address}")
    print(f"[starter] Connected.")

    await wait_for_worker(client, task_queue, timeout_seconds=args.wait_timeout)

    if args.batch:
        results = []
        for persona in PersonaType:
            r = await start_one(client, persona, phone, task_queue=task_queue, live=args.live)
            results.append(r)
        print("\n═══════════════════════ Batch Summary ═══════════════════════")
        for r in results:
            print(f"  {r['persona']:22} → {r['outcome']:12} ({r['total_turns']} turns)")
    else:
        await start_one(client, PersonaType(args.persona), phone, task_queue=task_queue, live=args.live)


if __name__ == "__main__":
    asyncio.run(main())
