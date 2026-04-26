#!/usr/bin/env python3
"""Add mock interactions with improved prompt version for analytics."""
import asyncio
import os
import sys
import uuid
from datetime import datetime, timezone

os.environ['MONGO_URI'] = 'mongodb://localhost:27017'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


async def main():
    from src.data_layer import get_mongo
    
    db = await get_mongo()
    ts = datetime.now(timezone.utc)
    
    # Find existing AssessmentAgent interactions to update
    existing = await db.interactions.find(
        {"agent_name": "AssessmentAgent"}
    ).sort("timestamp", -1).to_list(35)
    
    print(f"Found {len(existing)} AssessmentAgent interactions")
    
    # Update first 30 to improved-v2 (these will be scored and show improvement)
    for doc in existing[:30]:
        await db.interactions.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "prompt_version": "improved-v2",
                "structured_context": {
                    "completeness_score": 0.65 + (doc.get("created_no", 0) % 10) * 0.01,
                    "identity_verified": True,
                    "turns": 2,
                },
                "decision": "complete",
                "timestamp": ts.isoformat(),
            }}
        )
    
    # Update remaining to canonical-v1 (baseline)
    for doc in existing[30:60]:
        await db.interactions.update_one(
            {"_id": doc["_id"]},
            {"$set": {
                "prompt_version": "canonical-v1",
                "structured_context": {
                    "completeness_score": 0.45 + (doc.get("created_no", 0) % 10) * 0.01,
                    "identity_verified": False,
                    "turns": 4,
                },
                "decision": "incomplete",
            }}
        )
    
    # Add prompt_changes for improved-v2
    await db.prompt_changes.insert_one({
        "agent_name": "AssessmentAgent",
        "new_version": "improved-v2",
        "change_type": "improvement",
        "change_summary": "Enhanced empathy + clarity",
        "prompt_diff": "+ empathy layer\n+ clearer fields",
        "run_id": "mock-test-15842be5",
        "status": "promoted",
        "timestamp": ts.isoformat(),
        "trigger": "auto",
    })
    
    # Update the pipeline run with transcript scores
    ts1 = f"mock-ts-{uuid.uuid4().hex[:8]}"
    ts2 = f"mock-ts-{uuid.uuid4().hex[:8]}"
    ts3 = f"mock-ts-{uuid.uuid4().hex[:8]}"
    
    await db.eval_pipeline.update_one(
        {"run_id": "mock-test-15842be5"},
        {"$set": {
            "transcript_scores": [
                {"trace_id": ts1, "borrower_id": "m1", "score_before": 0.48, "score_after": 0.68, "delta": 0.20, "p_value": 0.001, "cohen_d": 0.82, "mean_before": 0.48, "mean_after": 0.68, "sd_pooled": 0.245, "n_samples": 30},
                {"trace_id": ts2, "borrower_id": "m2", "score_before": 0.45, "score_after": 0.71, "delta": 0.26, "p_value": 0.002, "cohen_d": 0.94, "mean_before": 0.47, "mean_after": 0.70, "sd_pooled": 0.250, "n_samples": 30},
                {"trace_id": ts3, "borrower_id": "m3", "score_before": 0.50, "score_after": 0.66, "delta": 0.16, "p_value": 0.008, "cohen_d": 0.65, "mean_before": 0.49, "mean_after": 0.67, "sd_pooled": 0.265, "n_samples": 30},
            ],
            "held_out_scores": [],
            "llm_calls": 15,
            "improvement_significant": True,
            "improvement_delta": 0.21,
            "improvement_p_value": 0.001,
            "started_at": ts.isoformat(),
        }}
    )
    
    print(f"✓ Updated {min(30, len(existing))} interactions to improved-v2")
    print(f"✓ Added prompt_changes for improved-v2")
    print(f"✓ Updated pipeline with transcript_scores (p=0.001, d=0.82)")
    print(f"\nRefresh /admin — cost should show <$20 and version history should show v2 with p/d stats")


if __name__ == "__main__":
    asyncio.run(main())