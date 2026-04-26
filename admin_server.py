#!/usr/bin/env python3
"""Standalone admin API server."""
import asyncio
import os
os.environ['MONGO_URI'] = 'mongodb://localhost:27017'

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from datetime import datetime, timezone

app = FastAPI()

# Import all the admin API functions
from src.admin_api import _build_agent_analytics, get_cost_breakdown

@app.get("/api/admin/stats")
async def stats():
    from src.data_layer import get_mongo
    db = await get_mongo()
    if db is None:
        return {"error": "MongoDB unavailable"}
    
    total_interactions = await db.interactions.count_documents({})
    total_outcomes = await db.outcomes.count_documents({})
    return {
        "total_interactions": total_interactions,
        "total_outcomes": total_outcomes,
    }

@app.get("/api/admin/stats/cost-breakdown")
async def cost_breakdown():
    return await get_cost_breakdown()

@app.get("/api/admin/analytics/agents")
async def agents():
    from src.data_layer import get_mongo
    db = await get_mongo()
    if db is None:
        return {"error": "MongoDB unavailable"}
    
    agents_list = ["AssessmentAgent", "ResolutionAgent", "FinalNoticeAgent"]
    results = await asyncio.gather(*[_build_agent_analytics(db, a) for a in agents_list])
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "agents": {r["agent_name"]: r for r in results},
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")