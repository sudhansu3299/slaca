#!/usr/bin/env python3
"""Simple test server - admin API only."""
import asyncio
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI()

@app.get("/api/admin/stats")
async def stats():
    return {"status": "ok", "message": "simple server works"}

@app.get("/api/admin/analytics/agents")
async def analytics():
    return {"agents": {}}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)