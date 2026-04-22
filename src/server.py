"""FastAPI server for the collections system."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from temporalio.client import Client
import uuid

app = FastAPI(title="AI Collections System")


class CollectionsRequest(BaseModel):
    borrower_id: str
    loan_id: str
    phone_number: str
    principal_amount: float
    outstanding_amount: float
    days_past_due: int


class CollectionsResponse(BaseModel):
    workflow_id: str
    status: str


@app.post("/collections/start", response_model=CollectionsResponse)
async def start_collections(request: CollectionsRequest):
    """Start a new collections workflow."""
    try:
        client = await Client.connect("localhost:7233")

        workflow_id = f"collections-{request.borrower_id}-{uuid.uuid4()}"

        await client.start_workflow(
            "src.temporal_workflow:CollectionsWorkflow.run",
            {
                "borrower_id": request.borrower_id,
                "loan_id": request.loan_id,
                "phone_number": request.phone_number,
                "principal_amount": request.principal_amount,
                "outstanding_amount": request.outstanding_amount,
                "days_past_due": request.days_past_due,
            },
            id=workflow_id,
            task_queue="collections-queue",
        )

        return CollectionsResponse(
            workflow_id=workflow_id,
            status="started"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/collections/{workflow_id}")
async def get_collections_status(workflow_id: str):
    """Get the status of a collections workflow."""
    try:
        client = await Client.connect("localhost:7233")
        handle = client.get_workflow_handle(workflow_id)
        result = await handle.result()

        return {
            "workflow_id": workflow_id,
            "result": result
        }

    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/collections/{workflow_id}/chat")
async def send_chat_message(workflow_id: str, message: str):
    """Send a chat message to an active collections workflow."""
    try:
        client = await Client.connect("localhost:7233")
        handle = client.get_workflow_handle(workflow_id)

        signal_result = await handle.signal("ReceiveMessage", {"message": message})

        return {"status": "sent", "workflow_id": workflow_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)