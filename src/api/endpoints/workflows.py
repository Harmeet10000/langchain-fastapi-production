"""LangGraph workflow API endpoints."""

from typing import List, Optional, Dict, Any, Literal
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime
import json

from src.graphs.workflows.rag_workflow import rag_workflow
from src.services.langgraph import workflow_service
from src.services.langsmith.client import langsmith_service
from src.core.config.logging_config import LoggerAdapter
from src.core.cache.redis_client import redis_cache

logger = LoggerAdapter(__name__)
router = APIRouter(prefix="/workflows", tags=["Workflows"])


class WorkflowInput(BaseModel):
    """Workflow input model."""
    workflow_id: str = Field(..., description="ID of the workflow to execute")
    inputs: Dict[str, Any] = Field(..., description="Input data for the workflow")
    config: Optional[Dict[str, Any]] = Field(default=None, description="Workflow configuration")
    thread_id: Optional[str] = Field(default=None, description="Thread ID for stateful workflows")
    stream: bool = Field(default=False, description="Stream workflow events")


class WorkflowState(BaseModel):
    """Workflow state model."""
    state_id: str
    workflow_id: str
    thread_id: str
    current_node: str
    status: Literal["pending", "running", "completed", "failed", "paused"]
    data: Dict[str, Any]
    metadata: Optional[Dict[str, Any]] = None
    created_at: str
    updated_at: str


class WorkflowResult(BaseModel):
    """Workflow execution result."""
    workflow_id: str
    thread_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    steps: Optional[List[Dict[str, Any]]] = None


class GraphVisualization(BaseModel):
    """Graph visualization data."""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    metadata: Optional[Dict[str, Any]] = None


@router.get("/list")
async def list_workflows():
    """List all available workflows."""
    try:
        workflows = workflow_service.get_available_workflows()
        
        return {
            "workflows": workflows,
            "count": len(workflows)
        }
        
    except Exception as e:
        logger.error("Failed to list workflows", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/schema")
async def get_workflow_schema(workflow_id: str):
    """Get workflow input/output schema."""
    try:
        schema = workflow_service.get_workflow_schema(workflow_id)
        
        if not schema:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return schema
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow schema", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/execute", response_model=WorkflowResult)
async def execute_workflow(
    background_tasks: BackgroundTasks,
    request: WorkflowInput
):
    """Execute a workflow."""
    try:
        # Generate thread ID if not provided
        thread_id = request.thread_id or f"thread_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.workflow_id}"
        
        # Get LangSmith callbacks
        callbacks = langsmith_service.get_callbacks()
        
        # Create workflow state
        state = {
            "state_id": f"state_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "workflow_id": request.workflow_id,
            "thread_id": thread_id,
            "current_node": "start",
            "status": "running",
            "data": request.inputs,
            "metadata": request.config,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat()
        }
        
        # Store state in cache
        await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
        
        if request.stream:
            # Execute with streaming
            # This would return an SSE stream in production
            result = await workflow_service.execute_workflow_stream(
                workflow_id=request.workflow_id,
                inputs=request.inputs,
                config=request.config,
                thread_id=thread_id,
                callbacks=callbacks
            )
        else:
            # Execute synchronously
            result = await workflow_service.execute_workflow(
                workflow_id=request.workflow_id,
                inputs=request.inputs,
                config=request.config,
                thread_id=thread_id,
                callbacks=callbacks
            )
        
        # Update state
        state["status"] = "completed"
        state["updated_at"] = datetime.now().isoformat()
        await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
        
        # Log to LangSmith
        await langsmith_service.log_run(
            run_type="workflow",
            inputs={"workflow_id": request.workflow_id, "inputs": request.inputs},
            outputs=result
        )
        
        return WorkflowResult(
            workflow_id=request.workflow_id,
            thread_id=thread_id,
            status="completed",
            result=result
        )
        
    except Exception as e:
        logger.error("Failed to execute workflow", error=str(e))
        
        # Update state with error
        if 'thread_id' in locals():
            state = await redis_cache.get(f"workflow_state:{thread_id}")
            if state:
                state["status"] = "failed"
                state["error"] = str(e)
                state["updated_at"] = datetime.now().isoformat()
                await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
        
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/rag", response_model=WorkflowResult)
async def execute_rag_workflow(
    query: str,
    namespace: str = "default",
    top_k: int = 5,
    rerank: bool = True,
    use_cache: bool = True
):
    """Execute the RAG workflow."""
    try:
        # Check cache if enabled
        cache_key = f"rag_result:{hash(query)}:{namespace}:{top_k}"
        if use_cache:
            cached_result = await redis_cache.get(cache_key)
            if cached_result:
                logger.info("Returning cached RAG result")
                return WorkflowResult(
                    workflow_id="rag_workflow",
                    thread_id="cached",
                    status="completed",
                    result=cached_result
                )
        
        # Execute RAG workflow
        result = await rag_workflow.run(
            query=query,
            metadata={
                "namespace": namespace,
                "top_k": top_k,
                "rerank": rerank
            }
        )
        
        # Cache result
        if use_cache:
            await redis_cache.set(cache_key, result, ttl=300)  # 5 minutes
        
        return WorkflowResult(
            workflow_id="rag_workflow",
            thread_id=f"rag_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            status="completed",
            result=result
        )
        
    except Exception as e:
        logger.error("Failed to execute RAG workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/state/{thread_id}", response_model=WorkflowState)
async def get_workflow_state(thread_id: str):
    """Get workflow execution state."""
    try:
        state = await redis_cache.get(f"workflow_state:{thread_id}")
        
        if not state:
            raise HTTPException(status_code=404, detail="Workflow state not found")
        
        return WorkflowState(**state)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow state", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/pause/{thread_id}")
async def pause_workflow(thread_id: str):
    """Pause a running workflow."""
    try:
        state = await redis_cache.get(f"workflow_state:{thread_id}")
        
        if not state:
            raise HTTPException(status_code=404, detail="Workflow state not found")
        
        if state["status"] != "running":
            raise HTTPException(status_code=400, detail=f"Workflow is not running (status: {state['status']})")
        
        # Update state
        state["status"] = "paused"
        state["updated_at"] = datetime.now().isoformat()
        await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
        
        # Actual pause logic would be implemented in the workflow service
        await workflow_service.pause_workflow(thread_id)
        
        return {"message": f"Workflow {thread_id} paused successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to pause workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/resume/{thread_id}")
async def resume_workflow(thread_id: str):
    """Resume a paused workflow."""
    try:
        state = await redis_cache.get(f"workflow_state:{thread_id}")
        
        if not state:
            raise HTTPException(status_code=404, detail="Workflow state not found")
        
        if state["status"] != "paused":
            raise HTTPException(status_code=400, detail=f"Workflow is not paused (status: {state['status']})")
        
        # Update state
        state["status"] = "running"
        state["updated_at"] = datetime.now().isoformat()
        await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
        
        # Actual resume logic would be implemented in the workflow service
        await workflow_service.resume_workflow(thread_id)
        
        return {"message": f"Workflow {thread_id} resumed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to resume workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/visualize", response_model=GraphVisualization)
async def visualize_workflow(workflow_id: str):
    """Get workflow graph visualization data."""
    try:
        graph_data = workflow_service.get_workflow_graph(workflow_id)
        
        if not graph_data:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        return GraphVisualization(
            nodes=graph_data["nodes"],
            edges=graph_data["edges"],
            metadata=graph_data.get("metadata")
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to visualize workflow", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{thread_id}")
async def get_workflow_history(thread_id: str):
    """Get workflow execution history."""
    try:
        # Get execution history from cache or database
        history_key = f"workflow_history:{thread_id}"
        history = await redis_cache.get(history_key)
        
        if not history:
            # Check if workflow exists
            state = await redis_cache.get(f"workflow_state:{thread_id}")
            if not state:
                raise HTTPException(status_code=404, detail="Workflow not found")
            
            # Return empty history if no history found
            history = []
        
        return {
            "thread_id": thread_id,
            "history": history,
            "count": len(history)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get workflow history", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/checkpoint/{thread_id}")
async def create_checkpoint(thread_id: str, checkpoint_id: Optional[str] = None):
    """Create a checkpoint for workflow state."""
    try:
        state = await redis_cache.get(f"workflow_state:{thread_id}")
        
        if not state:
            raise HTTPException(status_code=404, detail="Workflow state not found")
        
        # Generate checkpoint ID if not provided
        if not checkpoint_id:
            checkpoint_id = f"checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create checkpoint
        checkpoint = {
            "checkpoint_id": checkpoint_id,
            "thread_id": thread_id,
            "state": state,
            "created_at": datetime.now().isoformat()
        }
        
        # Store checkpoint
        await redis_cache.set(f"workflow_checkpoint:{thread_id}:{checkpoint_id}", checkpoint, ttl=86400)  # 24 hours
        
        return {
            "message": "Checkpoint created successfully",
            "checkpoint_id": checkpoint_id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to create checkpoint", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/restore/{thread_id}/{checkpoint_id}")
async def restore_checkpoint(thread_id: str, checkpoint_id: str):
    """Restore workflow state from a checkpoint."""
    try:
        checkpoint = await redis_cache.get(f"workflow_checkpoint:{thread_id}:{checkpoint_id}")
        
        if not checkpoint:
            raise HTTPException(status_code=404, detail="Checkpoint not found")
        
        # Restore state
        state = checkpoint["state"]
        state["updated_at"] = datetime.now().isoformat()
        await redis_cache.set(f"workflow_state:{thread_id}", state, ttl=3600)
        
        return {
            "message": "Checkpoint restored successfully",
            "checkpoint_id": checkpoint_id,
            "restored_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to restore checkpoint", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))