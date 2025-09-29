"""Chat API endpoints."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field

from src.services.langchain.gemini_service import gemini_service
from src.services.langsmith.client import langsmith_service
from src.core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)
router = APIRouter(prefix="/chat", tags=["Chat"])


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: user, assistant, or system")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat completion request model."""
    messages: List[ChatMessage]
    model: Optional[str] = Field(default="gemini-pro", description="Model to use")
    temperature: Optional[float] = Field(default=0.7, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=2048, ge=1, le=8192)
    session_id: Optional[str] = Field(default=None, description="Session ID for conversation memory")
    use_memory: bool = Field(default=False, description="Use conversation memory")


class ChatResponse(BaseModel):
    """Chat completion response model."""
    response: str
    model: str
    usage: Optional[Dict[str, int]] = None
    session_id: Optional[str] = None


@router.post("/completions", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Generate chat completion."""
    try:
        # Convert messages to dict format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Get LangSmith callbacks
        callbacks = langsmith_service.get_callbacks()
        
        # Generate response
        if request.use_memory and request.session_id:
            # Use conversation memory
            result = await gemini_service.generate_with_memory(
                message=messages[-1]["content"],  # Last message
                session_id=request.session_id,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                callbacks=callbacks
            )
            
            response = result["response"]
            session_id = result["session_id"]
        else:
            # Direct generation
            response = await gemini_service.generate_response(
                messages=messages,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                callbacks=callbacks
            )
            session_id = None
        
        # Log to LangSmith
        await langsmith_service.log_run(
            run_type="chat_completion",
            inputs={"messages": messages, "params": request.dict(exclude={"messages"})},
            outputs={"response": response}
        )
        
        return ChatResponse(
            response=response,
            model=request.model,
            session_id=session_id
        )
        
    except Exception as e:
        logger.error("Failed to generate chat completion", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stream")
async def chat_stream(request: ChatRequest):
    """Stream chat completion (SSE)."""
    # Implementation for streaming responses
    pass


@router.delete("/memory/{session_id}")
async def clear_memory(session_id: str):
    """Clear conversation memory for a session."""
    try:
        success = gemini_service.clear_memory(session_id)
        
        if success:
            return {"message": f"Memory cleared for session {session_id}"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except Exception as e:
        logger.error("Failed to clear memory", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))