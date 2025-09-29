"""Main API router configuration."""

from fastapi import APIRouter
from src.api.endpoints import chat, rag, documents, crawl, workflows

api_router = APIRouter()

# Health check endpoint in main router
@api_router.get("/health")
async def health_check():
    """API health check endpoint."""
    return {"status": "healthy", "service": "api"}

# Include all endpoint routers
api_router.include_router(chat.router)
api_router.include_router(rag.router)
api_router.include_router(documents.router)
api_router.include_router(crawl.router)
api_router.include_router(workflows.router)
