"""RAG API endpoints."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from pydantic import BaseModel, Field
import tempfile
import os

from src.services.docling.processor import document_processor
from src.services.pinecone.client import vector_store_service
from src.graphs.workflows.rag_workflow import rag_workflow
from src.core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)
router = APIRouter(prefix="/rag", tags=["RAG"])


class RAGQuery(BaseModel):
    """RAG query request model."""
    query: str = Field(..., description="Query to search for")
    namespace: Optional[str] = Field(default="default", description="Pinecone namespace")
    top_k: int = Field(default=5, ge=1, le=20)
    score_threshold: Optional[float] = Field(default=0.5, ge=0.0, le=1.0)
    use_workflow: bool = Field(default=True, description="Use LangGraph workflow")


class RAGResponse(BaseModel):
    """RAG response model."""
    answer: str
    sources: List[Dict[str, Any]]
    confidence: Optional[float] = None
    error: Optional[str] = None


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    message: str
    document_id: str
    chunks: int
    vector_ids: List[str]
    namespace: str


@router.post("/upload", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    namespace: str = Form(default="default"),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200)
):
    """Upload and process document for RAG."""
    try:
        # Validate file type
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in ["pdf", "txt", "docx", "md", "html", "csv", "xlsx"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}"
            )
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        try:
            # Process document
            chunks = await document_processor.process_document(
                tmp_path,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
            # Add document metadata
            for chunk in chunks:
                chunk["metadata"]["source"] = file.filename
                chunk["metadata"]["namespace"] = namespace
            
            # Add to vector store
            vector_ids = await vector_store_service.add_documents(
                chunks,
                namespace=namespace
            )
            
            logger.info(f"Uploaded document: {file.filename}, {len(chunks)} chunks")
            
            return DocumentUploadResponse(
                message="Document processed and indexed successfully",
                document_id=file.filename,
                chunks=len(chunks),
                vector_ids=vector_ids,
                namespace=namespace
            )
            
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
            
    except Exception as e:
        logger.error("Failed to upload document", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=RAGResponse)
async def query_rag(request: RAGQuery):
    """Query RAG system."""
    try:
        if request.use_workflow:
            # Use LangGraph workflow
            result = await rag_workflow.run(
                query=request.query,
                metadata={
                    "namespace": request.namespace,
                    "top_k": request.top_k,
                    "score_threshold": request.score_threshold
                }
            )
            
            return RAGResponse(
                answer=result["answer"],
                sources=result["sources"],
                error=result.get("error")
            )
        else:
            # Direct vector search
            results = await vector_store_service.similarity_search_with_relevance_scores(
                query=request.query,
                k=request.top_k,
                namespace=request.namespace,
                score_threshold=request.score_threshold
            )
            
            if not results:
                return RAGResponse(
                    answer="No relevant documents found.",
                    sources=[],
                    error="No matches found"
                )
            
            # Simple answer generation from top result
            answer = f"Based on the search, here's what I found:\n\n{results[0]['content'][:500]}..."
            
            return RAGResponse(
                answer=answer,
                sources=results,
                confidence=results[0]["score"] if results else 0.0
            )
            
    except Exception as e:
        logger.error("Failed to query RAG", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def vector_search(
    query: str,
    namespace: str = "default",
    k: int = 5
):
    """Perform vector similarity search."""
    try:
        results = await vector_store_service.similarity_search(
            query=query,
            k=k,
            namespace=namespace
        )
        
        return {
            "query": query,
            "results": results,
            "count": len(results)
        }
        
    except Exception as e:
        logger.error("Failed to perform vector search", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/vectors")
async def delete_vectors(
    ids: List[str],
    namespace: str = "default"
):
    """Delete vectors by IDs."""
    try:
        success = await vector_store_service.delete_by_ids(ids, namespace)
        
        if success:
            return {"message": f"Deleted {len(ids)} vectors from namespace {namespace}"}
        else:
            raise HTTPException(status_code=500, detail="Failed to delete vectors")
            
    except Exception as e:
        logger.error("Failed to delete vectors", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_vector_stats():
    """Get vector store statistics."""
    try:
        stats = vector_store_service.get_index_stats()
        return stats
    except Exception as e:
        logger.error("Failed to get vector stats", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))