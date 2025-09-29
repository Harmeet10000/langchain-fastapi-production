"""Document management API endpoints."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, UploadFile, File, HTTPException, Form, BackgroundTasks
from pydantic import BaseModel, Field
import tempfile
import os
from datetime import datetime

from src.services.docling.processor import document_processor
from src.services.pinecone.client import vector_store_service
from src.services.document_intelligence.advanced_processor import advanced_document_processor
from src.core.config.logging_config import LoggerAdapter
from src.core.cache.redis_client import redis_cache

logger = LoggerAdapter(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


class DocumentInfo(BaseModel):
    """Document information model."""
    document_id: str
    filename: str
    upload_time: str
    status: str
    chunks: Optional[int] = None
    namespace: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class DocumentProcessRequest(BaseModel):
    """Document processing request model."""
    document_id: str
    processing_type: str = Field(default="standard", description="Type of processing: standard, advanced, ocr")
    chunk_size: int = Field(default=1000, ge=100, le=5000)
    chunk_overlap: int = Field(default=200, ge=0, le=500)
    extract_tables: bool = Field(default=True)
    extract_images: bool = Field(default=False)


class DocumentSearchRequest(BaseModel):
    """Document search request model."""
    query: str
    document_ids: Optional[List[str]] = None
    limit: int = Field(default=10, ge=1, le=100)
    search_type: str = Field(default="semantic", description="Search type: semantic, keyword, hybrid")


@router.post("/upload", response_model=DocumentInfo)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    namespace: str = Form(default="default"),
    process_immediately: bool = Form(default=True),
    chunk_size: int = Form(default=1000),
    chunk_overlap: int = Form(default=200)
):
    """Upload a document for processing."""
    try:
        # Validate file type
        file_extension = file.filename.split(".")[-1].lower()
        supported_formats = ["pdf", "txt", "docx", "xlsx", "pptx", "md", "html", "csv", "json"]
        
        if file_extension not in supported_formats:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file_extension}. Supported formats: {', '.join(supported_formats)}"
            )
        
        # Generate document ID
        document_id = f"doc_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{file.filename}"
        
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file_extension}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name
        
        # Store document info in cache
        doc_info = {
            "document_id": document_id,
            "filename": file.filename,
            "upload_time": datetime.now().isoformat(),
            "status": "uploaded",
            "file_path": tmp_path,
            "namespace": namespace
        }
        
        await redis_cache.set(f"document:{document_id}", doc_info, ttl=86400)  # 24 hours
        
        if process_immediately:
            # Process in background
            background_tasks.add_task(
                process_document_background,
                document_id,
                tmp_path,
                namespace,
                chunk_size,
                chunk_overlap
            )
            doc_info["status"] = "processing"
        
        logger.info(f"Document uploaded: {document_id}")
        
        return DocumentInfo(**doc_info)
        
    except Exception as e:
        logger.error("Failed to upload document", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def process_document_background(
    document_id: str,
    file_path: str,
    namespace: str,
    chunk_size: int,
    chunk_overlap: int
):
    """Background task to process document."""
    try:
        # Process document
        chunks = await document_processor.process_document(
            file_path,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        # Add metadata
        for chunk in chunks:
            chunk["metadata"]["document_id"] = document_id
            chunk["metadata"]["namespace"] = namespace
        
        # Add to vector store
        vector_ids = await vector_store_service.add_documents(chunks, namespace=namespace)
        
        # Update document info
        doc_info = await redis_cache.get(f"document:{document_id}")
        if doc_info:
            doc_info["status"] = "processed"
            doc_info["chunks"] = len(chunks)
            doc_info["vector_ids"] = vector_ids
            await redis_cache.set(f"document:{document_id}", doc_info, ttl=86400)
        
        logger.info(f"Document processed: {document_id}, {len(chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Failed to process document {document_id}", error=str(e))
        # Update status to failed
        doc_info = await redis_cache.get(f"document:{document_id}")
        if doc_info:
            doc_info["status"] = "failed"
            doc_info["error"] = str(e)
            await redis_cache.set(f"document:{document_id}", doc_info, ttl=86400)
    finally:
        # Clean up temp file
        if os.path.exists(file_path):
            os.unlink(file_path)


@router.get("/status/{document_id}", response_model=DocumentInfo)
async def get_document_status(document_id: str):
    """Get document processing status."""
    try:
        doc_info = await redis_cache.get(f"document:{document_id}")
        
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        return DocumentInfo(**doc_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get document status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list", response_model=List[DocumentInfo])
async def list_documents(namespace: str = "default", limit: int = 100):
    """List all documents in a namespace."""
    try:
        # Get all document keys from cache
        documents = []
        # This is a simplified implementation - in production you'd want to use a database
        # or implement proper document management
        
        # For now, return empty list as we don't have persistence yet
        return documents
        
    except Exception as e:
        logger.error("Failed to list documents", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/process", response_model=DocumentInfo)
async def process_document(
    background_tasks: BackgroundTasks,
    request: DocumentProcessRequest
):
    """Process an already uploaded document with specific settings."""
    try:
        doc_info = await redis_cache.get(f"document:{request.document_id}")
        
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if "file_path" not in doc_info or not os.path.exists(doc_info["file_path"]):
            raise HTTPException(status_code=400, detail="Document file not found")
        
        # Update status
        doc_info["status"] = "reprocessing"
        await redis_cache.set(f"document:{request.document_id}", doc_info, ttl=86400)
        
        # Process in background
        background_tasks.add_task(
            process_document_background,
            request.document_id,
            doc_info["file_path"],
            doc_info.get("namespace", "default"),
            request.chunk_size,
            request.chunk_overlap
        )
        
        return DocumentInfo(**doc_info)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to process document", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_documents(request: DocumentSearchRequest):
    """Search within documents."""
    try:
        # Build filter for specific documents
        filter_dict = None
        if request.document_ids:
            filter_dict = {"document_id": {"$in": request.document_ids}}
        
        # Perform search based on type
        if request.search_type == "semantic":
            results = await vector_store_service.similarity_search(
                query=request.query,
                k=request.limit,
                filter=filter_dict
            )
        else:
            # For keyword search, we'd implement a different approach
            # For now, fallback to semantic search
            results = await vector_store_service.similarity_search(
                query=request.query,
                k=request.limit,
                filter=filter_dict
            )
        
        return {
            "query": request.query,
            "results": results,
            "count": len(results),
            "search_type": request.search_type
        }
        
    except Exception as e:
        logger.error("Failed to search documents", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{document_id}")
async def delete_document(document_id: str):
    """Delete a document and its vectors."""
    try:
        doc_info = await redis_cache.get(f"document:{document_id}")
        
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Delete vectors if they exist
        if "vector_ids" in doc_info:
            namespace = doc_info.get("namespace", "default")
            await vector_store_service.delete_by_ids(
                doc_info["vector_ids"],
                namespace=namespace
            )
        
        # Delete from cache
        await redis_cache.delete(f"document:{document_id}")
        
        # Delete file if it exists
        if "file_path" in doc_info and os.path.exists(doc_info["file_path"]):
            os.unlink(doc_info["file_path"])
        
        logger.info(f"Document deleted: {document_id}")
        
        return {"message": f"Document {document_id} deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to delete document", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-entities/{document_id}")
async def extract_entities(document_id: str):
    """Extract entities from a document."""
    try:
        doc_info = await redis_cache.get(f"document:{document_id}")
        
        if not doc_info:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if "file_path" not in doc_info or not os.path.exists(doc_info["file_path"]):
            raise HTTPException(status_code=400, detail="Document file not found")
        
        # Use advanced processor for entity extraction
        entities = await advanced_document_processor.extract_entities(doc_info["file_path"])
        
        return {
            "document_id": document_id,
            "entities": entities
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to extract entities", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))