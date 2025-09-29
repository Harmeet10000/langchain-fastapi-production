"""Unified content intelligence system for processing documents and web content."""

import os
import asyncio
from typing import List, Dict, Any, Optional, Union
from pathlib import Path
from datetime import datetime
import hashlib
import mimetypes

from src.services.document_intelligence.advanced_processor import DocumentIntelligence
from src.services.web_intelligence.smart_scraper import web_intelligence
from src.services.langchain.vectorstore_service import vectorstore_service
from src.services.langchain.gemini_service import gemini_service
from src.core.config.logging_config import LoggerAdapter
from src.core.cache.redis_client import cache_manager

logger = LoggerAdapter(__name__)


class UnifiedContentIntelligence:
    """Unified system for processing all types of content."""
    
    def __init__(self):
        """Initialize unified content intelligence."""
        self.doc_processor = DocumentIntelligence()
        self.web_processor = web_intelligence
        
    async def process_content(
        self,
        source: Union[str, Path, bytes],
        source_type: str = "auto",  # auto, file, url, text, bytes
        processing_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Process any content type with intelligent routing."""
        
        config = processing_config or {}
        
        try:
            # Auto-detect source type
            if source_type == "auto":
                source_type = self._detect_source_type(source)
            
            logger.info(f"Processing content of type: {source_type}")
            
            # Route to appropriate processor
            if source_type == "url":
                result = await self._process_web_content(str(source), config)
            elif source_type == "file":
                result = await self._process_file_content(source, config)
            elif source_type == "text":
                result = await self._process_text_content(str(source), config)
            elif source_type == "bytes":
                result = await self._process_bytes_content(source, config)
            else:
                raise ValueError(f"Unknown source type: {source_type}")
            
            # Add processing metadata
            result["processing_metadata"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "source_type": source_type,
                "processor_version": "1.0.0"
            }
            
            # Store in vector database if requested
            if config.get("store_vectors", False):
                await self._store_in_vectordb(result, config)
            
            # Cache result if requested
            if config.get("cache_result", False):
                await self._cache_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process content: {e}")
            raise
    
    def _detect_source_type(self, source: Any) -> str:
        """Auto-detect the type of source."""
        if isinstance(source, bytes):
            return "bytes"
        
        source_str = str(source)
        
        # Check if URL
        if source_str.startswith(('http://', 'https://', 'www.')):
            return "url"
        
        # Check if file path
        if os.path.exists(source_str):
            return "file"
        
        # Default to text
        return "text"
    
    async def _process_web_content(
        self,
        url: str,
        config: Dict
    ) -> Dict[str, Any]:
        """Process web content."""
        # Extract web scraping config
        web_config = {
            "extract_method": config.get("extract_method", "auto"),
            "javascript_render": config.get("javascript_render", False),
            "extract_metadata": config.get("extract_metadata", True),
            "extract_links": config.get("extract_links", True),
            "extract_images": config.get("extract_images", True),
            "analyze_content": config.get("analyze_content", True)
        }
        
        # Scrape web content
        web_result = await self.web_processor.intelligent_scrape(url, **web_config)
        
        # Process extracted text further if available
        if web_result.get("content", {}).get("text"):
            text_content = web_result["content"]["text"]
            
            # Apply document processing to web text
            doc_analysis = await self.doc_processor.process_text(
                text_content,
                metadata=web_result.get("metadata", {})
            )
            
            # Merge results
            web_result["document_analysis"] = doc_analysis
        
        return {
            "type": "web",
            "source": url,
            "content": web_result
        }
    
    async def _process_file_content(
        self,
        file_path: Union[str, Path],
        config: Dict
    ) -> Dict[str, Any]:
        """Process file content."""
        file_path = Path(file_path)
        
        # Read file
        with open(file_path, 'rb') as f:
            file_bytes = f.read()
        
        # Get file metadata
        file_stats = file_path.stat()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        
        metadata = {
            "filename": file_path.name,
            "path": str(file_path),
            "size": file_stats.st_size,
            "mime_type": mime_type,
            "created": datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
            "modified": datetime.fromtimestamp(file_stats.st_mtime).isoformat()
        }
        
        # Process with document intelligence
        doc_result = await self.doc_processor.process_document(
            file_bytes=file_bytes,
            filename=file_path.name,
            extract_images=config.get("extract_images", True),
            extract_tables=config.get("extract_tables", True),
            extract_metadata=config.get("extract_metadata", True),
            chunk_content=config.get("chunk_content", True),
            generate_summary=config.get("generate_summary", True)
        )
        
        return {
            "type": "file",
            "source": str(file_path),
            "metadata": metadata,
            "content": doc_result
        }
    
    async def _process_text_content(
        self,
        text: str,
        config: Dict
    ) -> Dict[str, Any]:
        """Process plain text content."""
        # Process with document intelligence
        doc_result = await self.doc_processor.process_text(
            text,
            metadata=config.get("metadata", {})
        )
        
        return {
            "type": "text",
            "source": "direct_input",
            "content": doc_result
        }
    
    async def _process_bytes_content(
        self,
        content_bytes: bytes,
        config: Dict
    ) -> Dict[str, Any]:
        """Process raw bytes content."""
        # Try to detect content type
        filename = config.get("filename", "unknown")
        
        # Process with document intelligence
        doc_result = await self.doc_processor.process_document(
            file_bytes=content_bytes,
            filename=filename,
            extract_images=config.get("extract_images", True),
            extract_tables=config.get("extract_tables", True),
            extract_metadata=config.get("extract_metadata", True),
            chunk_content=config.get("chunk_content", True),
            generate_summary=config.get("generate_summary", True)
        )
        
        return {
            "type": "bytes",
            "source": "bytes_input",
            "content": doc_result
        }
    
    async def _store_in_vectordb(self, result: Dict, config: Dict):
        """Store processed content in vector database."""
        try:
            # Extract chunks or create from content
            chunks = []
            
            if result["type"] in ["file", "text", "bytes"]:
                chunks = result.get("content", {}).get("chunks", [])
            elif result["type"] == "web":
                # Create chunks from web content
                text = result.get("content", {}).get("content", {}).get("text", "")
                if text:
                    chunks = await self.doc_processor.chunk_text(
                        text,
                        chunk_size=config.get("chunk_size", 1000),
                        chunk_overlap=config.get("chunk_overlap", 100)
                    )
            
            if chunks:
                # Prepare documents for vector storage
                documents = []
                for i, chunk in enumerate(chunks):
                    doc = {
                        "page_content": chunk.get("text", chunk) if isinstance(chunk, dict) else str(chunk),
                        "metadata": {
                            "source": result.get("source"),
                            "type": result.get("type"),
                            "chunk_index": i,
                            "total_chunks": len(chunks),
                            **chunk.get("metadata", {}) if isinstance(chunk, dict) else {}
                        }
                    }
                    documents.append(doc)
                
                # Store in vector database
                collection_name = config.get("collection_name", "content_intelligence")
                await vectorstore_service.add_documents(
                    collection_name=collection_name,
                    documents=documents
                )
                
                logger.info(f"Stored {len(documents)} chunks in vector database")
                
        except Exception as e:
            logger.error(f"Failed to store in vector database: {e}")
    
    async def _cache_result(self, result: Dict):
        """Cache processing result."""
        try:
            # Generate cache key
            source = result.get("source", "unknown")
            cache_key = f"content_intelligence:{hashlib.md5(str(source).encode()).hexdigest()}"
            
            # Store in cache
            await cache_manager.set_cache(
                key=cache_key,
                value=result,
                expire=3600  # 1 hour
            )
            
            logger.info(f"Cached result with key: {cache_key}")
            
        except Exception as e:
            logger.error(f"Failed to cache result: {e}")
    
    async def process_batch(
        self,
        sources: List[Union[str, Path, bytes]],
        source_types: Optional[List[str]] = None,
        processing_config: Optional[Dict] = None,
        max_concurrent: int = 5
    ) -> List[Dict[str, Any]]:
        """Process multiple content sources in batch."""
        if source_types is None:
            source_types = ["auto"] * len(sources)
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_with_semaphore(source, source_type):
            async with semaphore:
                try:
                    return await self.process_content(
                        source,
                        source_type,
                        processing_config
                    )
                except Exception as e:
                    logger.error(f"Failed to process {source}: {e}")
                    return {
                        "error": str(e),
                        "source": str(source),
                        "type": source_type
                    }
        
        tasks = [
            process_with_semaphore(source, source_type)
            for source, source_type in zip(sources, source_types)
        ]
        
        results = await asyncio.gather(*tasks)
        
        logger.info(f"Processed {len(results)} content sources")
        return results
    
    async def create_knowledge_base(
        self,
        sources: List[Union[str, Path]],
        name: str,
        description: str = "",
        processing_config: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Create a knowledge base from multiple sources."""
        logger.info(f"Creating knowledge base: {name}")
        
        # Process all sources
        results = await self.process_batch(
            sources,
            processing_config=processing_config
        )
        
        # Extract successful results
        successful = [r for r in results if "error" not in r]
        failed = [r for r in results if "error" in r]
        
        # Store in vector database
        if successful:
            config = processing_config or {}
            config["collection_name"] = name
            config["store_vectors"] = True
            
            for result in successful:
                await self._store_in_vectordb(result, config)
        
        # Create knowledge base metadata
        knowledge_base = {
            "name": name,
            "description": description,
            "created_at": datetime.utcnow().isoformat(),
            "sources_processed": len(successful),
            "sources_failed": len(failed),
            "statistics": self._calculate_statistics(successful),
            "failed_sources": failed
        }
        
        # Store metadata
        await cache_manager.set_cache(
            key=f"knowledge_base:{name}",
            value=knowledge_base,
            expire=86400  # 24 hours
        )
        
        logger.info(f"Knowledge base '{name}' created with {len(successful)} sources")
        return knowledge_base
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate statistics from processed results."""
        stats = {
            "total_sources": len(results),
            "by_type": {},
            "total_chunks": 0,
            "total_words": 0,
            "total_images": 0,
            "total_tables": 0,
            "total_links": 0
        }
        
        for result in results:
            # Count by type
            content_type = result.get("type", "unknown")
            stats["by_type"][content_type] = stats["by_type"].get(content_type, 0) + 1
            
            # Count content elements
            content = result.get("content", {})
            
            # Chunks
            chunks = content.get("chunks", [])
            stats["total_chunks"] += len(chunks)
            
            # Words
            if "word_count" in content:
                stats["total_words"] += content["word_count"]
            
            # Images
            images = content.get("images", [])
            stats["total_images"] += len(images)
            
            # Tables
            tables = content.get("tables", [])
            stats["total_tables"] += len(tables)
            
            # Links (for web content)
            if result["type"] == "web":
                links = result.get("content", {}).get("links", [])
                stats["total_links"] += len(links)
        
        return stats
    
    async def search_knowledge_base(
        self,
        query: str,
        knowledge_base_name: str,
        top_k: int = 5,
        search_type: str = "similarity"  # similarity, mmr, threshold
    ) -> Dict[str, Any]:
        """Search within a knowledge base."""
        try:
            # Search in vector database
            results = await vectorstore_service.search(
                collection_name=knowledge_base_name,
                query=query,
                k=top_k,
                search_type=search_type
            )
            
            # Get knowledge base metadata
            kb_metadata = await cache_manager.get_cache(f"knowledge_base:{knowledge_base_name}")
            
            return {
                "query": query,
                "knowledge_base": knowledge_base_name,
                "metadata": kb_metadata,
                "results": results,
                "count": len(results)
            }
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise
    
    async def generate_answer(
        self,
        question: str,
        knowledge_base_name: str,
        context_chunks: int = 5,
        answer_style: str = "concise"  # concise, detailed, technical
    ) -> Dict[str, Any]:
        """Generate an answer using knowledge base context."""
        try:
            # Search for relevant context
            search_result = await self.search_knowledge_base(
                query=question,
                knowledge_base_name=knowledge_base_name,
                top_k=context_chunks
            )
            
            # Build context from search results
            context_parts = []
            sources = []
            
            for result in search_result.get("results", []):
                context_parts.append(result.get("page_content", ""))
                sources.append(result.get("metadata", {}).get("source", "unknown"))
            
            context = "\n\n".join(context_parts)
            
            # Generate answer using Gemini
            style_instructions = {
                "concise": "Provide a brief, direct answer.",
                "detailed": "Provide a comprehensive answer with explanations.",
                "technical": "Provide a technical answer with specific details."
            }
            
            prompt = f"""
            Based on the following context, answer the question.
            {style_instructions.get(answer_style, "")}
            
            Context:
            {context}
            
            Question: {question}
            
            Answer:
            """
            
            response = await gemini_service.generate_response(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            return {
                "question": question,
                "answer": response,
                "sources": list(set(sources)),
                "context_used": len(context_parts),
                "answer_style": answer_style
            }
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            raise


# Global instance
unified_intelligence = UnifiedContentIntelligence()


# Convenience functions
async def process_any_content(
    source: Union[str, Path, bytes],
    **kwargs
) -> Dict[str, Any]:
    """Process any type of content."""
    return await unified_intelligence.process_content(source, **kwargs)


async def create_knowledge_base_from_sources(
    sources: List[Union[str, Path]],
    name: str,
    description: str = "",
    **kwargs
) -> Dict[str, Any]:
    """Create a knowledge base from multiple sources."""
    return await unified_intelligence.create_knowledge_base(
        sources, name, description, **kwargs
    )


async def ask_knowledge_base(
    question: str,
    knowledge_base_name: str,
    **kwargs
) -> Dict[str, Any]:
    """Ask a question to a knowledge base."""
    return await unified_intelligence.generate_answer(
        question, knowledge_base_name, **kwargs
    )