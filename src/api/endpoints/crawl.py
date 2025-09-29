"""Web crawling API endpoints."""

from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime

from src.services.crawl4ai.crawler import web_crawler
from src.services.web_intelligence.smart_scraper import smart_scraper
from src.services.pinecone.client import vector_store_service
from src.core.config.logging_config import LoggerAdapter
from src.core.cache.redis_client import redis_cache

logger = LoggerAdapter(__name__)
router = APIRouter(prefix="/crawl", tags=["Web Crawling"])


class CrawlRequest(BaseModel):
    """Web crawl request model."""
    url: HttpUrl
    max_depth: int = Field(default=2, ge=1, le=5, description="Maximum crawl depth")
    max_pages: int = Field(default=10, ge=1, le=100, description="Maximum pages to crawl")
    extract_content: bool = Field(default=True, description="Extract and clean content")
    extract_links: bool = Field(default=True, description="Extract links from pages")
    extract_images: bool = Field(default=False, description="Extract images from pages")
    follow_robots_txt: bool = Field(default=True, description="Respect robots.txt")
    wait_time: float = Field(default=1.0, ge=0, le=10, description="Wait time between requests (seconds)")
    user_agent: Optional[str] = Field(default=None, description="Custom user agent")
    save_to_vectors: bool = Field(default=False, description="Save content to vector store")
    namespace: str = Field(default="web_content", description="Vector store namespace")


class SmartCrawlRequest(BaseModel):
    """Smart web crawl request with AI extraction."""
    url: HttpUrl
    extraction_prompt: str = Field(..., description="What to extract from the page")
    css_selector: Optional[str] = Field(default=None, description="CSS selector for specific content")
    wait_for_selector: Optional[str] = Field(default=None, description="Wait for specific element")
    javascript_enabled: bool = Field(default=True, description="Enable JavaScript rendering")
    screenshot: bool = Field(default=False, description="Take screenshot of page")
    extract_structured_data: bool = Field(default=True, description="Extract structured data (JSON-LD, etc.)")


class CrawlStatus(BaseModel):
    """Crawl job status model."""
    job_id: str
    status: str  # pending, running, completed, failed
    url: str
    pages_crawled: int
    start_time: str
    end_time: Optional[str] = None
    error: Optional[str] = None
    results_summary: Optional[Dict[str, Any]] = None


class CrawlResult(BaseModel):
    """Crawl result model."""
    url: str
    title: Optional[str] = None
    content: Optional[str] = None
    links: Optional[List[str]] = None
    images: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str


@router.post("/", response_model=CrawlStatus)
async def crawl_website(
    background_tasks: BackgroundTasks,
    request: CrawlRequest
):
    """Crawl a website and extract content."""
    try:
        # Generate job ID
        job_id = f"crawl_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.url.host}"
        
        # Create job status
        job_status = {
            "job_id": job_id,
            "status": "pending",
            "url": str(request.url),
            "pages_crawled": 0,
            "start_time": datetime.now().isoformat()
        }
        
        # Store job status in cache
        await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)  # 1 hour
        
        # Start crawling in background
        background_tasks.add_task(
            crawl_website_background,
            job_id,
            request
        )
        
        job_status["status"] = "running"
        await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)
        
        logger.info(f"Crawl job started: {job_id}")
        
        return CrawlStatus(**job_status)
        
    except Exception as e:
        logger.error("Failed to start crawl job", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


async def crawl_website_background(job_id: str, request: CrawlRequest):
    """Background task to crawl website."""
    try:
        # Update job status
        job_status = await redis_cache.get(f"crawl_job:{job_id}")
        if job_status:
            job_status["status"] = "running"
            await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)
        
        # Perform crawling
        results = await web_crawler.crawl(
            url=str(request.url),
            max_depth=request.max_depth,
            max_pages=request.max_pages,
            extract_content=request.extract_content,
            extract_links=request.extract_links,
            extract_images=request.extract_images,
            follow_robots_txt=request.follow_robots_txt,
            wait_time=request.wait_time,
            user_agent=request.user_agent
        )
        
        # Save to vector store if requested
        if request.save_to_vectors and results:
            documents = []
            for result in results:
                if result.get("content"):
                    doc = {
                        "content": result["content"],
                        "metadata": {
                            "url": result["url"],
                            "title": result.get("title", ""),
                            "crawled_at": result.get("timestamp", ""),
                            "source": "web_crawl"
                        }
                    }
                    documents.append(doc)
            
            if documents:
                await vector_store_service.add_documents(
                    documents,
                    namespace=request.namespace
                )
        
        # Update job status
        if job_status:
            job_status["status"] = "completed"
            job_status["pages_crawled"] = len(results)
            job_status["end_time"] = datetime.now().isoformat()
            job_status["results_summary"] = {
                "total_pages": len(results),
                "total_links": sum(len(r.get("links", [])) for r in results),
                "saved_to_vectors": request.save_to_vectors
            }
            await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)
        
        # Store results
        await redis_cache.set(f"crawl_results:{job_id}", results, ttl=3600)
        
        logger.info(f"Crawl job completed: {job_id}, {len(results)} pages")
        
    except Exception as e:
        logger.error(f"Failed to complete crawl job {job_id}", error=str(e))
        # Update job status with error
        job_status = await redis_cache.get(f"crawl_job:{job_id}")
        if job_status:
            job_status["status"] = "failed"
            job_status["error"] = str(e)
            job_status["end_time"] = datetime.now().isoformat()
            await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)


@router.post("/smart", response_model=Dict[str, Any])
async def smart_crawl(request: SmartCrawlRequest):
    """Perform intelligent web crawling with AI extraction."""
    try:
        # Use smart scraper for intelligent extraction
        result = await smart_scraper.scrape_with_ai(
            url=str(request.url),
            extraction_prompt=request.extraction_prompt,
            css_selector=request.css_selector,
            wait_for_selector=request.wait_for_selector,
            javascript_enabled=request.javascript_enabled,
            screenshot=request.screenshot,
            extract_structured_data=request.extract_structured_data
        )
        
        logger.info(f"Smart crawl completed for: {request.url}")
        
        return {
            "url": str(request.url),
            "extracted_data": result.get("extracted_data"),
            "structured_data": result.get("structured_data"),
            "screenshot": result.get("screenshot"),
            "metadata": result.get("metadata")
        }
        
    except Exception as e:
        logger.error("Failed to perform smart crawl", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status/{job_id}", response_model=CrawlStatus)
async def get_crawl_status(job_id: str):
    """Get crawl job status."""
    try:
        job_status = await redis_cache.get(f"crawl_job:{job_id}")
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Crawl job not found")
        
        return CrawlStatus(**job_status)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get crawl status", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/results/{job_id}", response_model=List[CrawlResult])
async def get_crawl_results(job_id: str, limit: int = 100):
    """Get crawl job results."""
    try:
        # Check if job exists
        job_status = await redis_cache.get(f"crawl_job:{job_id}")
        if not job_status:
            raise HTTPException(status_code=404, detail="Crawl job not found")
        
        # Get results
        results = await redis_cache.get(f"crawl_results:{job_id}")
        if not results:
            if job_status["status"] == "running":
                return []  # Job still running
            raise HTTPException(status_code=404, detail="No results found")
        
        # Convert to response model
        crawl_results = []
        for result in results[:limit]:
            crawl_results.append(CrawlResult(**result))
        
        return crawl_results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to get crawl results", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/extract-from-url")
async def extract_from_url(
    url: HttpUrl,
    css_selectors: Optional[List[str]] = None,
    xpath_queries: Optional[List[str]] = None
):
    """Extract specific content from a URL."""
    try:
        result = await web_crawler.extract_specific_content(
            url=str(url),
            css_selectors=css_selectors,
            xpath_queries=xpath_queries
        )
        
        return {
            "url": str(url),
            "extracted_content": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error("Failed to extract from URL", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sitemap")
async def parse_sitemap(url: HttpUrl):
    """Parse and extract URLs from a sitemap."""
    try:
        urls = await web_crawler.parse_sitemap(str(url))
        
        return {
            "sitemap_url": str(url),
            "total_urls": len(urls),
            "urls": urls
        }
        
    except Exception as e:
        logger.error("Failed to parse sitemap", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/job/{job_id}")
async def cancel_crawl_job(job_id: str):
    """Cancel a running crawl job."""
    try:
        job_status = await redis_cache.get(f"crawl_job:{job_id}")
        
        if not job_status:
            raise HTTPException(status_code=404, detail="Crawl job not found")
        
        if job_status["status"] in ["completed", "failed"]:
            raise HTTPException(status_code=400, detail=f"Job already {job_status['status']}")
        
        # Update job status
        job_status["status"] = "cancelled"
        job_status["end_time"] = datetime.now().isoformat()
        await redis_cache.set(f"crawl_job:{job_id}", job_status, ttl=3600)
        
        # Clean up results if any
        await redis_cache.delete(f"crawl_results:{job_id}")
        
        logger.info(f"Crawl job cancelled: {job_id}")
        
        return {"message": f"Crawl job {job_id} cancelled successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to cancel crawl job", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))