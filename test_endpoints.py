#!/usr/bin/env python3
"""
Test script to verify all API endpoints in the LangChain FastAPI application.

Usage:
    python test_endpoints.py

This script will test all available endpoints with example requests.
Make sure the application is running before executing this script.
"""

import asyncio
import httpx
import json
from typing import Dict, Any, Optional
from datetime import datetime
import os
from pathlib import Path

# Configuration
BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"
FULL_URL = f"{BASE_URL}{API_PREFIX}"

# Test data storage
test_results = {
    "passed": [],
    "failed": [],
    "skipped": []
}

# Store IDs for cleanup
created_resources = {
    "documents": [],
    "crawl_jobs": [],
    "threads": []
}


def print_header(text: str):
    """Print formatted header."""
    print("\n" + "=" * 60)
    print(f" {text}")
    print("=" * 60)


def print_test(name: str, status: str, details: Optional[str] = None):
    """Print test result."""
    symbols = {"passed": "✅", "failed": "❌", "skipped": "⏭️"}
    print(f"{symbols.get(status, '❓')} {name}")
    if details:
        print(f"   {details}")


async def test_endpoint(
    client: httpx.AsyncClient,
    method: str,
    endpoint: str,
    name: str,
    data: Optional[Dict[str, Any]] = None,
    files: Optional[Dict] = None,
    params: Optional[Dict] = None
) -> Optional[Dict]:
    """Test a single endpoint."""
    try:
        url = f"{FULL_URL}{endpoint}"
        
        if method == "GET":
            response = await client.get(url, params=params)
        elif method == "POST":
            if files:
                response = await client.post(url, files=files, data=data or {})
            else:
                response = await client.post(url, json=data)
        elif method == "DELETE":
            response = await client.delete(url)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        if response.status_code in [200, 201]:
            test_results["passed"].append(name)
            print_test(name, "passed", f"Status: {response.status_code}")
            return response.json()
        else:
            test_results["failed"].append(name)
            print_test(name, "failed", f"Status: {response.status_code}, Error: {response.text[:200]}")
            return None
            
    except Exception as e:
        test_results["failed"].append(name)
        print_test(name, "failed", f"Error: {str(e)[:200]}")
        return None


async def test_health_endpoints(client: httpx.AsyncClient):
    """Test health check endpoints."""
    print_header("Testing Health Endpoints")
    
    # Main health check
    await test_endpoint(client, "GET", "/health", "API Health Check")
    
    # Root endpoint
    result = await test_endpoint(client, "GET", "", "API Root Info")
    if result:
        print(f"   API Version: {result.get('version', 'N/A')}")


async def test_chat_endpoints(client: httpx.AsyncClient):
    """Test chat completion endpoints."""
    print_header("Testing Chat Endpoints")
    
    # Basic chat completion
    chat_data = {
        "messages": [
            {"role": "user", "content": "Hello, what is LangChain?"}
        ],
        "model": "gemini-pro",
        "temperature": 0.7,
        "max_tokens": 150
    }
    
    result = await test_endpoint(
        client, "POST", "/chat/completions", 
        "Chat Completion", data=chat_data
    )
    
    if result:
        print(f"   Response preview: {result.get('response', '')[:100]}...")
    
    # Chat with memory
    chat_with_memory = {
        **chat_data,
        "use_memory": True,
        "session_id": "test_session_123"
    }
    
    result = await test_endpoint(
        client, "POST", "/chat/completions",
        "Chat with Memory", data=chat_with_memory
    )
    
    # Clear memory
    await test_endpoint(
        client, "DELETE", "/chat/memory/test_session_123",
        "Clear Chat Memory"
    )


async def test_document_endpoints(client: httpx.AsyncClient):
    """Test document management endpoints."""
    print_header("Testing Document Endpoints")
    
    # Create a test document
    test_file_path = Path("test_document.txt")
    test_file_path.write_text("This is a test document for the LangChain FastAPI application.\nIt contains sample text for testing.")
    
    try:
        # Upload document
        with open(test_file_path, "rb") as f:
            files = {"file": ("test_document.txt", f, "text/plain")}
            data = {
                "namespace": "test",
                "process_immediately": "true",
                "chunk_size": "500"
            }
            
            result = await test_endpoint(
                client, "POST", "/documents/upload",
                "Document Upload", files=files, data=data
            )
            
            if result:
                doc_id = result.get("document_id")
                created_resources["documents"].append(doc_id)
                print(f"   Document ID: {doc_id}")
                
                # Wait a bit for processing
                await asyncio.sleep(2)
                
                # Check document status
                await test_endpoint(
                    client, "GET", f"/documents/status/{doc_id}",
                    "Document Status Check"
                )
                
                # Search documents
                search_data = {
                    "query": "test document",
                    "document_ids": [doc_id],
                    "limit": 5
                }
                
                await test_endpoint(
                    client, "POST", "/documents/search",
                    "Document Search", data=search_data
                )
                
                # Extract entities (if document is processed)
                await test_endpoint(
                    client, "POST", f"/documents/extract-entities/{doc_id}",
                    "Extract Entities"
                )
        
        # List documents
        await test_endpoint(
            client, "GET", "/documents/list",
            "List Documents", params={"namespace": "test"}
        )
        
    finally:
        # Clean up test file
        if test_file_path.exists():
            test_file_path.unlink()


async def test_rag_endpoints(client: httpx.AsyncClient):
    """Test RAG endpoints."""
    print_header("Testing RAG Endpoints")
    
    # RAG query
    rag_query_data = {
        "query": "What is this document about?",
        "namespace": "test",
        "top_k": 3,
        "use_workflow": False
    }
    
    result = await test_endpoint(
        client, "POST", "/rag/query",
        "RAG Query", data=rag_query_data
    )
    
    if result:
        print(f"   Answer preview: {result.get('answer', '')[:100]}...")
    
    # Vector search
    await test_endpoint(
        client, "GET", "/rag/search",
        "Vector Search",
        params={"query": "test", "namespace": "test", "k": 5}
    )
    
    # Vector stats
    await test_endpoint(
        client, "GET", "/rag/stats",
        "Vector Store Statistics"
    )


async def test_crawl_endpoints(client: httpx.AsyncClient):
    """Test web crawling endpoints."""
    print_header("Testing Crawl Endpoints")
    
    # Basic crawl
    crawl_data = {
        "url": "https://example.com",
        "max_depth": 1,
        "max_pages": 2,
        "extract_content": True,
        "save_to_vectors": False
    }
    
    result = await test_endpoint(
        client, "POST", "/crawl/",
        "Start Web Crawl", data=crawl_data
    )
    
    if result:
        job_id = result.get("job_id")
        created_resources["crawl_jobs"].append(job_id)
        print(f"   Job ID: {job_id}")
        
        # Wait for crawl to start
        await asyncio.sleep(2)
        
        # Check crawl status
        await test_endpoint(
            client, "GET", f"/crawl/status/{job_id}",
            "Crawl Job Status"
        )
        
        # Get results
        await test_endpoint(
            client, "GET", f"/crawl/results/{job_id}",
            "Crawl Results", params={"limit": 5}
        )
    
    # Smart crawl with AI
    smart_crawl_data = {
        "url": "https://example.com",
        "extraction_prompt": "Extract the main heading and first paragraph",
        "javascript_enabled": False
    }
    
    await test_endpoint(
        client, "POST", "/crawl/smart",
        "Smart Crawl with AI", data=smart_crawl_data
    )
    
    # Extract from specific URL
    extract_data = {
        "url": "https://example.com",
        "css_selectors": ["h1", "p"]
    }
    
    await test_endpoint(
        client, "POST", "/crawl/extract-from-url",
        "Extract from URL", data=extract_data
    )
    
    # Parse sitemap
    await test_endpoint(
        client, "POST", "/crawl/sitemap",
        "Parse Sitemap", data={"url": "https://example.com/sitemap.xml"}
    )


async def test_workflow_endpoints(client: httpx.AsyncClient):
    """Test LangGraph workflow endpoints."""
    print_header("Testing Workflow Endpoints")
    
    # List workflows
    result = await test_endpoint(
        client, "GET", "/workflows/list",
        "List Available Workflows"
    )
    
    if result:
        workflows = result.get("workflows", [])
        print(f"   Available workflows: {len(workflows)}")
    
    # Execute RAG workflow
    rag_workflow_data = {
        "query": "What are the key features of LangChain?",
        "namespace": "default",
        "top_k": 3,
        "use_cache": False
    }
    
    result = await test_endpoint(
        client, "POST", "/workflows/rag",
        "Execute RAG Workflow", data=rag_workflow_data
    )
    
    if result:
        thread_id = result.get("thread_id")
        created_resources["threads"].append(thread_id)
        
        # Get workflow state
        await test_endpoint(
            client, "GET", f"/workflows/state/{thread_id}",
            "Get Workflow State"
        )
        
        # Get workflow history
        await test_endpoint(
            client, "GET", f"/workflows/history/{thread_id}",
            "Get Workflow History"
        )
        
        # Create checkpoint
        await test_endpoint(
            client, "POST", f"/workflows/checkpoint/{thread_id}",
            "Create Workflow Checkpoint"
        )
    
    # Execute generic workflow
    workflow_data = {
        "workflow_id": "test_workflow",
        "inputs": {"message": "Test input"},
        "config": {"temperature": 0.5}
    }
    
    result = await test_endpoint(
        client, "POST", "/workflows/execute",
        "Execute Generic Workflow", data=workflow_data
    )


async def cleanup_resources(client: httpx.AsyncClient):
    """Clean up created test resources."""
    print_header("Cleaning Up Test Resources")
    
    # Clean up documents
    for doc_id in created_resources["documents"]:
        await test_endpoint(
            client, "DELETE", f"/documents/{doc_id}",
            f"Delete Document {doc_id[:20]}..."
        )
    
    # Cancel crawl jobs
    for job_id in created_resources["crawl_jobs"]:
        await test_endpoint(
            client, "DELETE", f"/crawl/job/{job_id}",
            f"Cancel Crawl Job {job_id[:20]}..."
        )


async def main():
    """Main test execution."""
    print_header("LangChain FastAPI Endpoint Tests")
    print(f"Testing API at: {FULL_URL}")
    print(f"Started at: {datetime.now().isoformat()}")
    
    # Configure client with longer timeout for some operations
    timeout = httpx.Timeout(30.0)
    
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Check if API is accessible
        try:
            response = await client.get(f"{BASE_URL}/health")
            if response.status_code != 200:
                print("❌ API is not accessible. Please ensure the server is running.")
                return
        except Exception as e:
            print(f"❌ Cannot connect to API: {e}")
            print("Please ensure the server is running with:")
            print("  docker-compose up")
            print("  or")
            print("  uvicorn src.main:app --reload --host 0.0.0.0 --port 8000")
            return
        
        # Run tests
        await test_health_endpoints(client)
        await test_chat_endpoints(client)
        await test_document_endpoints(client)
        await test_rag_endpoints(client)
        await test_crawl_endpoints(client)
        await test_workflow_endpoints(client)
        
        # Cleanup
        await cleanup_resources(client)
    
    # Print summary
    print_header("Test Summary")
    print(f"✅ Passed: {len(test_results['passed'])}")
    print(f"❌ Failed: {len(test_results['failed'])}")
    print(f"⏭️  Skipped: {len(test_results['skipped'])}")
    
    if test_results["failed"]:
        print("\nFailed tests:")
        for test_name in test_results["failed"]:
            print(f"  - {test_name}")
    
    print(f"\nCompleted at: {datetime.now().isoformat()}")
    
    # Exit with appropriate code
    exit_code = 0 if not test_results["failed"] else 1
    exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())