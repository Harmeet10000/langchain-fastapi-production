# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

This is a production-grade FastAPI application integrating LangChain, LangGraph, and LangSmith with Google's Gemini models. It features Pinecone for vector storage, Docling for document processing, and Crawl4AI for web scraping.

## Essential Commands

### Development Server
```bash
# Start with Docker (recommended)
docker-compose up --build

# Start locally
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# Start in detached mode
docker-compose up -d

# View logs
docker-compose logs -f app
```

### Environment Setup
```bash
# Copy and configure environment
cp .env.example .env

# Install dependencies locally
pip install -r requirements.txt

# Install Playwright browsers (required for Crawl4AI)
playwright install chromium
```

### Testing
```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_api.py

# Run tests in parallel
pytest -n auto
```

### Code Quality
```bash
# Format code with black
black src/ tests/

# Sort imports
isort src/ tests/

# Type checking
mypy src/

# Linting
pylint src/

# All quality checks
black src/ tests/ && isort src/ tests/ && mypy src/ && pylint src/
```

### Docker Management
```bash
# Stop all services
docker-compose down

# Stop and remove volumes
docker-compose down -v

# Rebuild specific service
docker-compose build app

# Access container shell
docker exec -it langchain-fastapi /bin/bash

# View Redis data
docker exec -it langchain-redis redis-cli

# Access MongoDB
docker exec -it langchain-mongodb mongosh
```

## Architecture Overview

### Service Layer Architecture
The application follows a clean architecture pattern with clear separation of concerns:

1. **API Layer** (`src/api/`)
   - Handles HTTP requests/responses
   - Input validation with Pydantic schemas
   - FastAPI dependencies and middleware

2. **Service Layer** (`src/services/`)
   - Contains all business logic
   - Integrations with external services:
     - `langchain/`: LangChain operations
     - `langgraph/`: Graph-based workflows
     - `langsmith/`: Monitoring and tracing
     - `pinecone/`: Vector database operations
     - `docling/`: Document processing
     - `crawl4ai/`: Web scraping

3. **Chain & Graph Components** (`src/chains/`, `src/graphs/`)
   - RAG chains for retrieval-augmented generation
   - Conversation chains for chat functionality
   - Structured output chains for data extraction
   - LangGraph workflows for complex reasoning

4. **Core Infrastructure** (`src/core/`)
   - Configuration management
   - Database connections (MongoDB, Redis)
   - Security utilities
   - Caching layer

### Key Integration Points

#### LangChain & Gemini
- Models configured via `GEMINI_MODEL` environment variable
- Temperature and token limits configurable
- Chains defined in `src/chains/` directory
- Tools implemented in `src/tools/` directory

#### Vector Storage (Pinecone)
- Initialized in `src/services/pinecone/client.py`
- Embedding dimension: 768 (configurable)
- Index operations handled through service layer
- Automatic document chunking and embedding

#### Document Processing Flow
1. Documents uploaded via `/api/v1/documents/upload`
2. Processed by Docling service
3. Chunked and embedded
4. Stored in Pinecone with metadata
5. Available for RAG queries

#### Monitoring (LangSmith)
- Traces all LangChain operations when enabled
- Configure via `LANGCHAIN_TRACING_V2=true`
- Access traces at https://smith.langchain.com
- Project-based organization

### Async Patterns
The entire application uses async/await patterns:
- All database operations are async
- API endpoints use async handlers
- Background tasks for document processing
- Async Redis caching

## API Endpoints

### Core Endpoints
- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc
- **Health Check**: http://localhost:8000/health

### Management UIs
- **MongoDB Express**: http://localhost:8081 (admin/changeme)
- **Redis Commander**: http://localhost:8082

## Environment Variables

Critical variables that must be configured:
```bash
# Required API Keys
GOOGLE_API_KEY           # Google Gemini API
PINECONE_API_KEY         # Pinecone vector database
PINECONE_ENVIRONMENT     # Pinecone environment name

# Optional but recommended
LANGSMITH_API_KEY        # LangSmith monitoring
LANGSMITH_PROJECT        # Project name in LangSmith

# Service connections (auto-configured in Docker)
REDIS_HOST               # Default: redis (Docker) or localhost
MONGODB_URL              # Default: mongodb://mongodb:27017/langchain_db
```

## Common Development Patterns

### Adding a New Chain
1. Create chain module in `src/chains/[chain_name]/`
2. Define chain logic using LangChain components
3. Create service wrapper in `src/services/langchain/`
4. Add API endpoint in `src/api/endpoints/`
5. Update schemas in `src/schemas/`

### Adding a New LangGraph Workflow
1. Define workflow in `src/graphs/workflows/`
2. Create state management in `src/graphs/states/`
3. Implement graph components in `src/graphs/components/`
4. Integrate with service layer
5. Expose via API endpoint

### Implementing RAG Features
1. Upload documents via document API
2. Documents are automatically processed and indexed
3. Query using RAG endpoints with document filters
4. Customize retrieval in `src/chains/rag/`

### Working with Caching
- Redis caching is automatic for expensive operations
- Cache TTL configured via `CACHE_TTL` environment variable
- Manual cache invalidation available through Redis Commander

## Debugging Tips

### View Application Logs
```bash
# Docker logs
docker-compose logs -f app

# Local development logs
# Logs are in JSON format, configured in src/core/config/logging_config.py
tail -f logs/app.log
```

### Trace LangChain Operations
1. Ensure `LANGCHAIN_TRACING_V2=true` in `.env`
2. Set `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`
3. View traces at https://smith.langchain.com

### Database Inspection
```bash
# MongoDB shell
docker exec -it langchain-mongodb mongosh
use langchain_db
db.collections.find()

# Redis CLI
docker exec -it langchain-redis redis-cli
KEYS *
GET <key>
```

## Performance Considerations

- Document processing is async and queued
- Pinecone operations are batched for efficiency
- Redis caching reduces API calls to external services
- Connection pooling for MongoDB and Redis
- Rate limiting configured via `RATE_LIMIT_*` variables

## Security Notes

- JWT authentication configured but routes must be explicitly protected
- Rate limiting enabled by default
- CORS configured for local development
- All inputs validated with Pydantic
- Secrets managed via environment variables only