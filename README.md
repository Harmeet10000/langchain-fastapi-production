# LangChain FastAPI Production Template

A production-grade FastAPI application integrating LangChain, LangGraph, and LangSmith with Google's Gemini models, featuring Pinecone for vector storage, Docling for document processing, and Crawl4AI for web scraping.

## 🚀 Features

- **LangChain Integration**: Complete integration with Google Gemini models for LLM operations
- **LangGraph Workflows**: Graph-based reasoning and workflow management
- **LangSmith Monitoring**: Comprehensive tracing, evaluation, and feedback loops
- **Vector Store**: Pinecone integration for efficient semantic search and RAG
- **Document Processing**: Multi-format document parsing with Docling
- **Web Crawling**: Intelligent web scraping with Crawl4AI
- **Production Ready**: Docker, monitoring, caching, and security best practices
- **Async First**: Fully asynchronous architecture for high performance
- **Type Safe**: Complete type hints and Pydantic validation

## 📋 Prerequisites

- Python 3.11+
- Docker and Docker Compose
- API Keys:
  - Google Gemini API Key
  - Pinecone API Key and Environment
  - LangSmith API Key (optional)

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd langchain-fastapi-production
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 3. Using Docker (Recommended)

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d
```

### 4. Local Development

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (for Crawl4AI)
playwright install chromium

# Run the application
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

## 📁 Project Structure

```
langchain-fastapi-production/
├── src/
│   ├── api/                 # API layer
│   │   ├── endpoints/        # API endpoints
│   │   ├── dependencies/     # FastAPI dependencies
│   │   └── middleware/       # Custom middleware
│   ├── core/                 # Core functionality
│   │   ├── config/          # Configuration management
│   │   ├── security/        # Security utilities
│   │   ├── database/        # Database connections
│   │   └── cache/           # Redis cache
│   ├── services/            # Business logic services
│   │   ├── langchain/       # LangChain integration
│   │   ├── langgraph/       # LangGraph workflows
│   │   ├── langsmith/       # LangSmith monitoring
│   │   ├── pinecone/        # Vector store
│   │   ├── docling/         # Document processing
│   │   └── crawl4ai/        # Web crawling
│   ├── chains/              # LangChain chains
│   │   ├── rag/            # RAG chains
│   │   ├── conversation/   # Conversation chains
│   │   └── structured/      # Structured output chains
│   ├── graphs/              # LangGraph components
│   │   ├── workflows/       # Workflow definitions
│   │   ├── states/         # State management
│   │   └── components/      # Reusable graph components
│   ├── models/              # Data models
│   ├── schemas/             # Pydantic schemas
│   ├── tools/               # LangChain tools
│   ├── utils/               # Utility functions
│   └── main.py             # Application entry point
├── tests/                   # Test suite
├── docker/                  # Docker configurations
├── docs/                    # Documentation
├── scripts/                 # Utility scripts
└── data/                    # Data storage
```

## 🔧 Configuration

### Environment Variables

Key environment variables (see `.env.example` for full list):

```bash
# Google Gemini
GOOGLE_API_KEY=your-google-api-key
GEMINI_MODEL=gemini-pro
GEMINI_TEMPERATURE=0.7

# Pinecone
PINECONE_API_KEY=your-pinecone-api-key
PINECONE_ENVIRONMENT=your-environment
PINECONE_INDEX_NAME=langchain-index

# LangSmith (Optional)
LANGSMITH_API_KEY=your-langsmith-api-key
LANGSMITH_PROJECT=your-project
LANGCHAIN_TRACING_V2=true

# Redis Cache
REDIS_HOST=localhost
REDIS_PORT=6379

# MongoDB
MONGODB_URL=mongodb://localhost:27017/langchain_db
```

## 📚 API Documentation

Once the application is running, you can access:

- **Swagger UI**: http://localhost:8000/api/v1/docs
- **ReDoc**: http://localhost:8000/api/v1/redoc
- **OpenAPI JSON**: http://localhost:8000/api/v1/openapi.json

## 🔥 Quick Start Examples

### 1. Basic Chat Completion

```python
import httpx

response = httpx.post(
    "http://localhost:8000/api/v1/chat/completions",
    json={
        "messages": [
            {"role": "user", "content": "What is LangChain?"}
        ],
        "model": "gemini-pro",
        "temperature": 0.7
    }
)
print(response.json())
```

### 2. RAG Query

```python
# First, upload a document
files = {"file": open("document.pdf", "rb")}
upload_response = httpx.post(
    "http://localhost:8000/api/v1/documents/upload",
    files=files
)
document_id = upload_response.json()["document_id"]

# Query the document
query_response = httpx.post(
    "http://localhost:8000/api/v1/rag/query",
    json={
        "query": "What are the key points?",
        "document_ids": [document_id],
        "top_k": 5
    }
)
print(query_response.json())
```

### 3. Web Crawling

```python
response = httpx.post(
    "http://localhost:8000/api/v1/crawl",
    json={
        "url": "https://example.com",
        "max_depth": 2,
        "extract_content": True
    }
)
print(response.json())
```

## 🧪 Testing

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

## 📊 Monitoring

### LangSmith Integration

1. Set up LangSmith credentials in `.env`
2. Access traces at https://smith.langchain.com
3. Monitor:
   - Request traces
   - Token usage
   - Latency metrics
   - Error rates

### Application Metrics

- **Health Check**: http://localhost:8000/health
- **Prometheus Metrics**: http://localhost:8000/metrics (if enabled)

### Service UIs

- **MongoDB Express**: http://localhost:8081 (admin/changeme)
- **Redis Commander**: http://localhost:8082

## 🚀 Deployment

### Production Docker Build

```bash
# Build production image
docker build -f docker/production/Dockerfile -t langchain-fastapi:prod .

# Run with production config
docker run -p 8000:8000 --env-file .env.prod langchain-fastapi:prod
```

### Kubernetes Deployment

```yaml
# See kubernetes/ directory for manifests
kubectl apply -f kubernetes/
```

### Cloud Deployment

The application is ready for deployment on:
- AWS ECS/EKS
- Google Cloud Run/GKE
- Azure Container Instances/AKS
- Heroku
- Railway

## 🔒 Security

- JWT-based authentication
- Rate limiting
- Input validation with Pydantic
- SQL injection prevention
- XSS protection
- CORS configuration
- Secrets management via environment variables

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- LangChain team for the amazing framework
- Google for Gemini models
- Pinecone for vector database
- FastAPI for the web framework

## 📮 Contact

For questions and support, please open an issue on GitHub.

---

**Note**: This is a template project. Remember to:
1. Add your API keys to `.env`
2. Configure security settings for production
3. Set up proper monitoring and alerting
4. Review and adjust rate limits
5. Configure CORS for your domains