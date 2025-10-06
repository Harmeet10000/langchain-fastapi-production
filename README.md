# LangChain FastAPI Production Template

A production-grade FastAPI application integrating LangChain, LangGraph, and LangSmith with Google's Gemini models, featuring Pinecone for vector storage, Docling for document processing, Crawl4AI for web scraping, and **MCP (Model Context Protocol)** for dynamic tool integration.

## 🚀 Features

### Core Framework

-   **LangChain Integration**: Complete integration with Google Gemini models for LLM operations
-   **LangGraph Workflows**: Graph-based reasoning and workflow management
-   **LangSmith Monitoring**: Comprehensive tracing, evaluation, and feedback loops

### Advanced Capabilities

-   **MCP Protocol**: Dynamic tool discovery and multi-server communication
-   **Vector Store**: Pinecone integration for efficient semantic search and RAG
-   **Document Processing**: Multi-format document parsing with Docling (PDF, DOCX, PPTX, HTML, Markdown)
-   **Web Crawling**: Intelligent web scraping with Crawl4AI (JavaScript rendering, rate limiting)
-   **Structured Outputs**: Type-safe LLM responses with Pydantic models
-   **Agent Workflows**: ReAct, Plan-and-Execute, and custom agent patterns
-   **Memory Management**: Persistent conversation history and checkpointing

### Production Features

-   **Production Ready**: Docker, monitoring, caching, and security best practices
-   **Async First**: Fully asynchronous architecture for high performance
-   **Type Safe**: Complete type hints and Pydantic validation
-   **Multi-Server Support**: Connect to multiple MCP servers simultaneously
-   **Caching**: Redis-based caching for improved performance
-   **Rate Limiting**: Built-in rate limiting and throttling
-   **Error Handling**: Comprehensive error handling and logging
-   **Observability**: LangSmith integration for tracing and monitoring

## 📋 Prerequisites

-   Python 3.9+
-   [uv](https://docs.astral.sh/uv/) - Fast Python package manager (recommended)
-   Docker and Docker Compose
-   API Keys:
    -   Google Gemini API Key
    -   Pinecone API Key and Environment
    -   LangSmith API Key (optional)

## 🛠️ Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd langchain-fastapi-production
```

### 2. Install uv (if not already installed)

```bash
# On macOS and Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv
```

### 3. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 4. Using Docker (Recommended for Production)

```bash
# Build and start all services
docker-compose up --build

# Or run in detached mode
docker-compose up -d

# View logs
docker-compose logs -f app
```

### 5. Local Development with uv (Recommended)

```bash
# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project dependencies (reads pyproject.toml)
uv pip install -e .

# Install Playwright browsers (for Crawl4AI)
playwright install chromium

# Run the application
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 5000
```

### 6. Alternative: Traditional pip approach

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install MCP adapters
pip install langchain-mcp-adapters

# Install Playwright browsers (for Crawl4AI)
playwright install chromium

# Run the application
uvicorn src.main:app --reload --host 0.0.0.0 --port 5000
```

## ⚡ Why Use uv?

`uv` is a fast Python package manager that offers significant advantages:

-   **10-100x faster** than pip for dependency resolution and installation
-   **Better dependency resolution** with fewer conflicts
-   **Built-in virtual environment management**
-   **Compatible with pip** and existing workflows
-   **Deterministic builds** with better lock file support
-   **Parallel downloads** and installations



## � Deveelopment Workflow

### Docker + uv Hybrid Development

For the best development experience, you can combine Docker for services with local uv for the application:

```bash
# Start only the services (database, redis, etc.)
docker-compose up -d mongodb redis

# Run the application locally with uv
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 5000

# Or use the development script
chmod +x scripts/dev.sh
./scripts/dev.sh
```

### Development Scripts

```bash
# scripts/dev.sh
#!/bin/bash
set -e

echo "🚀 Starting development environment..."

# Start services
docker-compose up -d mongodb redis

# Wait for services to be ready
echo "⏳ Waiting for services..."
sleep 5

# Run the application with uv
echo "🏃 Starting FastAPI application..."
uv run uvicorn src.main:app --reload --host 0.0.0.0 --port 5000
```

### Hot Reloading with Docker

If you prefer full Docker development:

```bash
# Development with hot reloading
docker-compose -f docker-compose.dev.yml up --build

# View logs
docker-compose -f docker-compose.dev.yml logs -f app
```

## 📁 Project Structure

```
langchain-fastapi-production/
├── src/
│   ├── api/                 # API layer
│   │   ├── endpoints/        # API endpoints
│   │   │   ├── chat.py       # Chat endpoints
│   │   │   ├── rag.py        # RAG endpoints
│   │   │   ├── documents.py  # Document processing
│   │   │   ├── crawl.py      # Web crawling
│   │   │   ├── workflows.py  # LangGraph workflows
│   │   │   └── mcp_agents.py # MCP agent endpoints (NEW)
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
│   ├── mcp/                 # MCP Integration (NEW)
│   │   ├── client.py        # MCP client manager
│   │   ├── servers/         # MCP server implementations
│   │   │   ├── math_server.py      # Math operations
│   │   │   ├── weather_server.py   # Weather data
│   │   │   ├── database_server.py  # Database queries
│   │   │   └── filesystem_server.py # File operations
│   │   ├── adapters/        # Tool adapters
│   │   └── config/          # Server configurations
│   ├── agents/              # LangChain agents (NEW)
│   │   ├── mcp/             # MCP-enabled agents
│   │   │   └── mcp_agent.py # Main MCP agent
│   │   ├── rag/             # RAG agents
│   │   └── conversational/  # Chat agents
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
│   ├── test_mcp_integration.py  # MCP tests (NEW)
│   └── ...
├── docker/                  # Docker configurations
│   ├── mcp/                # MCP server Dockerfiles (NEW)
│   └── ...
├── docs/                    # Documentation
├── examples/                # Usage examples (NEW)
│   └── mcp_agent_example.py # MCP agent examples
├── scripts/                 # Utility scripts
└── data/                    # Data storage
```

## 🔧 Configuration

## 🎯 Core Features Detail

### 1. LangChain Integration

-   **Chat Models**: Google Gemini Pro, Flash, and custom models
-   **Chains**: RAG, Conversation, Summarization, Q&A
-   **Tools**: Web search, calculations, database queries, file operations
-   **Memory**: Conversation buffers, summaries, and entity tracking
-   **Callbacks**: Token counting, latency tracking, custom handlers

### 2. LangGraph Workflows

-   **State Management**: TypedDict-based state with checkpointing
-   **Conditional Routing**: Dynamic workflow paths based on state
-   **Human-in-the-Loop**: Approval gates and manual interventions
-   **Multi-Agent**: Orchestrate multiple specialized agents
-   **Streaming**: Real-time updates for long-running workflows

### 3. Vector Store & RAG

-   **Pinecone Integration**: Production-grade vector storage
-   **Embeddings**: Google Vertex AI, OpenAI, and custom embeddings
-   **Chunking Strategies**: Recursive, semantic, and custom splitters
-   **Retrieval**: Similarity search, MMR, and hybrid search
-   **Re-ranking**: Cross-encoder and LLM-based re-ranking

### 4. Document Processing

-   **Supported Formats**: PDF, DOCX, PPTX, XLSX, HTML, Markdown, TXT
-   **OCR Support**: Extract text from scanned documents
-   **Metadata Extraction**: Automatic metadata detection
-   **Batch Processing**: Parallel document processing
-   **Storage**: MongoDB-based document store

### 5. Web Crawling

-   **JavaScript Rendering**: Playwright-based crawling
-   **Smart Extraction**: Automatic content detection
-   **Rate Limiting**: Respectful crawling with delays
-   **Link Following**: Recursive crawling with depth control
-   **Content Cleaning**: Remove ads, navigation, and boilerplate

### 6. MCP (Model Context Protocol)

-   **Multi-Server**: Connect to unlimited MCP servers
-   **Transport Types**: stdio (local) and HTTP (remote)
-   **Built-in Servers**: Math, Weather, Database, Filesystem
-   **Custom Servers**: Easy extension with custom tools
-   **Auto-Discovery**: Automatic tool detection and registration

## 📚 API Documentation

Once the application is running, you can access:

-   **Swagger UI**: http://localhost:5000/api/v1/docs
-   **ReDoc**: http://localhost:5000/api/v1/redoc
-   **OpenAPI JSON**: http://localhost:5000/api/v1/openapi.json

### Available Endpoints

1. **Chat** - `/api/v1/chat` - Conversational AI with Gemini
2. **RAG Query** - `/api/v1/rag/query` - Semantic search and retrieval
3. **MCP Agents** - `/api/v1/mcp-agents/execute` - Multi-tool agent execution
4. **Document Upload** - `/api/v1/documents/upload` - Multi-format document processing
5. **Web Crawling** - `/api/v1/crawl` - Intelligent web scraping
6. **Workflows** - `/api/v1/workflows/execute` - LangGraph workflow execution

## 🤖 MCP (Model Context Protocol) Integration

### What is MCP?

MCP enables dynamic tool discovery and communication with multiple tool servers, allowing agents to access a wide range of capabilities:

-   **Math Operations**: Calculations, equations, and mathematical functions
-   **Weather Data**: Real-time weather information from OpenWeatherMap
-   **Database Queries**: MongoDB operations and data retrieval
-   **File System**: File operations and management
-   **Custom Tools**: Easily add your own MCP servers

### MCP Architecture

```
┌─────────────────┐
│   LangChain     │
│     Agent       │
└────────┬────────┘
         │
    ┌────▼────┐
    │   MCP   │
    │  Client │
    └────┬────┘
         │
    ┏━━━━┻━━━━━━━━━━━━━━━━━━━━━━━┓
    ┃                              ┃
┌───▼────┐  ┌───────┐  ┌────────┐ ┃
│  Math  │  │Weather│  │Database│ ┃
│ Server │  │Server │  │ Server │ ┃
│(stdio) │  │ (HTTP)│  │(stdio) │ ┃
└────────┘  └───────┘  └────────┘ ┃
                                   ┃
          MCP Servers              ┃
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

### Quick Example

```python
from src.agents.mcp.mcp_agent import MCPAgent

# Initialize and use MCP agent
agent = MCPAgent(model_name="gemini-pro")
await agent.initialize()
result = await agent.run("Calculate 25 * 4 and check weather in NYC")
await agent.cleanup()
```

### MCP Benefits

✅ **Dynamic Tool Discovery** - Automatically discover and use tools from multiple servers  
✅ **Mixed Transports** - Support both local (stdio) and remote (HTTP) servers  
✅ **Scalability** - Easily add new tool servers without modifying agent code  
✅ **Isolation** - Each server runs independently with its own dependencies  
✅ **Reusability** - Share MCP servers across multiple agents and applications

**📘 For detailed examples, server implementations, and integration patterns, see: [MCP_INTEGRATION_GUIDE.md](./MCP_INTEGRATION_GUIDE.md)**

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

-   **Health Check**: http://localhost:5000/health

### Service UIs

-   **MongoDB Express**: http://localhost:8081 (admin/changeme)
-   **Redis Commander**: http://localhost:8082
-   **MCP Weather Server**: http://localhost:8001/docs (if enabled)

## 🚀 Deployment

### Production Docker Build

```bash
# Build production image
docker build -f docker/production/Dockerfile -t langchain-fastapi:prod .

# Run with production config
docker run -p 5000:5000 --env-file .env.prod langchain-fastapi:prod
```

### Cloud Deployment

The application is ready for deployment on:

-   AWS ECS/EKS
-   Google Cloud Run/GKE
-   Azure Container Instances/AKS
-   Railway
-   Render

### MCP Servers in Production

```yaml
# docker-compose.yml additions for MCP
services:
    app:
        environment:
            - ENABLE_DB_MCP=true
            - ENABLE_WEATHER_MCP=true

    weather-mcp:
        build:
            context: .
            dockerfile: docker/mcp/weather.Dockerfile
        ports:
            - "8001:5000"
        environment:
            - OPENWEATHER_API_KEY=${OPENWEATHER_API_KEY}
        restart: unless-stopped
```

## 🧪 Testing

### Using uv (Recommended)

```bash
# Install test dependencies
uv add --dev pytest pytest-asyncio pytest-cov httpx

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src --cov-report=html

# Run specific test file
uv run pytest tests/test_mcp_integration.py

# Run MCP agent tests with verbose output
uv run pytest tests/test_mcp_integration.py -v

# Run tests in parallel
uv run pytest -n auto
```

### Using traditional pip

```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov httpx

# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_mcp_integration.py
```

### Docker Testing

```bash
# Run tests in Docker
docker-compose -f docker-compose.test.yml up --build

# Run specific test suite
docker-compose -f docker-compose.test.yml run --rm test pytest tests/test_mcp_integration.py
```

## 🔒 Security

-   Rate limiting on all endpoints
-   Input validation with Pydantic
-   CORS configuration
-   Secrets management via environment variables
-   MCP server isolation and sandboxing
-   API key rotation support

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

-   LangChain team for the amazing framework and MCP adapters
-   Google for Gemini models
-   Anthropic for the Model Context Protocol specification
-   Pinecone for vector database
-   FastAPI for the web framework
-   The open-source community

## 📮 Contact

For questions and support, please open an issue on GitHub.

---

## 🚀 Quick Start

```bash
# Clone and setup with uv (fastest way)
git clone <repository-url>
cd langchain-fastapi-production
chmod +x scripts/dev.sh
./scripts/dev.sh

# Or on Windows
scripts\dev.bat
```

**Note**: This is a template project. Remember to:

1. **Install uv** for faster dependency management: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Add your API keys to `.env`
3. Install `langchain-mcp-adapters` for MCP support: `uv add langchain-mcp-adapters`
4. Configure MCP servers in `src/mcp/config/server_config.py`
5. Configure security settings for production
6. Set up proper monitoring and alerting
7. Review and adjust rate limits
8. Configure CORS for your domains
9. Test MCP servers before deploying to production
10. Use `uv lock` to generate lock files for reproducible builds
