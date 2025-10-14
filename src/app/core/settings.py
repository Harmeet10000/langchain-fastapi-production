# src/settings.py

from functools import lru_cache
from typing import List, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Pydantic will now automatically look for the exact UPPERCASE name
    # defined in the class (e.g., Settings.APP_NAME will look for the APP_NAME env var).
    model_config = SettingsConfigDict(
        env_file=".env.development",
        env_file_encoding="utf-8",
        # Case sensitivity is set to True for clarity, but False works too
        # since we are enforcing the case in the model now.
        case_sensitive=True,
        extra="ignore",  # Ignore environment variables not explicitly defined in this model
    )

    # --- Application Settings ---
    APP_NAME: str = Field(default="LangChain FastAPI Production")
    APP_VERSION: str = Field(default="1.0.0")
    ENVIRONMENT: str = Field(default="development")
    DEBUG: bool = Field(default=False)
    API_PREFIX: str = Field(default="/api/v1")
    CORS_ORIGINS: List[str] = Field(default_factory=lambda: ["*"])

    # --- Server Configuration ---
    HOST: str = Field(default="0.0.0.0")
    PORT: int = Field(default=8000)
    WORKERS: int = Field(default=1)

    # --- Database ---
    MONGODB_URL: str = Field(default="mongodb://localhost:27017/langchain_db")
    MONGODB_DATABASE: str = Field(default="langchain_db")
    POSTGRES_URL: str = Field(
        default="postgresql://user:pass@host/db"
    )  # Added this missing field
    POSTGRES_MAX_OVERFLOW: int = Field(default=10)  # Added this missing field
    POSTGRES_POOL_SIZE: int = Field(default=5)  # Added this missing field
    # --- Redis Cache ---
    REDIS_HOST: str = Field(default="localhost")
    REDIS_PORT: int = Field(default=6379)
    REDIS_USERNAME: str = Field(default="default")
    REDIS_PASSWORD: Optional[str] = Field(default=None)

    # Note: REDIS_DB and CACHE_TTL were not in your ENV, so they remain as defaults
    REDIS_DB: int = Field(default=0)
    CACHE_TTL: int = Field(default=3600)

    # --- Google Gemini API ---
    GOOGLE_API_KEY: str = Field(default="")
    GEMINI_MODEL: str = Field(default="gemini-2.5-flash")
    GEMINI_VISION_MODEL: str = Field(default="gemini-2.5-flash-image")
    GEMINI_EMBEDDING_MODEL: str = Field(default="text-embedding-005")
    GEMINI_TEMPERATURE: float = Field(default=0.7)
    GEMINI_MAX_TOKENS: int = Field(default=2048)

    # --- Pinecone Vector Database ---
    PINECONE_API_KEY: str = Field(default="")
    PINECONE_ENVIRONMENT: str = Field(default="")
    PINECONE_INDEX_NAME: str = Field(default="langchain-index")
    PINECONE_DIMENSION: int = Field(default=768)
    PINECONE_METRIC: str = Field(default="cosine")

    # --- LangSmith ---
    # Renamed to match the variable in your ENV file: LANGSMITH_TRACING=true
    LANGSMITH_TRACING: bool = Field(default=False)
    LANGSMITH_ENDPOINT: str = Field(default="https://api.smith.langchain.com")
    LANGSMITH_API_KEY: str = Field(default="")  # Note: Your ENV had LANGSMITH_API_KEY
    LANGSMITH_PROJECT: str = Field(default="langchain-production")
    LANGCHAIN_PROJECT: str = Field(
        default="langchain-production"
    )  # Note: Your ENV had LANGCHAIN_PROJECT

    # --- Crawl4AI Configuration ---
    CRAWL4AI_HEADLESS: bool = Field(default=True)
    CRAWL4AI_TIMEOUT: int = Field(default=30000)
    CRAWL4AI_USER_AGENT: str = Field(
        default="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    )

    # --- Logging ---
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="json")
    LOG_FILE: str = Field(default="logs/app.log")

    # --- Rate Limiting ---
    RATE_LIMIT_ENABLED: bool = Field(default=True)
    RATE_LIMIT_REQUESTS: int = Field(default=100)
    RATE_LIMIT_PERIOD: int = Field(default=60)

    # --- File Upload ---
    MAX_UPLOAD_SIZE: int = Field(default=10485760)  # 10MB
    ALLOWED_EXTENSIONS: List[str] = Field(
        default_factory=lambda: ["pdf", "txt", "docx", "xlsx", "pptx", "md", "html"]
    )

    # --- OpenTelemetry ---
    OTEL_EXPORTER_OTLP_ENDPOINT: str = Field(default="http://localhost:4317")
    OTEL_SERVICE_NAME: str = Field(default="langchain-fastapi")
    OTEL_TRACES_EXPORTER: str = Field(default="otlp")
    OTEL_METRICS_EXPORTER: str = Field(default="otlp")


@lru_cache()
def get_settings() -> Settings:
    """Returns a cached instance of the application settings."""
    # Instantiating the class here ensures it's only done once (due to @lru_cache)
    return Settings()
