# Production-Grade Python/FastAPI/LangChain Template - Coding Standards & Examples

## Project Overview

This is a production-grade AI-powered service built with Python, FastAPI, LangChain, and modern async patterns following a **features-based layered architecture** with enterprise-level patterns focusing on security, scalability, and maintainability.
- Don't leave comments in code, unless they explain something complex and not trivial

## Architecture Principles

- **Features-Based Organization**: Each feature is self-contained with complete layer stack
- **Async-First**: Use async/await patterns throughout for better performance
- **Layered Architecture**: Router → Service → Repository → Model
- **Separation of Concerns**: Clear boundaries between layers
- **DRY Principle**: Don't repeat yourself, create reusable utilities
- **Type Safety**: Use Pydantic models and type hints everywhere

## Code Style Examples

### Repository Layer

```python
# ✅ Good - Async repository with proper error handling
from typing import Optional, List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, delete
from src.core.database import get_session
from src.models.user import User
from src.schemas.user import UserCreate, UserUpdate

class UserRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def find_by_email(self, email: str) -> Optional[User]:
        """Find user by email address."""
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def create(self, user_data: UserCreate) -> User:
        """Create new user."""
        user = User(**user_data.model_dump())
        self.session.add(user)
        await self.session.commit()
        await self.session.refresh(user)
        return user

    async def update(self, user_id: int, user_data: UserUpdate) -> Optional[User]:
        """Update existing user."""
        await self.session.execute(
            update(User)
            .where(User.id == user_id)
            .values(**user_data.model_dump(exclude_unset=True))
        )
        await self.session.commit()
        return await self.find_by_id(user_id)

    async def delete(self, user_id: int) -> bool:
        """Delete user by ID."""
        result = await self.session.execute(
            delete(User).where(User.id == user_id)
        )
        await self.session.commit()
        return result.rowcount > 0
```

### Service Layer

```python
# ✅ Good - Business logic with proper error handling and LangChain integration
from typing import Optional, List
from fastapi import HTTPException, status
from langchain.schema import BaseMessage
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from src.repositories.user import UserRepository
from src.repositories.conversation import ConversationRepository
from src.schemas.user import UserCreate, UserResponse
from src.schemas.conversation import ConversationCreate, ConversationResponse
from src.core.logging import logger
from src.core.exceptions import UserNotFoundError, ValidationError

class AuthService:
    def __init__(
        self,
        user_repo: UserRepository,
        conversation_repo: ConversationRepository
    ):
        self.user_repo = user_repo
        self.conversation_repo = conversation_repo
        self.llm = ChatOpenAI(temperature=0.7)
        self.memory = ConversationBufferMemory(return_messages=True)

    async def create_user(self, user_data: UserCreate) -> UserResponse:
        """Create new user with validation."""
        # Check if user already exists
        existing_user = await self.user_repo.find_by_email(user_data.email)
        if existing_user:
            raise ValidationError("User with this email already exists")

        # Create user
        user = await self.user_repo.create(user_data)
        logger.info(f"User created successfully", extra={
            "user_id": user.id,
            "email": user.email
        })

        return UserResponse.model_validate(user)

    async def process_chat_message(
        self,
        user_id: int,
        message: str,
        conversation_id: Optional[int] = None
    ) -> ConversationResponse:
        """Process chat message using LangChain."""
        # Verify user exists
        user = await self.user_repo.find_by_id(user_id)
        if not user:
            raise UserNotFoundError(f"User {user_id} not found")

        try:
            # Load conversation history if provided
            if conversation_id:
                history = await self.conversation_repo.get_history(conversation_id)
                for msg in history:
                    self.memory.chat_memory.add_message(msg)

            # Process with LangChain
            response = await self.llm.apredict(message)

            # Save conversation
            conversation_data = ConversationCreate(
                user_id=user_id,
                message=message,
                response=response,
                conversation_id=conversation_id
            )
            conversation = await self.conversation_repo.create(conversation_data)

            logger.info("Chat message processed", extra={
                "user_id": user_id,
                "conversation_id": conversation.id,
                "message_length": len(message)
            })

            return ConversationResponse.model_validate(conversation)

        except Exception as e:
            logger.error(f"Error processing chat message", extra={
                "user_id": user_id,
                "error": str(e)
            })
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to process message"
            )
```

### Router Layer (FastAPI)

```python
# ✅ Good - Clean FastAPI router with proper dependency injection
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Query
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from src.core.database import get_session
from src.core.auth import get_current_user
from src.repositories.user import UserRepository
from src.services.auth import AuthService
from src.schemas.user import UserCreate, UserResponse, UserUpdate
from src.schemas.conversation import ConversationResponse
from src.core.logging import logger

router = APIRouter(prefix="/api/v1/users", tags=["users"])
security = HTTPBearer()

async def get_user_service(session: AsyncSession = Depends(get_session)) -> AuthService:
    """Dependency to get user service."""
    user_repo = UserRepository(session)
    conversation_repo = ConversationRepository(session)
    return AuthService(user_repo, conversation_repo)

@router.post(
    "/",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create new user",
    description="Create a new user account with email and password"
)
async def create_user(
    user_data: UserCreate,
    service: AuthService = Depends(get_user_service)
) -> UserResponse:
    """Create new user endpoint."""
    try:
        return await service.create_user(user_data)
    except ValidationError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post(
    "/{user_id}/chat",
    response_model=ConversationResponse,
    summary="Process chat message",
    description="Process a chat message using LangChain AI"
)
async def chat_message(
    user_id: int,
    message: str,
    conversation_id: Optional[int] = Query(None),
    current_user = Depends(get_current_user),
    service: AuthService = Depends(get_user_service)
) -> ConversationResponse:
    """Process chat message endpoint."""
    # Verify user can access this resource
    if current_user.id != user_id and not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Access denied"
        )

    return await service.process_chat_message(user_id, message, conversation_id)
```

### Standardized HTTP Response Utility

```python
# ✅ Good - Standardized HTTP response utility
from typing import Any, Optional, Dict
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from src.core.config import settings
from src.core.logging import logger
import uuid

class StandardResponse:
    """Standardized HTTP response utility for consistent API responses."""

    @staticmethod
    def success(
        request: Request,
        status_code: int,
        message: str,
        data: Any = None,
        correlation_id: Optional[str] = None
    ) -> JSONResponse:
        """Create standardized success response."""

        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))

        response_data = {
            "success": True,
            "status_code": status_code,
            "request": {
                "ip": request.client.host if request.client else None,
                "method": request.method,
                "url": str(request.url),
                "correlation_id": correlation_id
            },
            "message": message,
            "data": data
        }

        # Remove sensitive info in production
        if settings.ENVIRONMENT == "production":
            del response_data["request"]["ip"]
            # Optionally remove correlation_id in production
            # del response_data["request"]["correlation_id"]

        # Log the response
        logger.info("CONTROLLER_RESPONSE", extra={"response": response_data})

        return JSONResponse(
            status_code=status_code,
            content=response_data
        )

    @staticmethod
    def error(
        request: Request,
        status_code: int,
        message: str,
        error_details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None
    ) -> JSONResponse:
        """Create standardized error response."""

        # Generate correlation ID if not provided
        if not correlation_id:
            correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))

        response_data = {
            "success": False,
            "status_code": status_code,
            "request": {
                "ip": request.client.host if request.client else None,
                "method": request.method,
                "url": str(request.url),
                "correlation_id": correlation_id
            },
            "message": message,
            "error": error_details
        }

        # Remove sensitive info in production
        if settings.ENVIRONMENT == "production":
            del response_data["request"]["ip"]
            # Remove detailed error info in production
            if error_details and not settings.DEBUG:
                response_data["error"] = "Internal server error"

        # Log the error response
        logger.error("CONTROLLER_ERROR_RESPONSE", extra={"response": response_data})

        return JSONResponse(
            status_code=status_code,
            content=response_data
        )

# Convenience functions for common use cases
def http_success(
    request: Request,
    status_code: int = 200,
    message: str = "Success",
    data: Any = None
) -> JSONResponse:
    """Convenience function for success responses."""
    return StandardResponse.success(request, status_code, message, data)

def http_created(
    request: Request,
    message: str = "Resource created successfully",
    data: Any = None
) -> JSONResponse:
    """Convenience function for 201 Created responses."""
    return StandardResponse.success(request, 201, message, data)

def http_error(
    request: Request,
    status_code: int = 500,
    message: str = "Internal server error",
    error_details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Convenience function for error responses."""
    return StandardResponse.error(request, status_code, message, error_details)

def http_not_found(
    request: Request,
    message: str = "Resource not found"
) -> JSONResponse:
    """Convenience function for 404 Not Found responses."""
    return StandardResponse.error(request, 404, message)

def http_bad_request(
    request: Request,
    message: str = "Bad request",
    error_details: Optional[Dict[str, Any]] = None
) -> JSONResponse:
    """Convenience function for 400 Bad Request responses."""
    return StandardResponse.error(request, 400, message, error_details)

def http_unauthorized(
    request: Request,
    message: str = "Unauthorized"
) -> JSONResponse:
    """Convenience function for 401 Unauthorized responses."""
    return StandardResponse.error(request, 401, message)

def http_forbidden(
    request: Request,
    message: str = "Forbidden"
) -> JSONResponse:
    """Convenience function for 403 Forbidden responses."""
    return StandardResponse.error(request, 403, message)
```

### Correlation ID Middleware

```python
# ✅ Good - Middleware to add correlation IDs to requests
import uuid
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

class CorrelationIDMiddleware(BaseHTTPMiddleware):
    """Middleware to add correlation ID to each request."""

    async def dispatch(self, request: Request, call_next):
        # Get correlation ID from header or generate new one
        correlation_id = request.headers.get('X-Correlation-ID', str(uuid.uuid4()))

        # Store in request state
        request.state.correlation_id = correlation_id

        # Process request
        response = await call_next(request)

        # Add correlation ID to response headers
        response.headers['X-Correlation-ID'] = correlation_id

        return response
```

### Updated Router Examples with Standardized Responses

```python
# ✅ Good - Router using standardized responses
from fastapi import APIRouter, Depends, Request
from src.core.responses import http_success, http_created, http_error, http_not_found
from src.schemas.user import UserCreate, UserResponse

router = APIRouter(prefix="/api/v1/users", tags=["users"])

@router.post("/", status_code=201)
async def create_user(
    request: Request,
    user_data: UserCreate,
    service: AuthService = Depends(get_user_service)
):
    """Create new user endpoint with standardized response."""
    try:
        user = await service.create_user(user_data)
        return http_created(
            request,
            message="User created successfully",
            data=user
        )
    except ValidationError as e:
        return http_bad_request(
            request,
            message="Validation failed",
            error_details={"validation_errors": str(e)}
        )
    except Exception as e:
        return http_error(
            request,
            message="Failed to create user",
            error_details={"error": str(e)} if settings.DEBUG else None
        )

@router.get("/{user_id}")
async def get_user(
    request: Request,
    user_id: int,
    service: AuthService = Depends(get_user_service)
):
    """Get user by ID with standardized response."""
    try:
        user = await service.get_user(user_id)
        if not user:
            return http_not_found(request, "User not found")

        return http_success(
            request,
            message="User retrieved successfully",
            data=user
        )
    except Exception as e:
        return http_error(
            request,
            message="Failed to retrieve user",
            error_details={"error": str(e)} if settings.DEBUG else None
        )

@router.post("/{user_id}/chat")
async def chat_message(
    request: Request,
    user_id: int,
    message: str,
    service: AuthService = Depends(get_user_service)
):
    """Process chat message with standardized response."""
    try:
        conversation = await service.process_chat_message(user_id, message)
        return http_success(
            request,
            message="Message processed successfully",
            data=conversation
        )
    except UserNotFoundError:
        return http_not_found(request, "User not found")
    except Exception as e:
        return http_error(
            request,
            message="Failed to process message",
            error_details={"error": str(e)} if settings.DEBUG else None
        )
```

### Pydantic Models/Schemas

```python
# ✅ Good - Comprehensive Pydantic models with validation
from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, validator
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    MODERATOR = "moderator"

class UserBase(BaseModel):
    email: EmailStr
    name: str = Field(..., min_length=1, max_length=100)
    role: UserRole = UserRole.USER
    is_active: bool = True

class UserCreate(UserBase):
    password: str = Field(..., min_length=8, max_length=128)

    @validator('password')
    def validate_password(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one digit')
        return v

class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    role: Optional[UserRole] = None
    is_active: Optional[bool] = None

class UserResponse(UserBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class ConversationCreate(BaseModel):
    user_id: int
    message: str = Field(..., min_length=1, max_length=4000)
    response: str
    conversation_id: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

class ConversationResponse(BaseModel):
    id: int
    user_id: int
    message: str
    response: str
    conversation_id: Optional[int]
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True
```

### Database Models (SQLAlchemy)

```python
# ✅ Good - SQLAlchemy models with proper relationships and indexes
from datetime import datetime
from typing import Optional
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Text, ForeignKey, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

Base = declarative_base()

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String(255), unique=True, index=True, nullable=False)
    name = Column(String(100), nullable=False)
    password_hash = Column(String(255), nullable=False)
    role = Column(String(20), default="user", nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    conversations = relationship("Conversation", back_populates="user")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}')>"

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    message = Column(Text, nullable=False)
    response = Column(Text, nullable=False)
    conversation_id = Column(Integer, ForeignKey("conversations.id"), nullable=True)
    metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    user = relationship("User", back_populates="conversations")
    parent = relationship("Conversation", remote_side=[id])

    def __repr__(self):
        return f"<Conversation(id={self.id}, user_id={self.user_id})>"
```

### LangChain Integration Patterns

```python
# ✅ Good - LangChain service with proper chain management
from typing import List, Dict, Any, Optional
from langchain.chat_models import ChatOpenAI
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationChain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.callbacks import AsyncCallbackHandler
from src.core.config import settings
from src.core.logging import logger

class ChatService:
    def __init__(self):
        self.llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL,
            temperature=settings.OPENAI_TEMPERATURE,
            openai_api_key=settings.OPENAI_API_KEY
        )
        self.memory = ConversationBufferWindowMemory(
            k=settings.CONVERSATION_MEMORY_SIZE,
            return_messages=True
        )
        self._setup_chain()

    def _setup_chain(self):
        """Setup the conversation chain with custom prompt."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful AI assistant. Be concise and accurate."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        self.chain = ConversationChain(
            llm=self.llm,
            memory=self.memory,
            prompt=prompt,
            verbose=settings.DEBUG
        )

    async def process_message(
        self,
        message: str,
        user_id: int,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a chat message and return AI response."""
        try:
            # Add context to the message if provided
            if context:
                enhanced_message = f"Context: {context}\n\nUser message: {message}"
            else:
                enhanced_message = message

            # Process with LangChain
            response = await self.chain.apredict(input=enhanced_message)

            logger.info("Message processed successfully", extra={
                "user_id": user_id,
                "message_length": len(message),
                "response_length": len(response)
            })

            return response

        except Exception as e:
            logger.error(f"Error processing message: {str(e)}", extra={
                "user_id": user_id,
                "error": str(e)
            })
            raise

class CustomCallbackHandler(AsyncCallbackHandler):
    """Custom callback handler for LangChain operations."""

    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        logger.debug("LLM started", extra={"prompts_count": len(prompts)})

    async def on_llm_end(self, response, **kwargs):
        logger.debug("LLM completed", extra={"response_length": len(str(response))})

    async def on_llm_error(self, error: Exception, **kwargs):
        logger.error(f"LLM error: {str(error)}")
```

### Logging Standards

```python
# ✅ Good - Structured logging with context
import logging
import json
from datetime import datetime
from typing import Dict, Any, Optional

class StructuredLogger:
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _log_with_context(
        self,
        level: int,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exc_info: bool = False
    ):
        """Log with structured context."""
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "message": message,
            "level": logging.getLevelName(level)
        }

        if extra:
            log_data.update(extra)

        self.logger.log(level, json.dumps(log_data), exc_info=exc_info)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self._log_with_context(logging.INFO, message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self._log_with_context(logging.ERROR, message, extra, exc_info=True)

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        self._log_with_context(logging.DEBUG, message, extra)

# Usage examples
logger = StructuredLogger(__name__)

# ✅ Good logging examples
logger.info("User authentication successful", extra={
    "user_id": user.id,
    "email": user.email,
    "login_method": "password"
})

logger.error("Database connection failed", extra={
    "database_url": settings.DATABASE_URL,
    "retry_count": 3,
    "error_code": "DB_CONNECTION_TIMEOUT"
})

logger.debug("LangChain chain execution", extra={
    "chain_type": "conversation",
    "input_tokens": 150,
    "output_tokens": 200,
    "execution_time_ms": 1250
})
```

## Coding Standards

### General Rules

- Use Python 3.11+ features and type hints everywhere
- Follow PEP 8 style guide with Black formatter
- Use async/await for all I/O operations
- Prefer composition over inheritance
- Use dataclasses or Pydantic models for data structures
- Add docstrings for all public functions and classes

### Naming Conventions

- **Files**: `snake_case.py`
- **Functions/Variables**: `snake_case`
- **Classes**: `PascalCase`
- **Constants**: `UPPER_SNAKE_CASE`
- **Environment Variables**: `UPPER_SNAKE_CASE`

### Error Handling

```python
# ✅ Good - Single error handling utility
from typing import Optional, Dict, Any, Union, Callable
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from src.core.config import settings
from src.core.logging import logger
from src.core.constants import SOMETHING_WENT_WRONG
import traceback
import uuid

def http_error(
    request: Request,
    error: Union[Exception, str],
    error_status_code: int = 500,
    next_handler: Optional[Callable] = None
) -> Union[JSONResponse, HTTPException]:
    """
    Centralized error handling utility.

    Args:
        request: FastAPI Request object
        error: Exception or error message
        error_status_code: HTTP status code for the error
        next_handler: Optional error handler function

    Returns:
        JSONResponse or raises HTTPException
    """
    error_obj = create_error_object(error, request, error_status_code)

    if next_handler and callable(next_handler):
        return next_handler(error_obj)

    # Return JSONResponse for API endpoints
    return JSONResponse(
        status_code=error_status_code,
        content=error_obj
    )

def create_error_object(
    error: Union[Exception, str],
    request: Request,
    error_status_code: int = 500
) -> Dict[str, Any]:
    """
    Create standardized error object.

    Args:
        error: Exception or error message
        request: FastAPI Request object
        error_status_code: HTTP status code

    Returns:
        Standardized error dictionary
    """
    # Get correlation ID from request state
    correlation_id = getattr(request.state, 'correlation_id', str(uuid.uuid4()))

    # Determine error name and message
    if isinstance(error, Exception):
        error_name = error.__class__.__name__
        error_message = str(error) or SOMETHING_WENT_WRONG
        error_trace = traceback.format_exc() if settings.DEBUG else None
    else:
        error_name = "Error"
        error_message = str(error) or SOMETHING_WENT_WRONG
        error_trace = None

    error_obj = {
        "name": error_name,
        "success": False,
        "status_code": error_status_code,
        "request": {
            "ip": request.client.host if request.client else None,
            "method": request.method,
            "url": str(request.url),
            "correlation_id": correlation_id
        },
        "message": error_message,
        "data": None,
        "trace": {"error": error_trace} if error_trace else None
    }

    # Log the error
    logger.error("CONTROLLER_ERROR", extra={"error": error_obj})

    # Remove sensitive information in production
    if settings.ENVIRONMENT == "production":
        del error_obj["request"]["ip"]
        if error_obj["trace"]:
            del error_obj["trace"]

    return error_obj

# Convenience functions for common error scenarios
def validation_error(request: Request, message: str = "Validation failed") -> JSONResponse:
    """Create validation error response."""
    return http_error(request, message, 400)

def not_found_error(request: Request, message: str = "Resource not found") -> JSONResponse:
    """Create not found error response."""
    return http_error(request, message, 404)

def unauthorized_error(request: Request, message: str = "Unauthorized") -> JSONResponse:
    """Create unauthorized error response."""
    return http_error(request, message, 401)

def forbidden_error(request: Request, message: str = "Forbidden") -> JSONResponse:
    """Create forbidden error response."""
    return http_error(request, message, 403)

def internal_server_error(request: Request, error: Exception) -> JSONResponse:
    """Create internal server error response."""
    return http_error(request, error, 500)

# Custom exception for application-specific errors
class AppError(Exception):
    """Base application error with status code."""

    def __init__(self, message: str, status_code: int = 500):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)

# Usage in services - raise AppError with specific status codes
async def get_user(user_id: int) -> User:
    user = await user_repo.find_by_id(user_id)
    if not user:
        raise AppError(f"User {user_id} not found", 404)
    return user

async def create_user(user_data: UserCreate) -> User:
    existing_user = await user_repo.find_by_email(user_data.email)
    if existing_user:
        raise AppError("User with this email already exists", 400)
    return await user_repo.create(user_data)

# Usage in routers - centralized error handling
@router.get("/{user_id}")
async def get_user_endpoint(request: Request, user_id: int):
    try:
        user = await service.get_user(user_id)
        return http_success(request, message="User retrieved successfully", data=user)
    except AppError as e:
        return http_error(request, e, e.status_code)
    except Exception as e:
        return http_error(request, e, 500)

@router.post("/")
async def create_user_endpoint(request: Request, user_data: UserCreate):
    try:
        user = await service.create_user(user_data)
        return http_created(request, message="User created successfully", data=user)
    except AppError as e:
        return http_error(request, e, e.status_code)
    except Exception as e:
        return http_error(request, e, 500)
```

### Global Exception Handler

```python
# ✅ Good - Global exception handler for FastAPI
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

def setup_exception_handlers(app: FastAPI):
    """Setup global exception handlers for the FastAPI app."""

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError):
        """Handle custom AppError exceptions."""
        return http_error(request, exc, exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(request: Request, exc: RequestValidationError):
        """Handle Pydantic validation errors."""
        error_details = []
        for error in exc.errors():
            error_details.append({
                "field": " -> ".join(str(x) for x in error["loc"]),
                "message": error["msg"],
                "type": error["type"]
            })

        return http_error(
            request,
            f"Validation failed: {'; '.join([e['message'] for e in error_details])}",
            400
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(request: Request, exc: StarletteHTTPException):
        """Handle HTTP exceptions."""
        return http_error(request, exc.detail, exc.status_code)

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle all other exceptions."""
        return http_error(request, exc, 500)

# Usage in main.py
from fastapi import FastAPI
from src.core.exceptions import setup_exception_handlers

app = FastAPI()
setup_exception_handlers(app)
```

### Dependency Injection Pattern

```python
# ✅ Good - FastAPI dependency injection
from fastapi import Depends
from typing import Annotated

# Database dependency
async def get_session() -> AsyncSession:
    async with async_session() as session:
        try:
            yield session
        finally:
            await session.close()

# Repository dependencies
def get_user_repository(
    session: Annotated[AsyncSession, Depends(get_session)]
) -> UserRepository:
    return UserRepository(session)

# Service dependencies
def get_auth_service(
    user_repo: Annotated[UserRepository, Depends(get_user_repository)]
) -> AuthService:
    return AuthService(user_repo)

# Usage in routes
@router.post("/users")
async def create_user(
    user_data: UserCreate,
    service: Annotated[AuthService, Depends(get_auth_service)]
):
    return await service.create_user(user_data)
```

## Feature Structure Template

```
src/features/[feature-name]/
├── __init__.py
├── router.py              # FastAPI route definitions
├── service.py             # Business logic and LangChain integration
├── repository.py          # Data access layer
├── schemas.py             # Pydantic models
├── models.py              # SQLAlchemy models
├── dependencies.py        # FastAPI dependencies
├── exceptions.py          # Feature-specific exceptions
└── constants.py           # Feature constants
```

### Constants File Example

```python
# ✅ Good - Application constants
# src/core/constants.py

# Error messages
SOMETHING_WENT_WRONG = "Something went wrong. Please try again."
INVALID_CREDENTIALS = "Invalid email or password"
USER_NOT_FOUND = "User not found"
USER_ALREADY_EXISTS = "User with this email already exists"
UNAUTHORIZED_ACCESS = "Unauthorized access"
FORBIDDEN_ACCESS = "Access forbidden"
VALIDATION_FAILED = "Validation failed"

# Application environments
class ApplicationEnvironment:
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

# HTTP status messages
HTTP_STATUS_MESSAGES = {
    200: "Success",
    201: "Created successfully",
    400: "Bad request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not found",
    422: "Validation error",
    500: "Internal server error"
}

# LangChain constants
DEFAULT_MODEL = "gpt-3.5-turbo"
MAX_TOKENS = 4000
DEFAULT_TEMPERATURE = 0.7
CONVERSATION_MEMORY_SIZE = 10

# Database constants
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
```

### Configuration Management

```python
# ✅ Good - Pydantic settings with environment variables
from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # Database
    DATABASE_URL: str
    DATABASE_POOL_SIZE: int = 10

    # Redis
    REDIS_URL: str
    REDIS_TTL: int = 3600

    # OpenAI/LangChain
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-3.5-turbo"
    OPENAI_TEMPERATURE: float = 0.7

    # Application
    SECRET_KEY: str
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    ENVIRONMENT: str = "development"

    # LangChain specific
    CONVERSATION_MEMORY_SIZE: int = 10
    MAX_TOKENS: int = 4000

    class Config:
        env_file = ".env"
        case_sensitive = True

settings = Settings()
```

## Development Workflow

1. Create feature directory under `src/features/`
2. Implement layered architecture within feature
3. Add comprehensive Pydantic model validation
4. Include OpenAPI documentation via FastAPI
5. Add feature routes to main app routing
6. Write unit and integration tests with pytest
7. Run `black`, `isort`, and `mypy` before commit
8. Update API documentation automatically via FastAPI

## Performance Considerations

- Use async/await throughout the application
- Implement proper database connection pooling
- Use Redis for caching and session management
- Optimize LangChain memory usage for conversations
- Monitor token usage and costs for AI operations
- Use database indexes effectively
- Implement proper pagination for large datasets

## Security Checklist

- [ ] Input validation with Pydantic models
- [ ] Authorization checks in place
- [ ] No hardcoded secrets (use environment variables)
- [ ] Proper error handling without information leakage
- [ ] Rate limiting configured
- [ ] CORS properly configured
- [ ] Security headers applied via middleware
- [ ] SQL injection prevention (use SQLAlchemy ORM)
- [ ] API key management for external services

## LangChain Best Practices

- Use conversation memory for chat ty
- Implement proper tounting and limits
- Handle API rate limits gracefully
- Use callbacks for monitoring and logging
- Implement proper error handling for AI operations
- Cache frequently used prompts and chains
- Monitor costs and usage metrics
- Use streaming for long responses when possible
