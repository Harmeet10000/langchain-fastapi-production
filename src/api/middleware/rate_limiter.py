"""Rate limiting middleware."""

from typing import Callable
import time
from collections import defaultdict

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config.settings import settings
from src.core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Middleware for rate limiting requests."""
    
    def __init__(self, app):
        super().__init__(app)
        self.requests = defaultdict(list)
        self.max_requests = settings.rate_limit_requests
        self.window_seconds = settings.rate_limit_period
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request with rate limiting."""
        # Skip rate limiting for health checks
        if request.url.path in ["/health", "/"]:
            return await call_next(request)
        
        # Get client identifier (IP address)
        client_id = request.client.host if request.client else "unknown"
        
        # Current timestamp
        now = time.time()
        
        # Clean old requests outside the window
        self.requests[client_id] = [
            timestamp for timestamp in self.requests[client_id]
            if now - timestamp < self.window_seconds
        ]
        
        # Check if limit exceeded
        if len(self.requests[client_id]) >= self.max_requests:
            logger.warning(
                "Rate limit exceeded",
                client=client_id,
                path=request.url.path,
                requests_count=len(self.requests[client_id]),
            )
            
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Too many requests",
                    "retry_after": self.window_seconds,
                },
                headers={
                    "Retry-After": str(self.window_seconds),
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(now + self.window_seconds)),
                },
            )
        
        # Add current request timestamp
        self.requests[client_id].append(now)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = self.max_requests - len(self.requests[client_id])
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(int(now + self.window_seconds))
        
        return response