"""Error handling middleware."""

from typing import Callable

from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config.logging_config import LoggerAdapter

logger = LoggerAdapter(__name__)


class ErrorHandlerMiddleware(BaseHTTPMiddleware):
    """Middleware for handling exceptions globally."""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Process request and handle exceptions."""
        try:
            response = await call_next(request)
            return response
            
        except ValueError as e:
            logger.warning(
                "Validation error",
                path=request.url.path,
                method=request.method,
                error=str(e)
            )
            return JSONResponse(
                status_code=400,
                content={
                    "detail": str(e),
                    "type": "validation_error"
                }
            )
            
        except PermissionError as e:
            logger.warning(
                "Permission denied",
                path=request.url.path,
                method=request.method,
                error=str(e)
            )
            return JSONResponse(
                status_code=403,
                content={
                    "detail": "Permission denied",
                    "type": "permission_error"
                }
            )
            
        except FileNotFoundError as e:
            logger.warning(
                "Resource not found",
                path=request.url.path,
                method=request.method,
                error=str(e)
            )
            return JSONResponse(
                status_code=404,
                content={
                    "detail": "Resource not found",
                    "type": "not_found"
                }
            )
            
        except Exception as e:
            logger.error(
                "Unhandled exception",
                path=request.url.path,
                method=request.method,
                error=str(e),
                exc_info=True
            )
            
            # Don't expose internal errors in production
            from src.core.config.settings import settings
            
            if settings.is_production:
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": "Internal server error",
                        "type": "internal_error"
                    }
                )
            else:
                return JSONResponse(
                    status_code=500,
                    content={
                        "detail": str(e),
                        "type": type(e).__name__
                    }
                )