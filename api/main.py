"""
Main module for the LlamaClaims API.

This is the entry point for the FastAPI application.
"""

import os
import logging
import uuid
from typing import Optional, Dict, Any

from fastapi import FastAPI, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import __version__
from api.routes import health, claims, analysis, models

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api.main")

# Create the FastAPI app
app = FastAPI(
    title="LlamaClaims API",
    description="API for the LlamaClaims insurance claims processing platform",
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add request ID middleware
@app.middleware("http")
async def add_request_id(request: Request, call_next) -> Response:
    """Add a request ID to the response headers."""
    # Get existing request ID from headers, or generate a new one
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    
    # Process the request
    response = await call_next(request)
    
    # Add the request ID to the response headers
    response.headers["X-Request-ID"] = request_id
    
    return response

# Add error handling middleware
@app.middleware("http")
async def catch_exceptions(request: Request, call_next) -> Response:
    """Catch exceptions and return a JSON response."""
    try:
        return await call_next(request)
    except Exception as e:
        # Log the exception
        logger.exception(f"Unhandled exception: {str(e)}")
        
        # Return a JSON response with the error
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": "Internal server error",
                "detail": str(e),
            },
        )

# Include routers
app.include_router(health.router)
app.include_router(claims.router)
app.include_router(analysis.router)
app.include_router(models.router)

# Root endpoint
@app.get("/", tags=["root"])
async def root() -> Dict[str, Any]:
    """Root endpoint that provides basic information about the API."""
    return {
        "name": "LlamaClaims API",
        "version": __version__,
        "documentation": "/docs",
    }

# Startup event
@app.on_event("startup")
async def startup_event() -> None:
    """Initialize the API on startup."""
    logger.info(f"Starting LlamaClaims API v{__version__}")
    
    # Check if running on Apple Silicon
    import platform
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        logger.info("Running on Apple Silicon - MLX optimization available")
    
    # Log environment settings
    env = os.environ.get("APP_ENV", "development")
    logger.info(f"Environment: {env}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event() -> None:
    """Clean up resources on shutdown."""
    logger.info("Shutting down LlamaClaims API")

# Run the application with uvicorn if executed directly
if __name__ == "__main__":
    import uvicorn
    
    # Configure port - default to 8000 if not specified
    port = int(os.environ.get("PORT", 8000))
    
    # Configure host - default to 127.0.0.1 if not in production
    host = "0.0.0.0" if os.environ.get("APP_ENV") == "production" else "127.0.0.1"
    
    # Start uvicorn server
    uvicorn.run(
        "api.main:app",
        host=host,
        port=port,
        reload=os.environ.get("APP_ENV") != "production",
        access_log=True,
    ) 