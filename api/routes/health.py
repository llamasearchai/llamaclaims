"""
Health API routes for LlamaClaims.

This module provides endpoints for monitoring the health and status
of the LlamaClaims API and its dependencies.
"""

import time
import platform
import os
from datetime import datetime, timezone
from typing import Dict, Any, Optional

from fastapi import APIRouter, Request, Depends

from api.dependencies import get_request_metadata

router = APIRouter(tags=["health"])

# Track startup time
START_TIME = time.time()

@router.get("/health")
async def health_check(request: Request) -> Dict[str, Any]:
    """
    Check the health of the API.
    
    Returns:
        Health status information
    """
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "uptime_seconds": int(time.time() - START_TIME)
    }

@router.get("/version")
async def version() -> Dict[str, str]:
    """
    Get API version information.
    
    Returns:
        Version information
    """
    from api import __version__
    
    return {
        "version": __version__,
        "name": "LlamaClaims API"
    }

@router.get("/system")
async def system_info() -> Dict[str, Any]:
    """
    Get system information.
    
    Returns:
        System information
    """
    is_apple_silicon = False
    mlx_available = False
    metal_available = False
    
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        is_apple_silicon = True
        
        # Check if MLX is available
        try:
            import mlx.core as mx
            mlx_available = True
            
            # Check if Metal is available
            metal_available = mx.metal.is_available()
        except (ImportError, AttributeError):
            pass
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "is_apple_silicon": is_apple_silicon,
        "mlx_available": mlx_available,
        "metal_available": metal_available,
        "cpu_count": os.cpu_count(),
        "env": os.environ.get("APP_ENV", "development")
    }

@router.get("/metrics")
async def metrics() -> Dict[str, Any]:
    """
    Get API metrics.
    
    Returns:
        API metrics for monitoring
    """
    # In a real app, this would retrieve metrics from a metrics collector
    return {
        "uptime_seconds": int(time.time() - START_TIME),
        "request_count": 0,  # Placeholder
        "error_count": 0,    # Placeholder
        "average_response_time_ms": 0  # Placeholder
    }

@router.get("/debug")
async def debug_info(metadata: Dict[str, Any] = Depends(get_request_metadata)) -> Dict[str, Any]:
    """
    Get debug information.
    
    Args:
        metadata: Request metadata
        
    Returns:
        Debug information
    """
    return {
        "request": metadata,
        "environment": {
            k: v for k, v in os.environ.items() 
            if k.startswith(("APP_", "API_", "MLX_")) and not "SECRET" in k and not "KEY" in k
        },
        "system": {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "is_apple_silicon": platform.system() == "Darwin" and platform.machine() == "arm64"
        }
    } 