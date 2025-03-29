"""
Dependencies for the LlamaClaims API.

This module provides dependencies for FastAPI dependency injection.
"""

from typing import Callable, Dict, Optional, Any
from functools import lru_cache
from fastapi import Depends, Request
import logging
import uuid
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api.dependencies")

# Service container for dependency injection
class ServiceContainer:
    """Container for service instances to enable dependency injection."""
    
    def __init__(self):
        """Initialize the service container."""
        self._services: Dict[str, Any] = {}
    
    def register(self, service_key: str, service_instance: Any) -> None:
        """Register a service instance."""
        self._services[service_key] = service_instance
    
    def get(self, service_key: str) -> Optional[Any]:
        """Get a service instance by key."""
        return self._services.get(service_key)

# Create a service container
service_container = ServiceContainer()

# Register services
from api.services.claims import ClaimsService
service_container.register("claims_service", ClaimsService())

from api.services.analysis import AnalysisService
service_container.register("analysis_service", AnalysisService())

# Register models service if available
try:
    from api.services.models import ModelsService
    
    # Create models directory if it doesn't exist
    models_dir = os.environ.get(
        "LLAMACLAIMS_MODELS_DIR", 
        os.path.join(os.path.dirname(__file__), "..", "data", "models")
    )
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    
    service_container.register("models_service", ModelsService(models_dir=models_dir))
    logger.info(f"Registered models service with models directory: {models_dir}")
except ImportError:
    logger.warning("Models service not available - model functionality will be limited")

# Dependency functions
def get_claims_service() -> ClaimsService:
    """Get the claims service instance."""
    return service_container.get("claims_service")

def get_analysis_service() -> AnalysisService:
    """Get the analysis service instance."""
    return service_container.get("analysis_service")

def get_models_service():
    """Get the models service instance."""
    service = service_container.get("models_service")
    if service is None:
        from api.services.models import ModelsService
        service = ModelsService()
        service_container.register("models_service", service)
    return service

def get_request_metadata(request: Request) -> Dict[str, Any]:
    """Get metadata for the current request."""
    # Get or generate a request ID
    request_id = request.headers.get("X-Request-ID")
    if not request_id:
        request_id = str(uuid.uuid4())
    
    # Return metadata
    return {
        "request_id": request_id,
        "user_agent": request.headers.get("User-Agent", ""),
        "remote_addr": request.client.host if request.client else "",
    } 