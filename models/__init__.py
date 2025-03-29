"""
LlamaClaims Model Package

This package provides utilities for downloading, optimizing, and using ML models with MLX.
"""

import os
import platform
import logging
from typing import Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("models")

# Check if running on Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Import MLX conditionally
HAS_MLX = False
if IS_APPLE_SILICON:
    try:
        import mlx.core
        HAS_MLX = True
        logger.info("MLX is available - GPU acceleration enabled")
    except ImportError:
        logger.warning("MLX not available. Install with: pip install mlx")

# Import interface components
from .interface import (
    ModelManager,
    ModelType,
    ModelLoadMode,
    get_model_manager,
)

# Import MLX wrapper components if MLX is available
if HAS_MLX:
    from .mlx_wrapper import (
        MLXModelWrapper,
        create_mlx_model_wrapper,
    )

# Version information
__version__ = "0.1.0"

# Determine default model directory
DEFAULT_MODELS_DIR = os.environ.get("LLAMACLAIMS_MODELS_DIR", os.path.join(os.path.dirname(__file__), "..", "data", "models"))

# Create default model manager
default_manager = ModelManager(models_dir=DEFAULT_MODELS_DIR)

# Re-export key functions at package level
def list_models() -> Dict[str, Dict[str, Any]]:
    """List available models."""
    return default_manager.list_available_models()

def load_model(model_type: Union[ModelType, str], **kwargs) -> Optional[Any]:
    """Load a model for inference."""
    return default_manager.load_model(model_type, **kwargs)

def benchmark_model(model_type: Union[ModelType, str], **kwargs) -> Dict[str, Any]:
    """Benchmark a model's inference speed."""
    return default_manager.benchmark_model(model_type, **kwargs) 