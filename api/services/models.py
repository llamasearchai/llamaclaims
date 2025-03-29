"""
Models Service for LlamaClaims API

This module provides a service for working with ML models through the API.
It integrates with the model interface to provide a clean API for model operations.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import asyncio
import time

# Import model types
try:
    import models
    from models import ModelType, ModelLoadMode
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    # Define fallback types for when the models package is not available
    class ModelType:
        DOCUMENT_CLASSIFIER = "document-classifier"
        DOCUMENT_EXTRACTOR = "document-extractor"
        CLAIMS_CLASSIFIER = "claims-classifier"
        FRAUD_DETECTOR = "fraud-detector"
        CLAIMS_LLM = "claims-llm"
    
    class ModelLoadMode:
        AUTO = "auto"
        MLX_ONLY = "mlx_only"
        PYTORCH_ONLY = "pytorch_only"

# Import API schemas
from api.schemas.models import (
    ModelInfo,
    ModelList,
    ModelStatus,
    InferenceRequest,
    InferenceResponse,
    BenchmarkResult,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("api.services.models")


class ModelsService:
    """
    Service for working with ML models.
    
    This service provides methods for listing, loading, and running inference
    with models. It uses the models package to handle the underlying model operations.
    """
    
    def __init__(self, models_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the models service.
        
        Args:
            models_dir: Directory containing the models (optional)
        """
        self.models_available = MODELS_AVAILABLE
        
        if not self.models_available:
            logger.warning("Models package not available. Running with limited functionality.")
            return
        
        # Get the model manager
        if models_dir:
            self.model_manager = models.ModelManager(models_dir=models_dir)
        else:
            self.model_manager = models.get_model_manager()
    
    async def list_models(self) -> ModelList:
        """
        List available models.
        
        Returns:
            ModelList containing information about available models
        """
        if not self.models_available:
            return ModelList(models=[])
        
        # Run model listing in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        available_models = await loop.run_in_executor(
            None, self.model_manager.list_available_models
        )
        
        # Convert to API schema
        model_infos = []
        for model_id, metadata in available_models.items():
            # Determine model status
            status = ModelStatus.AVAILABLE
            if metadata.get("optimized", False):
                status = ModelStatus.OPTIMIZED
            
            # Create model info
            model_info = ModelInfo(
                id=model_id,
                name=metadata.get("description", model_id),
                status=status,
                size_mb=metadata.get("size_mb", 0),
                download_date=metadata.get("download_date", ""),
                class_name=metadata.get("class", ""),
                is_optimized=metadata.get("optimized", False),
                mlx_metadata=metadata.get("mlx_metadata", {})
            )
            
            model_infos.append(model_info)
        
        return ModelList(models=model_infos)
    
    async def get_model(self, model_id: str) -> Optional[ModelInfo]:
        """
        Get information about a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            ModelInfo if the model exists, None otherwise
        """
        if not self.models_available:
            return None
        
        # Run in a separate thread to avoid blocking
        loop = asyncio.get_event_loop()
        metadata = await loop.run_in_executor(
            None, self.model_manager.get_model_metadata, model_id
        )
        
        if metadata is None:
            return None
        
        # Determine model status
        status = ModelStatus.AVAILABLE
        if metadata.get("optimized", False):
            status = ModelStatus.OPTIMIZED
        
        # Create model info
        return ModelInfo(
            id=model_id,
            name=metadata.get("description", model_id),
            status=status,
            size_mb=metadata.get("size_mb", 0),
            download_date=metadata.get("download_date", ""),
            class_name=metadata.get("class", ""),
            is_optimized=metadata.get("optimized", False),
            mlx_metadata=metadata.get("mlx_metadata", {})
        )
    
    async def run_inference(
        self, 
        model_id: str, 
        request: InferenceRequest
    ) -> InferenceResponse:
        """
        Run inference with a model.
        
        Args:
            model_id: ID of the model
            request: Inference request containing input data
            
        Returns:
            InferenceResponse containing the inference results
        """
        if not self.models_available:
            return InferenceResponse(
                model_id=model_id,
                success=False,
                error="Models package not available",
                results={},
                processing_time_ms=0
            )
        
        # Determine model load mode
        if request.use_mlx is not None:
            load_mode = ModelLoadMode.MLX_ONLY if request.use_mlx else ModelLoadMode.PYTORCH_ONLY
        else:
            load_mode = ModelLoadMode.AUTO
        
        try:
            # Load the model in a separate thread
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                None, 
                lambda: self.model_manager.load_model(model_id, load_mode=load_mode)
            )
            
            if model is None:
                return InferenceResponse(
                    model_id=model_id,
                    success=False,
                    error=f"Failed to load model {model_id}",
                    results={},
                    processing_time_ms=0
                )
            
            # Run inference
            start_time = time.time()
            results = await loop.run_in_executor(
                None,
                lambda: self.model_manager.run_inference(model, request.inputs)
            )
            end_time = time.time()
            
            # Calculate processing time
            processing_time_ms = (end_time - start_time) * 1000
            
            # Check for errors
            if isinstance(results, dict) and "error" in results:
                return InferenceResponse(
                    model_id=model_id,
                    success=False,
                    error=results["error"],
                    results={},
                    processing_time_ms=processing_time_ms
                )
            
            return InferenceResponse(
                model_id=model_id,
                success=True,
                error=None,
                results=results,
                processing_time_ms=processing_time_ms
            )
            
        except Exception as e:
            logger.error(f"Error running inference with model {model_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return InferenceResponse(
                model_id=model_id,
                success=False,
                error=str(e),
                results={},
                processing_time_ms=0
            )
    
    async def benchmark_model(
        self, 
        model_id: str,
        num_runs: int = 10,
        use_mlx: Optional[bool] = None
    ) -> BenchmarkResult:
        """
        Benchmark a model's inference speed.
        
        Args:
            model_id: ID of the model
            num_runs: Number of benchmark runs
            use_mlx: Whether to use MLX (if None, uses MLX if available)
            
        Returns:
            BenchmarkResult containing the benchmark results
        """
        if not self.models_available:
            return BenchmarkResult(
                model_id=model_id,
                success=False,
                error="Models package not available",
                backend="none",
                mean_ms=0,
                std_ms=0,
                num_runs=0
            )
        
        try:
            # Run benchmark in a separate thread
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self.model_manager.benchmark_model(
                    model_id,
                    num_runs=num_runs,
                    use_mlx=use_mlx
                )
            )
            
            # Check for errors
            if isinstance(results, dict) and "error" in results:
                return BenchmarkResult(
                    model_id=model_id,
                    success=False,
                    error=results["error"],
                    backend="none",
                    mean_ms=0,
                    std_ms=0,
                    num_runs=0
                )
            
            return BenchmarkResult(
                model_id=model_id,
                success=True,
                error=None,
                backend=results.get("backend", "unknown"),
                mean_ms=results.get("mean_ms", 0),
                std_ms=results.get("std_ms", 0),
                num_runs=results.get("num_runs", 0)
            )
            
        except Exception as e:
            logger.error(f"Error benchmarking model {model_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return BenchmarkResult(
                model_id=model_id,
                success=False,
                error=str(e),
                backend="none",
                mean_ms=0,
                std_ms=0,
                num_runs=0
            ) 