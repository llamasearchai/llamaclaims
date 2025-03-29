"""
Models API Routes for LlamaClaims

This module defines the FastAPI routes for the models API endpoints.
"""

import os
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query, Path

from api.services.models import ModelsService
from api.schemas.models import (
    ModelInfo,
    ModelList,
    InferenceRequest,
    InferenceResponse,
    BenchmarkRequest,
    BenchmarkResult,
    ModelDownloadRequest,
    ModelOptimizeRequest,
    ModelResponse,
)
from api.dependencies import get_models_service, get_request_metadata

# Create router
router = APIRouter(
    prefix="/models",
    tags=["models"],
    responses={
        404: {"description": "Item not found"},
        500: {"description": "Internal server error"},
    },
)


@router.get(
    "/",
    response_model=ModelList,
    summary="List available models",
    description="Get a list of all available models and their metadata.",
)
async def list_models(
    models_service: ModelsService = Depends(get_models_service),
    metadata: dict = Depends(get_request_metadata),
):
    """
    List all available models.
    
    Returns:
        ModelList: List of models
    """
    return await models_service.list_models()


@router.get(
    "/{model_id}",
    response_model=ModelInfo,
    summary="Get model information",
    description="Get detailed information about a specific model.",
)
async def get_model(
    model_id: str = Path(..., description="ID of the model"),
    models_service: ModelsService = Depends(get_models_service),
    metadata: dict = Depends(get_request_metadata),
):
    """
    Get information about a specific model.
    
    Args:
        model_id: ID of the model
        
    Returns:
        ModelInfo: Information about the model
        
    Raises:
        HTTPException: If the model is not found
    """
    model_info = await models_service.get_model(model_id)
    
    if model_info is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return model_info


@router.post(
    "/{model_id}/inference",
    response_model=InferenceResponse,
    summary="Run model inference",
    description="Run inference with a model using the provided input data.",
)
async def run_inference(
    model_id: str = Path(..., description="ID of the model"),
    request: InferenceRequest = ...,
    models_service: ModelsService = Depends(get_models_service),
    metadata: dict = Depends(get_request_metadata),
):
    """
    Run inference with a model.
    
    Args:
        model_id: ID of the model
        request: Inference request containing input data
        
    Returns:
        InferenceResponse: Inference results
        
    Raises:
        HTTPException: If the model is not found or inference fails
    """
    # Check if model exists
    model_info = await models_service.get_model(model_id)
    
    if model_info is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Run inference
    response = await models_service.run_inference(model_id, request)
    
    # Check for errors
    if not response.success:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {response.error or 'Unknown error'}"
        )
    
    return response


@router.post(
    "/{model_id}/benchmark",
    response_model=BenchmarkResult,
    summary="Benchmark model",
    description="Benchmark a model's inference speed.",
)
async def benchmark_model(
    model_id: str = Path(..., description="ID of the model"),
    request: BenchmarkRequest = ...,
    models_service: ModelsService = Depends(get_models_service),
    metadata: dict = Depends(get_request_metadata),
):
    """
    Benchmark a model's inference speed.
    
    Args:
        model_id: ID of the model
        request: Benchmark request
        
    Returns:
        BenchmarkResult: Benchmark results
        
    Raises:
        HTTPException: If the model is not found or benchmarking fails
    """
    # Check if model exists
    model_info = await models_service.get_model(model_id)
    
    if model_info is None:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    # Run benchmark
    result = await models_service.benchmark_model(
        model_id,
        num_runs=request.num_runs,
        use_mlx=request.use_mlx
    )
    
    # Check for errors
    if not result.success:
        raise HTTPException(
            status_code=500,
            detail=f"Benchmarking failed: {result.error or 'Unknown error'}"
        )
    
    return result


@router.post(
    "/download",
    response_model=ModelResponse,
    summary="Download a model",
    description="Download a model from HuggingFace.",
)
async def download_model(
    request: ModelDownloadRequest = ...,
    background_tasks: BackgroundTasks = ...,
    models_service: ModelsService = Depends(get_models_service),
    metadata: dict = Depends(get_request_metadata),
):
    """
    Download a model.
    
    Args:
        request: Download request
        
    Returns:
        ModelResponse: Response indicating that the download has started
    """
    try:
        # Try to import the downloader module
        from models.downloader import download_model as download_model_func
        
        # Define a background task for downloading
        def download_in_background(model_id: str, force: bool) -> None:
            try:
                import os
                from pathlib import Path
                
                # Get models directory from environment or default
                models_dir = os.environ.get(
                    "LLAMACLAIMS_MODELS_DIR", 
                    os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
                )
                
                # Create models directory if it doesn't exist
                models_dir = Path(models_dir)
                models_dir.mkdir(parents=True, exist_ok=True)
                
                # Download the model
                download_model_func(model_id, models_dir, force)
            except Exception as e:
                import logging
                logging.error(f"Error downloading model {model_id}: {e}")
        
        # Add the download task to the background tasks
        background_tasks.add_task(
            download_in_background,
            request.model_id,
            request.force
        )
        
        return ModelResponse(
            model_id=request.model_id,
            success=True,
            error=None,
            message=f"Model download started for {request.model_id}"
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Model download functionality is not available"
        )


@router.post(
    "/optimize",
    response_model=ModelResponse,
    summary="Optimize a model for MLX",
    description="Optimize a model for MLX, including quantization and other optimizations.",
)
async def optimize_model(
    request: ModelOptimizeRequest = ...,
    background_tasks: BackgroundTasks = ...,
    models_service: ModelsService = Depends(get_models_service),
    metadata: dict = Depends(get_request_metadata),
):
    """
    Optimize a model for MLX.
    
    Args:
        request: Optimize request
        
    Returns:
        ModelResponse: Response indicating that the optimization has started
    """
    try:
        # Check if model exists
        model_info = await models_service.get_model(request.model_id)
        
        if model_info is None:
            raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
        
        # Try to import the optimizer module
        from models.optimizer import optimize_model as optimize_model_func
        
        # Define a background task for optimizing
        def optimize_in_background(
            model_id: str,
            quantize: Optional[int],
            optimize_for_inference: bool
        ) -> None:
            try:
                import os
                from pathlib import Path
                
                # Get models directory from environment or default
                models_dir = os.environ.get(
                    "LLAMACLAIMS_MODELS_DIR", 
                    os.path.join(os.path.dirname(__file__), "..", "..", "data", "models")
                )
                
                # Create models directory if it doesn't exist
                models_dir = Path(models_dir)
                models_dir.mkdir(parents=True, exist_ok=True)
                
                # Optimize the model
                optimize_model_func(
                    model_id,
                    models_dir,
                    quantize,
                    optimize_for_inference
                )
            except Exception as e:
                import logging
                logging.error(f"Error optimizing model {model_id}: {e}")
        
        # Add the optimize task to the background tasks
        background_tasks.add_task(
            optimize_in_background,
            request.model_id,
            request.quantize,
            request.optimize_for_inference
        )
        
        return ModelResponse(
            model_id=request.model_id,
            success=True,
            error=None,
            message=f"Model optimization started for {request.model_id}"
        )
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="Model optimization functionality is not available"
        ) 