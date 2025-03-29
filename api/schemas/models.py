"""
Models Schemas for LlamaClaims API

This module defines the Pydantic models for the models API endpoints.
"""

from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


class ModelStatus(str, Enum):
    """Enum for model status."""
    AVAILABLE = "available"
    OPTIMIZED = "optimized"
    DOWNLOADING = "downloading"
    ERROR = "error"


class ModelInfo(BaseModel):
    """
    Information about a model.
    """
    
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Human-readable name of the model")
    status: ModelStatus = Field(..., description="Status of the model")
    size_mb: float = Field(0, description="Size of the model in MB")
    download_date: Optional[str] = Field(None, description="Date when the model was downloaded")
    class_name: Optional[str] = Field(None, description="Class name of the model from transformers")
    is_optimized: bool = Field(False, description="Whether the model is optimized for MLX")
    mlx_metadata: Optional[Dict[str, Any]] = Field(None, description="MLX-specific metadata")
    
    class Config:
        schema_extra = {
            "example": {
                "id": "document-classifier",
                "name": "Document Classification Model",
                "status": "optimized",
                "size_mb": 350.0,
                "download_date": "2023-03-14T12:34:56",
                "class_name": "LayoutLMv3ForSequenceClassification",
                "is_optimized": True,
                "mlx_metadata": {
                    "mlx_version": "0.3.0",
                    "quantization": 16,
                    "optimized_for_inference": True,
                    "conversion_date": "2023-03-14T13:45:12",
                    "metal_available": True
                }
            }
        }


class ModelList(BaseModel):
    """
    List of models.
    """
    
    models: List[ModelInfo] = Field(..., description="List of models")
    
    class Config:
        schema_extra = {
            "example": {
                "models": [
                    {
                        "id": "document-classifier",
                        "name": "Document Classification Model",
                        "status": "optimized",
                        "size_mb": 350.0,
                        "download_date": "2023-03-14T12:34:56",
                        "class_name": "LayoutLMv3ForSequenceClassification",
                        "is_optimized": True,
                        "mlx_metadata": {
                            "mlx_version": "0.3.0",
                            "quantization": 16,
                            "optimized_for_inference": True,
                            "conversion_date": "2023-03-14T13:45:12",
                            "metal_available": True
                        }
                    },
                    {
                        "id": "fraud-detector",
                        "name": "Fraud Detection Model",
                        "status": "available",
                        "size_mb": 480.0,
                        "download_date": "2023-03-14T12:40:23",
                        "class_name": "RobertaForSequenceClassification",
                        "is_optimized": False,
                        "mlx_metadata": None
                    }
                ]
            }
        }


class InferenceRequest(BaseModel):
    """
    Request for model inference.
    """
    
    inputs: Dict[str, Any] = Field(..., description="Input data for the model")
    use_mlx: Optional[bool] = Field(None, description="Whether to use MLX (if None, uses MLX if available)")
    
    class Config:
        schema_extra = {
            "example": {
                "inputs": {
                    "input_ids": [[101, 2054, 2003, 1037, 2609, 1012, 102]],
                    "attention_mask": [[1, 1, 1, 1, 1, 1, 1]]
                },
                "use_mlx": True
            }
        }


class InferenceResponse(BaseModel):
    """
    Response from model inference.
    """
    
    model_id: str = Field(..., description="ID of the model used for inference")
    success: bool = Field(..., description="Whether the inference was successful")
    error: Optional[str] = Field(None, description="Error message if inference failed")
    results: Dict[str, Any] = Field({}, description="Inference results")
    processing_time_ms: float = Field(..., description="Time taken for inference in milliseconds")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "document-classifier",
                "success": True,
                "error": None,
                "results": {
                    "logits": [[0.1, 0.9]],
                    "probs": [[0.1, 0.9]],
                    "predicted_class": [1]
                },
                "processing_time_ms": 112.5
            }
        }


class BenchmarkRequest(BaseModel):
    """
    Request for model benchmarking.
    """
    
    num_runs: int = Field(10, description="Number of benchmark runs")
    use_mlx: Optional[bool] = Field(None, description="Whether to use MLX (if None, uses MLX if available)")
    
    class Config:
        schema_extra = {
            "example": {
                "num_runs": 10,
                "use_mlx": True
            }
        }


class BenchmarkResult(BaseModel):
    """
    Result of model benchmarking.
    """
    
    model_id: str = Field(..., description="ID of the model benchmarked")
    success: bool = Field(..., description="Whether the benchmark was successful")
    error: Optional[str] = Field(None, description="Error message if benchmark failed")
    backend: str = Field(..., description="Backend used for benchmarking (mlx or pytorch)")
    mean_ms: float = Field(..., description="Mean inference time in milliseconds")
    std_ms: float = Field(..., description="Standard deviation of inference time in milliseconds")
    num_runs: int = Field(..., description="Number of benchmark runs")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "document-classifier",
                "success": True,
                "error": None,
                "backend": "mlx",
                "mean_ms": 112.5,
                "std_ms": 3.2,
                "num_runs": 10
            }
        }


class ModelDownloadRequest(BaseModel):
    """
    Request to download a model.
    """
    
    model_id: str = Field(..., description="ID of the model to download")
    force: bool = Field(False, description="Whether to force redownload if the model already exists")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "document-classifier",
                "force": False
            }
        }


class ModelOptimizeRequest(BaseModel):
    """
    Request to optimize a model for MLX.
    """
    
    model_id: str = Field(..., description="ID of the model to optimize")
    quantize: Optional[int] = Field(None, description="Quantization bits (8, 16, or None for no quantization)")
    optimize_for_inference: bool = Field(True, description="Whether to optimize the model for inference")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "document-classifier",
                "quantize": 16,
                "optimize_for_inference": True
            }
        }


class ModelResponse(BaseModel):
    """
    Response for model operations.
    """
    
    model_id: str = Field(..., description="ID of the model")
    success: bool = Field(..., description="Whether the operation was successful")
    error: Optional[str] = Field(None, description="Error message if the operation failed")
    message: str = Field(..., description="Message describing the result of the operation")
    
    class Config:
        schema_extra = {
            "example": {
                "model_id": "document-classifier",
                "success": True,
                "error": None,
                "message": "Model optimized successfully"
            }
        } 