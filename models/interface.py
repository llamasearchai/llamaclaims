#!/usr/bin/env python3
"""
LlamaClaims Model Interface

This module provides a unified interface for working with MLX-optimized models.
It abstracts away the details of model loading, inference, and optimization.
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
import platform
import threading
from enum import Enum

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model-interface")

# Check if running on Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Import MLX if available (only on Apple Silicon)
if IS_APPLE_SILICON:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        HAS_MLX = True
        logger.info("MLX is available - GPU acceleration enabled")
        
        # Check if Metal is available
        try:
            METAL_AVAILABLE = mx.metal.is_available()
            logger.info(f"Metal acceleration: {'Available' if METAL_AVAILABLE else 'Not available'}")
        except:
            METAL_AVAILABLE = False
            logger.warning("Metal acceleration check failed - assuming not available")
            
    except ImportError:
        HAS_MLX = False
        logger.warning("MLX not found - cannot use optimized models")
else:
    HAS_MLX = False
    logger.warning("Not running on Apple Silicon - MLX optimization not available")


class ModelType(Enum):
    """Enum representing different model types."""
    DOCUMENT_CLASSIFIER = "document-classifier"
    DOCUMENT_EXTRACTOR = "document-extractor"  
    CLAIMS_CLASSIFIER = "claims-classifier"
    FRAUD_DETECTOR = "fraud-detector"
    CLAIMS_LLM = "claims-llm"


class ModelLoadMode(Enum):
    """Enum representing different model loading modes."""
    AUTO = "auto"  # Automatically use MLX if available, fallback to PyTorch
    MLX_ONLY = "mlx_only"  # Only use MLX, fail if not available
    PYTORCH_ONLY = "pytorch_only"  # Only use PyTorch


class ModelCache:
    """A thread-safe cache for loaded models."""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ModelCache, cls).__new__(cls)
                cls._instance._models = {}
            return cls._instance
    
    def get(self, model_key: str) -> Optional[Any]:
        """Get a model from the cache."""
        with self._lock:
            return self._models.get(model_key)
    
    def set(self, model_key: str, model: Any) -> None:
        """Add a model to the cache."""
        with self._lock:
            self._models[model_key] = model
    
    def clear(self) -> None:
        """Clear the cache."""
        with self._lock:
            self._models.clear()
    
    def remove(self, model_key: str) -> None:
        """Remove a model from the cache."""
        with self._lock:
            if model_key in self._models:
                del self._models[model_key]


class ModelManager:
    """
    Manages loading, caching, and inference with models.
    
    This class handles both MLX and PyTorch models and provides a unified
    interface for model operations.
    """
    
    def __init__(
        self, 
        models_dir: Union[str, Path] = "./data/models",
        cache_enabled: bool = True,
        default_load_mode: ModelLoadMode = ModelLoadMode.AUTO
    ):
        """
        Initialize the model manager.
        
        Args:
            models_dir: Directory containing the models
            cache_enabled: Whether to cache loaded models
            default_load_mode: Default model loading mode
        """
        self.models_dir = Path(models_dir)
        self.cache_enabled = cache_enabled
        self.default_load_mode = default_load_mode
        self.model_cache = ModelCache() if cache_enabled else None
        
        # Create models directory if it doesn't exist
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def list_available_models(self) -> Dict[str, Dict[str, Any]]:
        """
        List available models and their metadata.
        
        Returns:
            Dict mapping model IDs to metadata
        """
        available_models = {}
        
        if not self.models_dir.exists():
            logger.warning(f"Models directory {self.models_dir} does not exist")
            return available_models
        
        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir():
                continue
            
            metadata_file = model_dir / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    
                    model_id = metadata.get("model_id", model_dir.name)
                    available_models[model_id] = metadata
                except Exception as e:
                    logger.warning(f"Error reading metadata for {model_dir.name}: {e}")
        
        return available_models
    
    def get_model_path(self, model_type: Union[ModelType, str]) -> Optional[Path]:
        """
        Get the path to a model directory.
        
        Args:
            model_type: Type of model to get path for
            
        Returns:
            Path to the model directory if it exists, None otherwise
        """
        if isinstance(model_type, ModelType):
            model_id = model_type.value
        else:
            model_id = model_type
        
        model_dir = self.models_dir / model_id
        
        if not model_dir.exists():
            logger.warning(f"Model directory {model_dir} does not exist")
            return None
        
        return model_dir
    
    def get_model_metadata(self, model_type: Union[ModelType, str]) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a model.
        
        Args:
            model_type: Type of model to get metadata for
            
        Returns:
            Model metadata if available, None otherwise
        """
        model_dir = self.get_model_path(model_type)
        
        if model_dir is None:
            return None
        
        metadata_file = model_dir / "metadata.json"
        
        if not metadata_file.exists():
            logger.warning(f"Metadata file {metadata_file} does not exist")
            return None
        
        try:
            with open(metadata_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error reading metadata for {model_type}: {e}")
            return None
    
    def is_model_optimized(self, model_type: Union[ModelType, str]) -> bool:
        """
        Check if a model is optimized for MLX.
        
        Args:
            model_type: Type of model to check
            
        Returns:
            True if the model is optimized, False otherwise
        """
        metadata = self.get_model_metadata(model_type)
        
        if metadata is None:
            return False
        
        return metadata.get("optimized", False)
    
    def load_model(
        self, 
        model_type: Union[ModelType, str],
        load_mode: Optional[ModelLoadMode] = None,
        force_reload: bool = False
    ) -> Optional[Any]:
        """
        Load a model for inference.
        
        Args:
            model_type: Type of model to load
            load_mode: Model loading mode (MLX, PyTorch, or auto)
            force_reload: Whether to reload the model even if it's cached
            
        Returns:
            Loaded model if successful, None otherwise
        """
        if isinstance(model_type, ModelType):
            model_id = model_type.value
        else:
            model_id = model_type
        
        # Use default load mode if not specified
        if load_mode is None:
            load_mode = self.default_load_mode
        
        # Check if model is cached
        cache_key = f"{model_id}_{load_mode.value}"
        if self.cache_enabled and not force_reload:
            cached_model = self.model_cache.get(cache_key)
            if cached_model is not None:
                logger.debug(f"Using cached model for {model_id}")
                return cached_model
        
        # Get model directory
        model_dir = self.get_model_path(model_id)
        if model_dir is None:
            return None
        
        # Get model metadata
        metadata = self.get_model_metadata(model_id)
        if metadata is None:
            return None
        
        # Check if MLX is required but not available
        is_optimized = metadata.get("optimized", False)
        if load_mode == ModelLoadMode.MLX_ONLY and (not HAS_MLX or not is_optimized):
            logger.error(f"MLX is required for {model_id} but not available")
            return None
        
        # Determine whether to use MLX or PyTorch
        use_mlx = False
        if load_mode == ModelLoadMode.MLX_ONLY:
            use_mlx = True
        elif load_mode == ModelLoadMode.PYTORCH_ONLY:
            use_mlx = False
        else:  # AUTO
            use_mlx = HAS_MLX and is_optimized
        
        # Load the model
        model = None
        try:
            if use_mlx:
                model = self._load_mlx_model(model_dir, metadata)
            else:
                model = self._load_pytorch_model(model_dir, metadata)
            
            # Cache the model if successful
            if model is not None and self.cache_enabled:
                self.model_cache.set(cache_key, model)
            
            return model
        except Exception as e:
            logger.error(f"Error loading model {model_id}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def _load_mlx_model(self, model_dir: Path, metadata: Dict[str, Any]) -> Optional[Any]:
        """
        Load an MLX model.
        
        Args:
            model_dir: Directory containing the model
            metadata: Model metadata
            
        Returns:
            Loaded MLX model if successful, None otherwise
        """
        if not HAS_MLX:
            logger.error("MLX is not available")
            return None
        
        mlx_dir = model_dir / "mlx"
        if not mlx_dir.exists():
            logger.error(f"MLX directory {mlx_dir} does not exist")
            return None
        
        weights_file = mlx_dir / "weights.safetensors"
        if not weights_file.exists():
            logger.error(f"MLX weights file {weights_file} does not exist")
            return None
        
        # Load MLX weights
        logger.info(f"Loading MLX model from {weights_file}")
        weights = mx.load(str(weights_file))
        
        # Get the model class from metadata
        model_class = metadata.get("class")
        if not model_class:
            logger.error("Model class not specified in metadata")
            return None
        
        # Create an MLX model wrapper
        from .mlx_wrapper import MLXModelWrapper
        model = MLXModelWrapper(
            model_id=metadata.get("model_id"),
            model_class=model_class,
            weights=weights,
            config_path=str(mlx_dir / "config.json") if (mlx_dir / "config.json").exists() else None,
            tokenizer_path=str(mlx_dir) if (mlx_dir / "tokenizer.json").exists() else None
        )
        
        return model
    
    def _load_pytorch_model(self, model_dir: Path, metadata: Dict[str, Any]) -> Optional[Any]:
        """
        Load a PyTorch model.
        
        Args:
            model_dir: Directory containing the model
            metadata: Model metadata
            
        Returns:
            Loaded PyTorch model if successful, None otherwise
        """
        # Import PyTorch and transformers
        try:
            import torch
            import transformers
        except ImportError:
            logger.error("PyTorch or transformers not available")
            return None
        
        # Get the model class from metadata
        model_class_name = metadata.get("class")
        if not model_class_name:
            logger.error("Model class not specified in metadata")
            return None
        
        # Get the model class
        model_class = getattr(transformers, model_class_name, None)
        if model_class is None:
            logger.error(f"Unknown model class: {model_class_name}")
            return None
        
        # Check if the model files exist
        if (model_dir / "pytorch_model.bin").exists() and (model_dir / "config.json").exists():
            # Load from local files
            logger.info(f"Loading PyTorch model from {model_dir}")
            model = model_class.from_pretrained(str(model_dir))
        else:
            # Load from HuggingFace
            repo_id = metadata.get("repo_id")
            if not repo_id:
                logger.error("Model repo_id not specified in metadata")
                return None
            
            logger.info(f"Loading PyTorch model from HuggingFace: {repo_id}")
            model = model_class.from_pretrained(repo_id)
        
        return model
    
    def run_inference(
        self, 
        model: Any, 
        inputs: Dict[str, Any],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run inference with a model.
        
        Args:
            model: Model to run inference with
            inputs: Input data for the model
            **kwargs: Additional arguments for inference
            
        Returns:
            Dictionary containing inference results
        """
        # Check if we're using an MLX wrapper
        if hasattr(model, "is_mlx_wrapper") and model.is_mlx_wrapper:
            return model.predict(inputs, **kwargs)
        
        # If it's a PyTorch model
        try:
            import torch
            
            # Check if CUDA is available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            # Convert inputs to tensors on the right device
            tensor_inputs = {}
            for k, v in inputs.items():
                if isinstance(v, (list, tuple)) and all(isinstance(x, (int, float)) for x in v):
                    tensor_inputs[k] = torch.tensor(v).to(device)
                elif isinstance(v, (int, float)):
                    tensor_inputs[k] = torch.tensor([v]).to(device)
                else:
                    tensor_inputs[k] = v
            
            # Run inference
            with torch.no_grad():
                outputs = model(**tensor_inputs)
            
            # Convert outputs to Python types
            results = {}
            for k, v in outputs.items():
                if hasattr(v, "detach"):
                    results[k] = v.detach().cpu().numpy().tolist()
                else:
                    results[k] = v
            
            return results
        except Exception as e:
            logger.error(f"Error running inference: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def benchmark_model(
        self, 
        model_type: Union[ModelType, str],
        num_runs: int = 10,
        warmup_runs: int = 3,
        use_mlx: Optional[bool] = None,
        sample_input: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Benchmark a model's inference speed.
        
        Args:
            model_type: Type of model to benchmark
            num_runs: Number of benchmark runs
            warmup_runs: Number of warmup runs
            use_mlx: Whether to use MLX (if None, uses MLX if available)
            sample_input: Sample input for inference (if None, generates one)
            
        Returns:
            Dictionary containing benchmark results
        """
        if isinstance(model_type, ModelType):
            model_id = model_type.value
        else:
            model_id = model_type
        
        # Determine whether to use MLX
        if use_mlx is None:
            use_mlx = HAS_MLX and self.is_model_optimized(model_id)
        elif use_mlx and (not HAS_MLX or not self.is_model_optimized(model_id)):
            logger.error(f"MLX requested for {model_id} but not available")
            return {"error": "MLX not available"}
        
        # Load the appropriate model
        load_mode = ModelLoadMode.MLX_ONLY if use_mlx else ModelLoadMode.PYTORCH_ONLY
        model = self.load_model(model_id, load_mode=load_mode, force_reload=True)
        
        if model is None:
            return {"error": f"Failed to load model {model_id}"}
        
        # Generate sample input if not provided
        if sample_input is None:
            sample_input = self._generate_sample_input(model_id)
        
        # Run warmup iterations
        for _ in range(warmup_runs):
            _ = self.run_inference(model, sample_input)
        
        # Run benchmark iterations
        times = []
        for _ in range(num_runs):
            start_time = time.time()
            _ = self.run_inference(model, sample_input)
            end_time = time.time()
            times.append((end_time - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        import numpy as np
        mean_time = np.mean(times)
        std_time = np.std(times)
        
        return {
            "model_id": model_id,
            "backend": "mlx" if use_mlx else "pytorch",
            "num_runs": num_runs,
            "mean_ms": float(mean_time),
            "std_ms": float(std_time),
            "times_ms": [float(t) for t in times]
        }
    
    def _generate_sample_input(self, model_id: str) -> Dict[str, Any]:
        """
        Generate a sample input for a model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Dictionary containing sample input data
        """
        # Default input sizes for different model types
        input_sizes = {
            "document-classifier": {"input_ids": [1, 512], "attention_mask": [1, 512]},
            "document-extractor": {"input_ids": [1, 512], "attention_mask": [1, 512]},
            "claims-classifier": {"input_ids": [1, 128], "attention_mask": [1, 128]},
            "fraud-detector": {"input_ids": [1, 256], "attention_mask": [1, 256]},
            "claims-llm": {"input_ids": [1, 64], "attention_mask": [1, 64]}
        }
        
        # Generate random input data
        import numpy as np
        
        input_size = input_sizes.get(model_id, {"input_ids": [1, 128], "attention_mask": [1, 128]})
        
        sample_input = {}
        for key, size in input_size.items():
            if key == "input_ids":
                sample_input[key] = np.random.randint(0, 1000, size=size).tolist()
            elif key == "attention_mask":
                sample_input[key] = np.ones(size, dtype=np.int32).tolist()
        
        return sample_input


# Default model manager instance
default_model_manager = ModelManager()

def get_model_manager() -> ModelManager:
    """Get the default model manager instance."""
    return default_model_manager 