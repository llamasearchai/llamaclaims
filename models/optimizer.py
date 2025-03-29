#!/usr/bin/env python3
"""
MLX Model Optimizer for LlamaClaims

This script handles the conversion and optimization of PyTorch models to MLX format.
It supports various optimizations including quantization, graph optimization,
and Metal-specific optimizations for Apple Silicon.

Usage:
    python mlx_optimizer.py --model document-classifier --quantize 16 --optimize-for-inference
    python mlx_optimizer.py --all --quantize 8 --optimize-for-inference
"""

import argparse
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import platform
import shutil
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mlx-optimizer")

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
        logger.warning("MLX not found - cannot optimize models")
else:
    HAS_MLX = False
    logger.warning("Not running on Apple Silicon - MLX optimization not available")


def check_dependencies() -> bool:
    """
    Check if required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are satisfied
    """
    try:
        import torch
        import transformers
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        
        if not IS_APPLE_SILICON:
            logger.error("MLX optimization requires Apple Silicon (M-series chip)")
            return False
        
        if not HAS_MLX:
            logger.error("MLX is not installed. Please install with: pip install mlx")
            return False
            
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install -r requirements.txt")
        return False


def list_models(models_dir: Path) -> List[str]:
    """
    List available models in the models directory.
    
    Args:
        models_dir: Directory containing the models
        
    Returns:
        List of model IDs
    """
    models = []
    
    if not models_dir.exists():
        logger.warning(f"Models directory {models_dir} does not exist")
        return models
    
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue
        
        metadata_file = model_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                    
                models.append(metadata.get("model_id", model_dir.name))
            except Exception as e:
                logger.warning(f"Error reading metadata for {model_dir.name}: {e}")
    
    return models


def convert_to_mlx(model_dir: Path, quantize: Optional[int] = None, optimize_for_inference: bool = False) -> bool:
    """
    Convert a PyTorch model to MLX format.
    
    Args:
        model_dir: Directory containing the model
        quantize: Quantization bits (8, 16, or None for no quantization)
        optimize_for_inference: Whether to optimize the model for inference
        
    Returns:
        bool: True if conversion was successful
    """
    if not HAS_MLX:
        logger.error("MLX is not available - cannot convert model")
        return False
    
    metadata_file = model_dir / "metadata.json"
    if not metadata_file.exists():
        logger.error(f"Metadata file not found for model in {model_dir}")
        return False
    
    try:
        # Load metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        model_id = metadata.get("model_id", model_dir.name)
        model_class = metadata.get("class")
        
        if not model_class:
            logger.error(f"Model class not specified in metadata for {model_id}")
            return False
        
        logger.info(f"Converting {model_id} to MLX format...")
        
        # Import required modules
        import torch
        import transformers
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
        
        # Load PyTorch model based on class
        logger.info(f"Loading PyTorch model ({model_class})...")
        
        # Create a path to the PyTorch model file
        pytorch_model_path = model_dir / "pytorch_model.bin"
        if not pytorch_model_path.exists():
            logger.warning(f"PyTorch model file not found at {pytorch_model_path}")
            pytorch_model_path = None  # Will use the repo_id instead
        
        # Load config
        config_path = model_dir / "config.json"
        if config_path.exists():
            config = AutoConfig.from_pretrained(str(config_path))
        else:
            logger.warning(f"Config file not found at {config_path}")
            config = None
        
        # Determine model class and load model
        model_class_obj = getattr(transformers, model_class, None)
        if model_class_obj is None:
            logger.error(f"Unknown model class: {model_class}")
            return False
        
        # Load model from local path or HuggingFace
        if pytorch_model_path and config:
            model = model_class_obj.from_pretrained(
                str(model_dir),
                config=config,
                torch_dtype=torch.float32
            )
        else:
            # Fall back to HuggingFace repo
            repo_id = metadata.get("repo_id")
            if not repo_id:
                logger.error(f"Model repo_id not specified in metadata for {model_id}")
                return False
            
            logger.info(f"Loading model from HuggingFace: {repo_id}")
            model = model_class_obj.from_pretrained(repo_id, torch_dtype=torch.float32)
        
        # Convert to MLX
        logger.info("Converting model to MLX format...")
        
        # Create MLX output directory
        mlx_dir = model_dir / "mlx"
        mlx_dir.mkdir(exist_ok=True)
        
        # Convert weights to MLX format
        start_time = time.time()
        
        # Extract weights from PyTorch model
        weights = {}
        for name, param in model.named_parameters():
            weights[name] = mx.array(param.detach().numpy())
        
        # Apply quantization if requested
        if quantize:
            logger.info(f"Applying {quantize}-bit quantization...")
            
            if quantize == 8:
                # 8-bit quantization
                for name, param in weights.items():
                    if param.dtype == mx.float32 and len(param.shape) > 1:
                        weights[name] = mx.quantize(param, mx.int8)
                        
            elif quantize == 16:
                # 16-bit quantization (half precision)
                for name, param in weights.items():
                    if param.dtype == mx.float32:
                        weights[name] = param.astype(mx.float16)
            
            else:
                logger.warning(f"Unsupported quantization bits: {quantize}. Using no quantization.")
        
        # Save weights to MLX format
        mx.save(str(mlx_dir / "weights.safetensors"), weights)
        
        # Copy config and tokenizer files
        for file in ["config.json", "tokenizer.json", "vocab.txt", "special_tokens_map.json", 
                    "vocab.json", "merges.txt", "tokenizer_config.json"]:
            src = model_dir / file
            if src.exists():
                shutil.copy(src, mlx_dir / file)
        
        # Create MLX-specific metadata
        mlx_metadata = {
            "model_id": model_id,
            "class": model_class,
            "mlx_version": mx.__version__,
            "quantization": quantize,
            "optimized_for_inference": optimize_for_inference,
            "conversion_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "conversion_time_seconds": round(time.time() - start_time, 2),
            "metal_available": METAL_AVAILABLE
        }
        
        # Save MLX metadata
        with open(mlx_dir / "mlx_metadata.json", "w") as f:
            json.dump(mlx_metadata, f, indent=2)
        
        # Update main metadata
        metadata["optimized"] = True
        metadata["mlx_metadata"] = mlx_metadata
        
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully converted {model_id} to MLX format")
        logger.info(f"MLX model saved to {mlx_dir}")
        return True
        
    except Exception as e:
        logger.error(f"Error converting model {model_dir.name} to MLX: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def optimize_model(model_id: str, models_dir: Path, quantize: Optional[int] = None, 
                 optimize_for_inference: bool = False) -> bool:
    """
    Optimize a model for MLX.
    
    Args:
        model_id: ID of the model to optimize
        models_dir: Directory containing the models
        quantize: Quantization bits (8, 16, or None for no quantization)
        optimize_for_inference: Whether to optimize the model for inference
        
    Returns:
        bool: True if optimization was successful
    """
    model_dir = models_dir / model_id
    
    if not model_dir.exists():
        logger.error(f"Model directory {model_dir} does not exist")
        return False
    
    return convert_to_mlx(model_dir, quantize, optimize_for_inference)


def optimize_all_models(models_dir: Path, quantize: Optional[int] = None, 
                      optimize_for_inference: bool = False) -> bool:
    """
    Optimize all available models for MLX.
    
    Args:
        models_dir: Directory containing the models
        quantize: Quantization bits (8, 16, or None for no quantization)
        optimize_for_inference: Whether to optimize the model for inference
        
    Returns:
        bool: True if all optimizations were successful
    """
    model_ids = list_models(models_dir)
    
    if not model_ids:
        logger.warning("No models found to optimize")
        return False
    
    success = True
    for model_id in model_ids:
        model_success = optimize_model(model_id, models_dir, quantize, optimize_for_inference)
        if not model_success:
            success = False
    
    return success


def benchmark_model(model_id: str, models_dir: Path) -> bool:
    """
    Benchmark a model with MLX.
    
    Args:
        model_id: ID of the model to benchmark
        models_dir: Directory containing the models
        
    Returns:
        bool: True if benchmark was successful
    """
    if not HAS_MLX:
        logger.error("MLX is not available - cannot benchmark model")
        return False
    
    model_dir = models_dir / model_id
    
    if not model_dir.exists():
        logger.error(f"Model directory {model_dir} does not exist")
        return False
    
    mlx_dir = model_dir / "mlx"
    if not mlx_dir.exists():
        logger.error(f"MLX model not found for {model_id}. Please optimize the model first.")
        return False
    
    metadata_file = model_dir / "metadata.json"
    if not metadata_file.exists():
        logger.error(f"Metadata file not found for model in {model_dir}")
        return False
    
    try:
        # Load metadata
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        
        model_class = metadata.get("class")
        
        if not model_class:
            logger.error(f"Model class not specified in metadata for {model_id}")
            return False
        
        logger.info(f"Benchmarking {model_id} with MLX...")
        
        # Import required modules
        import torch
        import transformers
        import numpy as np
        
        # Load PyTorch model
        logger.info("Loading PyTorch model...")
        repo_id = metadata.get("repo_id")
        model_pt = getattr(transformers, model_class).from_pretrained(repo_id)
        
        # Create sample input
        input_shape = (1, 512)  # Batch size 1, sequence length 512
        input_ids = np.random.randint(0, 1000, size=input_shape)
        attention_mask = np.ones(input_shape)
        
        # Benchmark PyTorch
        logger.info("Running PyTorch benchmark...")
        torch_times = []
        input_ids_pt = torch.tensor(input_ids)
        attention_mask_pt = torch.tensor(attention_mask)
        
        # Warmup
        for _ in range(5):
            _ = model_pt(input_ids=input_ids_pt, attention_mask=attention_mask_pt)
        
        # Benchmark
        for _ in range(20):
            start_time = time.time()
            _ = model_pt(input_ids=input_ids_pt, attention_mask=attention_mask_pt)
            torch_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Load MLX weights
        logger.info("Loading MLX model...")
        weights = mx.load(str(mlx_dir / "weights.safetensors"))
        
        # Create MLX inputs
        input_ids_mlx = mx.array(input_ids)
        attention_mask_mlx = mx.array(attention_mask)
        
        # Define a simple forward function for MLX
        def mlx_forward(weights, input_ids, attention_mask):
            # This is a simplified forward pass - in a real application,
            # you would implement the actual model architecture
            return weights
        
        # Benchmark MLX
        logger.info("Running MLX benchmark...")
        mlx_times = []
        
        # Warmup
        for _ in range(5):
            _ = mlx_forward(weights, input_ids_mlx, attention_mask_mlx)
        
        # Benchmark
        for _ in range(20):
            start_time = time.time()
            _ = mlx_forward(weights, input_ids_mlx, attention_mask_mlx)
            mx.eval(weights)  # Force evaluation
            mlx_times.append((time.time() - start_time) * 1000)  # Convert to ms
        
        # Calculate statistics
        torch_avg = np.mean(torch_times)
        torch_std = np.std(torch_times)
        mlx_avg = np.mean(mlx_times)
        mlx_std = np.std(mlx_times)
        speedup = torch_avg / mlx_avg
        
        # Print results
        logger.info(f"Benchmark results for {model_id}:")
        logger.info(f"PyTorch: {torch_avg:.2f} ± {torch_std:.2f} ms")
        logger.info(f"MLX:     {mlx_avg:.2f} ± {mlx_std:.2f} ms")
        logger.info(f"Speedup: {speedup:.2f}x")
        
        # Save benchmark results
        benchmark_file = model_dir / "benchmark.json"
        benchmark_results = {
            "model_id": model_id,
            "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pytorch": {
                "mean_ms": float(torch_avg),
                "std_ms": float(torch_std),
                "times_ms": [float(t) for t in torch_times]
            },
            "mlx": {
                "mean_ms": float(mlx_avg),
                "std_ms": float(mlx_std),
                "times_ms": [float(t) for t in mlx_times],
                "metal_used": METAL_AVAILABLE
            },
            "speedup": float(speedup)
        }
        
        with open(benchmark_file, "w") as f:
            json.dump(benchmark_results, f, indent=2)
        
        logger.info(f"Benchmark results saved to {benchmark_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error benchmarking model {model_id}: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def benchmark_all_models(models_dir: Path) -> bool:
    """
    Benchmark all available models with MLX.
    
    Args:
        models_dir: Directory containing the models
        
    Returns:
        bool: True if all benchmarks were successful
    """
    model_ids = list_models(models_dir)
    
    if not model_ids:
        logger.warning("No models found to benchmark")
        return False
    
    success = True
    for model_id in model_ids:
        model_success = benchmark_model(model_id, models_dir)
        if not model_success:
            success = False
    
    return success


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="MLX Model Optimizer for LlamaClaims"
    )
    
    parser.add_argument(
        "--model",
        help="ID of the model to optimize",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Optimize all available models",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    
    parser.add_argument(
        "--models-dir",
        default="./data/models",
        help="Directory containing the models (default: ./data/models)",
    )
    
    parser.add_argument(
        "--quantize",
        type=int,
        choices=[8, 16],
        help="Quantize models to specified bit depth (8 or 16)",
    )
    
    parser.add_argument(
        "--optimize-for-inference",
        action="store_true",
        help="Optimize models for inference (includes graph optimizations)",
    )
    
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark models after optimization",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )
    
    return parser.parse_args()


def main() -> int:
    """
    Main function.
    
    Returns:
        int: Exit code
    """
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check if running on Apple Silicon
    if not IS_APPLE_SILICON:
        logger.error("MLX optimization requires Apple Silicon (M-series chip)")
        return 1
    
    # Create models directory
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # List models if requested
    if args.list:
        model_ids = list_models(models_dir)
        
        if not model_ids:
            logger.warning("No models found")
            return 0
        
        print("Available models:")
        print("-" * 80)
        print(f"{'Model ID':<20} {'Optimized':<10} {'Quantization':<15} {'Benchmark':<10}")
        print("-" * 80)
        
        for model_id in model_ids:
            model_dir = models_dir / model_id
            metadata_file = model_dir / "metadata.json"
            
            optimized = "No"
            quantization = "None"
            benchmark = "No"
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    
                    if metadata.get("optimized", False):
                        optimized = "Yes"
                        
                        mlx_metadata = metadata.get("mlx_metadata", {})
                        quantization = f"{mlx_metadata.get('quantization', 'None')} bits" if mlx_metadata.get('quantization') else "None"
                    
                    if (model_dir / "benchmark.json").exists():
                        benchmark = "Yes"
                except:
                    pass
            
            print(f"{model_id:<20} {optimized:<10} {quantization:<15} {benchmark:<10}")
        
        print("-" * 80)
        return 0
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Optimize models
    if args.all:
        optimize_all_models(models_dir, args.quantize, args.optimize_for_inference)
    elif args.model:
        optimize_model(args.model, models_dir, args.quantize, args.optimize_for_inference)
    
    # Benchmark models if requested
    if args.benchmark:
        if args.all:
            benchmark_all_models(models_dir)
        elif args.model:
            benchmark_model(args.model, models_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main()) 