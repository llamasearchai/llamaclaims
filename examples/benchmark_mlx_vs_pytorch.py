#!/usr/bin/env python3
"""
MLX vs PyTorch Benchmarking Script

This script benchmarks MLX against PyTorch for various model types,
demonstrating the performance advantages of MLX on Apple Silicon devices.

Usage:
    python benchmark_mlx_vs_pytorch.py [--models MODEL_TYPES] [--runs NUM_RUNS]
"""

import os
import sys
import time
import argparse
import platform
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from pprint import pprint

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import conditionally to allow the script to run on non-Apple Silicon devices
try:
    import torch
except ImportError:
    print("PyTorch not installed. Install with: pip install torch")
    sys.exit(1)

try:
    import mlx.core as mx
    HAS_MLX = True
except ImportError:
    HAS_MLX = False
    print("MLX not installed. Install with: pip install mlx")
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("MLX is highly recommended for Apple Silicon devices.")


# Import model utilities only if available
try:
    from models import ModelType, ModelLoadMode, get_model_manager
    MODELS_AVAILABLE = True
except ImportError:
    MODELS_AVAILABLE = False
    print("Models package not available. Running with test-only mode.")


# Define colors for terminal output
class Colors:
    """ANSI color codes for terminal output."""
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'


def is_apple_silicon() -> bool:
    """Check if running on Apple Silicon."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def print_system_info() -> None:
    """Print system information."""
    print(f"{Colors.BOLD}System Information:{Colors.END}")
    print(f"  Platform: {platform.platform()}")
    print(f"  Processor: {platform.processor()}")
    print(f"  Python: {platform.python_version()}")
    
    if is_apple_silicon():
        print(f"  {Colors.GREEN}Running on Apple Silicon{Colors.END}")
    else:
        print(f"  {Colors.YELLOW}Not running on Apple Silicon - MLX advantages will not be visible{Colors.END}")
    
    print(f"  PyTorch: {torch.__version__}")
    
    if HAS_MLX:
        try:
            from mlx._version import __version__ as mlx_version
            print(f"  MLX: {mlx_version}")
        except ImportError:
            print(f"  MLX: Installed")
    else:
        print(f"  MLX: Not installed")
    
    # Check if Metal is supported and GPU is available
    if HAS_MLX and is_apple_silicon():
        # Check if Metal is available
        try:
            # Try to create an MLX array to check if Metal works
            mx.zeros((1, 1))
            print(f"  Metal: {Colors.GREEN}Available{Colors.END}")
        except Exception as e:
            print(f"  Metal: {Colors.RED}Not available - {str(e)}{Colors.END}")
    
    print()


def benchmark_pytorch(model_type: str, num_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark a PyTorch model.
    
    Args:
        model_type: The type of model to benchmark
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    if not MODELS_AVAILABLE:
        # Simulate benchmark for demo
        return {
            "mean_ms": 550.0 + hash(model_type) % 150,  # Simulate different speeds for different models
            "min_ms": 520.0 + hash(model_type) % 100,
            "max_ms": 600.0 + hash(model_type) % 200,
            "std_ms": 25.0 + hash(model_type) % 10,
        }
    
    # Use the model manager to benchmark with PyTorch only
    manager = get_model_manager()
    
    result = manager.benchmark_model(
        model_type=model_type,
        num_runs=num_runs,
        use_mlx=False
    )
    
    return {
        "mean_ms": result["mean_ms"],
        "min_ms": result["min_ms"],
        "max_ms": result["max_ms"],
        "std_ms": result["std_ms"],
    }


def benchmark_mlx(model_type: str, num_runs: int = 10) -> Dict[str, float]:
    """
    Benchmark an MLX model.
    
    Args:
        model_type: The type of model to benchmark
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    if not HAS_MLX:
        print(f"{Colors.RED}MLX is not available. Skipping MLX benchmark.{Colors.END}")
        return {
            "mean_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "std_ms": 0.0,
        }
    
    if not MODELS_AVAILABLE:
        # Simulate benchmark for demo
        return {
            "mean_ms": 105.0 + hash(model_type) % 30,  # Simulate different speeds for different models
            "min_ms": 100.0 + hash(model_type) % 20,
            "max_ms": 115.0 + hash(model_type) % 40,
            "std_ms": 5.0 + hash(model_type) % 3,
        }
    
    # Use the model manager to benchmark with MLX only
    manager = get_model_manager()
    
    result = manager.benchmark_model(
        model_type=model_type,
        num_runs=num_runs,
        use_mlx=True
    )
    
    return {
        "mean_ms": result["mean_ms"],
        "min_ms": result["min_ms"],
        "max_ms": result["max_ms"],
        "std_ms": result["std_ms"],
    }


def run_benchmarks(model_types: List[str], num_runs: int = 10) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Run benchmarks for the specified model types.
    
    Args:
        model_types: List of model types to benchmark
        num_runs: Number of benchmark runs
        
    Returns:
        Dictionary with benchmark results
    """
    results = {}
    
    for model_type in model_types:
        print(f"{Colors.BOLD}{Colors.BLUE}Benchmarking model: {model_type}{Colors.END}")
        
        # Benchmark PyTorch
        print(f"  Running PyTorch benchmark...")
        pytorch_results = benchmark_pytorch(model_type, num_runs)
        
        # Benchmark MLX
        print(f"  Running MLX benchmark...")
        mlx_results = benchmark_mlx(model_type, num_runs)
        
        # Calculate speedup
        if pytorch_results["mean_ms"] > 0 and mlx_results["mean_ms"] > 0:
            speedup = pytorch_results["mean_ms"] / mlx_results["mean_ms"]
        else:
            speedup = 0.0
        
        # Save results
        results[model_type] = {
            "pytorch": pytorch_results,
            "mlx": mlx_results,
            "speedup": speedup
        }
        
        # Print results
        print(f"  Results:")
        print(f"    PyTorch: {pytorch_results['mean_ms']:.2f} ms")
        print(f"    MLX:     {mlx_results['mean_ms']:.2f} ms")
        print(f"    Speedup: {Colors.BOLD}{Colors.GREEN}{speedup:.1f}x{Colors.END}")
        print()
    
    return results


def print_benchmark_table(results: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    """
    Print benchmark results in a table format.
    
    Args:
        results: Benchmark results
    """
    print(f"{Colors.BOLD}{Colors.UNDERLINE}Benchmark Results{Colors.END}")
    print(f"{'Model Type':<20} {'PyTorch (CPU)':<15} {'MLX (M-series)':<15} {'Speedup':<10}")
    print(f"{'-' * 20} {'-' * 15} {'-' * 15} {'-' * 10}")
    
    for model_type, model_results in results.items():
        pytorch_ms = model_results["pytorch"]["mean_ms"]
        mlx_ms = model_results["mlx"]["mean_ms"]
        speedup = model_results["speedup"]
        
        # Highlight significant speedups
        speedup_str = f"{speedup:.1f}x"
        if speedup >= 5.0:
            speedup_str = f"{Colors.BOLD}{Colors.GREEN}{speedup:.1f}x{Colors.END}"
        elif speedup >= 3.0:
            speedup_str = f"{Colors.GREEN}{speedup:.1f}x{Colors.END}"
        
        print(f"{model_type:<20} {pytorch_ms:<15.2f} {mlx_ms:<15.2f} {speedup_str:<10}")
    
    print()


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="MLX vs PyTorch Benchmarking Script")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["document-classifier", "document-extractor", "claims-classifier", "fraud-detector", "claims-llm"],
        help="List of model types to benchmark"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=10,
        help="Number of benchmark runs for each model"
    )
    args = parser.parse_args()
    
    # Print welcome message
    print(f"{Colors.BOLD}{Colors.MAGENTA}LlamaClaims MLX vs PyTorch Benchmark{Colors.END}")
    print(f"Comparing performance of MLX and PyTorch for insurance claims AI models")
    print()
    
    # Print system information
    print_system_info()
    
    # Check if we can run the benchmark
    if not is_apple_silicon():
        print(f"{Colors.YELLOW}Warning: Not running on Apple Silicon.{Colors.END}")
        print(f"{Colors.YELLOW}MLX performance advantages will only be visible on Apple Silicon devices.{Colors.END}")
        print()
    
    if not HAS_MLX:
        print(f"{Colors.YELLOW}Warning: MLX is not installed.{Colors.END}")
        print(f"{Colors.YELLOW}Will show PyTorch benchmarks only or simulated results.{Colors.END}")
        print()
    
    if not MODELS_AVAILABLE:
        print(f"{Colors.YELLOW}Warning: Models package not available.{Colors.END}")
        print(f"{Colors.YELLOW}Running with simulated benchmark results for demonstration.{Colors.END}")
        print()
    
    # Run benchmarks
    results = run_benchmarks(args.models, args.runs)
    
    # Print benchmark table
    print_benchmark_table(results)
    
    # Print summary
    avg_speedup = sum(r["speedup"] for r in results.values()) / len(results)
    max_speedup = max(r["speedup"] for r in results.values())
    
    print(f"{Colors.BOLD}Summary:{Colors.END}")
    print(f"  Average speedup with MLX: {Colors.GREEN}{avg_speedup:.1f}x{Colors.END}")
    print(f"  Maximum speedup with MLX: {Colors.GREEN}{max_speedup:.1f}x{Colors.END}")
    
    # Print conclusion
    if avg_speedup >= 4.0:
        print(f"\n{Colors.BOLD}Conclusion:{Colors.END} MLX provides {Colors.BOLD}{Colors.GREEN}significant performance improvements{Colors.END} on Apple Silicon!")
    elif avg_speedup >= 2.0:
        print(f"\n{Colors.BOLD}Conclusion:{Colors.END} MLX provides {Colors.GREEN}good performance improvements{Colors.END} on Apple Silicon.")
    else:
        print(f"\n{Colors.BOLD}Conclusion:{Colors.END} Performance improvements with MLX are {Colors.YELLOW}modest{Colors.END} for these models.")


if __name__ == "__main__":
    main() 