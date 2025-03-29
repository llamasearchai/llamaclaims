#!/usr/bin/env python3
"""
LlamaClaims Model Downloader

This script downloads and prepares the ML models needed for LlamaClaims.
It supports downloading from Hugging Face and local model repositories.
"""

import os
import sys
import argparse
import logging
import json
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import shutil
import time
import platform
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("model-downloader")

# Model definitions - mapping model IDs to their HuggingFace repos
MODEL_DEFINITIONS = {
    "document-classifier": {
        "repo_id": "microsoft/layoutlmv3-base",
        "description": "Document classification model based on LayoutLMv3",
        "size_mb": 350,
        "files": ["config.json", "pytorch_model.bin", "tokenizer.json", "special_tokens_map.json", "vocab.txt"],
        "class": "LayoutLMv3ForSequenceClassification"
    },
    "document-extractor": {
        "repo_id": "microsoft/layoutlmv3-base",
        "description": "Document information extraction model based on LayoutLMv3",
        "size_mb": 420,
        "files": ["config.json", "pytorch_model.bin", "tokenizer.json", "special_tokens_map.json", "vocab.txt"],
        "class": "LayoutLMv3ForTokenClassification"
    },
    "claims-classifier": {
        "repo_id": "distilbert-base-uncased",
        "description": "Claims classification model based on DistilBERT",
        "size_mb": 260,
        "files": ["config.json", "pytorch_model.bin", "tokenizer.json", "vocab.txt"],
        "class": "DistilBertForSequenceClassification"
    },
    "fraud-detector": {
        "repo_id": "roberta-base",
        "description": "Fraud detection model based on RoBERTa",
        "size_mb": 480,
        "files": ["config.json", "pytorch_model.bin", "tokenizer.json", "vocab.json", "merges.txt"],
        "class": "RobertaForSequenceClassification"
    },
    "claims-llm": {
        "repo_id": "mistralai/Mistral-7B-v0.1",
        "description": "Claims analysis LLM based on Mistral-7B",
        "size_mb": 4200,
        "files": ["config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"],
        "class": "AutoModelForCausalLM",
        "quantized": True
    }
}

def check_dependencies() -> bool:
    """
    Check if required dependencies are installed.
    
    Returns:
        bool: True if all dependencies are satisfied
    """
    try:
        import torch
        import transformers
        from huggingface_hub import hf_hub_download
        
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        
        # Check if running on Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            logger.info("Running on Apple Silicon - MLX optimization available")
        
        return True
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        logger.error("Please install required packages: pip install -r requirements.txt")
        return False

def download_model(model_id: str, output_dir: str, force: bool = False) -> bool:
    """
    Download a model from Hugging Face.
    
    Args:
        model_id: ID of the model to download
        output_dir: Directory to save the model to
        force: Whether to force download even if the model already exists
        
    Returns:
        bool: True if successful, False otherwise
    """
    if model_id not in MODEL_DEFINITIONS:
        logger.error(f"Unknown model ID: {model_id}")
        return False
    
    model_def = MODEL_DEFINITIONS[model_id]
    repo_id = model_def["repo_id"]
    model_dir = os.path.join(output_dir, model_id)
    
    # Check if model already exists
    if os.path.exists(model_dir) and not force:
        logger.info(f"Model {model_id} already exists at {model_dir}")
        return True
    
    # Create model directory
    os.makedirs(model_dir, exist_ok=True)
    
    # Download model files
    try:
        logger.info(f"Downloading model {model_id} from {repo_id}...")
        
        # Download configuration
        logger.info("Downloading config.json...")
        config_path = hf_hub_download(repo_id=repo_id, filename="config.json")
        config_size = os.path.getsize(config_path) / (1024 * 1024)
        logger.info(f"Downloaded config.json ({config_size:.2f} MB)")
        shutil.copy(config_path, os.path.join(model_dir, "config.json"))
        
        # Download model weights
        logger.info("Downloading pytorch_model.bin...")
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Downloaded pytorch_model.bin ({model_size:.2f} MB)")
        shutil.copy(model_path, os.path.join(model_dir, "pytorch_model.bin"))
        
        # Try to download tokenizer files
        try:
            logger.info("Downloading tokenizer.json...")
            tokenizer_path = hf_hub_download(repo_id=repo_id, filename="tokenizer.json")
            tokenizer_size = os.path.getsize(tokenizer_path) / (1024 * 1024)
            logger.info(f"Downloaded tokenizer.json ({tokenizer_size:.2f} MB)")
            shutil.copy(tokenizer_path, os.path.join(model_dir, "tokenizer.json"))
        except Exception as e:
            logger.warning(f"Tokenizer.json not found, trying tokenizer_config.json instead: {str(e)}")
            try:
                logger.info("Downloading tokenizer_config.json...")
                tokenizer_config_path = hf_hub_download(repo_id=repo_id, filename="tokenizer_config.json")
                tokenizer_config_size = os.path.getsize(tokenizer_config_path) / (1024 * 1024)
                logger.info(f"Downloaded tokenizer_config.json ({tokenizer_config_size:.2f} MB)")
                shutil.copy(tokenizer_config_path, os.path.join(model_dir, "tokenizer_config.json"))
            except Exception as e2:
                logger.warning(f"Tokenizer_config.json not found: {str(e2)}")
                logger.warning("Continuing without tokenizer files...")
        
        # Try to download special tokens map
        try:
            logger.info("Downloading special_tokens_map.json...")
            special_tokens_path = hf_hub_download(repo_id=repo_id, filename="special_tokens_map.json")
            special_tokens_size = os.path.getsize(special_tokens_path) / (1024 * 1024)
            logger.info(f"Downloaded special_tokens_map.json ({special_tokens_size:.2f} MB)")
            shutil.copy(special_tokens_path, os.path.join(model_dir, "special_tokens_map.json"))
        except Exception as e:
            logger.warning(f"Special tokens map not found: {str(e)}")
            logger.warning("Continuing without special tokens map...")
        
        # Create metadata file
        metadata = {
            "id": model_id,
            "name": model_def.get("name", model_id),
            "description": model_def.get("description", ""),
            "repo_id": repo_id,
            "size_mb": model_size,
            "downloaded_at": datetime.now().isoformat(),
            "files": [
                "config.json",
                "pytorch_model.bin"
            ],
            "optimized": False,
            "optimized_formats": []
        }
        
        # Add tokenizer files to metadata if they exist
        if os.path.exists(os.path.join(model_dir, "tokenizer.json")):
            metadata["files"].append("tokenizer.json")
        if os.path.exists(os.path.join(model_dir, "tokenizer_config.json")):
            metadata["files"].append("tokenizer_config.json")
        if os.path.exists(os.path.join(model_dir, "special_tokens_map.json")):
            metadata["files"].append("special_tokens_map.json")
        
        # Write metadata
        with open(os.path.join(model_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Successfully downloaded model {model_id} to {model_dir}")
        return True
    
    except Exception as e:
        logger.error(f"Error downloading model {model_id}: {str(e)}")
        return False

def download_all_models(output_dir: Path, force: bool = False) -> bool:
    """
    Download all defined models.
    
    Args:
        output_dir: Directory to save the models to
        force: Force redownload even if files exist
        
    Returns:
        bool: True if all downloads were successful
    """
    success = True
    for model_id in MODEL_DEFINITIONS:
        model_success = download_model(model_id, output_dir, force)
        if not model_success:
            success = False
    
    return success

def list_models(output_dir: Optional[Path] = None) -> None:
    """
    List available models and their download status.
    
    Args:
        output_dir: Optional directory to check download status
    """
    print("Available models:")
    print("-" * 80)
    print(f"{'Model ID':<20} {'Description':<40} {'Size':<10} {'Status':<10}")
    print("-" * 80)
    
    for model_id, model_def in MODEL_DEFINITIONS.items():
        status = "Not downloaded"
        
        if output_dir:
            metadata_file = output_dir / model_id / "metadata.json"
            if metadata_file.exists():
                try:
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    
                    if metadata.get("optimized", False):
                        status = "Optimized"
                    else:
                        status = "Downloaded"
                except:
                    status = "Error"
        
        print(f"{model_id:<20} {model_def['description']:<40} {model_def['size_mb']:<10}MB {status:<10}")
    
    print("-" * 80)

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="LlamaClaims Model Downloader - Download and prepare models for LlamaClaims"
    )
    
    parser.add_argument(
        "--model",
        help="ID of the model to download (default: all models)",
    )
    
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download all available models",
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models",
    )
    
    parser.add_argument(
        "--output-dir",
        default="./data/models",
        help="Directory to save models to (default: ./data/models)",
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force redownload even if model is already downloaded",
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
        Exit code
    """
    args = parse_args()
    
    # Configure logging level
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List models if requested
    if args.list:
        list_models(output_dir)
        return 0
    
    # Check dependencies
    if not check_dependencies():
        return 1
    
    # Download models
    if args.all:
        success = download_all_models(output_dir, args.force)
    elif args.model:
        success = download_model(args.model, output_dir, args.force)
    else:
        list_models(output_dir)
        logger.info("No model specified. Use --model <model_id> or --all to download models.")
        return 0
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 