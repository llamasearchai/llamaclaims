#!/usr/bin/env python3
"""
MLX Model Wrapper

This module provides a wrapper class for MLX models that standardizes
the interface for model loading, inference, and optimization.
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import platform
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("mlx-wrapper")

# Check if running on Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Import MLX if available (only on Apple Silicon)
if IS_APPLE_SILICON:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        from mlx.utils import tree_unflatten
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
        logger.warning("MLX not found - running without GPU acceleration")
else:
    HAS_MLX = False
    logger.warning("Not running on Apple Silicon - MLX not available")


class MLXModelWrapper:
    """
    A wrapper for MLX models providing a standardized interface.
    
    This class handles loading MLX models, running inference, and managing
    the model's weights and tokenizer.
    """
    
    is_mlx_wrapper = True
    
    def __init__(
        self,
        model_id: str,
        model_class: str,
        weights: Dict[str, Any],
        config_path: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
    ):
        """
        Initialize the MLX model wrapper.
        
        Args:
            model_id: ID of the model
            model_class: Class name of the model (from transformers)
            weights: MLX weights dictionary
            config_path: Path to the model config file
            tokenizer_path: Path to the tokenizer files
        """
        if not HAS_MLX:
            raise ImportError("MLX is not available. Cannot use MLX models.")
        
        self.model_id = model_id
        self.model_class = model_class
        self.weights = weights
        self.config_path = config_path
        self.tokenizer_path = tokenizer_path
        
        # Load config and tokenizer if available
        self._load_config_and_tokenizer()
        
        # Set up input processing based on model class
        self._setup_processing()
    
    def _load_config_and_tokenizer(self):
        """Load the model config and tokenizer if available."""
        self.config = None
        self.tokenizer = None
        
        try:
            from transformers import AutoConfig, AutoTokenizer
            
            # Load config
            if self.config_path:
                logger.info(f"Loading config from {self.config_path}")
                self.config = AutoConfig.from_pretrained(self.config_path)
            
            # Load tokenizer
            if self.tokenizer_path:
                logger.info(f"Loading tokenizer from {self.tokenizer_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        except ImportError:
            logger.warning("Transformers not available. Running without config and tokenizer.")
        except Exception as e:
            logger.error(f"Error loading config or tokenizer: {e}")
            import traceback
            logger.error(traceback.format_exc())
    
    def _setup_processing(self):
        """Set up input processing based on model type."""
        # Extract model type from the class name
        self.input_processor = None
        self.output_processor = None
        
        # Set up processors based on model class
        if "LayoutLM" in self.model_class:
            self._setup_layoutlm_processors()
        elif "Bert" in self.model_class:
            self._setup_bert_processors()
        elif "Roberta" in self.model_class:
            self._setup_roberta_processors()
        elif "CausalLM" in self.model_class:
            self._setup_causal_lm_processors()
    
    def _setup_layoutlm_processors(self):
        """Set up processors for LayoutLM models."""
        def process_layoutlm_input(inputs):
            # Convert input tensors to MLX arrays
            processed = {}
            for key, value in inputs.items():
                if key in ["input_ids", "attention_mask", "token_type_ids"]:
                    processed[key] = mx.array(value)
            return processed
        
        def process_layoutlm_output(outputs):
            if "SequenceClassification" in self.model_class:
                # Classification output
                logits = outputs["logits"]
                # Apply softmax
                probs = mx.softmax(logits, axis=-1)
                # Convert to Python types
                return {
                    "logits": logits.tolist(),
                    "probs": probs.tolist(),
                    "predicted_class": mx.argmax(probs, axis=-1).tolist()
                }
            elif "TokenClassification" in self.model_class:
                # Token classification output
                logits = outputs["logits"]
                # Apply softmax
                probs = mx.softmax(logits, axis=-1)
                # Convert to Python types
                return {
                    "logits": logits.tolist(),
                    "probs": probs.tolist(),
                    "predicted_class": mx.argmax(probs, axis=-1).tolist()
                }
            return outputs
        
        self.input_processor = process_layoutlm_input
        self.output_processor = process_layoutlm_output
    
    def _setup_bert_processors(self):
        """Set up processors for BERT models."""
        def process_bert_input(inputs):
            # Convert input tensors to MLX arrays
            processed = {}
            for key, value in inputs.items():
                if key in ["input_ids", "attention_mask", "token_type_ids"]:
                    processed[key] = mx.array(value)
            return processed
        
        def process_bert_output(outputs):
            if "SequenceClassification" in self.model_class:
                # Classification output
                logits = outputs["logits"]
                # Apply softmax
                probs = mx.softmax(logits, axis=-1)
                # Convert to Python types
                return {
                    "logits": logits.tolist(),
                    "probs": probs.tolist(),
                    "predicted_class": mx.argmax(probs, axis=-1).tolist()
                }
            return outputs
        
        self.input_processor = process_bert_input
        self.output_processor = process_bert_output
    
    def _setup_roberta_processors(self):
        """Set up processors for RoBERTa models."""
        def process_roberta_input(inputs):
            # Convert input tensors to MLX arrays
            processed = {}
            for key, value in inputs.items():
                if key in ["input_ids", "attention_mask"]:
                    processed[key] = mx.array(value)
            return processed
        
        def process_roberta_output(outputs):
            if "SequenceClassification" in self.model_class:
                # Classification output
                logits = outputs["logits"]
                # Apply softmax
                probs = mx.softmax(logits, axis=-1)
                # Convert to Python types
                return {
                    "logits": logits.tolist(),
                    "probs": probs.tolist(),
                    "predicted_class": mx.argmax(probs, axis=-1).tolist()
                }
            return outputs
        
        self.input_processor = process_roberta_input
        self.output_processor = process_roberta_output
    
    def _setup_causal_lm_processors(self):
        """Set up processors for Causal LM models."""
        def process_causal_lm_input(inputs):
            # Convert input tensors to MLX arrays
            processed = {}
            for key, value in inputs.items():
                if key in ["input_ids", "attention_mask"]:
                    processed[key] = mx.array(value)
            return processed
        
        def process_causal_lm_output(outputs):
            # For generation, return the logits and tokens
            if "logits" in outputs:
                logits = outputs["logits"]
                # For text generation, we might get the generated text
                return {
                    "logits": logits.tolist(),
                    "next_token_logits": logits[:, -1, :].tolist()
                }
            return outputs
        
        self.input_processor = process_causal_lm_input
        self.output_processor = process_causal_lm_output
    
    def predict(self, inputs: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Run inference with the MLX model.
        
        Args:
            inputs: Dictionary of input tensors
            **kwargs: Additional keyword arguments for inference
            
        Returns:
            Dictionary of output tensors
        """
        if not HAS_MLX:
            return {"error": "MLX is not available"}
        
        try:
            # Process inputs
            if self.input_processor is not None:
                processed_inputs = self.input_processor(inputs)
            else:
                # Default processing
                processed_inputs = {}
                for key, value in inputs.items():
                    processed_inputs[key] = mx.array(value)
            
            # Run forward pass
            outputs = self._forward(processed_inputs)
            
            # Process outputs
            if self.output_processor is not None:
                return self.output_processor(outputs)
            else:
                # Default processing
                result = {}
                for key, value in outputs.items():
                    if hasattr(value, "tolist"):
                        result[key] = value.tolist()
                    else:
                        result[key] = value
                return result
                
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {"error": str(e)}
    
    def _forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the forward pass through the model.
        
        This is a simplified implementation that assumes the model follows
        a standard transformer architecture. For complex models, this might
        need to be overridden.
        
        Args:
            inputs: Dictionary of input tensors
            
        Returns:
            Dictionary of output tensors
        """
        # Extract input tensors
        input_ids = inputs.get("input_ids")
        attention_mask = inputs.get("attention_mask")
        
        if input_ids is None:
            raise ValueError("input_ids is required")
        
        # Prepare a state dict from the flattened weights
        # This is a simplified approach that won't work for all model types
        if "SequenceClassification" in self.model_class:
            # Extract flattened weights
            embeddings = self._extract_embeddings(self.weights)
            
            # Simple linear model for demonstration
            logits = mx.matmul(input_ids.reshape(-1, input_ids.shape[-1]), embeddings)
            
            return {"logits": logits}
        
        elif "TokenClassification" in self.model_class:
            # Similar approach for token classification
            embeddings = self._extract_embeddings(self.weights)
            
            # Simplified token classification
            logits = mx.matmul(input_ids.reshape(-1, input_ids.shape[-1]), embeddings)
            logits = logits.reshape(input_ids.shape[0], input_ids.shape[1], -1)
            
            return {"logits": logits}
        
        elif "CausalLM" in self.model_class:
            # For causal LM, we would need a more complex architecture
            # This is just a placeholder
            embeddings = self._extract_embeddings(self.weights)
            
            # Simplified language model
            logits = mx.matmul(input_ids.reshape(-1, input_ids.shape[-1]), embeddings)
            logits = logits.reshape(input_ids.shape[0], input_ids.shape[1], -1)
            
            return {"logits": logits}
        
        else:
            # For other model types
            logger.warning(f"Unsupported model class: {self.model_class}")
            # Return dummy output
            return {"logits": mx.zeros((input_ids.shape[0], 2))}
    
    def _extract_embeddings(self, weights: Dict[str, Any]) -> mx.array:
        """
        Extract embeddings from the weights.
        
        This is a simplified version that won't work for all model types.
        
        Args:
            weights: Dictionary of model weights
            
        Returns:
            Embeddings matrix
        """
        # Look for embedding weights in the model
        embedding_keys = [key for key in weights.keys() if "embeddings" in key]
        
        if embedding_keys:
            # Use the first embedding layer found
            return weights[embedding_keys[0]]
        else:
            # If no embeddings found, use a dummy matrix
            logger.warning("No embedding weights found, using dummy matrix")
            return mx.ones((768, 2))  # Dummy matrix
    
    def tokenize(self, text: Union[str, List[str]]) -> Dict[str, List[int]]:
        """
        Tokenize text using the model's tokenizer.
        
        Args:
            text: Text or list of texts to tokenize
            
        Returns:
            Dictionary of tokenized inputs
        """
        if self.tokenizer is None:
            logger.warning("Tokenizer not available")
            return {"error": "Tokenizer not available"}
        
        try:
            # Tokenize input text
            tokenized = self.tokenizer(
                text,
                padding=True,
                truncation=True,
                return_tensors="pt"  # We'll convert from PyTorch format
            )
            
            # Convert to lists for JSON serialization
            result = {}
            for key, value in tokenized.items():
                result[key] = value.tolist()
            
            return result
        except Exception as e:
            logger.error(f"Error during tokenization: {e}")
            return {"error": str(e)}
    
    def get_config(self) -> Optional[Dict[str, Any]]:
        """
        Get the model configuration.
        
        Returns:
            Model configuration dictionary
        """
        if self.config is None:
            return None
        
        return self.config.to_dict()


# Function to create an MLX model wrapper
def create_mlx_model_wrapper(
    model_id: str,
    model_dir: Union[str, Path],
    model_class: Optional[str] = None,
) -> Optional[MLXModelWrapper]:
    """
    Create an MLX model wrapper from a model directory.
    
    Args:
        model_id: ID of the model
        model_dir: Directory containing the model
        model_class: Class name of the model (from transformers)
        
    Returns:
        MLX model wrapper if successful, None otherwise
    """
    if not HAS_MLX:
        logger.error("MLX is not available")
        return None
    
    model_dir = Path(model_dir)
    mlx_dir = model_dir / "mlx"
    
    if not mlx_dir.exists():
        logger.error(f"MLX directory {mlx_dir} does not exist")
        return None
    
    weights_file = mlx_dir / "weights.safetensors"
    if not weights_file.exists():
        logger.error(f"MLX weights file {weights_file} does not exist")
        return None
    
    # Get model class from metadata if not provided
    if model_class is None:
        metadata_file = model_dir / "metadata.json"
        if metadata_file.exists():
            try:
                with open(metadata_file, "r") as f:
                    metadata = json.load(f)
                model_class = metadata.get("class")
            except Exception as e:
                logger.error(f"Error reading metadata: {e}")
                return None
    
    if model_class is None:
        logger.error("Model class not specified")
        return None
    
    try:
        # Load MLX weights
        weights = mx.load(str(weights_file))
        
        # Create model wrapper
        return MLXModelWrapper(
            model_id=model_id,
            model_class=model_class,
            weights=weights,
            config_path=str(mlx_dir / "config.json") if (mlx_dir / "config.json").exists() else None,
            tokenizer_path=str(mlx_dir) if (mlx_dir / "tokenizer.json").exists() else None
        )
    except Exception as e:
        logger.error(f"Error creating MLX model wrapper: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None 