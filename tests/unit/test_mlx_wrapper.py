"""
Unit tests for the MLX model wrapper class.
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Skip these tests if MLX is not available
mlx_available = False
try:
    import mlx.core
    mlx_available = True
except ImportError:
    pass

# Skip these tests if not on Apple Silicon
is_apple_silicon = (
    sys.platform == "darwin" and 
    os.uname().machine == "arm64"
)

# Determine if these tests should be run based on environment
should_run_mlx_tests = mlx_available and is_apple_silicon

# Conditionally import MLXModelWrapper
if should_run_mlx_tests:
    from models.mlx_wrapper import MLXModelWrapper, create_mlx_model_wrapper


@pytest.mark.skipif(
    not should_run_mlx_tests, 
    reason="MLX tests require Apple Silicon and MLX package"
)
class TestMLXWrapper:
    """Test suite for MLXModelWrapper."""
    
    def test_initialization(self):
        """Test initializing the MLX wrapper."""
        # Create a mock weights dictionary
        weights = {
            "weight1": MagicMock(),
            "weight2": MagicMock()
        }
        
        # Create a wrapper instance
        wrapper = MLXModelWrapper(
            model_id="test-model",
            model_class="TestModel",
            weights=weights
        )
        
        # Check basic properties
        assert wrapper.model_id == "test-model"
        assert wrapper.model_class == "TestModel"
        assert wrapper.weights is weights
        assert wrapper.is_mlx_wrapper == True
    
    @patch('models.mlx_wrapper.mx.load')
    @patch('models.mlx_wrapper.AutoTokenizer.from_pretrained')
    def test_create_mlx_model_wrapper(self, mock_tokenizer, mock_mx_load):
        """Test creating an MLX model wrapper from a directory."""
        # Set up mocks
        mock_mx_load.return_value = {"weight1": MagicMock(), "weight2": MagicMock()}
        mock_tokenizer.return_value = MagicMock()
        
        # Create a temporary directory structure
        model_dir = Path("test_model_dir")
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(model_dir / "mlx", exist_ok=True)
        
        # Create metadata file
        metadata = {
            "description": "Test Model",
            "class": "BertForSequenceClassification",
            "optimized": True,
            "mlx_metadata": {
                "mlx_version": "0.5.0",
                "quantization": 16
            }
        }
        
        # Mock the metadata file read
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__.return_value.read.return_value = str(metadata)
            
            # Call the function
            wrapper = create_mlx_model_wrapper(
                model_id="test-model",
                model_dir=model_dir
            )
        
        # Check that the wrapper was created
        assert wrapper is not None
        assert wrapper.model_id == "test-model"
        assert wrapper.model_class == "BertForSequenceClassification"
        
        # Clean up
        import shutil
        shutil.rmtree(model_dir, ignore_errors=True)
    
    @patch('models.mlx_wrapper.mx')
    def test_predict_method(self, mock_mx):
        """Test the predict method of MLXModelWrapper."""
        # Create a mock weights dictionary
        weights = {
            "weight1": MagicMock(),
            "weight2": MagicMock()
        }
        
        # Create mock processors
        process_input = MagicMock(return_value={"processed": True})
        process_output = MagicMock(return_value={"result": "success"})
        
        # Create a wrapper instance with mock processors
        wrapper = MLXModelWrapper(
            model_id="test-model",
            model_class="BertForSequenceClassification",
            weights=weights
        )
        
        # Set mock processors
        wrapper.process_input = process_input
        wrapper.process_output = process_output
        wrapper._forward = MagicMock(return_value={"logits": MagicMock()})
        
        # Test predict method
        inputs = {"input_ids": [[1, 2, 3]], "attention_mask": [[1, 1, 1]]}
        result = wrapper.predict(inputs)
        
        # Verify results
        assert result == {"result": "success"}
        process_input.assert_called_once_with(inputs)
        wrapper._forward.assert_called_once()
        process_output.assert_called_once() 