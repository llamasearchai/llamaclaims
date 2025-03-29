"""
Unit tests for the models API endpoints.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from fastapi import status

from api.schemas.models import ModelStatus


@pytest.mark.asyncio
async def test_list_models(api_client, mock_model_manager):
    """Test listing models endpoint."""
    # Patch the get_model_manager function to return our mock
    with patch('models.interface.get_model_manager', return_value=mock_model_manager):
        # Make request to list models endpoint
        response = api_client.get("/models/")
        
        # Check response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check data structure
        assert "models" in data
        assert len(data["models"]) == 2
        
        # Verify first model
        model = next(m for m in data["models"] if m["id"] == "document-classifier")
        assert model["name"] == "Document Classification Model"
        assert model["status"] == ModelStatus.OPTIMIZED
        assert model["is_optimized"] == True
        assert "mlx_metadata" in model
        assert model["mlx_metadata"]["mlx_version"] == "0.5.0"
        
        # Verify second model
        model = next(m for m in data["models"] if m["id"] == "fraud-detector")
        assert model["name"] == "Fraud Detection Model"
        assert model["status"] == ModelStatus.AVAILABLE
        assert model["is_optimized"] == False


@pytest.mark.asyncio
async def test_get_model(api_client, mock_model_manager):
    """Test getting a specific model endpoint."""
    # Patch the get_model_manager function to return our mock
    with patch('models.interface.get_model_manager', return_value=mock_model_manager):
        # Make request to get model endpoint
        response = api_client.get("/models/document-classifier")
        
        # Check response
        assert response.status_code == status.HTTP_200_OK
        model = response.json()
        
        # Check data
        assert model["id"] == "document-classifier"
        assert model["name"] == "Document Classification Model"
        assert model["status"] == ModelStatus.OPTIMIZED
        assert model["is_optimized"] == True


@pytest.mark.asyncio
async def test_get_nonexistent_model(api_client, mock_model_manager):
    """Test getting a model that doesn't exist."""
    # Patch the get_model_manager function to return our mock
    with patch('models.interface.get_model_manager', return_value=mock_model_manager):
        # Make request to get a nonexistent model
        response = api_client.get("/models/nonexistent-model")
        
        # Check response
        assert response.status_code == status.HTTP_404_NOT_FOUND
        data = response.json()
        assert "detail" in data
        assert "not found" in data["detail"]


@pytest.mark.asyncio
async def test_inference(api_client, mock_model_manager):
    """Test model inference endpoint."""
    # Patch the get_model_manager function to return our mock
    with patch('models.interface.get_model_manager', return_value=mock_model_manager):
        # Prepare inference request
        inference_request = {
            "inputs": {
                "input_ids": [[101, 2054, 2003, 1037, 2609, 1012, 102]],
                "attention_mask": [[1, 1, 1, 1, 1, 1, 1]]
            },
            "use_mlx": True
        }
        
        # Make request to inference endpoint
        response = api_client.post(
            "/models/document-classifier/inference",
            json=inference_request
        )
        
        # Check response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check data structure
        assert data["model_id"] == "document-classifier"
        assert data["success"] == True
        assert data["error"] is None
        assert "results" in data
        assert "logits" in data["results"]
        assert "probs" in data["results"]
        assert "predicted_class" in data["results"]
        assert "processing_time_ms" in data


@pytest.mark.asyncio
async def test_benchmark(api_client, mock_model_manager):
    """Test model benchmark endpoint."""
    # Patch the get_model_manager function to return our mock
    with patch('models.interface.get_model_manager', return_value=mock_model_manager):
        # Prepare benchmark request
        benchmark_request = {
            "num_runs": 5,
            "use_mlx": True
        }
        
        # Make request to benchmark endpoint
        response = api_client.post(
            "/models/document-classifier/benchmark",
            json=benchmark_request
        )
        
        # Check response
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        # Check data structure
        assert data["model_id"] == "document-classifier"
        assert data["success"] == True
        assert data["error"] is None
        assert data["backend"] == "mlx"
        assert data["mean_ms"] == 112.5
        assert data["std_ms"] == 3.2
        assert data["num_runs"] == 10 