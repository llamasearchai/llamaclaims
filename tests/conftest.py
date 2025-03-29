"""
Pytest configuration file for LlamaClaims tests.

This file contains fixtures used across multiple test modules.
"""

import os
import sys
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Environment setup for testing
os.environ["APP_ENV"] = "testing"
os.environ["LOG_LEVEL"] = "ERROR"
os.environ["LLAMACLAIMS_MODELS_DIR"] = str(Path(__file__).parent / "test_data" / "models")
os.environ["LLAMACLAIMS_UPLOADS_DIR"] = str(Path(__file__).parent / "test_data" / "uploads")
os.environ["LLAMACLAIMS_CACHE_DIR"] = str(Path(__file__).parent / "test_data" / "cache")


@pytest.fixture
def api_client():
    """
    Create a FastAPI test client for testing API endpoints.
    
    Returns:
        TestClient: A FastAPI test client
    """
    # Import here to avoid circular imports
    from api.main import app
    
    # Create test client
    client = TestClient(app)
    
    return client


@pytest.fixture
def sample_claim_data():
    """
    Provide sample claim data for testing.
    
    Returns:
        dict: Sample claim data
    """
    return {
        "claim_number": "CLM-2023-12345",
        "policy_number": "POL-2023-67890",
        "incident_date": "2023-03-14T12:00:00",
        "filing_date": "2023-03-15T09:30:00",
        "claimant": {
            "first_name": "John",
            "last_name": "Doe",
            "email": "john.doe@example.com",
            "phone": "555-123-4567"
        },
        "incident_description": "Water damage from burst pipe in kitchen",
        "claim_type": "property",
        "status": "new",
        "estimated_value": 5000.00
    }


@pytest.fixture
def sample_document_data():
    """
    Provide sample document data for testing.
    
    Returns:
        dict: Sample document data
    """
    return {
        "document_type": "claim_form",
        "file_name": "claim_form.pdf",
        "mime_type": "application/pdf",
        "upload_date": "2023-03-15T10:00:00",
        "status": "uploaded"
    }


@pytest.fixture
def mock_model_manager():
    """
    Create a mock model manager for testing.
    
    Returns:
        MagicMock: A mock model manager
    """
    from unittest.mock import MagicMock
    
    # Create mock manager
    mock_manager = MagicMock()
    
    # Mock list_available_models
    mock_manager.list_available_models.return_value = {
        "document-classifier": {
            "description": "Document Classification Model",
            "size_mb": 350.0,
            "download_date": "2023-03-14T12:34:56",
            "class": "LayoutLMv3ForSequenceClassification",
            "optimized": True,
            "mlx_metadata": {
                "mlx_version": "0.5.0",
                "quantization": 16,
                "optimized_for_inference": True,
                "conversion_date": "2023-03-14T13:45:12",
                "metal_available": True
            }
        },
        "fraud-detector": {
            "description": "Fraud Detection Model",
            "size_mb": 480.0,
            "download_date": "2023-03-14T12:40:23",
            "class": "RobertaForSequenceClassification",
            "optimized": False
        }
    }
    
    # Mock get_model_metadata
    mock_manager.get_model_metadata.side_effect = lambda model_id: (
        mock_manager.list_available_models().get(model_id)
    )
    
    # Mock is_model_optimized
    mock_manager.is_model_optimized.side_effect = lambda model_id: (
        mock_manager.list_available_models().get(model_id, {}).get("optimized", False)
    )
    
    # Mock load_model
    mock_manager.load_model.return_value = MagicMock()
    
    # Mock run_inference
    mock_manager.run_inference.return_value = {
        "logits": [[0.1, 0.9]],
        "probs": [[0.1, 0.9]],
        "predicted_class": [1]
    }
    
    # Mock benchmark_model
    mock_manager.benchmark_model.return_value = {
        "backend": "mlx",
        "mean_ms": 112.5,
        "std_ms": 3.2,
        "num_runs": 10
    }
    
    return mock_manager


@pytest.fixture
def test_data_dir():
    """
    Create and return a temporary directory for test data.
    
    Returns:
        Path: Path to test data directory
    """
    test_data_dir = Path(__file__).parent / "test_data"
    
    # Create test data directories
    models_dir = test_data_dir / "models"
    uploads_dir = test_data_dir / "uploads"
    cache_dir = test_data_dir / "cache"
    
    models_dir.mkdir(exist_ok=True, parents=True)
    uploads_dir.mkdir(exist_ok=True, parents=True)
    cache_dir.mkdir(exist_ok=True, parents=True)
    
    return test_data_dir 