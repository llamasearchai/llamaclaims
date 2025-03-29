"""
Integration tests for the LlamaClaims API.

These tests verify API endpoints work correctly together.
"""

import os
import pytest
import requests
import time
from typing import Dict, Any

API_BASE_URL = "http://localhost:8000"


def is_api_running() -> bool:
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=1)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.mark.skipif(not is_api_running(), reason="API server is not running")
class TestApiIntegration:
    """Test suite for API integration tests."""
    
    @pytest.fixture
    def api_client(self):
        """Create a requests session for API testing."""
        session = requests.Session()
        return session
    
    @pytest.fixture
    def claim_data(self) -> Dict[str, Any]:
        """Sample claim data for testing."""
        return {
            "claim_number": f"CLM-TEST-{int(time.time())}",
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
    
    def test_api_health(self, api_client):
        """Test API health endpoint."""
        response = api_client.get(f"{API_BASE_URL}/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
    
    def test_create_and_get_claim(self, api_client, claim_data):
        """Test creating and retrieving a claim."""
        # Create a claim
        response = api_client.post(f"{API_BASE_URL}/claims/", json=claim_data)
        assert response.status_code == 201
        created_claim = response.json()
        claim_id = created_claim["id"]
        
        # Get the claim
        response = api_client.get(f"{API_BASE_URL}/claims/{claim_id}")
        assert response.status_code == 200
        retrieved_claim = response.json()
        
        # Verify claim data
        assert retrieved_claim["id"] == claim_id
        assert retrieved_claim["claim_number"] == claim_data["claim_number"]
        assert retrieved_claim["policy_number"] == claim_data["policy_number"]
        assert retrieved_claim["status"] == claim_data["status"]
    
    def test_list_claims(self, api_client, claim_data):
        """Test listing claims."""
        # Create a claim to ensure there's at least one
        response = api_client.post(f"{API_BASE_URL}/claims/", json=claim_data)
        assert response.status_code == 201
        
        # List claims
        response = api_client.get(f"{API_BASE_URL}/claims/")
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "claims" in data
        assert isinstance(data["claims"], list)
        assert len(data["claims"]) > 0
    
    def test_update_claim(self, api_client, claim_data):
        """Test updating a claim."""
        # Create a claim
        response = api_client.post(f"{API_BASE_URL}/claims/", json=claim_data)
        assert response.status_code == 201
        created_claim = response.json()
        claim_id = created_claim["id"]
        
        # Update the claim
        update_data = {
            "status": "in_progress",
            "estimated_value": 7500.00,
            "incident_description": "Updated: Water damage from burst pipe in kitchen and bathroom"
        }
        
        response = api_client.put(f"{API_BASE_URL}/claims/{claim_id}", json=update_data)
        assert response.status_code == 200
        updated_claim = response.json()
        
        # Verify updates
        assert updated_claim["id"] == claim_id
        assert updated_claim["status"] == "in_progress"
        assert updated_claim["estimated_value"] == 7500.00
        assert "Updated:" in updated_claim["incident_description"]
    
    def test_models_endpoints(self, api_client):
        """Test models API endpoints."""
        # List models
        response = api_client.get(f"{API_BASE_URL}/models/")
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "models" in data
        assert isinstance(data["models"], list)
        
        # If there are models, test getting a specific model
        if data["models"]:
            model_id = data["models"][0]["id"]
            response = api_client.get(f"{API_BASE_URL}/models/{model_id}")
            assert response.status_code == 200
            model_data = response.json()
            assert model_data["id"] == model_id

def test_api_version():
    """Test the API version endpoint."""
    response = requests.get(f"{API_BASE_URL}/version")
    assert response.status_code == 200
    data = response.json()
    assert "version" in data
    assert data["version"] == "1.0.0"

def test_get_claims():
    """Test the get claims endpoint."""
    response = requests.get(f"{API_BASE_URL}/claims")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0

def test_get_claim():
    """Test the get claim endpoint."""
    response = requests.get(f"{API_BASE_URL}/claims/1")
    assert response.status_code == 200
    data = response.json()
    assert data["id"] == 1
    assert "title" in data
    assert "status" in data

def test_analyze_document():
    """Test the analyze document endpoint."""
    payload = {
        "document_type": "claim_form",
        "document_content": "This is a test document content.",
    }
    response = requests.post(f"{API_BASE_URL}/analyze", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "document_type" in data
    assert "analysis" in data
    assert "classification" in data["analysis"]
    assert "confidence" in data["analysis"]
    assert "extracted_info" in data["analysis"]

def test_api_response_time():
    """Test the API response time."""
    start_time = time.time()
    response = requests.get(f"{API_BASE_URL}/health")
    end_time = time.time()
    
    assert response.status_code == 200
    response_time = end_time - start_time
    
    # Response time should be less than 1 second
    assert response_time < 1.0, f"Response time too slow: {response_time:.2f} seconds" 