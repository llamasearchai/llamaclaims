import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "ok"
    assert "timestamp" in response.json()

def test_version_endpoint():
    """Test the version endpoint."""
    response = client.get("/version")
    assert response.status_code == 200
    assert "version" in response.json()
    assert response.json()["version"] == "1.0.0"

def test_root_endpoint():
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "name" in response.json()
    assert "description" in response.json()
    assert "version" in response.json()
    assert "documentation" in response.json()

def test_get_claims():
    """Test the get claims endpoint."""
    response = client.get("/claims")
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0
    
    # Check the structure of the first claim
    claim = response.json()[0]
    assert "id" in claim
    assert "title" in claim
    assert "status" in claim
    assert "created_at" in claim

def test_get_claim():
    """Test the get claim endpoint."""
    response = client.get("/claims/1")
    assert response.status_code == 200
    assert "id" in response.json()
    assert "title" in response.json()
    assert "description" in response.json()
    assert "status" in response.json()
    assert "created_at" in response.json()
    assert "amount" in response.json()
    assert "policy_number" in response.json()
    assert "documents" in response.json()

def test_get_claim_not_found():
    """Test the get claim endpoint with an invalid ID."""
    response = client.get("/claims/999")
    assert response.status_code == 404
    assert "detail" in response.json()
    assert response.json()["detail"] == "Claim not found"

def test_analyze_document():
    """Test the analyze document endpoint."""
    response = client.post(
        "/analyze",
        json={
            "document_type": "claim_form",
            "document_content": "This is a test document content.",
        },
    )
    assert response.status_code == 200
    assert "document_type" in response.json()
    assert "analysis" in response.json()
    assert "classification" in response.json()["analysis"]
    assert "confidence" in response.json()["analysis"]
    assert "extracted_info" in response.json()["analysis"] 