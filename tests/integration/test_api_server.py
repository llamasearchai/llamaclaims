"""
Integration tests for the API server.

These tests verify that the API server starts up correctly and basic endpoints work.
"""

import os
import pytest
import subprocess
import time
import requests
from pathlib import Path

# Root directory of the project
ROOT_DIR = Path(__file__).parent.parent.parent


def is_server_running(port=8000):
    """Check if the server is running on the given port."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=1)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


class TestAPIServer:
    """Test suite for API server integration tests."""
    
    @pytest.fixture
    def server_process(self):
        """Start the API server as a subprocess and yield the process."""
        # Set environment variables for testing
        env = os.environ.copy()
        env["APP_ENV"] = "testing"
        env["LOG_LEVEL"] = "ERROR"
        env["API_PORT"] = "8765"  # Use a different port for testing
        
        # Start the server
        process = subprocess.Popen(
            ["python", "run.py", "--port", "8765"],
            cwd=ROOT_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        # Wait for the server to start
        for _ in range(10):  # Try for 5 seconds
            if is_server_running(port=8765):
                break
            time.sleep(0.5)
        else:
            process.terminate()
            stdout, stderr = process.communicate()
            pytest.fail(f"Server failed to start. Stdout: {stdout.decode()}, Stderr: {stderr.decode()}")
        
        # Yield the process
        yield process
        
        # Clean up
        process.terminate()
        process.wait(timeout=5)
    
    def test_health_endpoint(self, server_process):
        """Test that the health endpoint returns a 200 OK status."""
        response = requests.get("http://localhost:8765/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "version" in data
        assert "uptime" in data
        
    def test_root_endpoint(self, server_process):
        """Test that the root endpoint returns the API info."""
        response = requests.get("http://localhost:8765/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data
        assert "documentation" in data
        assert data["name"] == "LlamaClaims API"
    
    def test_documentation_endpoints(self, server_process):
        """Test that the documentation endpoints are available."""
        # Check Swagger UI
        response = requests.get("http://localhost:8765/docs")
        assert response.status_code == 200
        assert "swagger" in response.text.lower()
        
        # Check ReDoc UI
        response = requests.get("http://localhost:8765/redoc")
        assert response.status_code == 200
        assert "redoc" in response.text.lower() 