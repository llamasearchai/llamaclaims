"""
Performance tests for the LlamaClaims API.

These tests measure response times for API endpoints under various load conditions.
"""

import os
import time
import statistics
import pytest
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# API endpoint for testing
API_BASE_URL = "http://localhost:8000"
HEALTH_ENDPOINT = f"{API_BASE_URL}/health"
MODELS_LIST_ENDPOINT = f"{API_BASE_URL}/models/"


def measure_response_time(url, method="get", json_data=None, timeout=10):
    """Measure the response time for a request to the given URL."""
    start_time = time.time()
    try:
        if method.lower() == "get":
            response = requests.get(url, timeout=timeout)
        elif method.lower() == "post":
            response = requests.post(url, json=json_data, timeout=timeout)
        else:
            raise ValueError(f"Unsupported method: {method}")
            
        end_time = time.time()
        response_time = end_time - start_time
        
        # Return the response time and status code
        return response_time, response.status_code
    except requests.exceptions.RequestException as e:
        end_time = time.time()
        return end_time - start_time, None


def is_api_running():
    """Check if the API is running."""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=2)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False


@pytest.mark.skipif(not is_api_running(), reason="API server is not running")
class TestAPIPerformance:
    """Test suite for API performance tests."""
    
    def test_health_endpoint_response_time(self):
        """Test the response time of the health endpoint."""
        # Measure response time
        response_time, status_code = measure_response_time(HEALTH_ENDPOINT)
        
        # Log results
        print(f"\nHealth endpoint response time: {response_time * 1000:.2f} ms")
        
        # Assert that the response was successful
        assert status_code == 200
        
        # Assert that the response time is within an acceptable range
        assert response_time < 0.5, f"Response time ({response_time:.2f}s) exceeded 500ms threshold"
    
    def test_models_list_endpoint_response_time(self):
        """Test the response time of the models list endpoint."""
        # Measure response time
        response_time, status_code = measure_response_time(MODELS_LIST_ENDPOINT)
        
        # Log results
        print(f"\nModels list endpoint response time: {response_time * 1000:.2f} ms")
        
        # Assert that the response was successful
        assert status_code == 200
        
        # Assert that the response time is within an acceptable range
        assert response_time < 1.0, f"Response time ({response_time:.2f}s) exceeded 1s threshold"
    
    def test_concurrent_health_requests(self):
        """Test the API performance under concurrent load."""
        # Number of concurrent requests
        num_requests = 10
        
        # Store response times
        response_times = []
        
        # Send concurrent requests
        with ThreadPoolExecutor(max_workers=num_requests) as executor:
            futures = [executor.submit(measure_response_time, HEALTH_ENDPOINT) for _ in range(num_requests)]
            
            # Collect results
            for future in as_completed(futures):
                response_time, status_code = future.result()
                response_times.append(response_time)
                assert status_code == 200, f"Request failed with status code {status_code}"
        
        # Calculate statistics
        avg_time = statistics.mean(response_times)
        max_time = max(response_times)
        min_time = min(response_times)
        percentile_95 = sorted(response_times)[int(len(response_times) * 0.95)]
        
        # Log results
        print(f"\nConcurrent health requests ({num_requests} requests):")
        print(f"  Average: {avg_time * 1000:.2f} ms")
        print(f"  Min: {min_time * 1000:.2f} ms")
        print(f"  Max: {max_time * 1000:.2f} ms")
        print(f"  95th percentile: {percentile_95 * 1000:.2f} ms")
        
        # Assert that the response times are within acceptable ranges
        assert avg_time < 0.5, f"Average response time ({avg_time:.2f}s) exceeded 500ms threshold"
        assert max_time < 1.0, f"Maximum response time ({max_time:.2f}s) exceeded 1s threshold"


if __name__ == "__main__":
    # Run the performance tests explicitly
    pytest.main(["-v", __file__]) 