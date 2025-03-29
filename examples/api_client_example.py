#!/usr/bin/env python3
"""
LlamaClaims API Client Example

This script demonstrates how to use the LlamaClaims API to perform various operations:
1. Checking API health
2. Listing available models
3. Running inference with a model
4. Creating and managing claims
5. Processing documents

Usage:
    python api_client_example.py [--api-url URL]
"""

import argparse
import json
import os
import sys
import requests
from pprint import pprint
from typing import Dict, Any, Optional


class LlamaClaimsClient:
    """Client for interacting with the LlamaClaims API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize the API client.
        
        Args:
            base_url: Base URL of the LlamaClaims API
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    
    def check_health(self) -> Dict[str, Any]:
        """
        Check the health of the API.
        
        Returns:
            Health status information
        """
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> Dict[str, Any]:
        """
        List available models.
        
        Returns:
            List of available models
        """
        response = self.session.get(f"{self.base_url}/models/")
        response.raise_for_status()
        return response.json()
    
    def get_model(self, model_id: str) -> Dict[str, Any]:
        """
        Get information about a specific model.
        
        Args:
            model_id: ID of the model
            
        Returns:
            Model information
        """
        response = self.session.get(f"{self.base_url}/models/{model_id}")
        response.raise_for_status()
        return response.json()
    
    def run_inference(self, model_id: str, inputs: Dict[str, Any], use_mlx: Optional[bool] = None) -> Dict[str, Any]:
        """
        Run inference with a model.
        
        Args:
            model_id: ID of the model
            inputs: Input data for the model
            use_mlx: Whether to use MLX (if None, uses MLX if available)
            
        Returns:
            Inference results
        """
        payload = {
            "inputs": inputs,
            "use_mlx": use_mlx
        }
        
        response = self.session.post(
            f"{self.base_url}/models/{model_id}/inference",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def benchmark_model(self, model_id: str, num_runs: int = 10, use_mlx: Optional[bool] = None) -> Dict[str, Any]:
        """
        Benchmark a model's inference speed.
        
        Args:
            model_id: ID of the model
            num_runs: Number of benchmark runs
            use_mlx: Whether to use MLX (if None, uses MLX if available)
            
        Returns:
            Benchmark results
        """
        payload = {
            "num_runs": num_runs,
            "use_mlx": use_mlx
        }
        
        response = self.session.post(
            f"{self.base_url}/models/{model_id}/benchmark",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def create_claim(self, claim_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create a new claim.
        
        Args:
            claim_data: Claim data
            
        Returns:
            Created claim
        """
        response = self.session.post(
            f"{self.base_url}/claims/",
            json=claim_data
        )
        response.raise_for_status()
        return response.json()
    
    def list_claims(self) -> Dict[str, Any]:
        """
        List claims.
        
        Returns:
            List of claims
        """
        response = self.session.get(f"{self.base_url}/claims/")
        response.raise_for_status()
        return response.json()
    
    def get_claim(self, claim_id: str) -> Dict[str, Any]:
        """
        Get a specific claim.
        
        Args:
            claim_id: ID of the claim
            
        Returns:
            Claim information
        """
        response = self.session.get(f"{self.base_url}/claims/{claim_id}")
        response.raise_for_status()
        return response.json()
    
    def analyze_document(self, file_path: str) -> Dict[str, Any]:
        """
        Analyze a document.
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Analysis results
        """
        with open(file_path, "rb") as f:
            files = {"file": (os.path.basename(file_path), f)}
            response = self.session.post(
                f"{self.base_url}/analyze/document",
                files=files
            )
        response.raise_for_status()
        return response.json()


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="LlamaClaims API Client Example")
    parser.add_argument("--api-url", default="http://localhost:8000", help="URL of the LlamaClaims API")
    args = parser.parse_args()
    
    # Create API client
    client = LlamaClaimsClient(base_url=args.api_url)
    
    # Check API health
    try:
        print("Checking API health...")
        health = client.check_health()
        print("API is healthy!")
        print(f"Status: {health['status']}")
        print(f"Version: {health['version']}")
        print(f"Uptime: {health['uptime']} seconds")
        print()
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to connect to the API. Make sure the API server is running.\n{e}")
        sys.exit(1)
    
    # List available models
    try:
        print("Listing available models...")
        models = client.list_models()
        if models["models"]:
            print(f"Found {len(models['models'])} models:")
            for model in models["models"]:
                print(f"  - {model['id']}: {model['name']} (Status: {model['status']})")
        else:
            print("No models available.")
        print()
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to list models.\n{e}")
    
    # Example: Get a specific model (if available)
    try:
        if models["models"]:
            model_id = models["models"][0]["id"]
            print(f"Getting information about model '{model_id}'...")
            model_info = client.get_model(model_id)
            print("Model information:")
            pprint(model_info)
            print()
            
            # Example: Benchmark the model (if available)
            print(f"Benchmarking model '{model_id}'...")
            benchmark_results = client.benchmark_model(model_id, num_runs=5)
            print("Benchmark results:")
            pprint(benchmark_results)
            print()
    except (NameError, IndexError, requests.exceptions.RequestException) as e:
        print(f"Error: Failed to get or benchmark model.\n{e}")
    
    # Example: Create a claim
    try:
        print("Creating a sample claim...")
        claim_data = {
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
        
        created_claim = client.create_claim(claim_data)
        print("Claim created successfully:")
        print(f"  Claim ID: {created_claim['id']}")
        print(f"  Claim Number: {created_claim['claim_number']}")
        print(f"  Status: {created_claim['status']}")
        print()
        
        # Example: List claims
        print("Listing claims...")
        claims = client.list_claims()
        print(f"Found {len(claims['claims'])} claims.")
        print()
    except requests.exceptions.RequestException as e:
        print(f"Error: Failed to create or list claims.\n{e}")
    
    print("Example completed!")


if __name__ == "__main__":
    main() 