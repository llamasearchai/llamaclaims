# LlamaClaims

<div align="center" markdown>
![LlamaClaims Logo](assets/logo.png){ width="150" }

**AI-powered insurance claims processing platform**

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/llamaclaims/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://pypi.org/project/llamaclaims/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/llamasearchai/llamaclaims/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://llamasearchai.github.io/llamaclaims/)
</div>

## Overview

LlamaClaims is a modular framework for processing insurance claims using artificial intelligence and machine learning. It provides a comprehensive set of tools to analyze policy documents, detect fraud, and automate claims handling with state-of-the-art language models.

## Key Features

- **AI-Powered Document Analysis**: Extract and interpret information from policy documents, claims forms, and supporting evidence
- **Fraud Detection**: Identify suspicious patterns and anomalies in claims data
- **MLX Acceleration**: Optimized for Apple Silicon with MLX for high-performance inference
- **Comprehensive API**: RESTful API for integration with existing insurance systems
- **Modular Architecture**: Easily extensible with custom processing components
- **Secure & Compliant**: Designed with data protection and compliance in mind

## Quick Start

```bash
# Clone the repository
git clone https://github.com/llamasearchai/llamaclaims.git
cd llamaclaims

# Setup environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the API server
python run.py
```

## Example: Processing a Claim

```python
import requests

# API endpoint
url = "http://localhost:8000/api/claims/analyze"

# Claim data
claim_data = {
    "policy_number": "POL-12345",
    "claim_amount": 5000.00,
    "description": "Water damage to kitchen floor due to pipe burst",
    "date_of_incident": "2024-03-01",
    "claimant_info": {
        "name": "Jane Smith",
        "contact": "jane.smith@example.com"
    }
}

# Send request
response = requests.post(url, json=claim_data)
result = response.json()

print(f"Claim ID: {result['claim_id']}")
print(f"Risk Score: {result['risk_score']}")
print(f"Estimated Processing Time: {result['estimated_processing_time']} days")
```

## Documentation

For full documentation, visit [llamasearchai.github.io/llamaclaims](https://llamasearchai.github.io/llamaclaims/).

- [User Guide](user_guide/getting_started.md) - Getting started with LlamaClaims
- [API Reference](api_reference/overview.md) - Detailed API documentation
- [Developer Guide](developer_guide/architecture.md) - Architecture and contributing guidelines
- [Deployment](deployment/requirements.md) - Production deployment options
- [Examples](examples/basic_usage.md) - Example usage scenarios

## Requirements

- Python 3.9+
- FastAPI
- PyTorch (automatically replaced with MLX on Apple Silicon)
- Hugging Face Transformers
- Additional dependencies listed in `requirements.txt`

## License

LlamaClaims is released under the [MIT License](https://github.com/llamasearchai/llamaclaims/blob/main/LICENSE). 