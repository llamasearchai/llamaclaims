# Getting Started with LlamaClaims

This guide will help you get up and running with LlamaClaims for processing insurance claims using AI.

## Prerequisites

Before you begin, make sure you have:

- Python 3.9 or higher installed
- pip (Python package manager)
- 8GB+ RAM for running models locally
- For Apple Silicon users: macOS 12.0+ for MLX acceleration
- For GPU acceleration on other platforms: CUDA compatible GPU

## Installation

### Option 1: Using pip

```bash
# Install from PyPI
pip install llamaclaims

# Install with all optional dependencies
pip install "llamaclaims[all]"
```

### Option 2: From source

```bash
# Clone the repository
git clone https://github.com/llamasearchai/llamaclaims.git
cd llamaclaims

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Configuration

LlamaClaims uses environment variables for configuration. Copy the example file to create your own configuration:

```bash
cp .env.example .env
```

Edit the `.env` file to configure:

- API settings (host, port)
- Model settings
- Security settings
- Storage paths

See the [Configuration](configuration.md) page for detailed information on all available options.

## Running the API Server

LlamaClaims provides a REST API for processing claims. To start the server:

```bash
# Using the CLI script
python run.py

# With custom settings
python run.py --host 0.0.0.0 --port 8080 --log-level debug
```

Once running, you can access:

- API documentation: http://localhost:8000/docs
- ReDoc alternative: http://localhost:8000/redoc
- API status: http://localhost:8000/health

## First Request

Let's send a simple request to analyze a claim:

```python
import requests

# API endpoint
url = "http://localhost:8000/api/claims/analyze"

# Sample claim data
claim_data = {
    "policy_number": "POL-12345",
    "claim_amount": 1500.00,
    "description": "Laptop damaged by coffee spill during work hours",
    "date_of_incident": "2024-03-15",
    "claimant_info": {
        "name": "John Doe",
        "contact": "john.doe@example.com"
    }
}

# Send request
response = requests.post(url, json=claim_data)

# Process response
if response.status_code == 200:
    result = response.json()
    print(f"Analysis complete!")
    print(f"Claim ID: {result['claim_id']}")
    print(f"Risk Score: {result['risk_score']}")
    print(f"Classification: {result['classification']}")
    print(f"Estimated Processing Time: {result['estimated_processing_time']} days")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
```

## Document Processing

LlamaClaims can extract information from policy documents and claims forms:

```python
import requests

# Document processing endpoint
url = "http://localhost:8000/api/documents/process"

# Send a document for processing
with open("claim_form.pdf", "rb") as f:
    files = {"file": ("claim_form.pdf", f, "application/pdf")}
    data = {"document_type": "claim_form"}
    response = requests.post(url, files=files, data=data)

# Process extracted data
if response.status_code == 200:
    extracted_data = response.json()
    print(f"Extracted {len(extracted_data['fields'])} fields from document")
    for field in extracted_data['fields']:
        print(f"{field['name']}: {field['value']} (confidence: {field['confidence']})")
```

## Next Steps

Now that you've got LlamaClaims running, explore:

- [Processing Claims](processing_claims.md) - In-depth guide to claims processing
- [Analysis Reports](analysis_reports.md) - Generating and interpreting analysis reports
- [Models & Optimization](models.md) - Understanding and configuring the ML models
- [API Reference](../api_reference/overview.md) - Complete API documentation
- [Examples](../examples/basic_usage.md) - More example code and use cases 