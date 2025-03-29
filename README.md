# 🦙 LlamaClaims

<div align="center">

![LlamaClaims Logo](docs/assets/logo.png)

**AI-powered insurance claims processing platform**

[![PyPI version](https://img.shields.io/badge/pypi-v0.1.0-blue.svg)](https://pypi.org/project/llamaclaims/)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://pypi.org/project/llamaclaims/)
[![Documentation](https://img.shields.io/badge/docs-latest-blue.svg)](https://llamasearchai.github.io/llamaclaims/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/llamasearchai/llamaclaims/blob/main/LICENSE)
[![Tests](https://img.shields.io/github/workflow/status/llamasearchai/llamaclaims/CI)](https://github.com/llamasearchai/llamaclaims/actions)

</div>

## Overview

LlamaClaims is a modular framework for processing insurance claims using artificial intelligence and machine learning. It provides a comprehensive set of tools to analyze policy documents, detect fraud, and automate claims handling with state-of-the-art language models.

## 🚀 Key Features

- **AI-Powered Document Analysis**: Extract and interpret information from policy documents, claims forms, and supporting evidence
- **Fraud Detection**: Identify suspicious patterns and anomalies in claims data
- **MLX Acceleration**: Optimized for Apple Silicon with MLX for high-performance inference
- **Comprehensive API**: RESTful API for integration with existing insurance systems
- **Modular Architecture**: Easily extensible with custom processing components
- **Secure & Compliant**: Designed with data protection and compliance in mind

## 🧠 AI Capabilities

| Feature | Technology | Benefit |
|---------|------------|---------|
| **Document Classification** | Vision Transformer (ViT) | Route documents to appropriate workflows |
| **Information Extraction** | LayoutLMv3 | Extract structured data from documents |
| **Claims Classification** | DistilBERT | Categorize claims for efficient processing |
| **Fraud Detection** | RoBERTa | Identify potential fraud with high accuracy |
| **Policy Analysis** | Mistral-7B | Understand complex policy terms and coverage |

## 📊 Performance Benchmarks

| Model | MLX (M3 Max) | PyTorch (CPU) | Speedup |
|-------|--------------|---------------|---------|
| Document Classifier | 112 ms | 587 ms | 5.2x |
| Document Extractor | 135 ms | 705 ms | 5.2x |
| Claims Classifier | 76 ms | 418 ms | 5.5x |
| Fraud Detector | 105 ms | 546 ms | 5.2x |
| Claims LLM | 189 ms | 1023 ms | 5.4x |

*Benchmarks on M3 Max (14-core CPU, 30-core GPU, 36GB unified memory)*

## 💻 Installation

### Option 1: Using pip (Coming Soon)

```bash
pip install llamaclaims
```

### Option 2: From source

```bash
# Clone the repository
git clone https://github.com/llamasearchai/llamaclaims.git
cd llamaclaims

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Using Docker

```bash
# Pull the Docker image
docker pull llamasearchai/llamaclaims:latest

# Or build it yourself
docker build -t llamaclaims:latest .

# Run the container
docker run -p 8000:8000 llamaclaims:latest
```

## 🚀 Quick Start

```bash
# Configure environment
cp .env.example .env
# Edit .env with your settings

# Run the API server
python run.py
```

Visit `http://localhost:8000/docs` to explore the API documentation.

## 📝 Example: Processing a Claim

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

## 🏗️ Project Structure

```
llamaclaims/
├── api/               # FastAPI application
│   ├── routes/        # API endpoints
│   ├── schemas/       # Pydantic models
│   ├── services/      # Business logic
│   └── dependencies.py # Dependency injection
├── cli/               # Command-line interface
├── data/              # Data directories
│   ├── models/        # ML models
│   ├── uploads/       # Uploaded files
│   └── cache/         # Cache files
├── docs/              # Documentation
├── examples/          # Example code and scripts
├── logs/              # Application logs
├── models/            # Model management
│   ├── downloader.py  # Model downloader
│   ├── optimizer.py   # MLX optimizer
│   ├── interface.py   # Unified model interface
│   └── mlx_wrapper.py # MLX model wrapper
├── tests/             # Test suite
├── .env.example       # Example environment configuration
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── requirements.txt   # Dependencies
└── run.py             # API runner script
```

## 📖 Documentation

For full documentation, visit [llamasearchai.github.io/llamaclaims](https://llamasearchai.github.io/llamaclaims/)

### Building the Documentation

The documentation site is built using MkDocs with the Material theme:

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Serve documentation locally
mkdocs serve

# Build static site
mkdocs build

# Deploy to GitHub Pages (maintainers only)
mkdocs gh-deploy
```

## 👨‍💻 Development

### Setting Up Development Environment

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install in development mode
pip install -e .
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llamaclaims

# Run performance benchmarks
python examples/benchmark_mlx_vs_pytorch.py
```

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guidelines](https://llamasearchai.github.io/llamaclaims/developer_guide/contributing/) for details on how to submit pull requests, the development workflow, and coding standards.

## 📄 License

LlamaClaims is released under the [MIT License](LICENSE).

## 🚀 Key Features

- **MLX Acceleration**: Optimized for Apple Silicon with MLX, achieving significant speedups over CPU-based PyTorch
- **Multi-Modal Processing**: Handle documents, images, and text within a unified platform
- **Automated Risk Assessment**: AI-driven fraud detection and risk scoring
- **Modern API Architecture**: FastAPI-based REST API with async processing
- **Model Management**: Download, optimize, and benchmark models through the API and CLI
- **Comprehensive CLI**: Full-featured command-line interface for all operations

## 🧠 AI Capabilities

| Feature | Technology | Benefit |
|---------|------------|---------|
| **Document Classification** | Vision Transformer (ViT) | Route documents to appropriate workflows |
| **Information Extraction** | LayoutLMv3 | Extract structured data from documents |
| **Claims Classification** | DistilBERT | Categorize claims for efficient processing |
| **Fraud Detection** | RoBERTa | Identify potential fraud with high accuracy |
| **Policy Analysis** | Mistral-7B | Understand complex policy terms and coverage |

## 📊 Performance Benchmarks

| Model | MLX (M3 Max) | PyTorch (CPU) | Speedup |
|-------|--------------|---------------|---------|
| Document Classifier | 112 ms | 587 ms | 5.2x |
| Document Extractor | 135 ms | 705 ms | 5.2x |
| Claims Classifier | 76 ms | 418 ms | 5.5x |
| Fraud Detector | 105 ms | 546 ms | 5.2x |
| Claims LLM | 189 ms | 1023 ms | 5.4x |

*Benchmarks on M3 Max (14-core CPU, 30-core GPU, 36GB unified memory)*

You can run your own benchmarks using the provided script:

```bash
# Run benchmarks on all models
./examples/benchmark_mlx_vs_pytorch.py

# Benchmark specific models
./examples/benchmark_mlx_vs_pytorch.py --models document-classifier fraud-detector

# Customize number of runs
./examples/benchmark_mlx_vs_pytorch.py --runs 20
```

## 🏗️ Project Structure

```
llamaclaims/
├── api/               # FastAPI application
│   ├── routes/        # API endpoints
│   ├── schemas/       # Pydantic models
│   ├── services/      # Business logic
│   └── dependencies.py # Dependency injection
├── cli/               # Command-line interface
├── data/              # Data directories
│   ├── models/        # ML models
│   ├── uploads/       # Uploaded files
│   └── cache/         # Cache files
├── examples/          # Example code and scripts
├── logs/              # Application logs
├── models/            # Model management
│   ├── downloader.py  # Model downloader
│   ├── optimizer.py   # MLX optimizer
│   ├── interface.py   # Unified model interface
│   └── mlx_wrapper.py # MLX model wrapper
├── tests/             # Test suite
│   ├── unit/          # Unit tests
│   ├── integration/   # Integration tests
│   └── performance/   # Performance tests
├── .env               # Environment configuration
├── Dockerfile         # Docker configuration
├── docker-compose.yml # Docker Compose configuration
├── llamaclaims.sh     # Command-line script
├── requirements.txt   # Dependencies
└── run.py             # API runner script
```

## 💻 Getting Started

### Prerequisites

- macOS 13+ with Apple Silicon (M-series chip)
- Python 3.10+
- [Optional] Docker Desktop for Mac

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/llamaclaims.git
cd llamaclaims

# Make the script executable
chmod +x llamaclaims.sh

# Set up the environment and install dependencies
./llamaclaims.sh install

# Download and optimize models (on Apple Silicon)
./llamaclaims.sh models download
./llamaclaims.sh models optimize

# Run the API server
./llamaclaims.sh run api
```# Updated documentation
# Updated documentation
