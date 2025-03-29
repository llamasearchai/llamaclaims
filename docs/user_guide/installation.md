# Installation Guide

This page provides detailed instructions for installing LlamaClaims in various environments.

## System Requirements

### Minimum Requirements

- **Python**: 3.9 or higher
- **RAM**: 8GB minimum (16GB recommended)
- **Disk Space**: 5GB for base installation (10GB+ with models)
- **OS**: Linux, macOS, or Windows 10/11

### For GPU Acceleration

- CUDA-compatible NVIDIA GPU (8GB+ VRAM recommended)
- CUDA Toolkit 11.7+ and cuDNN

### For Apple Silicon (M1/M2/M3)

- macOS 12.0 (Monterey) or higher
- MLX package will be automatically installed for optimized performance

## Installation Methods

### Option 1: PyPI Installation (Recommended)

The simplest way to install LlamaClaims is via pip:

```bash
# Basic installation
pip install llamaclaims

# With plotting support
pip install "llamaclaims[plot]"

# With development tools
pip install "llamaclaims[dev]"

# With all optional dependencies
pip install "llamaclaims[all]"
```

### Option 2: From Source

```bash
# Clone the repository
git clone https://github.com/llamasearchai/llamaclaims.git
cd llamaclaims

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option 3: Docker Installation

LlamaClaims provides Docker images for easy deployment:

```bash
# Pull the latest image
docker pull llamasearchai/llamaclaims:latest

# Run the container
docker run -p 8000:8000 -v $(pwd)/data:/app/data llamasearchai/llamaclaims:latest
```

Or build and run using docker-compose:

```bash
# Clone the repository
git clone https://github.com/llamasearchai/llamaclaims.git
cd llamaclaims

# Build and start the containers
docker-compose up -d
```

## Post-Installation Setup

### Environment Configuration

Copy the example environment file and configure it for your setup:

```bash
cp .env.example .env
```

Edit the `.env` file to match your requirements. See the [Configuration](configuration.md) guide for details.

### Model Downloads

The first time you run LlamaClaims, it will automatically download the necessary models. You can also manually download them:

```bash
# Using the CLI
python -m llamaclaims.cli.download_models

# With specific options
python -m llamaclaims.cli.download_models --model small --quantization 4bit
```

### Verify Installation

To verify that LlamaClaims is installed correctly:

```bash
# Start the API server
python run.py

# In a separate terminal, check the health endpoint
curl http://localhost:8000/health
```

You should receive a response indicating the system is operational.

## Platform-Specific Notes

### macOS

For macOS users, we recommend using Homebrew to install Python:

```bash
brew install python@3.11
```

For Apple Silicon users, MLX will be automatically used for optimal performance.

### Linux

On Ubuntu/Debian systems, you may need to install additional system dependencies:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv tesseract-ocr poppler-utils
```

### Windows

For Windows users, we recommend installing Python from the [official website](https://www.python.org/downloads/windows/) and ensuring it's added to your PATH.

For PDF processing, you'll need to install:
- [Tesseract OCR](https://github.com/UB-Mannheim/tesseract/wiki)
- [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)

Add both to your PATH environment variable.

## Troubleshooting

### Common Issues

**Issue**: `ImportError: No module named 'llamaclaims'`
**Solution**: Ensure you've activated the virtual environment or install the package in development mode with `pip install -e .`

**Issue**: PDF processing fails
**Solution**: Verify Tesseract OCR and Poppler are installed and in your PATH

**Issue**: Model download fails
**Solution**: Check your internet connection and try the manual download with `--verbose` flag

For more issues and solutions, see the [Troubleshooting](../developer_guide/troubleshooting.md) page.

## Next Steps

- [Configuration](configuration.md) - Configure LlamaClaims for your environment
- [Getting Started](getting_started.md) - Start using LlamaClaims
- [Developer Guide](../developer_guide/contributing.md) - Contribute to the project 