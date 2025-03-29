# Architecture Overview

This document describes the architecture of LlamaClaims, explaining the system design, components, and their interactions.

## High-Level Architecture

LlamaClaims follows a modern, modular architecture designed for scalability, maintainability, and performance:

```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │
│  API Layer     │────▶│  Service Layer │────▶│  Model Layer   │
│  (FastAPI)     │     │  (Business     │     │  (ML/AI        │
│                │     │   Logic)       │     │   Models)      │
└────────────────┘     └────────────────┘     └────────────────┘
        │                      │                      │
        │                      │                      │
        ▼                      ▼                      ▼
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│                │     │                │     │                │
│  Data Access   │     │  Document      │     │  Optimization  │
│  Layer         │     │  Processing    │     │  Layer (MLX)   │
│                │     │                │     │                │
└────────────────┘     └────────────────┘     └────────────────┘
```

### Core Components

1. **API Layer**: FastAPI-based REST API endpoints that handle HTTP requests/responses
2. **Service Layer**: Core business logic for claims processing and analysis
3. **Model Layer**: Machine learning models for document understanding and claims analysis
4. **Data Access Layer**: Interfaces for data storage and retrieval
5. **Document Processing Layer**: PDF/image processing and text extraction
6. **Optimization Layer**: Performance optimizations for different hardware (MLX for Apple Silicon)

## Component Details

### API Layer (`api/`)

The API layer is implemented using FastAPI and provides the following endpoints:

- **Health**: System health and status (`/health`)
- **Claims**: Claims submission and processing (`/claims/*`)
- **Analysis**: Analysis of claims and documents (`/analysis/*`)
- **Models**: Model management and information (`/models/*`)

Key files:
- `api/main.py`: Main FastAPI application setup
- `api/routes/`: API route definitions by domain
- `api/schemas/`: Pydantic models for request/response validation
- `api/dependencies.py`: Dependency injection for services

### Service Layer (`api/services/`)

The service layer contains the core business logic:

- **ClaimsService**: Handling and processing of insurance claims
- **AnalysisService**: Analysis of claims data and fraud detection
- **DocumentService**: Document processing and information extraction
- **ModelService**: Model management and inference orchestration

### Model Layer (`models/`)

The model layer manages the machine learning models:

- **Model Interface**: Common interface for all models (`models/interface.py`)
- **Model Downloader**: Model downloading and caching (`models/downloader.py`)
- **MLX Wrapper**: Apple Silicon optimization with MLX (`models/mlx_wrapper.py`)
- **Model Optimizer**: Quantization and optimization (`models/optimizer.py`)

### Data Flow

The typical data flow for claims processing:

1. Client submits a claim via the API
2. API layer validates the request format
3. Service layer processes the business logic
4. Model layer performs AI-powered analysis
5. Results are returned to the client via the API

## Directory Structure

```
llamaclaims/
├── api/                  # API implementation
│   ├── routes/           # API routes
│   ├── schemas/          # Data models
│   ├── services/         # Business logic
│   ├── main.py           # FastAPI app
│   └── dependencies.py   # Dependency injection
├── cli/                  # Command-line interface
├── data/                 # Data storage
│   ├── models/           # ML models
│   ├── uploads/          # User uploads
│   └── cache/            # Temporary cache
├── docs/                 # Documentation
├── examples/             # Example code
├── models/               # ML model integration
├── tests/                # Automated tests
├── ui/                   # Web UI (if applicable)
├── .env.example          # Environment variables template
├── docker-compose.yml    # Docker Compose configuration
├── Dockerfile            # Docker configuration
├── requirements.txt      # Dependencies
└── run.py                # Entry point
```

## Dependency Injection

LlamaClaims uses FastAPI's dependency injection system to manage service dependencies. This approach:

- Improves testability by making it easy to mock dependencies
- Enhances maintainability by clearly defining component relationships
- Enables flexibility in implementation details

Example:

```python
# In dependencies.py
def get_claims_service():
    return ClaimsService()

# In routes
@router.post("/claims")
def create_claim(
    claim: ClaimCreate,
    claims_service: ClaimsService = Depends(get_claims_service)
):
    return claims_service.create_claim(claim)
```

## Model Loading and Optimization

LlamaClaims intelligently manages model loading and optimization:

1. **Lazy Loading**: Models are loaded only when needed
2. **Hardware Detection**: Automatically detects Apple Silicon and uses MLX
3. **Memory Management**: Unloads models when not in use
4. **Quantization**: Supports various levels of quantization for performance

## Scaling Considerations

For high-load deployments, consider:

1. **Horizontal Scaling**: Deploy multiple API instances behind a load balancer
2. **Model Serving**: Separate model serving with dedicated resources
3. **Caching**: Implement Redis or similar for caching frequently accessed data
4. **Database**: Add a persistent database for claims storage

## Future Architecture Directions

Planned architectural improvements:

1. **Async Processing**: Message queue for asynchronous claims processing
2. **Microservices**: Split into domain-specific microservices
3. **Event Sourcing**: Event-driven architecture for claims state management
4. **Streaming**: Real-time claims processing updates

## Design Principles

LlamaClaims follows these architectural principles:

1. **Modularity**: Components are designed to be independent and replaceable
2. **Testability**: All components can be tested in isolation
3. **Performance**: Optimized for both CPU and GPU/MLX acceleration
4. **Security**: Security-first design with proper authentication/authorization
5. **Extensibility**: Easy to extend with new models and capabilities 