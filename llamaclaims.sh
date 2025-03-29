#!/bin/bash
#==========================================================================#
# LlamaClaims Master Command Line Tool                                     #
# -------------------------------------------                              #
# A complete MLX-optimized AI insurance claims processing platform         #
# Designed for Apple Silicon (M-series) Macs                               #
#                                                                          #
# Version: 1.0.0                                                           #
# License: Apache 2.0                                                      #
#==========================================================================#

set -e  # Exit on error

# ANSI Colors for better output
RED='\033[0;31m'
YELLOW='\033[1;33m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${WORKSPACE_DIR}/logs"
DATA_DIR="${WORKSPACE_DIR}/data"
MODELS_DIR="${DATA_DIR}/models"
UPLOADS_DIR="${DATA_DIR}/uploads"
CACHE_DIR="${DATA_DIR}/cache"
CONFIG_FILE="${WORKSPACE_DIR}/.env"

# Print banner
print_banner() {
    echo -e "${BOLD}${BLUE}LlamaClaims${NC} - MLX-Optimized Insurance Claims Platform"
    echo -e "Version: 1.0.0"
    echo -e "Run ${YELLOW}./llamaclaims.sh help${NC} for usage information"
    echo
}

# Check prerequisites
check_prereqs() {
    echo -e "${BLUE}Checking prerequisites...${NC}"
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}ERROR: Python 3 is required but not installed.${NC}"
        exit 1
    fi
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        echo -e "${RED}ERROR: pip3 is required but not installed.${NC}"
        exit 1
    fi
    
    # Check Apple Silicon
    if [[ $(uname -m) == "arm64" && $(uname -s) == "Darwin" ]]; then
        echo -e "Running on ${GREEN}Apple Silicon${NC} - MLX optimization available"
        HAS_APPLE_SILICON=true
    else
        echo -e "${YELLOW}Not running on Apple Silicon. MLX optimizations will not be available.${NC}"
        HAS_APPLE_SILICON=false
    fi
    
    # Check Docker if needed
    if [[ "$1" == "docker" ]]; then
        if ! command -v docker &> /dev/null; then
            echo -e "${RED}ERROR: Docker is required but not installed.${NC}"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
            echo -e "${RED}ERROR: Docker Compose is required but not installed.${NC}"
            exit 1
        fi
    fi
}

# Setup the environment
setup_env() {
    echo -e "${BLUE}Setting up environment...${NC}"
    
    # Create directories
    mkdir -p "${LOG_DIR}"
    mkdir -p "${MODELS_DIR}"
    mkdir -p "${UPLOADS_DIR}"
    mkdir -p "${CACHE_DIR}"
    
    # Create or update .env file if it doesn't exist
    if [ ! -f "${CONFIG_FILE}" ]; then
        echo -e "Creating default configuration file..."
        cat > "${CONFIG_FILE}" << EOF
# LlamaClaims Configuration
APP_NAME=LlamaClaims
APP_ENV=development
LOG_LEVEL=INFO
API_HOST=0.0.0.0
API_PORT=8000
UI_PORT=3000
LLAMACLAIMS_MODELS_DIR=${MODELS_DIR}
LLAMACLAIMS_UPLOADS_DIR=${UPLOADS_DIR}
LLAMACLAIMS_CACHE_DIR=${CACHE_DIR}
EOF
    fi
    
    echo -e "${GREEN}Environment setup complete!${NC}"
}

# Install dependencies
install_deps() {
    local install_type=$1
    
    echo -e "${BLUE}Installing dependencies...${NC}"
    
    case "${install_type}" in
        minimal)
            echo -e "Installing minimal dependencies..."
            pip3 install -r requirements-minimal.txt
            ;;
        dev)
            echo -e "Installing development dependencies..."
            pip3 install -r requirements-dev.txt
            ;;
        full|*)
            echo -e "Installing all dependencies..."
            pip3 install -r requirements.txt
            
            # Install MLX if on Apple Silicon
            if [[ "${HAS_APPLE_SILICON}" == true ]]; then
                echo -e "Installing MLX for Apple Silicon..."
                pip3 install mlx
            fi
            ;;
    esac
    
    echo -e "${GREEN}Dependencies installed successfully!${NC}"
}

# Download models
download_models() {
    local model_id=$1
    local force=$2
    
    echo -e "${BLUE}Downloading models...${NC}"
    
    # Check if model downloader exists
    if [ ! -f "${WORKSPACE_DIR}/models/downloader.py" ]; then
        echo -e "${RED}ERROR: Model downloader not found.${NC}"
        exit 1
    fi
    
    # Set environment variable for models directory
    export LLAMACLAIMS_MODELS_DIR="${MODELS_DIR}"
    
    # Download all models or a specific model
    if [ -z "${model_id}" ]; then
        echo -e "Downloading all models..."
        python3 -m models.downloader --all --output-dir "${MODELS_DIR}" ${force:+--force}
    else
        echo -e "Downloading model: ${model_id}..."
        python3 -m models.downloader --model "${model_id}" --output-dir "${MODELS_DIR}" ${force:+--force}
    fi
    
    echo -e "${GREEN}Model download complete!${NC}"
}

# Optimize models for MLX
optimize_models() {
    local model_id=$1
    local quantize=$2
    
    echo -e "${BLUE}Optimizing models for MLX...${NC}"
    
    # Check if running on Apple Silicon
    if [[ "${HAS_APPLE_SILICON}" != true ]]; then
        echo -e "${RED}ERROR: MLX optimization requires Apple Silicon.${NC}"
        exit 1
    fi
    
    # Check if model optimizer exists
    if [ ! -f "${WORKSPACE_DIR}/models/optimizer.py" ]; then
        echo -e "${RED}ERROR: Model optimizer not found.${NC}"
        exit 1
    fi
    
    # Set environment variable for models directory
    export LLAMACLAIMS_MODELS_DIR="${MODELS_DIR}"
    
    # Optimize all models or a specific model
    if [ -z "${model_id}" ]; then
        echo -e "Optimizing all models..."
        python3 -m models.optimizer --all --models-dir "${MODELS_DIR}" ${quantize:+--quantize "${quantize}"} --optimize-for-inference
    else
        echo -e "Optimizing model: ${model_id}..."
        python3 -m models.optimizer --model "${model_id}" --models-dir "${MODELS_DIR}" ${quantize:+--quantize "${quantize}"} --optimize-for-inference
    fi
    
    echo -e "${GREEN}Model optimization complete!${NC}"
}

# Benchmark models
benchmark_models() {
    local model_id=$1
    
    echo -e "${BLUE}Benchmarking models...${NC}"
    
    # Check if running on Apple Silicon
    if [[ "${HAS_APPLE_SILICON}" != true ]]; then
        echo -e "${YELLOW}WARNING: Running on non-Apple Silicon. Benchmark results will not include MLX.${NC}"
    fi
    
    # Check if model optimizer exists
    if [ ! -f "${WORKSPACE_DIR}/models/optimizer.py" ]; then
        echo -e "${RED}ERROR: Model optimizer not found.${NC}"
        exit 1
    fi
    
    # Set environment variable for models directory
    export LLAMACLAIMS_MODELS_DIR="${MODELS_DIR}"
    
    # Benchmark all models or a specific model
    if [ -z "${model_id}" ]; then
        echo -e "Benchmarking all models..."
        python3 -m models.optimizer --all --models-dir "${MODELS_DIR}" --benchmark
    else
        echo -e "Benchmarking model: ${model_id}..."
        python3 -m models.optimizer --model "${model_id}" --models-dir "${MODELS_DIR}" --benchmark
    fi
    
    echo -e "${GREEN}Model benchmarking complete!${NC}"
}

# List models
list_models() {
    echo -e "${BLUE}Available models:${NC}"
    
    # Check if model optimizer exists
    if [ ! -f "${WORKSPACE_DIR}/models/optimizer.py" ]; then
        echo -e "${RED}ERROR: Model optimizer not found.${NC}"
        exit 1
    fi
    
    # Set environment variable for models directory
    export LLAMACLAIMS_MODELS_DIR="${MODELS_DIR}"
    
    # List models
    python3 -m models.optimizer --models-dir "${MODELS_DIR}" --list
}

# Run the API server
run_api() {
    local dev_mode=$1
    
    echo -e "${BLUE}Starting API server...${NC}"
    
    # Set environment variables
    export LLAMACLAIMS_MODELS_DIR="${MODELS_DIR}"
    export LLAMACLAIMS_UPLOADS_DIR="${UPLOADS_DIR}"
    export LLAMACLAIMS_CACHE_DIR="${CACHE_DIR}"
    
    # Source .env file if it exists
    if [ -f "${CONFIG_FILE}" ]; then
        source "${CONFIG_FILE}"
    fi
    
    # Determine run mode
    if [ "${dev_mode}" == "dev" ]; then
        echo -e "Running in development mode with auto-reload..."
        python3 run.py --reload --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --log-level "${LOG_LEVEL:-info}"
    else
        echo -e "Running in standard mode..."
        python3 run.py --host "${API_HOST:-0.0.0.0}" --port "${API_PORT:-8000}" --log-level "${LOG_LEVEL:-info}"
    fi
}

# Run the UI server (placeholder)
run_ui() {
    echo -e "${BLUE}UI server functionality will be available soon.${NC}"
    echo -e "${YELLOW}For now, please use the API directly or via Swagger UI at http://localhost:8000/docs${NC}"
}

# Docker operations
docker_ops() {
    local operation=$1
    
    echo -e "${BLUE}Docker operations...${NC}"
    
    # Check Docker prerequisites
    check_prereqs "docker"
    
    case "${operation}" in
        build)
            echo -e "Building Docker image..."
            docker build -t llamaclaims:latest .
            echo -e "${GREEN}Docker image built successfully!${NC}"
            ;;
        up)
            echo -e "Starting Docker containers..."
            if command -v docker-compose &> /dev/null; then
                docker-compose up -d
            else
                docker compose up -d
            fi
            echo -e "${GREEN}Docker containers started!${NC}"
            echo -e "API is available at ${YELLOW}http://localhost:8000${NC}"
            echo -e "Swagger UI is available at ${YELLOW}http://localhost:8000/docs${NC}"
            ;;
        down)
            echo -e "Stopping Docker containers..."
            if command -v docker-compose &> /dev/null; then
                docker-compose down
            else
                docker compose down
            fi
            echo -e "${GREEN}Docker containers stopped!${NC}"
            ;;
        logs)
            echo -e "Showing Docker logs..."
            if command -v docker-compose &> /dev/null; then
                docker-compose logs -f
            else
                docker compose logs -f
            fi
            ;;
        *)
            echo -e "${RED}Unknown Docker operation: ${operation}${NC}"
            echo -e "Available operations: build, up, down, logs"
            exit 1
            ;;
    esac
}

# Run tests
run_tests() {
    local test_type=$1
    
    echo -e "${BLUE}Running tests...${NC}"
    
    case "${test_type}" in
        unit)
            echo -e "Running unit tests..."
            pytest tests/unit -v
            ;;
        integration)
            echo -e "Running integration tests..."
            pytest tests/integration -v
            ;;
        performance)
            echo -e "Running performance tests..."
            pytest tests/performance -v
            ;;
        *)
            echo -e "Running all tests..."
            pytest
            ;;
    esac
}

# Clean up resources
clean() {
    echo -e "${BLUE}Cleaning up resources...${NC}"
    
    # Ask for confirmation
    read -p "This will delete all logs, cached data, and downloaded models. Are you sure? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo -e "Cleanup canceled."
        return
    fi
    
    # Remove logs, cache, and models
    echo -e "Removing logs..."
    rm -rf "${LOG_DIR}"/*
    
    echo -e "Removing cache..."
    rm -rf "${CACHE_DIR}"/*
    
    echo -e "Removing uploaded files..."
    rm -rf "${UPLOADS_DIR}"/*
    
    # Ask if models should be deleted too
    read -p "Do you want to delete downloaded models too? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo -e "Removing models..."
        rm -rf "${MODELS_DIR}"/*
    fi
    
    echo -e "${GREEN}Cleanup complete!${NC}"
}

# Show help
show_help() {
    echo -e "${BLUE}Available commands:${NC}"
    echo -e "  ${GREEN}help${NC}                           - Show this help message"
    echo -e "  ${GREEN}install${NC} [minimal|full|dev]     - Install dependencies"
    echo -e "  ${GREEN}setup${NC}                          - Set up the environment"
    echo
    echo -e "  ${GREEN}models list${NC}                    - List available models"
    echo -e "  ${GREEN}models download${NC} [<model_id>]   - Download models (all or specific model)"
    echo -e "  ${GREEN}models optimize${NC} [<model_id>] [<bits>] - Optimize models for MLX (8 or 16 bits)"
    echo -e "  ${GREEN}models benchmark${NC} [<model_id>]  - Benchmark models"
    echo
    echo -e "  ${GREEN}run api${NC} [dev]                  - Run the API server"
    echo -e "  ${GREEN}run ui${NC}                         - Run the UI server"
    echo
    echo -e "  ${GREEN}docker build${NC}                   - Build Docker image"
    echo -e "  ${GREEN}docker up${NC}                      - Start services with Docker"
    echo -e "  ${GREEN}docker down${NC}                    - Stop Docker services"
    echo -e "  ${GREEN}docker logs${NC}                    - View Docker logs"
    echo
    echo -e "  ${GREEN}test${NC} [unit|integration|performance] - Run tests"
    echo -e "  ${GREEN}clean${NC}                          - Clean up resources"
}

# Main function
main() {
    # Print banner
    print_banner
    
    # Check prerequisites
    check_prereqs
    
    # Process commands
    case "$1" in
        help)
            show_help
            ;;
        install)
            setup_env
            install_deps "$2"
            ;;
        setup)
            setup_env
            ;;
        models)
            case "$2" in
                list)
                    list_models
                    ;;
                download)
                    local force=""
                    if [ "$4" == "--force" ]; then
                        force="--force"
                    fi
                    download_models "$3" "$force"
                    ;;
                optimize)
                    optimize_models "$3" "$4"
                    ;;
                benchmark)
                    benchmark_models "$3"
                    ;;
                *)
                    echo -e "${RED}Unknown models command: $2${NC}"
                    echo -e "Run ${YELLOW}./llamaclaims.sh help${NC} for usage information."
                    exit 1
                    ;;
            esac
            ;;
        run)
            case "$2" in
                api)
                    run_api "$3"
                    ;;
                ui)
                    run_ui
                    ;;
                *)
                    echo -e "${RED}Unknown run command: $2${NC}"
                    echo -e "Run ${YELLOW}./llamaclaims.sh help${NC} for usage information."
                    exit 1
                    ;;
            esac
            ;;
        docker)
            docker_ops "$2"
            ;;
        test)
            run_tests "$2"
            ;;
        clean)
            clean
            ;;
        *)
            echo -e "${RED}Unknown command: $1${NC}"
            echo -e "Run ${YELLOW}./llamaclaims.sh help${NC} for usage information."
            exit 1
            ;;
    esac
}

# Execute main function with all arguments
main "$@" 