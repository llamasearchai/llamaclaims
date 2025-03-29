#!/usr/bin/env python3
"""
LlamaClaims API Runner

This script serves as the entry point for running the LlamaClaims API server.
It configures logging, sets environment variables, and launches the FastAPI application.
"""

import os
import sys
import argparse
import logging
import uvicorn
from pathlib import Path
from dotenv import load_dotenv

# Add the current directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

logger = logging.getLogger("llamaclaims")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="LlamaClaims API Server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--host", 
        type=str, 
        default=os.getenv("API_HOST", "127.0.0.1"),
        help="Host to run the server on"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=int(os.getenv("API_PORT", 8000)),
        help="Port to run the server on"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "info"),
        choices=["debug", "info", "warning", "error", "critical"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    return parser.parse_args()


def setup_env():
    """Load environment variables and set up app environment."""
    # Load .env file
    load_dotenv()
    
    # Define important directories
    models_dir = os.getenv("MODELS_DIR", "./data/models")
    uploads_dir = os.getenv("UPLOADS_DIR", "./data/uploads")
    cache_dir = os.getenv("CACHE_DIR", "./data/cache")
    logs_dir = os.getenv("LOGS_DIR", "./logs")
    
    # Create directories if they don't exist
    for directory in [models_dir, uploads_dir, cache_dir, logs_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Set environment variables for use throughout the application
    os.environ["LLAMACLAIMS_MODELS_DIR"] = models_dir
    os.environ["LLAMACLAIMS_UPLOADS_DIR"] = uploads_dir
    os.environ["LLAMACLAIMS_CACHE_DIR"] = cache_dir
    os.environ["LLAMACLAIMS_LOGS_DIR"] = logs_dir
    
    # Set app environment if not already set
    if "APP_ENV" not in os.environ:
        os.environ["APP_ENV"] = os.getenv("ENVIRONMENT", "development")
    
    # Log environment
    env = os.environ["APP_ENV"]
    logger.info(f"Starting LlamaClaims API in {env.upper()} mode")
    if env == "development":
        logger.info(f"Models directory: {models_dir}")
        logger.info(f"Uploads directory: {uploads_dir}")
        logger.info(f"Cache directory: {cache_dir}")
    
    # Configure logging level
    return env


def main():
    """Main entry point for the API server."""
    # Parse command line arguments
    args = parse_args()
    
    # Setup environment
    env = setup_env()
    
    # Configure log level
    log_level = args.log_level.upper()
    logging.getLogger().setLevel(log_level)
    
    # Print startup banner
    print("\n" + "=" * 60)
    print(" " * 15 + "LlamaClaims API Server Starting")
    print("-" * 60)
    print(f" Host:        {args.host}")
    print(f" Port:        {args.port}")
    print(f" Environment: {env}")
    print(f" Log Level:   {log_level}")
    print(f" Auto-Reload: {'Enabled' if args.reload else 'Disabled'}")
    print("=" * 60 + "\n")
    
    # Start the server using Uvicorn
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        reload=args.reload
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Server shutdown initiated by user")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Error starting server: {str(e)}")
        sys.exit(1) 