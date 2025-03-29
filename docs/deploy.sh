#!/bin/bash
# Script to build and deploy documentation to GitHub Pages

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'  # No Color

echo -e "${YELLOW}Starting documentation deployment...${NC}"

# Ensure we're in the project root
if [ ! -f "mkdocs.yml" ]; then
    echo "Error: mkdocs.yml not found, make sure you're in the project root."
    exit 1
fi

# Install or update dependencies
echo -e "${YELLOW}Installing documentation dependencies...${NC}"
pip install -r docs/requirements.txt

# Build documentation
echo -e "${YELLOW}Building documentation...${NC}"
mkdocs build --clean

# Deploy to GitHub Pages
echo -e "${YELLOW}Deploying to GitHub Pages...${NC}"
mkdocs gh-deploy --force

echo -e "${GREEN}Documentation has been successfully deployed to GitHub Pages!${NC}"
echo -e "${GREEN}View it at: https://llamasearchai.github.io/llamaclaims/${NC}" 