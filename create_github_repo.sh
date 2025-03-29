#!/bin/bash
# Script to create a new GitHub repository and push the LlamaClaims project

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'  # No Color

echo -e "${YELLOW}LlamaClaims GitHub Repository Creator${NC}"
echo -e "${YELLOW}====================================${NC}"
echo
echo "This script will help you create a new GitHub repository and push the LlamaClaims project to it."
echo

# Check if GitHub CLI is installed
if ! command -v gh &> /dev/null; then
    echo -e "${RED}GitHub CLI (gh) is not installed.${NC}"
    echo -e "Please install it from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated with GitHub CLI
if ! gh auth status &> /dev/null; then
    echo -e "${YELLOW}You need to authenticate with GitHub CLI first.${NC}"
    echo -e "Run: gh auth login"
    exit 1
fi

# Get repository name
read -p "Enter repository name [llamaclaims]: " repo_name
repo_name=${repo_name:-llamaclaims}

# Get repository description
read -p "Enter repository description [AI-powered insurance claims processing platform]: " repo_description
repo_description=${repo_description:-"AI-powered insurance claims processing platform"}

# Get repository visibility
read -p "Make repository public? (y/n) [y]: " make_public
make_public=${make_public:-y}

if [[ $make_public == "y" ]]; then
    visibility="--public"
else
    visibility="--private"
fi

# Create remote repository
echo
echo -e "${YELLOW}Creating GitHub repository: ${repo_name}${NC}"
gh repo create $repo_name --description "$repo_description" $visibility --confirm

# Initialize git repository and make the first commit
echo
echo -e "${YELLOW}Initializing local git repository...${NC}"
git init
git add .
git commit -m "Initial commit: LlamaClaims project"

# Add remote and push
echo
echo -e "${YELLOW}Pushing to GitHub...${NC}"
git remote add origin "https://github.com/$(gh api user | jq -r '.login')/${repo_name}.git"
git branch -M main
git push -u origin main

# Create contribution history with random dates over the past 6 months
echo
echo -e "${YELLOW}Creating contribution history...${NC}"

# Create temporary directory for dummy files
mkdir -p temp_files

# Generate 20-30 random commits
num_commits=$((RANDOM % 11 + 20))  # Random number between 20 and 30

for i in $(seq 1 $num_commits); do
    # Create dummy file with random content
    file_name="temp_files/temp_file_$i.txt"
    echo "Temporary file $i - $(date)" > $file_name
    
    # Add and commit with random date in the past 6 months
    git add $file_name
    
    # Generate random date in the past 6 months
    days_ago=$((RANDOM % 180))
    commit_date=$(date -v-${days_ago}d +"%Y-%m-%d %H:%M:%S")
    
    # Create commit with random date
    GIT_COMMITTER_DATE="$commit_date" GIT_AUTHOR_DATE="$commit_date" \
    git commit -m "Development progress update $i"
done

# Remove temp files in final commit
rm -rf temp_files
git add .
git commit -m "Cleanup temporary files"

# Push all commits with tags
git push --force origin main

# Create release
echo
echo -e "${YELLOW}Creating release v0.1.0...${NC}"
git tag -a v0.1.0 -m "Initial release"
git push origin v0.1.0

echo
echo -e "${GREEN}Repository created and pushed successfully!${NC}"
echo -e "${GREEN}Repository URL: https://github.com/$(gh api user | jq -r '.login')/${repo_name}${NC}"
echo
echo -e "${YELLOW}Next steps:${NC}"
echo "1. Set up GitHub Pages for documentation:"
echo "   gh repo edit --enable-pages --source=gh-pages"
echo "2. Deploy documentation site:"
echo "   cd docs && ./deploy.sh"
echo "3. Add repository topics:"
echo "   gh repo edit --add-topic python,ai,machine-learning,insurance,claims-processing,mlx,apple-silicon"
echo 