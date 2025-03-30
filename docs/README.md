# LlamaClaims Documentation

This directory contains the source files for the LlamaClaims documentation site.

## Overview

The documentation is built using [MkDocs](https://www.mkdocs.org/) with the [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/) theme. The API reference is generated using [mkdocstrings](https://mkdocstrings.github.io/).

## Building the Documentation

### Prerequisites

Install the documentation dependencies:

```bash
pip install -r requirements.txt
```

### Local Development

To serve the documentation locally:

```bash
# From the project root
mkdocs serve
```

This will start a local server at http://localhost:8000 where you can preview the documentation. The site will automatically reload when you make changes to the documentation files.

### Building the Static Site

To build the static site:

```bash
# From the project root
mkdocs build
```

This will generate the static site in the `site` directory.

## Deploying the Documentation

### Manual Deployment

To deploy the documentation to GitHub Pages manually:

```bash
# From the project root
mkdocs gh-deploy
```

This will build the site and push it to the `gh-pages` branch of the repository.

### Using the Deploy Script

Alternatively, you can use the provided deploy script:

```bash
# From the project root
./docs/deploy.sh
```

## Documentation Structure

- `docs/index.html`: Home page
- `docs/user_guide/`: User guide documentation
- `docs/api_reference/`: API reference documentation
- `docs/developer_guide/`: Documentation for developers
- `docs/deployment/`: Deployment guides
- `docs/examples/`: Example code and usage scenarios
- `docs/assets/`: Images, logos, and other assets
- `docs/stylesheets/`: Custom CSS files

## Adding New Documentation

1. Create a new Markdown file in the appropriate directory
2. Add the file to the navigation in `mkdocs.yml`
3. Build and preview the documentation to ensure it looks correct

## Style Guide

- Use ATX-style headers (`#` for main headers, `##` for subheaders)
- Use fenced code blocks with language specifiers (````python`)
- Use relative links for internal links (`[link text](../path/to/file.html)`)
- Use reference-style links for external links
- Include alt text for all images
- Keep lines to a reasonable length (80-100 characters)

## API Documentation

The API documentation is generated from docstrings in the codebase. To ensure proper generation:

1. Use Google-style docstrings
2. Include type hints in function signatures
3. Document parameters, return values, and exceptions

Example:

```python
def process_claim(claim_id: str, force: bool = False) -> dict:
    """Process an insurance claim.
    
    Args:
        claim_id: The unique identifier of the claim to process
        force: If True, force reprocessing even if already processed
        
    Returns:
        A dictionary containing the processing results
        
    Raises:
        ClaimNotFoundError: If the claim could not be found
        ValidationError: If the claim data is invalid
    """
    # Implementation
``` 