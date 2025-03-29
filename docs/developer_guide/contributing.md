# Contributing to LlamaClaims

Thank you for your interest in contributing to LlamaClaims! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

Please read and follow our [Code of Conduct](https://github.com/llamasearchai/llamaclaims/blob/main/CODE_OF_CONDUCT.md) to foster an inclusive and respectful community.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- A GitHub account

### Development Environment Setup

1. Fork the repository on GitHub.

2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/llamaclaims.git
   cd llamaclaims
   ```

3. Set up the upstream remote:
   ```bash
   git remote add upstream https://github.com/llamasearchai/llamaclaims.git
   ```

4. Create a virtual environment and install development dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements-dev.txt
   ```

5. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Development Workflow

### Branching Strategy

We use a simplified Git flow approach:

- `main`: Stable production-ready code
- `development`: Integration branch for new features
- Feature branches: Individual features and bug fixes

Always create feature branches from the `development` branch:

```bash
git checkout development
git pull upstream development
git checkout -b feature/your-feature-name
```

### Coding Standards

We follow these standards:

- **Code Style**: [PEP 8](https://pep8.org/)
- **Docstrings**: [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
- **Type Hints**: Use type hints for all function parameters and return values

Use the provided linting and formatting tools:

```bash
# Run the linter
flake8 llamaclaims tests

# Run the type checker
mypy llamaclaims

# Run code formatter
black llamaclaims tests

# Sort imports
isort llamaclaims tests
```

### Testing

Write tests for all new features and bug fixes. We use `pytest` for testing:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=llamaclaims

# Run a specific test
pytest tests/test_specific_module.py::test_specific_function
```

Test files should be placed in the `tests/` directory with a structure that mirrors the package.

### Commit Messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification for commit messages:

```
feat: add new fraud detection algorithm
^--^  ^----------------------------^
|     |
|     +-> Summary in present tense
|
+-------> Type: feat, fix, docs, style, refactor, test, chore
```

Common types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring without functionality changes
- `test`: Adding or improving tests
- `chore`: Maintenance tasks, dependency updates, etc.

### Pull Requests

1. Ensure all tests pass and linting issues are resolved.

2. Update documentation if needed.

3. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```

4. Create a pull request to the `development` branch of the main repository.

5. Fill out the pull request template with all relevant information.

6. Wait for code review and address any feedback.

## Documentation

When adding new features or making significant changes, update the documentation:

1. Update docstrings for new functions, classes, and modules.

2. Update or create markdown files in the `docs/` directory.

3. If your changes affect the API, update the API reference.

4. Add examples if applicable.

We use MkDocs with the Material theme for documentation. To preview changes:

```bash
# Install documentation dependencies
pip install -r docs/requirements.txt

# Serve the documentation locally
mkdocs serve
```

## API Changes

When making changes to the API:

1. Maintain backward compatibility whenever possible.

2. If breaking changes are necessary, discuss them in an issue first.

3. Document all changes in the API reference and highlight them in CHANGELOG.md.

## Performance Considerations

When working on performance-sensitive code:

1. Add benchmarks to measure the performance impact of your changes.

2. Optimize for both Apple Silicon (using MLX) and other platforms.

3. Document performance trade-offs and considerations.

## Release Process

Releases are managed by maintainers, but here's an overview:

1. We follow [Semantic Versioning](https://semver.org/).

2. Release notes are maintained in CHANGELOG.md.

3. Releases are tagged in Git and published to PyPI.

## Issue Reporting

If you find a bug or have a feature request:

1. Check if the issue already exists in the [GitHub Issues](https://github.com/llamasearchai/llamaclaims/issues).

2. If not, create a new issue using the appropriate template.

3. Provide as much relevant information as possible, including:
   - For bugs: Steps to reproduce, expected vs. actual behavior, environment details
   - For features: Use case, benefits, potential implementation approach

## Getting Help

If you need help with your contribution:

- Ask questions in the relevant GitHub issue
- Join our [Discord server](https://discord.gg/llamasearchai)
- Reach out to the maintainers via email

## Acknowledgements

Contributors are acknowledged in our [Contributors List](https://github.com/llamasearchai/llamaclaims/graphs/contributors). We appreciate all contributions, no matter how small!

Thank you for contributing to LlamaClaims! 