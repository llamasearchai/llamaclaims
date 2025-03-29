# LlamaClaims API Examples

This directory contains examples of how to use the LlamaClaims API.

## API Client Example

`api_client_example.py` demonstrates how to use the LlamaClaims API client to:

- Check API health
- List and get information about available models
- Run inference with models
- Benchmark model performance
- Create and manage claims
- Process documents

### Usage

Make sure the LlamaClaims API server is running:

```bash
# Start the API server
./llamaclaims.sh run api
```

Then run the example:

```bash
# Run the example with the default API URL (http://localhost:8000)
python examples/api_client_example.py

# Or specify a custom API URL
python examples/api_client_example.py --api-url http://example.com:8000
```

## Python Client Library

The `LlamaClaimsClient` class in `api_client_example.py` can be reused in your own applications:

```python
from examples.api_client_example import LlamaClaimsClient

# Create a client
client = LlamaClaimsClient(base_url="http://localhost:8000")

# Check API health
health = client.check_health()
print(f"API Status: {health['status']}")

# List available models
models = client.list_models()
```

## Adding More Examples

Feel free to add more examples to this directory. Some ideas:

- Document processing workflow
- Custom ML model integration
- Batch claims processing
- Analytics and reporting examples 