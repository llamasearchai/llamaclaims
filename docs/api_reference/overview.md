# API Reference Overview

LlamaClaims provides a comprehensive REST API for insurance claims processing and analysis. This document provides an overview of the API structure, authentication, and common patterns.

## Base URL

All API endpoints are accessible under the base URL:

```
http://<host>:<port>/api/
```

For example, if the server is running on localhost port 8000, the base URL would be:

```
http://localhost:8000/api/
```

## API Structure

The API is organized into the following resource groups:

- **Claims**: `/api/claims/*` - Handling insurance claims
- **Analysis**: `/api/analysis/*` - Analysis of claims and policies
- **Documents**: `/api/documents/*` - Document processing and extraction
- **Models**: `/api/models/*` - Model management and information
- **Health**: `/health` - System health and status

## Authentication

The API supports JWT (JSON Web Token) based authentication. Most endpoints require authentication.

### Obtaining an Access Token

To obtain an access token, send a POST request to the `/api/auth/token` endpoint:

```bash
curl -X POST http://localhost:8000/api/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=password"
```

The response will contain an access token:

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

### Using the Access Token

Include the access token in the Authorization header for authenticated requests:

```bash
curl -X GET http://localhost:8000/api/claims \
  -H "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
```

## Response Format

All API responses follow a standard format:

```json
{
  "success": true,
  "data": { ... },
  "meta": { ... }
}
```

Or for errors:

```json
{
  "success": false,
  "error": "Error message",
  "detail": "Detailed error information"
}
```

## Pagination

List endpoints support pagination using the following query parameters:

- `page`: Page number (starts at 1)
- `limit`: Items per page (default: 20, max: 100)

Example:

```
GET /api/claims?page=2&limit=50
```

Response:

```json
{
  "success": true,
  "data": [ ... ],
  "meta": {
    "page": 2,
    "limit": 50,
    "total": 157,
    "pages": 4
  }
}
```

## Filtering and Sorting

List endpoints support filtering and sorting with query parameters:

- Filtering: `field=value`
- Sorting: `sort=field` or `sort=-field` (descending)

Example:

```
GET /api/claims?status=pending&sort=-created_at
```

## Rate Limiting

The API implements rate limiting to prevent abuse. Limits are included in response headers:

- `X-Rate-Limit-Limit`: The maximum number of requests allowed per period
- `X-Rate-Limit-Remaining`: The number of requests remaining in the current period
- `X-Rate-Limit-Reset`: The time when the current period will reset (Unix timestamp)

## Versioning

The API version is included in the response headers:

- `X-API-Version`: The current API version

## Error Codes

The API uses standard HTTP status codes to indicate the success or failure of a request:

- `200 OK`: The request was successful
- `201 Created`: The resource was created successfully
- `400 Bad Request`: The request was invalid
- `401 Unauthorized`: Authentication is required
- `403 Forbidden`: The authenticated user doesn't have permission
- `404 Not Found`: The requested resource was not found
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error

## Detailed API Sections

For detailed information about specific API endpoints, refer to the following sections:

- [Claims API](claims.md): Manage insurance claims
- [Analysis API](analysis.md): Analyze claims and policies
- [Documents API](documents.md): Process documents and extract information
- [Models API](models.md): Manage machine learning models
- [Authentication](authentication.md): Authentication and authorization details

## OpenAPI Specification

The complete OpenAPI specification is available at:

```
http://localhost:8000/docs
```

Or in JSON format:

```
http://localhost:8000/openapi.json
```

## API Client

For Python users, we provide an official Python client:

```python
from llamaclaims.client import LlamaClaimsClient

# Initialize client
client = LlamaClaimsClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# List claims
claims = client.claims.list(status="pending")

# Create a claim
new_claim = client.claims.create({
    "policy_number": "POL-12345",
    "claim_amount": 1500.00,
    "description": "Water damage to kitchen floor"
})
```

See the [API Client Example](../examples/api_client.md) for more detailed usage examples. 