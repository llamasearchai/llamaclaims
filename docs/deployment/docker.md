# Docker Deployment

This guide covers deploying LlamaClaims using Docker and Docker Compose, which is the recommended way to deploy in production environments.

## Prerequisites

Before you begin, ensure you have the following installed:

- [Docker](https://docs.docker.com/get-docker/) (version 20.10.0 or higher)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 2.0.0 or higher)
- Git (for cloning the repository)

## Quick Start with Docker Compose

The simplest way to get LlamaClaims running with Docker is to use Docker Compose:

1. Clone the repository:
   ```bash
   git clone https://github.com/llamasearchai/llamaclaims.git
   cd llamaclaims
   ```

2. Create environment configuration:
   ```bash
   cp .env.example .env
   ```
   
   Edit the `.env` file to customize your deployment settings.

3. Start the services:
   ```bash
   docker-compose up -d
   ```

4. Check the logs:
   ```bash
   docker-compose logs -f
   ```

5. Access the API at: `http://localhost:8000/docs`

## Manual Docker Deployment

If you prefer to manage the containers manually or need more customization:

1. Build the Docker image:
   ```bash
   docker build -t llamaclaims:latest .
   ```

2. Create data directories:
   ```bash
   mkdir -p data/models data/uploads data/cache logs
   ```

3. Run the container:
   ```bash
   docker run -d \
     --name llamaclaims \
     -p 8000:8000 \
     -v $(pwd)/data:/app/data \
     -v $(pwd)/logs:/app/logs \
     -v $(pwd)/.env:/app/.env \
     llamaclaims:latest
   ```

## Environment Configuration

The Docker deployment uses environment variables for configuration. Key variables include:

- `API_HOST`: The host to bind the server to (default: `0.0.0.0`)
- `API_PORT`: The port to bind the server to (default: `8000`)
- `LOG_LEVEL`: Logging level (default: `info`)
- `ENVIRONMENT`: `development`, `staging`, or `production` (default: `production`)
- `SECRET_KEY`: Security key for JWT token generation

See the [Configuration](../user_guide/configuration.md) guide for a complete list of environment variables.

## Docker Compose Configuration

The default `docker-compose.yml` file includes the following services:

```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
    env_file:
      - .env
    restart: unless-stopped
```

### Adding Redis for Caching (Optional)

To enhance performance with Redis caching, add this to your `docker-compose.yml`:

```yaml
services:
  # ... existing api service ...
  
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    restart: unless-stopped

volumes:
  redis-data:
```

And update your `.env` file:

```
REDIS_HOST=redis
REDIS_PORT=6379
```

### Adding a Database (Optional)

For persistent storage, add a PostgreSQL database:

```yaml
services:
  # ... existing services ...
  
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: llamaclaims
      POSTGRES_PASSWORD: securepassword
      POSTGRES_DB: llamaclaims
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres-data:
```

And update your `.env` file:

```
DB_HOST=db
DB_PORT=5432
DB_USER=llamaclaims
DB_PASSWORD=securepassword
DB_NAME=llamaclaims
```

## Production Deployment Considerations

For production environments, consider the following enhancements:

### 1. Using a Reverse Proxy

For production, use a reverse proxy like Nginx:

```yaml
services:
  # ... existing services ...
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf
      - ./nginx/certs:/etc/nginx/certs
    depends_on:
      - api
    restart: unless-stopped
```

### 2. Implementing Health Checks

Add health checks to your services:

```yaml
services:
  api:
    # ... existing configuration ...
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
```

### 3. Setting Resource Limits

Set memory and CPU limits for containers:

```yaml
services:
  api:
    # ... existing configuration ...
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## Container Security

Enhance security with these practices:

1. **Run as non-root user**: The Dockerfile already configures this
2. **Use secrets management**: For production, use Docker secrets instead of environment variables
3. **Scan images for vulnerabilities**: Use tools like Docker Scout or Trivy
4. **Use read-only file systems**: Add `read_only: true` to your service

## Backup and Restore

### Backup

Backup the data directories:

```bash
# Stop the containers
docker-compose down

# Backup the data
tar -czvf llamaclaims-backup-$(date +%Y%m%d).tar.gz data logs

# Restart the containers
docker-compose up -d
```

### Restore

Restore from a backup:

```bash
# Stop the containers
docker-compose down

# Restore the data
tar -xzvf llamaclaims-backup-20240315.tar.gz

# Restart the containers
docker-compose up -d
```

## Monitoring

For monitoring the Docker deployment, consider:

- **Prometheus & Grafana**: For metrics collection and visualization
- **Loki**: For log aggregation
- **Docker stats**: Use `docker stats` for basic monitoring

## Troubleshooting

Common issues and solutions:

**Issue**: Container fails to start
**Solution**: Check logs with `docker-compose logs api`

**Issue**: API is unreachable
**Solution**: Ensure ports are correctly mapped and not blocked by firewalls

**Issue**: High memory usage
**Solution**: Adjust model size or quantization level in the `.env` file

```
DEFAULT_QUANTIZATION_BITS=4
```

## Next Steps

- [Kubernetes Deployment](kubernetes.md) - For scaling to multiple nodes
- [Cloud Deployment](cloud.md) - For deploying to cloud providers
- [Configuration](../user_guide/configuration.md) - Detailed configuration options 