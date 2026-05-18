---
name: docker
description: Manage Docker containers, images, volumes, and networks. Build and publish images. Run multi-service apps with Docker Compose. Debug container issues with logs, exec, and inspect.
version: 1.0.0
author: Hermes Agent
license: MIT
metadata:
  hermes:
    tags: [Docker, Containers, DevOps, Compose, Images, Networking, Volumes]
    related_skills: [systematic-debugging]
---

# Docker

Manage containers, images, volumes, and networks using the Docker CLI. Covers single-container workflows and multi-service apps via Docker Compose.

## When to Use

- Start, stop, or restart containers
- Build, tag, push, or pull images
- Run multi-service applications with `docker compose`
- Debug a failing container (logs, exec, inspect)
- Clean up unused containers, images, volumes, or networks

## Prerequisites

```bash
# Verify Docker is installed and the daemon is running
docker info
```

If `docker info` fails:
- **Linux**: `sudo systemctl start docker`
- **macOS**: Start Docker Desktop from the Applications menu

---

## Quick Reference

| Task | Command |
|------|---------|
| List running containers | `docker ps` |
| List all containers | `docker ps -a` |
| Start a container | `docker start <name>` |
| Stop a container | `docker stop <name>` |
| Remove a container | `docker rm <name>` |
| View container logs | `docker logs -f <name>` |
| Execute a shell | `docker exec -it <name> sh` |
| List images | `docker images` |
| Pull an image | `docker pull <image>:<tag>` |
| Build an image | `docker build -t <name>:<tag> .` |
| Push an image | `docker push <name>:<tag>` |
| Compose up | `docker compose up -d` |
| Compose down | `docker compose down` |
| Compose logs | `docker compose logs -f` |
| Prune everything | `docker system prune -af --volumes` |

---

## Container Management

### Run a Container

```bash
# Basic run (pulls image automatically if absent)
docker run --name my-app -p 8080:80 -d nginx:alpine

# With environment variables and a volume
docker run --name my-app \
  -e DATABASE_URL=postgres://user:pass@db:5432/mydb \
  -v "$(pwd)/data:/app/data" \
  -p 8080:8080 \
  -d my-image:latest
```

### Inspect a Running Container

```bash
# Human-readable summary
docker inspect my-app

# Pull a specific field (e.g. IP address)
docker inspect --format '{{.NetworkSettings.IPAddress}}' my-app

# Helper script for a structured overview (see scripts/docker_inspect.py)
docker inspect my-app | python3 scripts/docker_inspect.py
```

### Container Lifecycle

```bash
docker stop  my-app        # graceful stop (SIGTERM → SIGKILL after 10 s)
docker kill  my-app        # immediate stop (SIGKILL)
docker start my-app        # start a stopped container
docker restart my-app      # stop + start
docker rm    my-app        # remove (must be stopped first)
docker rm -f my-app        # force remove (running containers too)
```

### Interactive Shell

```bash
# For images that have bash
docker exec -it my-app bash

# For minimal images (alpine, distroless)
docker exec -it my-app sh

# Run a one-off command
docker exec my-app cat /etc/os-release
```

---

## Logs

```bash
# Follow logs in real time
docker logs -f my-app

# Last 100 lines
docker logs --tail 100 my-app

# With timestamps
docker logs -f --timestamps my-app

# Since a specific time
docker logs --since 2024-01-01T00:00:00 my-app
```

---

## Image Management

### Build

```bash
# Build from Dockerfile in current directory
docker build -t my-app:latest .

# Build with a specific Dockerfile
docker build -f docker/Dockerfile.prod -t my-app:prod .

# Build-time arguments
docker build --build-arg NODE_ENV=production -t my-app:prod .

# Multi-platform build (push to registry)
docker buildx build --platform linux/amd64,linux/arm64 \
  -t myregistry/my-app:latest --push .
```

### Tag and Push

```bash
# Tag an existing image
docker tag my-app:latest myregistry/my-app:1.2.3

# Login to Docker Hub
docker login

# Login to a private registry
docker login myregistry.example.com

# Push
docker push myregistry/my-app:1.2.3
docker push myregistry/my-app:latest
```

### Pull and List

```bash
docker pull postgres:16-alpine       # pull a specific tag
docker images                        # list local images
docker images --filter dangling=true # list untagged (dangling) images
```

---

## Docker Compose

### Basic Workflow

```bash
# Start all services in the background
docker compose up -d

# Start specific services
docker compose up -d web db

# Rebuild images before starting
docker compose up -d --build

# Follow logs for all services
docker compose logs -f

# Follow logs for a specific service
docker compose logs -f web

# Stop and remove containers (preserves volumes)
docker compose down

# Stop and remove containers + volumes
docker compose down -v
```

### Compose File Example

```yaml
# docker-compose.yml
version: "3.9"

services:
  web:
    build: .
    ports:
      - "8080:8080"
    environment:
      - DATABASE_URL=postgres://user:pass@db:5432/mydb
    depends_on:
      db:
        condition: service_healthy
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: user
      POSTGRES_PASSWORD: pass
      POSTGRES_DB: mydb
    volumes:
      - pg_data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U user -d mydb"]
      interval: 10s
      timeout: 5s
      retries: 5

volumes:
  pg_data:
```

### Scale and Execute

```bash
# Scale a service to 3 replicas
docker compose up -d --scale worker=3

# Execute a command in a running Compose service
docker compose exec web bash
docker compose exec db psql -U user -d mydb

# Run a one-off command (new container, then remove)
docker compose run --rm web python manage.py migrate
```

---

## Volumes and Networks

### Volumes

```bash
# Create a named volume
docker volume create my-data

# List volumes
docker volume ls

# Inspect a volume (shows mount point)
docker volume inspect my-data

# Remove a volume
docker volume rm my-data

# Remove all unused volumes
docker volume prune
```

### Networks

```bash
# Create a user-defined network
docker network create my-network

# Run a container attached to the network
docker run --network my-network --name app my-image

# Connect a running container to a network
docker network connect my-network my-app

# Inspect a network (shows connected containers and their IPs)
docker network inspect my-network

# Remove unused networks
docker network prune
```

---

## Cleanup

```bash
# Remove stopped containers
docker container prune

# Remove dangling images (untagged)
docker image prune

# Remove all unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove unused networks
docker network prune

# Remove EVERYTHING unused at once (containers, images, volumes, networks)
docker system prune -af --volumes

# Show disk usage breakdown
docker system df
```

---

## Pitfalls

- **Port conflicts**: `docker: Error response from daemon: Ports are not available`. Use `lsof -i :<port>` (macOS/Linux) or `netstat -ano | findstr :<port>` (Windows) to find the conflicting process.
- **Container already exists**: Run `docker rm -f <name>` before restarting with the same name.
- **Permission denied on Docker socket**: Add user to the `docker` group with `sudo usermod -aG docker $USER`, then log out and back in (Linux only).
- **Volume mount shows empty directory**: Ensure the host path exists before starting the container; Docker creates a new empty volume rather than erroring if the path is missing.
- **Compose `depends_on` doesn't wait for readiness**: Add a `healthcheck` to the dependency and use `condition: service_healthy` in the dependent service's `depends_on` block.
- **Image not updated after code change**: Always pass `--build` to `docker compose up` or run `docker compose build` explicitly when there are Dockerfile changes.
- **Dangling images filling disk**: Run `docker image prune` regularly or add it to a cron job.

---

## Verification

```bash
# Confirm a container is running and healthy
docker ps --filter "name=my-app" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Confirm a Compose stack is up
docker compose ps

# Check container resource usage in real time
docker stats my-app

# Verify a specific port is exposed
docker port my-app
```
