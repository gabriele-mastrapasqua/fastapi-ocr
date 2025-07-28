compose = @docker compose

# Detect GPU at runtime
ifeq ($(shell nvidia-smi > /dev/null 2>&1 && echo yes || echo no),yes)
  compose_file = -f docker-compose.yml -f docker-compose.nvidia.yml
else
  compose_file = -f docker-compose.yml
endif


.PHONY: help build up down

# Default target
help:
	@echo "FastAPI PaddleOCR API - Available commands:"
	@echo ""
	@echo "🚀 Development:"
	@echo "  make up    - Quick start development"
	@echo "  make build    - Build development"
	@echo "  make down    - Shutdown development"

# Build Docker images
build:
	$(compose) $(compose_file) build --no-cache --pull

# Start services
up:
	$(compose) $(compose_file) up -d

# Stop services
down:
	@echo "calling docker compose down..."
	$(compose) $(compose_file) down
	@echo "called docker compose down..."
