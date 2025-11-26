# MetaGuard Production Dockerfile
# Author: Moslem Mohseni
#
# Multi-stage build for minimal production image

# ============================================================================
# Stage 1: Builder
# ============================================================================
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install package with API dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[api]"

# ============================================================================
# Stage 2: Production
# ============================================================================
FROM python:3.11-slim as production

# Labels
LABEL maintainer="Moslem Mohseni <moslem.mohseni@example.com>"
LABEL org.opencontainers.image.title="MetaGuard"
LABEL org.opencontainers.image.description="Fraud detection for metaverse transactions"
LABEL org.opencontainers.image.version="1.1.0"
LABEL org.opencontainers.image.source="https://github.com/moslem-mohseni/MetaGuard"
LABEL org.opencontainers.image.licenses="MIT"

# Create non-root user for security
RUN groupadd --gid 1000 metaguard && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home metaguard

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=metaguard:metaguard src/ ./src/

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    METAGUARD_LOG_LEVEL=INFO \
    METAGUARD_RISK_THRESHOLD=0.5

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()" || exit 1

# Switch to non-root user
USER metaguard

# Run the API server
CMD ["python", "-m", "uvicorn", "metaguard.api.rest:app", "--host", "0.0.0.0", "--port", "8000"]
