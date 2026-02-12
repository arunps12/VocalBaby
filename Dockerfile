FROM python:3.10-slim-bookworm

# noninteractive installs
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
        awscli \
        ffmpeg \
        libsndfile1 \
        libstdc++6 \
        curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files first (layer caching)
COPY pyproject.toml uv.lock ./

# Install dependencies (frozen from lock file)
RUN uv sync --frozen --no-dev --no-editable

# Copy source code
COPY src/ ./src/
COPY configs/ ./configs/

# Copy final model artifacts (if available)
COPY final_model/ ./final_model/

# Expose FastAPI port
EXPOSE 8000

# Run VocalBaby server via console script
CMD ["uv", "run", "vocalbaby-serve"]