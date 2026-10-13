# DeepGuard Containerized Deployment
# Multi-stage or slim Python 3.11 image compatible with OpenCV and TensorFlow

FROM python:3.11-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for OpenCV and video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and static web assets
COPY . /app

# Ensure runtime directories exist
RUN mkdir -p /app/experiments/temp_uploads /app/experiments/api_artifacts

EXPOSE 8000

# Healthcheck probe using built-in /api/v1/health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/api/v1/health || exit 1

# Launch production server
CMD ["python", "scripts/run_server.py", "--host", "0.0.0.0", "--port", "8000"]
