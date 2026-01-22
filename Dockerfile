# Standard Dockerfile for API and Services

FROM python:3.10-slim-bookworm

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_DEFAULT_TIMEOUT=1000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    tesseract-ocr \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy requirements file (ensure it exists)
COPY requirements.txt .

# Install dependencies (use index-url for torch if needed, or rely on requirements.txt)
# Using standard index for generalized deps
RUN pip install --no-cache-dir -r requirements.txt

# Set working directory
WORKDIR /app

# Copy application code
COPY . /app

# Expose API port
EXPOSE 8000

# Default command (overridden by docker-compose)
CMD ["python", "api_server.py"]
