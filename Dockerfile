# Use official Python 3.11 image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN pip install uv

# Copy project configuration and source code
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install dependencies using uv
RUN uv pip install --system -e .

# Copy remaining directories
COPY tests/ ./tests/
COPY notebooks/ ./notebooks/

# Expose MLflow port
EXPOSE 5000

# Default command: Run MLflow server
CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "./mlruns", "--host", "0.0.0.0"]