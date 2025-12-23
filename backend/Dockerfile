FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e .

# Copy application code
COPY tunix_rt_backend/ ./tunix_rt_backend/

# Expose port
EXPOSE 8000

# Run application
CMD ["uvicorn", "tunix_rt_backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
