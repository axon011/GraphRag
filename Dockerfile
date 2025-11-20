FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory for results
RUN mkdir -p output

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV GRAPHRAG_LLM_TYPE=ollama
ENV OLLAMA_BASE_URL=http://ollama:11434

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "from src.query_engine import MatchingEngine; print('OK')" || exit 1

# Command to run the application
CMD ["python", "-m", "pytest", "tests/", "-v", "--cov=src"]
