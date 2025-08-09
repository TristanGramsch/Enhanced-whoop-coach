FROM python:3.11

WORKDIR /app

# Install OS deps (if any needed later)
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy orchestrator requirements and install
COPY requirements.txt /app/orchestrator/requirements.txt
RUN pip install --no-cache-dir -r /app/orchestrator/requirements.txt mlflow==2.16.0

# Copy orchestrator code
COPY run_pipeline.py /app/orchestrator/run_pipeline.py

# Default command
WORKDIR /app/orchestrator
CMD ["python", "-u", "run_pipeline.py"]