FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set workdir to the module so relative paths (../data) resolve to /app/data
WORKDIR /app/whoop_api_client

# System dependencies (minimal)
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	&& rm -rf /var/lib/apt/lists/*

# Install Python deps first for layer caching
COPY whoop_api_client/requirements.txt /app/whoop_api_client/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY whoop_api_client/ /app/whoop_api_client/

# Create data directory at /app/data (used by WHOOPDataFetcher via ../data)
RUN mkdir -p /app/data

EXPOSE 8000

CMD ["uvicorn", "whoop_server:app", "--host", "0.0.0.0", "--port", "8000"]