# WHOOP API Client

This module handles WHOOP API interactions to pull and process sensor data. It implements OAuth2 authentication and exposes FastAPI endpoints to trigger data pulls on-demand. It retrieves daily recovery‑related metrics including HRV, Resting Heart Rate (RHR), Respiratory Rate (RR), sleep, strain, and recovery scores. Data is stored locally (JSON/CSV/Parquet) under `data/` for downstream analytics and modeling across all recovery metrics.

# Resources
https://developer.whoop.com/docs/introduction