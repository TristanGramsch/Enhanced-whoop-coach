# WHOOP API Client

This module handles all interactions with the WHOOP API to pull and process sensor data. It implements OAuth2 authentication using client ID and secret credentials to securely access user data. The module uses FastAPI to create endpoints that can trigger data pulls on-demand or on a schedule, retrieving daily metrics like HRV, sleep data, strain, and recovery scores. All pulled data is processed locally and stored in structured formats (CSV or Parquet) for downstream analysis by other modules in the HRV prediction pipeline. 

# Resources
https://developer.whoop.com/docs/introduction