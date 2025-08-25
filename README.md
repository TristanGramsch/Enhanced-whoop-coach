# Recovery Prediction Infrastructure

This is the MVP of a modular system for predicting future recovery. Recovery is composed by: Heart Rate Variability, Resting Heart Rate, and Respiratory rate (HRV, RHR, RR). Prediction happens using WHOOP sensor data and a journal. 

The architecture is a full analytics environment. With ML experimentation, data intelligence, and voice journal analysis. A savy LLM sits on top of the infrastructure. Reads reports. And makes an opinionated recovery prediction for tomorrow. Then checks accuracy every day, and remembers. 

## Project Structure

```
recovery-infrastructure/
├── whoop_client/       # Pull and process WHOOP data (FastAPI)
├── training/           # Train ML models and track experiments with MLflow
├── analytics/          # Analyze recovery trends and model outputs
├── journal/            # Record and extract insights from voice journals
├── coach/              # Reason over data and predict recovery
├── orchestrator/       # Schedule and orchestrate all workflows (Dagster)
├── outputs/            # Artifacts, predictions, metrics, MLflow store
├── data/               # WHOOP data (raw/processed)
├── tests/              # Test suite
├── frontend/           # Minimal Streamlit app with links and charts
└── README.md
```

## Components

### 1. `whoop_client/`

- Continously pull WHOOP data. 
- Use OAuth2 to authenticate (client ID & secret).
- Process data and store it in a duckDB.

### 2. `training/`

- Use PyCaret, scikit-learn, XGBoost, or PyTorch for model training. The more models, the better. 
- Track experiments using **MLflow**:
  - Log model parameters, training metrics, and artifacts.
  - Register top N models (e.g., best 10) for downstream use.
- Predict all recovery metrics (HRV, RHR, RR). Train models for each and track their performance. 
- Registered models continously re-train and report predictions. 

### 3. `analytics/`

- Analyze data and model predictions.
- Identify trends and forecast reliability.
- Visualize prediction accuracy and trends for HRV, RHR, and RR, plus anomalies.
- Output daily and weekly structured summaries for the agent.

### 4. `journal/`

- Accept daily voice recordings.
- Transcribe audio.
- Perform voice analysis. 

### 5. `coach/`

- An agent (like GPT-5) that:
  - Reads in data summaries and model outputs.
  - Remembers past choices and outcomes.
  - Produces probabilistic forecasts for HRV, RHR, and RR.
  - Justifies its prediction in natural language.

### 6. `orchestrator/`

- Use **Dagster** to:
  - Define daily and on-demand pipelines.
  - Connect all modules: data pull, journal parse, model run, agent output.
  - Monitor and track task runs.

### 7. `frontend/`

  - For humans. Showing links to all tools. 
 
## Setup 

Each module is containerized. A compose file orchestrates. 
Test.  

## MVP Objective

The MVP should:

- Pull one full day of WHOOP data
- Train at least one model using historical data
- Transcribe one journal entry
- Produce an LLM-based probabilistic recovery forecast (HRV, RHR, RR)
- Log outputs, and allow inspection of where predictions succeed or fail