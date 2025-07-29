# HRV Prediction Infrastructure

This is the MVP of a modular system for predicting future Heart Rate Variability (HRV) based on WHOOP sensor data and daily journal input. 

The architecture is a full analytics environment. With ML experimentation, data intelligence, and voice journal analysis. Each component generates reports. A savy LLM sits on top of the infrastructure. Reads reports. And makes an opinionated HRV prediction for tomorrow and a week in advance. Checks accuracy every day. 

---

## Project Structure

```
hrv-infrastructure/
├── whoop_api_client/       # Pull and process WHOOP data (FastAPI)
├── model_training/         # Train ML models and track experiments with MLflow
├── data_intelligence/      # Analyze HRV trends and model outputs
├── journal_analysis/       # Extract behavioral features from voice journals
├── llm_agent/              # Reason over data and predict HRV (prompt-based)
├── orchestrator/           # Schedule and orchestrate all workflows (Dagster)
├── shared/                 # Common constants, schemas, and utilities
└── README.md
```

Each subdirectory can be developed as a Git submodule.

---

## Components

### 1. `whoop_api_client/`

- Pull WHOOP data daily using OAuth2 (client ID & secret).
- Use FastAPI to implement and trigger data pulls.
- Process data locally and store it in structured formats (e.g., CSV or Parquet).

### 2. `model_training/`

- Use PyCaret, scikit-learn, XGBoost, or PyTorch for model training.
- Track all experiments using **MLflow**:
  - Log model parameters, training metrics, and artifacts.
  - Register top N models (e.g., best 10) for downstream use.
- Train models to predict:
  - HRV for the next day.
  - HRV for the next 7 days (rolling forecast).

### 3. `data_intelligence/`

- Analyze raw WHOOP data and model predictions.
- Identify trends and forecast reliability.
- Visualize prediction accuracy, HRV trends, and anomalies.
- Output daily and weekly structured summaries for the agent.

### 4. `journal_analysis/`

- Accept daily voice recordings.
- Transcribe audio (e.g., via Whisper).
- Extract structured behavioral/contextual features.
- Time-align journal insights with WHOOP and model data.

### 5. `llm_agent/`

- A simple prompt-based LLM that:
  - Reads in data summaries and model outputs.
  - Produces a probabilistic HRV forecast.
  - Justifies its prediction in natural language.

### 6. `orchestrator/`

- Use **Dagster** to:
  - Define daily and on-demand pipelines.
  - Connect all modules: data pull, journal parse, model run, agent output.
  - Monitor and track task runs.

### 7. `shared/`

- Store commonly used constants (e.g., metric thresholds, file paths).
- Shared schemas for structured data across modules.
- Utility functions reused by more than one submodule (e.g., date formatting, data validators).

---

## Setup Instructions

### Install Python Dependencies

Each module will have its own `conda` environment. Activate the environment per submodule and install dependencies as defined in each `environment.yml`.

--

## MVP Objective

The MVP should:

- Pull one full day of WHOOP data
- Train at least one model using historical data
- Transcribe one journal entry
- Produce an LLM-based probabilistic HRV forecast
- Log outputs, and allow inspection of where predictions succeed or fail

---

## Notes

- All processing is local.
- Results and logs will be reviewed manually for now (visualization tools to come).

---

## Next Steps

- Implement data connectors and base training pipelines.
- Set up MLflow tracking.
- Begin defining agent prompts and format of prediction records.
- Establish Dagster pipelines to automate the daily flow.
