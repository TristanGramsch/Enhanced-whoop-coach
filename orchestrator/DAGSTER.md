# Dagster in this Project: Why, How, and Key Concepts

Dagster is our data orchestrator. It gives us a typed, testable way to define and observe the end‑to‑end recovery pipeline (journal → training → analytics → agent), run it on demand or on a schedule, and inspect runs, logs, and artifacts in a friendly UI.

Why Dagster
- Reliability and visibility: rich run metadata and logs, easy failure surfacing.
- Modularity: each step is an op; dependencies are explicit and composable.
- Local to prod parity: same code runs in the web UI or headless in CI/CD.
- Concurrency controls: multiprocess execution for parallel safe steps.

How it’s used here
- Location: `orchestrator/dagster_app/` holds the Dagster workspace.
- Entry: `repository.py` defines `hrv_daily_job` and `defs` that the webserver loads via `workspace.yaml`.
- Ops:
  - `journal_op` → extract features from the daily journal and record optional user HRV guess.
  - `training_op` → train/select model, write `model_predictions.json` and `training_metrics.json`.
  - `tracking_op` → append next‑day prediction to `predictions_registry.csv` for backtesting.
  - `intelligence_op` → summarize data and training metrics into `intelligence_summary.json`.
  - `agent_op` → read intelligence summary (and optional user guess) to produce `agent_forecast.json`.
  - `reconcile_op` → reconcile predictions with actuals when available.
- Job wiring: `training_op` feeds both `intelligence_op` and `tracking_op`; `agent_op` depends on `intelligence_op`; `reconcile_op` runs last, after `agent_op`.

Key Dagster concepts used
- op: a unit of computation (our functions wrapped for orchestration).
- job: a directed acyclic graph (DAG) of ops with explicit dependencies.
- Definitions (`defs`): what the workspace loads (jobs, schedules, sensors).
- Executor: configured as `multiprocess_executor` to enable parallel execution of independent ops.

Running it
- Compose service: `dagster_webserver` in `docker-compose.yml` builds from `orchestrator/Dockerfile.dagster` and exposes the UI on port 3000.
- The server loads `defs` from `workspace.yaml`; data and outputs are volume‑mounted from the repo.

Answers to the questions
- Is the whole pipeline integrated in Dagster? Yes. The daily workflow—journal, training, tracking, intelligence, agent, and reconciliation—is encapsulated in the `hrv_daily_job` DAG in `repository.py`.
- Can Dagster work concurrently? Yes. We enable `multiprocess_executor` so independent ops (e.g., `tracking_op` can run alongside downstream steps that don’t depend on it). Dagster parallelizes ops that have no dependency edges between them and isolates each in its own process.