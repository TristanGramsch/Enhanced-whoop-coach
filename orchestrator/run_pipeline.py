import sys
import os
from pathlib import Path
from datetime import datetime
import logging

# Ensure module paths are available inside the container
REPO_ROOT = Path(__file__).resolve().parents[1]
# Add repo root so package-qualified imports work
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from model_training.train import run_training_pipeline  # type: ignore
from data_intelligence.analyze import run_intelligence  # type: ignore
from llm_agent.agent import run_agent  # type: ignore
from journal_analysis.process_journal import run_journal_processing  # type: ignore
from data_intelligence.prediction_tracking import append_next_day_prediction, reconcile_with_actuals  # type: ignore


def ensure_directories(outputs_dir: Path, models_dir: Path):
    outputs_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)


def main():
    outputs_dir_env = os.environ.get("OUTPUTS_DIR")
    shared_dir = REPO_ROOT / "shared"
    outputs_dir = Path(outputs_dir_env) if outputs_dir_env else (shared_dir / "outputs")
    models_dir = shared_dir / "models"
    ensure_directories(outputs_dir, models_dir)

    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    # Configure logging to file and stdout
    logger = logging.getLogger("orchestrator")
    logger.setLevel(logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler = logging.FileHandler(outputs_dir / "pipeline.log")
    file_handler.setFormatter(log_formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(log_formatter)
    # Avoid duplicating handlers on rerun
    if not logger.handlers:
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    logger.info(f"Starting pipeline run {run_id}")

    # Stage 0: Journal processing
    logger.info("Stage 0: journal_analysis")
    journal_artifacts = run_journal_processing(
        outputs_dir=str(outputs_dir)
    )

    # Stage 1: Model training and predictions
    logger.info("Stage 1: model_training")
    train_artifacts = run_training_pipeline(
        data_dir=str(REPO_ROOT / "data"),
        models_dir=str(models_dir),
        outputs_dir=str(outputs_dir),
    )

    # Record the next-day prediction for tracking
    pred_path = train_artifacts.get("predictions_path")
    if pred_path:
        rec = append_next_day_prediction(
            outputs_dir=str(outputs_dir),
            predictions_path=pred_path,
        )
        if rec:
            logger.info(
                f"Tracked next-day prediction for {rec.forecast_for_date}: {rec.predicted_hrv:.2f} ms"
            )

    # Stage 2: Data intelligence summaries
    logger.info("Stage 2: data_intelligence")
    intelligence_artifacts = run_intelligence(
        data_dir=str(REPO_ROOT / "data"),
        outputs_dir=str(outputs_dir),
        training_metrics=train_artifacts.get("metrics", {}),
    )

    # Stage 3: LLM agent forecast report
    logger.info("Stage 3: llm_agent")
    agent_artifacts = run_agent(
        outputs_dir=str(outputs_dir)
    )

    # Reconcile predictions with actuals if available
    recon = reconcile_with_actuals(
        data_dir=str(REPO_ROOT / "data"),
        outputs_dir=str(outputs_dir),
    )
    logger.info(
        f"Reconcile: updated={recon.get('updated',0)} evaluated={recon.get('n_evaluated',0)} RMSE={recon.get('rmse','NA')}"
    )

    # Write simple run log
    run_log_path = outputs_dir / "last_run.json"
    with open(run_log_path, "w") as f:
        import json
        json.dump(
            {
                "run_id": run_id,
                "journal_artifacts": journal_artifacts,
                "train_artifacts": train_artifacts,
                "intelligence_artifacts": intelligence_artifacts,
                "agent_artifacts": agent_artifacts,
                "reconciliation": recon,
                "timestamp": datetime.utcnow().isoformat(),
            },
            f,
            indent=2,
        )

    logger.info("Pipeline completed successfully.")


if __name__ == "__main__":
    main()