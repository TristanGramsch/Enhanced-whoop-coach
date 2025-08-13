import os
import json
from pathlib import Path


def test_data_presence():
    repo = Path(__file__).resolve().parents[1]
    rec_path = repo / "data" / "recovery" / "recoveries.json"
    assert rec_path.exists(), "recoveries.json missing"
    data = json.loads(rec_path.read_text())
    assert isinstance(data, list) and len(data) > 0


def test_training_produces_artifacts():
    repo = Path(__file__).resolve().parents[1]
    outputs = repo / "shared" / "outputs" / "test"
    outputs.mkdir(parents=True, exist_ok=True)

    os.environ["OUTPUTS_DIR"] = str(outputs)
    import importlib
    train = importlib.import_module("model_training.train")
    artifacts = train.run_training_pipeline(
        data_dir=str(repo / "data"),
        models_dir=str(repo / "shared" / "models"),
        outputs_dir=str(outputs),
    )
    assert Path(artifacts["predictions_path"]).exists()
    metrics = artifacts["metrics"]
    assert isinstance(metrics, dict)


def test_intelligence_creates_summary():
    repo = Path(__file__).resolve().parents[1]
    outputs = repo / "shared" / "outputs" / "test"
    os.environ["OUTPUTS_DIR"] = str(outputs)

    import importlib
    analyze = importlib.import_module("data_intelligence.analyze")
    # training metrics may not exist; pass empty
    artifacts = analyze.run_intelligence(
        data_dir=str(repo / "data"),
        outputs_dir=str(outputs),
        training_metrics={},
    )
    summary_path = Path(artifacts["summary_path"])
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "trend" in summary


def test_agent_writes_forecast_json():
    repo = Path(__file__).resolve().parents[1]
    outputs = repo / "shared" / "outputs" / "test"
    os.environ["OUTPUTS_DIR"] = str(outputs)

    import importlib
    agent = importlib.import_module("llm_agent.agent")
    artifacts = agent.run_agent(outputs_dir=str(outputs))
    out_path = Path(artifacts["agent_output_path"])
    assert out_path.exists()
    data = json.loads(out_path.read_text())
    assert "final_pred" in data or "next_day_hrv" in data


def test_prediction_tracking_roundtrip():
    repo = Path(__file__).resolve().parents[1]
    outputs = repo / "shared" / "outputs" / "test"
    os.environ["OUTPUTS_DIR"] = str(outputs)

    import importlib
    train = importlib.import_module("model_training.train")
    artifacts = train.run_training_pipeline(
        data_dir=str(repo / "data"),
        models_dir=str(repo / "shared" / "models"),
        outputs_dir=str(outputs),
    )

    tracking = importlib.import_module("data_intelligence.prediction_tracking")
    rec = tracking.append_next_day_prediction(outputs_dir=str(outputs), predictions_path=artifacts["predictions_path"]) 
    assert rec is not None
    recon = tracking.reconcile_with_actuals(data_dir=str(repo / "data"), outputs_dir=str(outputs))
    assert "updated" in recon 