from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pathlib import Path
import json
import os

from .analyze import run_intelligence

app = FastAPI(title="Data Intelligence API")

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/run")
async def run():
    # Attempt to load training metrics if present
    metrics_path = OUTPUTS_DIR / "training_metrics.json"
    training_metrics = {}
    if metrics_path.exists():
        try:
            training_metrics = json.loads(metrics_path.read_text())
        except Exception:
            training_metrics = {}

    try:
        artifacts = run_intelligence(
            data_dir=str(DATA_DIR),
            outputs_dir=str(OUTPUTS_DIR),
            training_metrics=training_metrics,
        )
        summary = json.loads((OUTPUTS_DIR / "intelligence_summary.json").read_text())
        return {"artifacts": artifacts, "summary": summary}
    except FileNotFoundError as e:
        return JSONResponse(status_code=404, content={"error": str(e)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/summary")
async def summary():
    path = OUTPUTS_DIR / "intelligence_summary.json"
    if not path.exists():
        return JSONResponse(status_code=404, content={"error": "summary not found; POST /run first"})
    return json.loads(path.read_text()) 