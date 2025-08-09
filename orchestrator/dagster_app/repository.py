from __future__ import annotations

import os
from pathlib import Path
from dagster import job, op, Definitions

# Extend sys.path for module imports
import sys
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in [
    REPO_ROOT / "journal_analysis",
    REPO_ROOT / "model_training",
    REPO_ROOT / "data_intelligence",
    REPO_ROOT / "llm_agent",
]:
    if str(p) not in sys.path:
        sys.path.append(str(p))

from process_journal import run_journal_processing  # type: ignore
from train import run_training_pipeline  # type: ignore
from analyze import run_intelligence  # type: ignore
from agent import run_agent  # type: ignore


@op
def journal_op():
    return run_journal_processing(outputs_dir=str(REPO_ROOT / "shared" / "outputs"))


@op
def training_op():
    return run_training_pipeline(
        data_dir=str(REPO_ROOT / "data"),
        models_dir=str(REPO_ROOT / "shared" / "models"),
        outputs_dir=str(REPO_ROOT / "shared" / "outputs"),
    )


@op
def intelligence_op(train_artifacts):
    return run_intelligence(
        data_dir=str(REPO_ROOT / "data"),
        outputs_dir=str(REPO_ROOT / "shared" / "outputs"),
        training_metrics=train_artifacts.get("metrics", {}),
    )


@op
def agent_op():
    return run_agent(outputs_dir=str(REPO_ROOT / "shared" / "outputs"))


@job
def hrv_daily_job():
    _j = journal_op()
    t = training_op()
    i = intelligence_op(t)
    _a = agent_op()


defs = Definitions(jobs=[hrv_daily_job])