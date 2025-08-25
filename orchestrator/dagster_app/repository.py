from __future__ import annotations

import os
from pathlib import Path
from dagster import job, op, Definitions, ScheduleDefinition

# Extend sys.path for module imports
import sys
# repository.py is in /app/dagster_app; repo root is its parent (/app)
REPO_ROOT = Path(__file__).resolve().parents[1]
# Ensure package imports resolve from repo root
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from journal.process_journal import run_journal_processing  # type: ignore
from training.train import run_training_pipeline  # type: ignore
from analytics.analyze import run_intelligence  # type: ignore
from coach.agent import run_agent  # type: ignore

OUTPUTS_DIR = os.environ.get("OUTPUTS_DIR", str(REPO_ROOT / "outputs"))


@op
def journal_op():
    return run_journal_processing(outputs_dir=str(OUTPUTS_DIR))


@op
def training_op():
    return run_training_pipeline(
        data_dir=str(REPO_ROOT / "data"),
        models_dir=str(REPO_ROOT / "outputs" / "models"),
        outputs_dir=str(OUTPUTS_DIR),
    )


@op
def intelligence_op(train_artifacts):
    return run_intelligence(
        data_dir=str(REPO_ROOT / "data"),
        outputs_dir=str(OUTPUTS_DIR),
        training_metrics=train_artifacts.get("metrics", {}),
    )


@op
def agent_op():
    return run_agent(outputs_dir=str(OUTPUTS_DIR))


@job
def hrv_daily_job():
    _j = journal_op()
    t = training_op()
    i = intelligence_op(t)
    _a = agent_op()


daily_schedule = ScheduleDefinition(
    job=hrv_daily_job,
    cron_schedule="0 5 * * *",  # 05:00 UTC daily
    execution_timezone="UTC",
)

defs = Definitions(jobs=[hrv_daily_job], schedules=[daily_schedule])