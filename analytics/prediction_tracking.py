from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional


@dataclass
class PredictionRecord:
    created_at: str
    training_end_date: str
    forecast_for_date: str
    horizon_days: int
    model_name: str
    predicted_hrv: float
    actual_hrv: Optional[float] = None
    error: Optional[float] = None
    abs_error: Optional[float] = None


def _registry_path(outputs_dir: str) -> Path:
    return Path(outputs_dir) / "predictions_registry.csv"


def _ensure_registry_with_header(path: Path) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "created_at",
                "training_end_date",
                "forecast_for_date",
                "horizon_days",
                "model_name",
                "predicted_hrv",
                "actual_hrv",
                "error",
                "abs_error",
            ])


def append_next_day_prediction(outputs_dir: str, predictions_path: str) -> Optional[PredictionRecord]:
    path = Path(predictions_path)
    if not path.exists():
        return None
    preds = json.loads(path.read_text())
    fcst = preds.get("next_7_day_forecast", [])
    if not fcst:
        return None
    next_day = fcst[0]
    record = PredictionRecord(
        created_at=datetime.utcnow().isoformat(),
        training_end_date=str(preds.get("last_observed_date")),
        forecast_for_date=str(next_day.get("date")),
        horizon_days=1,
        model_name=str(preds.get("selected_model", "unknown")),
        predicted_hrv=float(next_day.get("predicted_hrv")),
    )

    reg_path = _registry_path(outputs_dir)
    _ensure_registry_with_header(reg_path)
    with open(reg_path, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            record.created_at,
            record.training_end_date,
            record.forecast_for_date,
            record.horizon_days,
            record.model_name,
            f"{record.predicted_hrv:.6f}",
            "",
            "",
            "",
        ])
    return record


def _load_daily_actuals(data_dir: str) -> Dict[str, float]:
    rec_path = Path(data_dir) / "recovery" / "recoveries.json"
    if not rec_path.exists():
        return {}
    records = json.loads(rec_path.read_text())
    from collections import defaultdict
    day_to_vals: Dict[str, List[float]] = defaultdict(list)
    for rec in records:
        created_at = rec.get("created_at")
        score = rec.get("score", {})
        if not created_at:
            continue
        hrv = score.get("hrv_rmssd_milli")
        if hrv is None:
            continue
        # Normalize to date string
        day = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date().isoformat()
        try:
            day_to_vals[day].append(float(hrv))
        except Exception:
            continue
    return {d: (sum(v) / len(v)) for d, v in day_to_vals.items() if v}


def reconcile_with_actuals(data_dir: str, outputs_dir: str) -> Dict[str, Any]:
    reg_path = _registry_path(outputs_dir)
    if not reg_path.exists():
        return {"updated": 0, "total": 0}

    actuals = _load_daily_actuals(data_dir)

    # Load all rows
    with open(reg_path, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    updated = 0
    for row in rows:
        if not row.get("actual_hrv") and row.get("forecast_for_date") in actuals:
            actual = float(actuals[row["forecast_for_date"]])
            try:
                pred = float(row["predicted_hrv"])
            except Exception:
                continue
            row["actual_hrv"] = f"{actual:.6f}"
            err = pred - actual
            row["error"] = f"{err:.6f}"
            row["abs_error"] = f"{abs(err):.6f}"
            updated += 1

    # Write back
    with open(reg_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "created_at",
                "training_end_date",
                "forecast_for_date",
                "horizon_days",
                "model_name",
                "predicted_hrv",
                "actual_hrv",
                "error",
                "abs_error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    # Compute simple metrics on evaluated rows
    evaluated_rows = [r for r in rows if r.get("abs_error")]
    if not evaluated_rows:
        return {"updated": updated, "total": total, "n_evaluated": 0}

    import math
    abs_errors = [float(r["abs_error"]) for r in evaluated_rows]
    errors = [float(r["error"]) for r in evaluated_rows]
    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(e * e for e in errors) / len(errors))
    return {
        "updated": updated,
        "total": total,
        "n_evaluated": len(evaluated_rows),
        "mae": mae,
        "rmse": rmse,
    } 