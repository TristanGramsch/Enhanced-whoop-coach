from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime
import os

import numpy as np


def parse_date(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))


def load_history(data_dir: str) -> List[Tuple[datetime, float]]:
    recovery_path = Path(data_dir) / "recovery" / "recoveries.json"
    with open(recovery_path, "r") as f:
        records = json.load(f)

    daily_map: Dict[str, List[float]] = {}
    for rec in records:
        created_at = rec.get("created_at")
        score = rec.get("score", {})
        if created_at is None:
            continue
        hrv = score.get("hrv_rmssd_milli")
        if hrv is None:
            continue
        day_key = parse_date(created_at).date().isoformat()
        daily_map.setdefault(day_key, []).append(float(hrv))

    daily_items: List[Tuple[datetime, float]] = []
    for day_key, values in daily_map.items():
        daily_items.append((datetime.fromisoformat(day_key), float(np.mean(values))))
    daily_items.sort(key=lambda x: x[0])
    return daily_items


def compute_trend_metrics(daily: List[Tuple[datetime, float]]) -> Dict[str, Any]:
    if not daily:
        return {"avg_hrv_30d": 0.0, "trend_slope": 0.0, "volatility_30d": 0.0}

    values = np.array([v for _, v in daily], dtype=float)
    last_30 = values[-30:] if len(values) >= 30 else values
    if len(last_30) < 2:
        return {
            "avg_hrv_30d": float(np.mean(last_30)),
            "trend_slope": 0.0,
            "volatility_30d": 0.0,
        }

    x = np.arange(len(last_30), dtype=float)
    x_centered = x - x.mean()
    y_centered = last_30 - last_30.mean()
    denom = float((x_centered ** 2).sum())
    slope = float((x_centered * y_centered).sum() / denom) if denom != 0 else 0.0
    volatility = float(np.std(last_30))
    return {
        "avg_hrv_30d": float(np.mean(last_30)),
        "trend_slope": slope,
        "volatility_30d": volatility,
    }


def run_intelligence(data_dir: str, outputs_dir: str, training_metrics: Dict[str, Any]) -> Dict[str, Any]:
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    daily = load_history(data_dir)
    trend = compute_trend_metrics(daily)

    # Load model predictions
    pred_path = Path(outputs_dir) / "model_predictions.json"
    with open(pred_path, "r") as f:
        preds = json.load(f)

    last_date = daily[-1][0].date().isoformat() if daily else None
    last_hrv = float(daily[-1][1]) if daily else None

    summary = {
        "history_days": int(len(daily)),
        "last_observed": {
            "date": last_date,
            "hrv": last_hrv,
        },
        "trend": trend,
        "training_metrics": training_metrics,
        "next_7_day_forecast": preds.get("next_7_day_forecast", []),
    }

    out_path = Path(outputs_dir) / "intelligence_summary.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    return {
        "summary_path": str(out_path),
        "n_history_days": int(len(daily)),
    }


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = os.environ.get("OUTPUTS_DIR", str(repo_root / "shared" / "outputs"))
    artifacts = run_intelligence(
        data_dir=str(repo_root / "data"),
        outputs_dir=outputs_dir,
        training_metrics={},
    )
    print(json.dumps(artifacts, indent=2))