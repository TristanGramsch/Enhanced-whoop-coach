from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

import math
import numpy as np


def parse_date(date_str: str) -> datetime:
    return datetime.fromisoformat(date_str.replace("Z", "+00:00"))


def load_hrv_series(data_dir: str) -> List[Tuple[datetime, float]]:
    recovery_path = Path(data_dir) / "recovery" / "recoveries.json"
    if not recovery_path.exists():
        raise FileNotFoundError(f"Recovery data not found at {recovery_path}")

    with open(recovery_path, "r") as f:
        records = json.load(f)

    daily_map: Dict[str, List[float]] = {}
    for rec in records:
        created_at = rec.get("created_at")
        score = rec.get("score", {})
        hrv = score.get("hrv_rmssd_milli")
        if created_at is None or hrv is None:
            continue
        day_key = parse_date(created_at).date().isoformat()
        daily_map.setdefault(day_key, []).append(float(hrv))

    daily_items: List[Tuple[datetime, float]] = []
    for day_key, values in daily_map.items():
        daily_items.append((datetime.fromisoformat(day_key), float(np.mean(values))))

    daily_items.sort(key=lambda x: x[0])
    if not daily_items:
        raise ValueError("No valid HRV records found in recovery data")

    return daily_items


def build_features(daily: List[Tuple[datetime, float]], max_lag: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    dates = [d for d, _ in daily]
    values = np.array([v for _, v in daily], dtype=float)

    rows: List[List[float]] = []
    targets: List[float] = []

    for idx in range(max_lag, len(values)):
        features: List[float] = []
        # lag features
        for lag in range(1, max_lag + 1):
            features.append(values[idx - lag])
        # moving averages
        ma3 = float(np.mean(values[idx - 3:idx])) if idx >= 3 else values[idx - 1]
        ma7 = float(np.mean(values[idx - 7:idx])) if idx >= 7 else ma3
        features.extend([ma3, ma7])
        # day of week
        dow = dates[idx].weekday()
        features.append(float(dow))

        rows.append(features)
        targets.append(values[idx])

    X = np.array(rows, dtype=float)
    y = np.array(targets, dtype=float)
    return X, y


def fit_ridge_closed_form(X: np.ndarray, y: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    # Add bias term
    ones = np.ones((X.shape[0], 1))
    Xb = np.hstack([ones, X])
    I = np.eye(Xb.shape[1])
    I[0, 0] = 0.0  # don't regularize bias
    w = np.linalg.pinv(Xb.T @ Xb + alpha * I) @ Xb.T @ y
    return w  # shape (p+1,)


def predict_with_weights(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    ones = np.ones((X.shape[0], 1))
    Xb = np.hstack([ones, X])
    return Xb @ w


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    return {"mae": mae, "rmse": rmse}


def forecast_next_days(daily: List[Tuple[datetime, float]], w: np.ndarray, horizon: int = 7) -> List[Dict[str, Any]]:
    history_dates = [d for d, _ in daily]
    history_values = [v for _, v in daily]
    max_lag = 7

    forecasts: List[Dict[str, Any]] = []

    for step in range(1, horizon + 1):
        # Build a single-row feature vector from current history
        recent = history_values[-max_lag:][::-1]
        lag_feats = []
        for lag in range(1, max_lag + 1):
            if lag <= len(recent):
                lag_feats.append(recent[lag - 1])
            else:
                lag_feats.append(recent[-1])
        ma3 = float(np.mean(history_values[-3:])) if len(history_values) >= 3 else history_values[-1]
        ma7 = float(np.mean(history_values[-7:])) if len(history_values) >= 7 else ma3
        dow = (history_dates[-1].weekday() + step) % 7
        feats = np.array(lag_feats + [ma3, ma7, float(dow)], dtype=float).reshape(1, -1)
        pred = float(predict_with_weights(feats, w)[0])
        future_dt = history_dates[-1] + timedelta(days=1)
        forecasts.append({"date": future_dt.date().isoformat(), "predicted_hrv": pred})
        history_values.append(pred)
        history_dates.append(future_dt)

    return forecasts


def run_training_pipeline(data_dir: str, models_dir: str, outputs_dir: str) -> Dict[str, Any]:
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    daily = load_hrv_series(data_dir)
    X, y = build_features(daily)

    if len(y) < 10:
        # Fallback to last observed value forecasting
        last_date, last_val = daily[-1]
        fcst = []
        for step in range(1, 8):
            future_dt = last_date + timedelta(days=step)
            fcst.append({"date": future_dt.date().isoformat(), "predicted_hrv": float(last_val)})
        metrics = {"note": "insufficient data, using naive forecast", "n_samples": int(len(y))}

        predictions_path = Path(outputs_dir) / "model_predictions.json"
        with open(predictions_path, "w") as f:
            json.dump({
                "next_7_day_forecast": fcst,
                "last_observed_date": daily[-1][0].date().isoformat(),
                "last_observed_hrv": float(daily[-1][1]),
            }, f, indent=2)

        model_path = Path(models_dir) / "naive_model.json"
        with open(model_path, "w") as f:
            json.dump({"type": "naive_last"}, f)

        metrics_path = Path(outputs_dir) / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        return {
            "model_path": str(model_path),
            "predictions_path": str(predictions_path),
            "metrics": metrics,
        }

    # Train/validation split
    split = max(10, int(len(y) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    w = fit_ridge_closed_form(X_train, y_train, alpha=1.0)

    if len(y_test) > 0:
        y_pred = predict_with_weights(X_test, w)
        metrics = compute_metrics(y_test, y_pred)
        metrics.update({"n_train": int(len(y_train)), "n_test": int(len(y_test))})
    else:
        y_pred = predict_with_weights(X_train, w)
        metrics = compute_metrics(y_train, y_pred)
        metrics.update({"n_train": int(len(y_train)), "n_test": 0, "note": "no holdout"})

    # Save weights
    model_path = Path(models_dir) / "hrv_ridge_weights.json"
    with open(model_path, "w") as f:
        json.dump({"weights": w.tolist()}, f)

    # Forecast next 7 days
    fcst = forecast_next_days(daily, w, horizon=7)

    predictions_path = Path(outputs_dir) / "model_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump({
            "next_7_day_forecast": fcst,
            "last_observed_date": daily[-1][0].date().isoformat(),
            "last_observed_hrv": float(daily[-1][1]),
        }, f, indent=2)

    metrics_path = Path(outputs_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return {
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
        "metrics": metrics,
    }


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = run_training_pipeline(
        data_dir=str(repo_root / "data"),
        models_dir=str(repo_root / "shared" / "models"),
        outputs_dir=str(repo_root / "shared" / "outputs"),
    )
    print(json.dumps(artifacts, indent=2))