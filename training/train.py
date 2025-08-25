from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple
from datetime import datetime, timedelta

import os
import math
import numpy as np

# Optional sklearn/xgboost
try:
    from sklearn.linear_model import Ridge as SkRidge, Lasso, ElasticNet
    from sklearn.ensemble import RandomForestRegressor
    import joblib  # sklearn dependency, used for persistence
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb  # type: ignore
    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

# Optional MLflow
try:
    import mlflow  # type: ignore
    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False


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


def forecast_next_days_regressor(
    daily: List[Tuple[datetime, float]],
    model: Any,
    max_lag: int = 7,
    horizon: int = 7,
) -> List[Dict[str, Any]]:
    history_dates = [d for d, _ in daily]
    history_values = [v for _, v in daily]

    forecasts: List[Dict[str, Any]] = []

    for step in range(1, horizon + 1):
        recent = history_values[-max_lag:][::-1]
        lag_feats: List[float] = []
        for lag in range(1, max_lag + 1):
            if lag <= len(recent):
                lag_feats.append(recent[lag - 1])
            else:
                lag_feats.append(recent[-1])
        ma3 = float(np.mean(history_values[-3:])) if len(history_values) >= 3 else history_values[-1]
        ma7 = float(np.mean(history_values[-7:])) if len(history_values) >= 7 else ma3
        dow = (history_dates[-1].weekday() + step) % 7
        feats = np.array(lag_feats + [ma3, ma7, float(dow)], dtype=float).reshape(1, -1)
        pred = float(model.predict(feats)[0])
        future_dt = history_dates[-1] + timedelta(days=1)
        forecasts.append({"date": future_dt.date().isoformat(), "predicted_hrv": pred})
        history_values.append(pred)
        history_dates.append(future_dt)

    return forecasts

def moving_average_baseline(daily: List[Tuple[datetime, float]], window: int = 3) -> Dict[str, Any]:
    values = np.array([v for _, v in daily], dtype=float)
    preds = []
    trues = []
    for idx in range(window, len(values)):
        preds.append(float(np.mean(values[idx - window:idx])))
        trues.append(float(values[idx]))
    if not trues:
        return {"rmse": float("inf"), "mae": float("inf"), "n_train": 0, "n_test": 0}
    metrics = compute_metrics(np.array(trues), np.array(preds))
    metrics.update({"n_train": int(max(0, len(values) - len(trues))), "n_test": int(len(trues))})
    return metrics


def run_training_pipeline(data_dir: str, models_dir: str, outputs_dir: str) -> Dict[str, Any]:
    Path(models_dir).mkdir(parents=True, exist_ok=True)
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)

    daily = load_hrv_series(data_dir)
    X, y = build_features(daily)

    use_mlflow = MLFLOW_AVAILABLE and bool(os.environ.get("MLFLOW_TRACKING_URI"))
    run_ctx = None
    if use_mlflow:
        mlflow.set_experiment("hrv_mvp")
        run_ctx = mlflow.start_run(run_name="ridge_and_ma")
        mlflow.log_param("feature_max_lag", 7)

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
                "selected_model": "naive_last",
            }, f, indent=2)

        model_path = Path(models_dir) / "naive_model.json"
        with open(model_path, "w") as f:
            json.dump({"type": "naive_last"}, f)

        metrics_path = Path(outputs_dir) / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        if use_mlflow:
            mlflow.log_params({"model": "naive_last"})
            mlflow.log_metrics({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
            mlflow.log_artifact(str(predictions_path))
            mlflow.log_artifact(str(metrics_path))
            mlflow.end_run()

        return {
            "model_path": str(model_path),
            "predictions_path": str(predictions_path),
            "metrics": metrics,
        }

    # Train/validation split
    split = max(10, int(len(y) * 0.8))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # Candidate models
    candidate_metrics: Dict[str, Dict[str, Any]] = {}
    trained_models: Dict[str, Any] = {}

    # Model A: Closed-form ridge
    w = fit_ridge_closed_form(X_train, y_train, alpha=1.0)
    if len(y_test) > 0:
        y_pred_cf = predict_with_weights(X_test, w)
        ridge_metrics = compute_metrics(y_test, y_pred_cf)
        ridge_metrics.update({"n_train": int(len(y_train)), "n_test": int(len(y_test))})
    else:
        y_pred_cf = predict_with_weights(X_train, w)
        ridge_metrics = compute_metrics(y_train, y_pred_cf)
        ridge_metrics.update({"n_train": int(len(y_train)), "n_test": 0, "note": "no holdout"})
    candidate_metrics["ridge_closed_form"] = ridge_metrics

    # Model B: Moving average baseline (uses only history)
    ma_metrics = moving_average_baseline(daily, window=3)
    candidate_metrics["moving_average_3"] = ma_metrics

    # Additional models (scikit-learn)
    if SKLEARN_AVAILABLE:
        # sklearn Ridge (optimizer-based)
        try:
            sk_ridge = SkRidge(alpha=1.0)
            sk_ridge.fit(X_train, y_train)
            trained_models["sk_ridge"] = sk_ridge
            y_pred = sk_ridge.predict(X_test) if len(y_test) > 0 else sk_ridge.predict(X_train)
            m = compute_metrics(y_test if len(y_test) > 0 else y_train, y_pred)
            m.update({"n_train": int(len(y_train)), "n_test": int(len(y_test))})
            candidate_metrics["sklearn_ridge"] = m
        except Exception:
            pass

        # Lasso
        try:
            lasso = Lasso(alpha=0.001, max_iter=10000)
            lasso.fit(X_train, y_train)
            trained_models["lasso"] = lasso
            y_pred = lasso.predict(X_test) if len(y_test) > 0 else lasso.predict(X_train)
            m = compute_metrics(y_test if len(y_test) > 0 else y_train, y_pred)
            m.update({"n_train": int(len(y_train)), "n_test": int(len(y_test))})
            candidate_metrics["lasso"] = m
        except Exception:
            pass

        # ElasticNet
        try:
            en = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=10000)
            en.fit(X_train, y_train)
            trained_models["elasticnet"] = en
            y_pred = en.predict(X_test) if len(y_test) > 0 else en.predict(X_train)
            m = compute_metrics(y_test if len(y_test) > 0 else y_train, y_pred)
            m.update({"n_train": int(len(y_train)), "n_test": int(len(y_test))})
            candidate_metrics["elasticnet"] = m
        except Exception:
            pass

        # RandomForest
        try:
            rf = RandomForestRegressor(n_estimators=200, random_state=42)
            rf.fit(X_train, y_train)
            trained_models["random_forest"] = rf
            y_pred = rf.predict(X_test) if len(y_test) > 0 else rf.predict(X_train)
            m = compute_metrics(y_test if len(y_test) > 0 else y_train, y_pred)
            m.update({"n_train": int(len(y_train)), "n_test": int(len(y_test))})
            candidate_metrics["random_forest"] = m
        except Exception:
            pass

    # XGBoost model
    if XGBOOST_AVAILABLE:
        try:
            xgbr = xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method="hist",
            )
            xgbr.fit(X_train, y_train)
            trained_models["xgboost"] = xgbr
            y_pred = xgbr.predict(X_test) if len(y_test) > 0 else xgbr.predict(X_train)
            # xgb returns numpy array
            m = compute_metrics(y_test if len(y_test) > 0 else y_train, np.array(y_pred))
            m.update({"n_train": int(len(y_train)), "n_test": int(len(y_test))})
            candidate_metrics["xgboost"] = m
        except Exception:
            pass

    # MLflow logging per candidate (nested runs)
    if use_mlflow:
        try:
            for name, m in candidate_metrics.items():
                with mlflow.start_run(run_name=f"candidate_{name}", nested=True):
                    mlflow.log_param("model", name)
                    for mk, mv in m.items():
                        if isinstance(mv, (int, float)) and not math.isnan(mv) and not math.isinf(mv):
                            mlflow.log_metric(mk, float(mv))
        except Exception:
            # Avoid training failure due to logging issues
            pass

    # Select best by RMSE across all candidates
    best_name = min(candidate_metrics.keys(), key=lambda n: candidate_metrics[n].get("rmse", float("inf")))

    # Save model artifact(s) and produce forecast
    if best_name == "ridge_closed_form":
        model_path = Path(models_dir) / "hrv_ridge_weights.json"
        with open(model_path, "w") as f:
            json.dump({"weights": w.tolist(), "type": "ridge_closed_form"}, f)
        fcst = forecast_next_days(daily, w, horizon=7)
    elif best_name == "moving_average_3":
        model_path = Path(models_dir) / "ma3_model.json"
        with open(model_path, "w") as f:
            json.dump({"type": "moving_average_3"}, f)
        last_ma3 = float(np.mean([v for _, v in daily][-3:])) if len(daily) >= 3 else float(daily[-1][1])
        fcst = []
        last_date = daily[-1][0]
        for step in range(1, 8):
            future_dt = last_date + timedelta(days=step)
            fcst.append({"date": future_dt.date().isoformat(), "predicted_hrv": last_ma3})
    else:
        # sklearn/xgboost model selected
        selected_model = trained_models.get(best_name if best_name != "sklearn_ridge" else "sk_ridge")
        if selected_model is None and best_name in ("lasso", "elasticnet", "random_forest", "xgboost"):
            selected_model = trained_models.get(best_name)
        model_path = Path(models_dir) / f"{best_name}.pkl"
        try:
            if SKLEARN_AVAILABLE:
                joblib.dump(selected_model, model_path)
            else:
                # Fallback to pickle if joblib unavailable
                import pickle
                with open(model_path, "wb") as f:
                    pickle.dump(selected_model, f)
        except Exception:
            # As a last resort, write a stub json
            with open(Path(models_dir) / f"{best_name}.json", "w") as f:
                json.dump({"type": best_name}, f)
        fcst = forecast_next_days_regressor(daily, selected_model, max_lag=7, horizon=7)

    # Persist predictions
    predictions_path = Path(outputs_dir) / "model_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump({
            "next_7_day_forecast": fcst,
            "last_observed_date": daily[-1][0].date().isoformat(),
            "last_observed_hrv": float(daily[-1][1]),
            "selected_model": best_name,
        }, f, indent=2)

    # Compose combined metrics
    metrics = {"selected_model": best_name}
    # fold in all candidate metrics with namespaced keys
    for name, m in candidate_metrics.items():
        for mk, mv in m.items():
            if mk in ("n_train", "n_test"):
                # keep a single n_train/n_test representative
                metrics[mk] = m[mk]
            else:
                metrics[f"{name}_{mk}"] = mv

    metrics_path = Path(outputs_dir) / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    if use_mlflow:
        # Summarize in parent run
        try:
            mlflow.log_param("candidate_models", ",".join(candidate_metrics.keys()))
            # Log selected model and top-line metrics
            mlflow.log_param("selected_model", best_name)
            for k, v in metrics.items():
                if isinstance(v, (int, float)) and not math.isnan(v) and not math.isinf(v):
                    mlflow.log_metric(k, float(v))
            # Log artifacts
            if Path(model_path).exists():
                mlflow.log_artifact(str(model_path))
            mlflow.log_artifact(str(predictions_path))
            mlflow.log_artifact(str(metrics_path))

            # If the selected model is a sklearn/xgboost model, also log via MLflow flavor and register
            try:
                if best_name in ("sklearn_ridge", "lasso", "elasticnet", "random_forest"):
                    import mlflow.sklearn  # type: ignore
                    mlflow.sklearn.log_model(
                        sk_model=selected_model, artifact_path="model", registered_model_name="hrv_forecaster"
                    )
                elif best_name == "xgboost":
                    import mlflow.xgboost  # type: ignore
                    mlflow.xgboost.log_model(
                        xgb_model=selected_model, artifact_path="model", registered_model_name="hrv_forecaster"
                    )
            except Exception:
                # Non-fatal if flavor logging fails
                pass
        finally:
            mlflow.end_run()

    return {
        "model_path": str(model_path),
        "predictions_path": str(predictions_path),
        "metrics": metrics,
    }


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = os.environ.get("OUTPUTS_DIR", str(repo_root / "outputs"))
    artifacts = run_training_pipeline(
        data_dir=str(repo_root / "data"),
        models_dir=str(repo_root / "outputs" / "models"),
        outputs_dir=outputs_dir,
    )
    print(json.dumps(artifacts, indent=2))