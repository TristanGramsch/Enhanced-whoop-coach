from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Tuple
import numpy as np


def _parse_iso_date(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def build_timeseries(data_dir: str, outputs_dir: str) -> Dict[str, Any]:
    """Aggregate daily metrics: HRV, RHR, RR, strain, sleep duration, disturbances.

    Writes timeseries.json with a list of daily records sorted by date.
    """
    data_root = Path(data_dir)
    # Recovery: HRV, RHR, RR
    recov_path = data_root / "recovery" / "recoveries.json"
    recov = json.loads(recov_path.read_text()) if recov_path.exists() else []
    hrvs_by_day: Dict[str, List[float]] = {}
    rhr_by_day: Dict[str, List[float]] = {}
    rr_by_day: Dict[str, List[float]] = {}
    for r in recov:
        created_at = r.get("created_at")
        score = r.get("score") or {}
        if not created_at:
            continue
        day = _parse_iso_date(created_at).date().isoformat()
        hrv = score.get("hrv_rmssd_milli")
        rhr = score.get("resting_heart_rate")
        rr = score.get("respiratory_rate") or r.get("respiratory_rate")
        if hrv is not None:
            hrvs_by_day.setdefault(day, []).append(float(hrv))
        if rhr is not None:
            rhr_by_day.setdefault(day, []).append(float(rhr))
        if rr is not None:
            rr_by_day.setdefault(day, []).append(float(rr))

    # Workouts: daily mean strain
    workouts_path = data_root / "workouts" / "workouts.json"
    workouts = json.loads(workouts_path.read_text()) if workouts_path.exists() else []
    strain_by_day: Dict[str, List[float]] = {}
    for w in workouts:
        end_ts = w.get("end") or w.get("created_at")
        score = w.get("score") or {}
        if not end_ts:
            continue
        day = _parse_iso_date(end_ts).date().isoformat()
        strain = score.get("strain")
        if strain is not None:
            strain_by_day.setdefault(day, []).append(float(strain))

    # Sleep: duration (hours) and disturbance_count
    sleep_path = data_root / "sleep" / "sleep_activities.json"
    sleeps = json.loads(sleep_path.read_text()) if sleep_path.exists() else []
    sleep_hours_by_day: Dict[str, List[float]] = {}
    disturb_by_day: Dict[str, List[int]] = {}
    for s in sleeps:
        end_ts = s.get("end") or s.get("created_at")
        score = s.get("score") or {}
        stage = score.get("stage_summary") or {}
        if not end_ts:
            continue
        day = _parse_iso_date(end_ts).date().isoformat()
        total_in_bed_ms = stage.get("total_in_bed_time_milli")
        disturbances = stage.get("disturbance_count")
        if total_in_bed_ms is not None:
            sleep_hours_by_day.setdefault(day, []).append(float(total_in_bed_ms) / 3_600_000.0)
        if disturbances is not None:
            disturb_by_day.setdefault(day, []).append(int(disturbances))

    # Merge days
    all_days = set(hrvs_by_day) | set(rhr_by_day) | set(rr_by_day) | set(strain_by_day) | set(sleep_hours_by_day) | set(disturb_by_day)
    daily: List[Dict[str, Any]] = []
    for day in sorted(all_days):
        def mean_or_none(arr: List[float] | List[int] | None) -> float | None:
            if not arr:
                return None
            return float(np.mean(arr))

        daily.append({
            "date": day,
            "hrv": mean_or_none(hrvs_by_day.get(day)),
            "rhr": mean_or_none(rhr_by_day.get(day)),
            "rr": mean_or_none(rr_by_day.get(day)),
            "strain": mean_or_none(strain_by_day.get(day)),
            "sleep_hours": mean_or_none(sleep_hours_by_day.get(day)),
            "disturbances": mean_or_none(disturb_by_day.get(day)),
        })

    out = {"daily": daily}
    Path(outputs_dir).mkdir(parents=True, exist_ok=True)
    (Path(outputs_dir) / "timeseries.json").write_text(json.dumps(out, indent=2))
    return out


def _valid_numeric_pairs(x: List[float | None], y: List[float | None]) -> Tuple[np.ndarray, np.ndarray]:
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None]
    if not pairs:
        return np.array([]), np.array([])
    ax, ay = zip(*pairs)
    return np.array(ax, dtype=float), np.array(ay, dtype=float)


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    c = np.corrcoef(x, y)
    if np.isnan(c).any():
        return None
    return float(c[0, 1])


def _spearman(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or y.size < 2:
        return None
    # Rank data (average ranks for ties via argsort twice)
    def rankdata(a: np.ndarray) -> np.ndarray:
        order = a.argsort()
        ranks = np.empty_like(order, dtype=float)
        ranks[order] = np.arange(1, a.size + 1)
        # Simple tie handling by averaging contiguous equal segments
        sorted_a = a[order]
        start = 0
        for i in range(1, a.size + 1):
            if i == a.size or sorted_a[i] != sorted_a[start]:
                avg = (start + i - 1) / 2 + 1
                ranks[start:i] = avg
                start = i
        return ranks
    xr, yr = rankdata(x), rankdata(y)
    return _pearson(xr, yr)


def compute_correlations(timeseries: Dict[str, Any], outputs_dir: str) -> Dict[str, Any]:
    daily = timeseries.get("daily", [])
    if not daily:
        result = {"pearson": {}, "spearman": {}, "lag": {}}
        (Path(outputs_dir) / "correlations.json").write_text(json.dumps(result, indent=2))
        return result

    # Collect vectors
    dates = [d["date"] for d in daily]
    fields = ["hrv", "rhr", "rr", "strain", "sleep_hours", "disturbances"]
    series: Dict[str, List[float | None]] = {f: [row.get(f) for row in daily] for f in fields}

    # Pairwise correlations
    pearson: Dict[str, Dict[str, float | None]] = {}
    spearman: Dict[str, Dict[str, float | None]] = {}
    for i, a in enumerate(fields):
        pearson[a] = {}
        spearman[a] = {}
        for b in fields:
            x, y = _valid_numeric_pairs(series[a], series[b])
            pearson[a][b] = _pearson(x, y)
            spearman[a][b] = _spearman(x, y)

    # Lag correlations: var_k vs hrv_0 for k in [-7..+7]
    lag_window = list(range(-7, 8))
    lag: Dict[str, Dict[str, float | None]] = {}
    for var in ["rhr", "rr", "strain", "sleep_hours", "disturbances"]:
        lag[var] = {}
        v = np.array([np.nan if x is None else float(x) for x in series[var]], dtype=float)
        h = np.array([np.nan if x is None else float(x) for x in series["hrv"]], dtype=float)
        for k in lag_window:
            if k < 0:
                v_shift = v[-k:]
                h_align = h[:v_shift.size]
            elif k > 0:
                v_shift = v[:-k]
                h_align = h[k:]
            else:
                v_shift = v
                h_align = h
            mask = ~np.isnan(v_shift) & ~np.isnan(h_align)
            if mask.sum() < 3:
                lag[var][str(k)] = None
                continue
            lag[var][str(k)] = _pearson(v_shift[mask], h_align[mask])

    result = {"dates": dates, "pearson": pearson, "spearman": spearman, "lag": lag}
    (Path(outputs_dir) / "correlations.json").write_text(json.dumps(result, indent=2))
    return result


def compute_seasonality(timeseries: Dict[str, Any], outputs_dir: str) -> Dict[str, Any]:
    daily = timeseries.get("daily", [])
    if not daily:
        res = {"weekday_means": {}}
        (Path(outputs_dir) / "seasonality.json").write_text(json.dumps(res, indent=2))
        return res
    # Map date to weekday 0..6
    wd_to_vals: Dict[int, List[float]] = {i: [] for i in range(7)}
    for row in daily:
        date = row.get("date")
        val = row.get("hrv")
        if date and val is not None:
            wd = datetime.fromisoformat(date).weekday()
            wd_to_vals[wd].append(float(val))
    weekday_means = {str(k): (float(np.mean(v)) if v else None) for k, v in wd_to_vals.items()}
    res = {"weekday_means": weekday_means}
    (Path(outputs_dir) / "seasonality.json").write_text(json.dumps(res, indent=2))
    return res


def compute_anomalies(timeseries: Dict[str, Any], outputs_dir: str) -> Dict[str, Any]:
    daily = timeseries.get("daily", [])
    if not daily:
        res = {"anomalies": []}
        (Path(outputs_dir) / "anomalies.json").write_text(json.dumps(res, indent=2))
        return res
    vals = np.array([row.get("hrv") if row.get("hrv") is not None else np.nan for row in daily], dtype=float)
    mean = np.nanmean(vals)
    std = np.nanstd(vals)
    anoms: List[Dict[str, Any]] = []
    if np.isfinite(std) and std > 0:
        z = (vals - mean) / std
        for i, zi in enumerate(z):
            if np.isnan(zi):
                continue
            if abs(zi) >= 2.5:
                anoms.append({"date": daily[i]["date"], "hrv": float(vals[i]), "zscore": float(zi)})
    res = {"anomalies": anoms, "mean": float(mean) if np.isfinite(mean) else None, "std": float(std) if np.isfinite(std) else None}
    (Path(outputs_dir) / "anomalies.json").write_text(json.dumps(res, indent=2))
    return res


def summarize_backtests(outputs_dir: str) -> Dict[str, Any]:
    """Summarize forecast tracking from predictions_registry.csv if available."""
    reg_path = Path(outputs_dir) / "predictions_registry.csv"
    if not reg_path.exists():
        res = {"tracking": {"count": 0, "mae": None, "rmse": None}}
        (Path(outputs_dir) / "forecast_backtests.json").write_text(json.dumps(res, indent=2))
        return res
    import csv
    rows = []
    with reg_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    # Compute errors where actual exists
    errors: List[float] = []
    abs_errors: List[float] = []
    for r in rows:
        try:
            pred = float(r.get("predicted_hrv")) if r.get("predicted_hrv") not in (None, "") else None
            actual = float(r.get("actual_hrv")) if r.get("actual_hrv") not in (None, "") else None
            if pred is None or actual is None:
                continue
            e = pred - actual
            errors.append(e)
            abs_errors.append(abs(e))
        except Exception:
            continue
    count = len(abs_errors)
    mae = float(np.mean(abs_errors)) if count > 0 else None
    rmse = float(np.sqrt(np.mean(np.square(errors)))) if count > 0 else None
    res = {"tracking": {"count": count, "mae": mae, "rmse": rmse}}
    (Path(outputs_dir) / "forecast_backtests.json").write_text(json.dumps(res, indent=2))
    return res 