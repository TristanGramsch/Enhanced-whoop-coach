from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

import statistics


def generate_explanation(summary: Dict[str, Any]) -> str:
    trend = summary.get("trend", {})
    slope = trend.get("trend_slope", 0.0)
    avg_30 = trend.get("avg_hrv_30d", 0.0)
    vol = trend.get("volatility_30d", 0.0)

    metrics = summary.get("training_metrics", {})
    rmse = metrics.get("rmse") or metrics.get("rmse_train")

    forecast = summary.get("next_7_day_forecast", [])
    next_day = forecast[0]["predicted_hrv"] if forecast else None

    direction = "increasing" if slope > 0 else ("decreasing" if slope < 0 else "stable")

    parts = []
    parts.append(f"30-day average HRV is {avg_30:.1f} ms with {direction} trend (slope={slope:.3f}).")
    parts.append(f"Short-term volatility is {vol:.1f} ms.")

    if rmse is not None:
        parts.append(f"Model RMSE on holdout is {rmse:.1f} ms, indicating expected prediction error magnitude.")

    if next_day is not None:
        parts.append(f"Predicted HRV for tomorrow is {next_day:.1f} ms.")

    return " ".join(parts)


def run_agent(outputs_dir: str) -> Dict[str, Any]:
    outputs_path = Path(outputs_dir)
    summary_path = outputs_path / "intelligence_summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)

    forecast = summary.get("next_7_day_forecast", [])
    next_day_value = forecast[0]["predicted_hrv"] if forecast else None
    week_avg = statistics.mean([x["predicted_hrv"] for x in forecast]) if forecast else None

    explanation = generate_explanation(summary)

    agent_output = {
        "next_day_hrv": next_day_value,
        "seven_day_avg_hrv": week_avg,
        "explanation": explanation,
    }

    out_path = outputs_path / "agent_forecast.json"
    with open(out_path, "w") as f:
        json.dump(agent_output, f, indent=2)

    return {
        "agent_output_path": str(out_path),
    }


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    artifacts = run_agent(outputs_dir=str(repo_root / "shared" / "outputs"))
    print(json.dumps(artifacts, indent=2))