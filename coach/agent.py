from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, Optional
import os
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


def _read_user_guess(outputs_dir: Path) -> Optional[float]:
    log_path = outputs_dir / "user_guess.json"
    if not log_path.exists():
        return None
    try:
        data = json.loads(log_path.read_text())
        if not isinstance(data, list) or not data:
            return None
        # take most recent record for today if present, else most recent overall
        today = os.environ.get("FORECAST_FOR_DATE") or None
        if today:
            todays = [r for r in data if r.get("date") == today]
            if todays:
                return float(todays[-1].get("value"))
        return float(data[-1].get("value"))
    except Exception:
        return None


def _llm_final_prediction(summary: Dict[str, Any], base_next_day: Optional[float], user_guess: Optional[float]) -> Optional[Dict[str, Any]]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        system = (
            "You are an expert HRV forecasting assistant. Weigh model predictions, recent trends, "
            "volatility, tracking errors, and a human user's self-reported HRV guess to produce a single next-day HRV prediction in milliseconds. "
            "Respond with strict JSON with keys: final_pred (float), confidence (0-1), reasoning (string), used_signals (array of strings)."
        )
        user_payload = {
            "base_next_day": base_next_day,
            "summary": summary,
            "user_guess": user_guess,
        }
        completion = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload)},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        content = completion.choices[0].message.content
        result = json.loads(content)
        if isinstance(result, dict) and "final_pred" in result:
            return result
        return None
    except Exception:
        return None


def run_agent(outputs_dir: str) -> Dict[str, Any]:
    outputs_path = Path(outputs_dir)
    summary_path = outputs_path / "intelligence_summary.json"
    with open(summary_path, "r") as f:
        summary = json.load(f)

    forecast = summary.get("next_7_day_forecast", [])
    next_day_value = forecast[0]["predicted_hrv"] if forecast else None
    week_avg = statistics.mean([x["predicted_hrv"] for x in forecast]) if forecast else None

    explanation = generate_explanation(summary)

    # Try LLM-based final prediction
    user_guess = _read_user_guess(outputs_path)
    llm_result = _llm_final_prediction(summary, next_day_value, user_guess)

    agent_output = {
        "next_day_hrv": next_day_value,
        "seven_day_avg_hrv": week_avg,
        "explanation": explanation,
    }

    if llm_result:
        agent_output.update({
            "final_pred": llm_result.get("final_pred"),
            "confidence": llm_result.get("confidence"),
            "reasoning": llm_result.get("reasoning"),
            "used_signals": llm_result.get("used_signals"),
            "user_guess": user_guess,
            "model": "gpt-5",
        })
    else:
        # Fallback: keep model next_day and include user guess for transparency
        agent_output.update({
            "final_pred": next_day_value,
            "confidence": None,
            "reasoning": "LLM unavailable; using model next-day prediction.",
            "used_signals": ["model_forecast"],
            "user_guess": user_guess,
            "model": None,
        })

    out_path = outputs_path / "agent_forecast.json"
    with open(out_path, "w") as f:
        json.dump(agent_output, f, indent=2)

    return {
        "agent_output_path": str(out_path),
    }


if __name__ == "__main__":
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = os.environ.get("OUTPUTS_DIR", str(repo_root / "outputs"))
    artifacts = run_agent(outputs_dir=outputs_dir)
    print(json.dumps(artifacts, indent=2))