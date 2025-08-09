import json
from pathlib import Path
from datetime import datetime

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "shared" / "outputs"

st.set_page_config(page_title="HRV MVP Dashboard", layout="wide")
st.title("HRV Prediction MVP")

# Load historical HRV
recovery_path = DATA_DIR / "recovery" / "recoveries.json"
with open(recovery_path, "r") as f:
    records = json.load(f)

daily_map = {}
for rec in records:
    created_at = rec.get("created_at")
    score = rec.get("score", {})
    if not created_at:
        continue
    hrv = score.get("hrv_rmssd_milli")
    if hrv is None:
        continue
    day = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date().isoformat()
    daily_map.setdefault(day, []).append(float(hrv))

history = sorted([(k, float(np.mean(v))) for k, v in daily_map.items()], key=lambda x: x[0])

# Load predictions and artifacts
pred_path = OUTPUTS_DIR / "model_predictions.json"
agent_path = OUTPUTS_DIR / "agent_forecast.json"
summary_path = OUTPUTS_DIR / "intelligence_summary.json"
journal_path = OUTPUTS_DIR / "journal_features.json"

preds = {"next_7_day_forecast": []}
if pred_path.exists():
    preds = json.loads(pred_path.read_text())
agent = {}
if agent_path.exists():
    agent = json.loads(agent_path.read_text())
summary = {}
if summary_path.exists():
    summary = json.loads(summary_path.read_text())
journal = {}
if journal_path.exists():
    journal = json.loads(journal_path.read_text())

# Plot simple line chart using Streamlit built-in
obs_dates = [d for d, _ in history]
obs_values = [v for _, v in history]

st.line_chart({"Observed HRV": obs_values})
st.caption(f"Observed days: {len(obs_values)}")

fcst = preds.get("next_7_day_forecast", [])
if fcst:
    fcst_values = [x.get("predicted_hrv") for x in fcst]
    st.line_chart({"Forecast HRV (next 7 days)": fcst_values})

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader("Next Day HRV")
    val = agent.get("next_day_hrv")
    st.metric("Predicted", f"{val:.1f}" if isinstance(val, (int, float)) else "N/A")
with col2:
    st.subheader("7-Day Avg HRV")
    val = agent.get("seven_day_avg_hrv")
    st.metric("Predicted", f"{val:.1f}" if isinstance(val, (int, float)) else "N/A")
with col3:
    st.subheader("Model Error (RMSE)")
    metrics = summary.get("training_metrics", {})
    rmse = metrics.get("rmse") or metrics.get("rmse_train")
    st.metric("RMSE", f"{rmse:.1f} ms" if isinstance(rmse, (int, float)) else "N/A")

st.subheader("Agent Explanation")
st.write(agent.get("explanation", "Run the pipeline to generate an explanation."))

st.subheader("Journal Features")
if journal:
    st.json(journal)
else:
    st.write("No journal features yet.")

st.caption("Data source: WHOOP recovery HRV; MVP forecasts via ridge-style closed-form regression with lag features.")