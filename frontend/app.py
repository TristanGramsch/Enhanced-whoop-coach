import json
from pathlib import Path
from datetime import datetime
import os

import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(REPO_ROOT / "shared" / "outputs")))

st.set_page_config(page_title="HRV MVP Dashboard", layout="wide")
st.title("HRV Prediction MVP")

# Control Center quick links
with st.expander("Control Center: Services & Actions", expanded=True):
    host = os.environ.get("HOST_IP", "localhost")
    st.markdown(
        f"- [MLflow](http://{host}:5000)  \n"
        f"- [Dagster](http://{host}:3000)  \n"
        f"- [Journal Web](http://{host}:8080)  \n"
        f"- [Data Intelligence API](http://{host}:7000/health)  \n"
        f"- [WHOOP Server](http://{host}:8000)  \n"
        f"- [This Dashboard](http://{host}:8501)"
    )
    colA, colB = st.columns(2)
    with colA:
        if st.button("Fetch WHOOP Data"):
            try:
                import requests
                r = requests.get("http://whoop_api:8000/fetch-data", timeout=5)
                st.write("Triggered WHOOP fetch:", r.status_code)
            except Exception as e:
                st.write("Failed to trigger WHOOP fetch:", e)
    with colB:
        if st.button("Run Pipeline Now"):
            try:
                st.write("Run: docker compose run --rm orchestrator python -u run_pipeline.py")
            except Exception as e:
                st.write("Failed to trigger pipeline:", e)
    last_run = OUTPUTS_DIR / "last_run.json"
    if last_run.exists():
        with st.expander("Last Run Status"):
            try:
                st.json(json.loads(last_run.read_text()))
            except Exception:
                st.write("Could not read last_run.json")

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
registry_path = OUTPUTS_DIR / "predictions_registry.csv"
debrief_path = OUTPUTS_DIR / "dev_debrief.json"

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
    val = agent.get("final_pred") if agent.get("final_pred") is not None else agent.get("next_day_hrv")
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
st.write(agent.get("reasoning") or agent.get("explanation", "Run the pipeline to generate an explanation."))

st.subheader("Journal Features")
if journal:
    st.json(journal)
else:
    st.write("No journal features yet.")

# New: Prediction tracking
st.subheader("Prediction Tracking")
if registry_path.exists():
    import pandas as pd

    df = pd.read_csv(registry_path)
    # Convert numeric columns
    for c in ["predicted_hrv", "actual_hrv", "error", "abs_error"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    recent = df.sort_values("forecast_for_date").tail(14)
    st.dataframe(recent[[
        "forecast_for_date", "model_name", "predicted_hrv", "actual_hrv", "error", "abs_error"
    ]], hide_index=True)

    evaluated = df[df["abs_error"].notna()]
    if not evaluated.empty:
        mae = float(evaluated["abs_error"].mean())
        rmse = float(np.sqrt((evaluated["error"] ** 2).mean()))
        st.metric("Tracking MAE", f"{mae:.2f} ms")
        st.metric("Tracking RMSE", f"{rmse:.2f} ms")
else:
    st.write("No predictions tracked yet. Run the pipeline to log next-day predictions.")

# Development Debrief
st.subheader("Development Debrief")
if debrief_path.exists():
    try:
        debrief = json.loads(debrief_path.read_text())
        st.json(debrief)
    except Exception:
        st.write("Debrief not readable.")
else:
    st.write("No debrief available yet.")

st.caption("Data source: WHOOP recovery HRV; MVP forecasts via ridge-style closed-form regression with lag features. LLM final prediction enabled when OPENAI_API_KEY is set.")