import json
from pathlib import Path
import os
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(REPO_ROOT / "shared" / "outputs")))

st.title("Forecasts")

# Predictions
pred = {}
p = OUTPUTS_DIR / "model_predictions.json"
if p.exists():
    try:
        pred = json.loads(p.read_text())
    except Exception:
        pass

series = pred.get("next_7_day_forecast", [])
if series:
    st.subheader("Next 7 Days (point forecasts)")
    st.line_chart({"Forecast HRV": [x.get("predicted_hrv") for x in series]})
else:
    st.write("No forecast series available. Train or run the model to populate predictions.")

# Backtests / live tracking
bt = {}
p2 = OUTPUTS_DIR / "forecast_backtests.json"
if p2.exists():
    try:
        bt = json.loads(p2.read_text())
    except Exception:
        pass

trk = bt.get("tracking", {})
st.subheader("Live Tracking Metrics")
st.metric("MAE", f"{trk.get('mae'):.2f}" if trk.get("mae") is not None else "N/A")
st.metric("RMSE", f"{trk.get('rmse'):.2f}" if trk.get("rmse") is not None else "N/A")

st.caption("Forecasts communicate expected range; backtests reflect reliability and calibration over time.") 