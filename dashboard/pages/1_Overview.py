import json
from pathlib import Path
import os
import numpy as np
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(REPO_ROOT / "shared" / "outputs")))

st.title("Overview")

# Load artifacts
summary = {}
for name in ["intelligence_summary.json", "timeseries.json", "anomalies.json"]:
    p = OUTPUTS_DIR / name
    if p.exists():
        try:
            summary[name] = json.loads(p.read_text())
        except Exception:
            pass

trend = summary.get("intelligence_summary.json", {}).get("trend", {})
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("30‑day Avg HRV", f"{trend.get('avg_hrv_30d', 0):.1f}")
with col2:
    st.metric("Trend Slope (30d)", f"{trend.get('trend_slope', 0):.3f}")
with col3:
    st.metric("Volatility (30d)", f"{trend.get('volatility_30d', 0):.2f}")

# Time series
_ts = summary.get("timeseries.json", {}).get("daily", [])
if _ts:
    st.subheader("HRV: Last 180 Days")
    vals = [row.get("hrv") for row in _ts[-180:]]
    st.line_chart({"HRV": vals})

    st.subheader("RHR, Strain, Sleep (Last 90 Days)")
    rhr = [row.get("rhr") for row in _ts[-90:]]
    strain = [row.get("strain") for row in _ts[-90:]]
    sleep = [row.get("sleep_hours") for row in _ts[-90:]]
    st.line_chart({"RHR": rhr, "Strain": strain, "Sleep (h)": sleep})

# Anomalies table
anom = summary.get("anomalies.json", {}).get("anomalies", [])
if anom:
    st.subheader("Anomalies (|z| ≥ 2.5)")
    st.dataframe(anom, hide_index=True, use_container_width=True)

st.caption("Trend level (baseline), slope (direction), volatility (stability) provide the headline view of recovery.") 