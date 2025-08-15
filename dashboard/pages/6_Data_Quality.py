import json
from pathlib import Path
import os
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(REPO_ROOT / "shared" / "outputs")))

st.title("Data Quality")

# Basic missingness from timeseries
p = OUTPUTS_DIR / "timeseries.json"
if p.exists():
    try:
        ts = json.loads(p.read_text()).get("daily", [])
    except Exception:
        ts = []
else:
    ts = []

if ts:
    missing = {
        "hrv": sum(1 for r in ts if r.get("hrv") is None),
        "rhr": sum(1 for r in ts if r.get("rhr") is None),
        "strain": sum(1 for r in ts if r.get("strain") is None),
        "sleep_hours": sum(1 for r in ts if r.get("sleep_hours") is None),
        "disturbances": sum(1 for r in ts if r.get("disturbances") is None),
    }
    st.subheader("Missing Values (count)")
    st.json(missing)
else:
    st.write("Timeseries artifact not found. Generate analytics first.")

st.caption("Data quality checks ensure insights rest on reliable inputs.") 