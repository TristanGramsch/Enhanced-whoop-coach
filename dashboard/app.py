import json
from pathlib import Path
import os
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(REPO_ROOT / "shared" / "outputs")))

st.set_page_config(page_title="Recovery Intelligence Dashboard", layout="wide")
st.title("Recovery Intelligence Dashboard")

st.write("Use the sidebar to navigate between pages: Overview, Drivers, Seasonality, Anomalies, Forecasts, and Data Quality.")

missing = []
for name in [
    "intelligence_summary.json",
    "timeseries.json",
    "correlations.json",
    "seasonality.json",
    "anomalies.json",
    "forecast_backtests.json",
]:
    if not (OUTPUTS_DIR / name).exists():
        missing.append(name)

if missing:
    st.warning(
        "Some analytics artifacts are missing: " + ", ".join(missing) + ".\n"
        "Run the pipeline or call the Data Intelligence API /run to generate them."
    )

st.caption("Artifacts are read from OUTPUTS_DIR: " + str(OUTPUTS_DIR)) 