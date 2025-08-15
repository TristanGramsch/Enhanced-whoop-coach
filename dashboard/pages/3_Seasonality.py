import json
from pathlib import Path
import os
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(REPO_ROOT / "shared" / "outputs")))

st.title("Seasonality")

season = {}
p = OUTPUTS_DIR / "seasonality.json"
if p.exists():
    try:
        season = json.loads(p.read_text())
    except Exception:
        pass

wd = season.get("weekday_means", {})
if wd:
    st.subheader("Weekday Pattern (HRV)")
    ordered = [wd.get(str(i)) for i in range(7)]
    st.bar_chart(ordered)

st.caption("Seasonality captures recurring weekly rhythms distinct from long‑term trend.") 