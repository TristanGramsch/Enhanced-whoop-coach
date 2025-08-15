import json
from pathlib import Path
import os
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(REPO_ROOT / "shared" / "outputs")))

st.title("Anomalies")

anom = {}
p = OUTPUTS_DIR / "anomalies.json"
if p.exists():
    try:
        anom = json.loads(p.read_text())
    except Exception:
        pass

items = anom.get("anomalies", [])
if items:
    st.subheader("Outlier Days")
    st.dataframe(items, hide_index=True, use_container_width=True)
else:
    st.write("No significant anomalies detected.")

st.caption("Anomalies (e.g., |z| ≥ 2.5) surface unusual deviations and regime shifts for review.") 