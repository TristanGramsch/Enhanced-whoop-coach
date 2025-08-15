import json
from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(REPO_ROOT / "shared" / "outputs")))

st.title("Drivers")

corr = {}
for name in ["correlations.json"]:
    p = OUTPUTS_DIR / name
    if p.exists():
        try:
            corr = json.loads(p.read_text())
        except Exception:
            pass

fields = ["hrv", "rhr", "strain", "sleep_hours", "disturbances"]
pear = corr.get("pearson", {})
if pear:
    st.subheader("Correlation Matrix (Pearson)")
    df = pd.DataFrame(pear).reindex(index=fields, columns=fields)
    st.dataframe(df.style.background_gradient(cmap="RdBu", axis=None), use_container_width=True)

st.subheader("Lag Correlation vs HRV (−7…+7 days)")
lag = corr.get("lag", {})
if lag:
    cols = st.columns(2)
    for i, var in enumerate(["rhr", "strain", "sleep_hours", "disturbances"]):
        series = lag.get(var, {})
        if series:
            ks = sorted(series.keys(), key=lambda x: int(x))
            vals = [series[k] if series[k] is not None else np.nan for k in ks]
            with cols[i % 2]:
                st.line_chart({f"{var} → HRV": vals})

st.caption("Correlation shows association strength; lag correlation detects delayed effects (e.g., strain today, HRV tomorrow).") 