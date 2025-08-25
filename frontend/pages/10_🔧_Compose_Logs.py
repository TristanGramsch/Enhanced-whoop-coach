import os
import time
import shutil
import subprocess
from pathlib import Path
from collections import deque

import streamlit as st


# Determine repo root from within frontend/pages/
REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE_FILE = REPO_ROOT / "docker-compose.yml"
OUTPUTS_DIR = Path(os.environ.get("OUTPUTS_DIR", str(REPO_ROOT / "outputs")))


st.set_page_config(page_title="Docker Compose Logs", layout="wide")
st.title("Docker Compose Logs")

st.write(
    "Stream live logs. Attempts to run `docker compose logs -f | cat` when available. "
    "If Docker/compose is not available in this environment, falls back to streaming `outputs/pipeline.log`."
)


def can_run_compose() -> bool:
    docker_path = shutil.which("docker")
    return bool(docker_path) and COMPOSE_FILE.exists()


source = st.radio(
    "Log source",
    [
        "Docker Compose (if available)",
        "Fallback: orchestrator outputs/pipeline.log",
    ],
    index=0,
)

service_filter = st.text_input("Service name (optional)", placeholder="e.g., orchestrator")
no_color = st.checkbox("Strip ANSI colors", value=True)

cmd = "docker compose logs -f"
if service_filter.strip():
    cmd += f" {service_filter.strip()}"
if no_color:
    cmd += " --no-color"
cmd += " | cat"

st.caption(f"Command: {cmd}")


if "logs_running" not in st.session_state:
    st.session_state["logs_running"] = False

output_area = st.empty()


def stop_process():
    proc = st.session_state.get("_proc")
    if proc is not None:
        try:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except Exception:
                proc.kill()
        except Exception:
            pass
        st.session_state.pop("_proc", None)


colA, colB = st.columns(2)
with colA:
    start_clicked = st.button("Start streaming", disabled=st.session_state["logs_running"])
with colB:
    stop_clicked = st.button("Stop", disabled=not st.session_state["logs_running"])

if stop_clicked:
    st.session_state["logs_running"] = False
    stop_process()

if start_clicked:
    st.session_state["logs_running"] = True


def stream_compose_logs():
    if not can_run_compose():
        st.warning(
            "Docker/compose not available or compose file missing in this environment. "
            "Use the fallback source or run the frontend outside containers."
        )
        return

    if "_proc" not in st.session_state:
        st.session_state["_proc"] = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=str(REPO_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
        )

    proc = st.session_state["_proc"]
    lines = st.session_state.get("_lines", deque(maxlen=2000))
    st.session_state["_lines"] = lines

    assert proc.stdout is not None
    while st.session_state.get("logs_running", False):
        line = proc.stdout.readline()
        if line == "" and proc.poll() is not None:
            break
        if line:
            lines.append(line.rstrip("\n"))
            output_area.code("\n".join(lines))
        time.sleep(0.05)


def stream_pipeline_log():
    log_path = OUTPUTS_DIR / "pipeline.log"
    if not log_path.exists():
        st.info(f"Log file not found: {log_path}")
        return

    # Tail-like behavior
    lines = st.session_state.get("_lines", deque(maxlen=2000))
    st.session_state["_lines"] = lines

    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        # Jump near end for large files
        try:
            f.seek(0, os.SEEK_END)
        except Exception:
            pass
        while st.session_state.get("logs_running", False):
            chunk = f.read()
            if not chunk:
                time.sleep(0.2)
                continue
            for ln in chunk.splitlines():
                lines.append(ln)
            output_area.code("\n".join(lines))


if st.session_state.get("logs_running", False):
    try:
        if source.startswith("Docker Compose"):
            stream_compose_logs()
        else:
            stream_pipeline_log()
    finally:
        # Clean up process if any
        stop_process()
        st.session_state["logs_running"] = False

