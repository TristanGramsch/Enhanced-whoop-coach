from __future__ import annotations

from fastapi import FastAPI, Form, Request, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
import os
import json
import shutil

# Import processing util
from process_journal import process_journal_text
from transcribe import transcribe_audio_to_text

app = FastAPI(title="Journal Web")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS_DIR = REPO_ROOT / "shared" / "outputs"
UPLOADS_DIR = REPO_ROOT / "shared" / "uploads"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


INDEX_HTML = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Journal Entry</title>
    <style>
      body { font-family: system-ui, Arial, sans-serif; margin: 40px; background: #f7f7f9; }
      .shell { display: grid; gap: 20px; max-width: 820px; margin: 0 auto; }
      .card { background: white; padding: 20px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }
      textarea { width: 100%; height: 160px; padding: 12px; font-size: 16px; }
      button { padding: 10px 16px; background: #0d6efd; color: white; border: none; border-radius: 6px; cursor: pointer; }
      .row { display: flex; align-items: center; gap: 10px; }
      .muted { color: #6c757d; }
      pre { background: #f1f3f5; padding: 12px; border-radius: 8px; overflow-x: auto; }
      .btn-secondary { background: #6c757d; }
      .btn-danger { background: #dc3545; }
    </style>
  </head>
  <body>
    <div class="shell">
      <div class="card">
        <h2>Record Audio Journal</h2>
        <p class="muted">Record a short voice note about sleep, training, stress, or anything relevant.</p>
        <div class="row">
          <button id="startBtn">Start</button>
          <button id="stopBtn" class="btn-danger" disabled>Stop</button>
          <audio id="player" controls></audio>
        </div>
        <div class="row">
          <button id="uploadBtn" class="btn-secondary" disabled>Transcribe & Process</button>
          <span id="status" class="muted"></span>
        </div>
      </div>

      <div class="card">
        <h2>Or Paste Text</h2>
        <form method="post" action="/submit">
          <textarea name="text" placeholder="How are you feeling? Sleep, training, stress, recovery..." required></textarea>
          <div class="row">
            <button type="submit">Process</button>
            <span class="muted">Outputs saved to shared/outputs/journal_features.json</span>
          </div>
        </form>
      </div>
      <p class="muted">API: POST file to <code>/api/upload-audio</code> (multipart/form-data) or JSON to <code>/api/submit</code> with {"text": "..."}</p>
    </div>

    <script>
      let mediaRecorder;
      let recordedChunks = [];
      const startBtn = document.getElementById('startBtn');
      const stopBtn = document.getElementById('stopBtn');
      const player = document.getElementById('player');
      const uploadBtn = document.getElementById('uploadBtn');
      const statusEl = document.getElementById('status');

      startBtn.onclick = async () => {
        recordedChunks = [];
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder = new MediaRecorder(stream);
        mediaRecorder.ondataavailable = (e) => {
          if (e.data.size > 0) recordedChunks.push(e.data);
        };
        mediaRecorder.onstop = () => {
          const blob = new Blob(recordedChunks, { type: 'audio/webm' });
          player.src = URL.createObjectURL(blob);
          uploadBtn.disabled = false;
        };
        mediaRecorder.start();
        startBtn.disabled = true;
        stopBtn.disabled = false;
        statusEl.textContent = 'Recording...';
      };

      stopBtn.onclick = () => {
        mediaRecorder.stop();
        startBtn.disabled = false;
        stopBtn.disabled = true;
        statusEl.textContent = 'Recorded. Ready to upload.';
      };

      uploadBtn.onclick = async () => {
        statusEl.textContent = 'Uploading...';
        uploadBtn.disabled = true;
        const blob = new Blob(recordedChunks, { type: 'audio/webm' });
        const formData = new FormData();
        formData.append('file', blob, 'journal.webm');
        const res = await fetch('/api/upload-audio', { method: 'POST', body: formData });
        const json = await res.json();
        statusEl.textContent = json.error ? ('Error: ' + json.error) : 'Done. Saved ' + (json.artifacts?.journal_features_path || '');
        uploadBtn.disabled = false;
      };
    </script>
  </body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return HTMLResponse(content=INDEX_HTML)


@app.post("/submit")
async def submit_form(text: str = Form(...)):
    artifacts = process_journal_text(text, outputs_dir=str(OUTPUTS_DIR))
    return HTMLResponse(
        content=f"""
        <html><body>
        <div class=\"card\">
        <h3>Processed ✅</h3>
        <p>Saved: <code>{artifacts['journal_features_path']}</code></p>
        <a href=\"/\">Back</a>
        </div>
        </body></html>
        """,
    )


@app.post("/api/submit")
async def submit_json(payload: dict):
    text = (payload or {}).get("text", "").strip()
    if not text:
        return JSONResponse(status_code=400, content={"error": "text is required"})
    artifacts = process_journal_text(text, outputs_dir=str(OUTPUTS_DIR))
    meta = {
        "artifacts": artifacts,
    }
    # echo features
    try:
        features = json.loads((OUTPUTS_DIR / "journal_features.json").read_text())
        meta["features"] = features
    except Exception:
        pass
    return meta


@app.post("/api/upload-audio")
async def upload_audio(file: UploadFile = File(...)):
    try:
        # Save upload
        dest = UPLOADS_DIR / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Transcribe
        text = transcribe_audio_to_text(str(dest), model_size=os.getenv("WHISPER_MODEL", "tiny"))
        if not text.strip():
            return JSONResponse(status_code=400, content={"error": "transcription empty"})

        # Process features
        artifacts = process_journal_text(text, outputs_dir=str(OUTPUTS_DIR))
        # Attach transcribed text for reference
        with open(OUTPUTS_DIR / "journal_transcript.txt", "w") as tf:
            tf.write(text)

        return {"artifacts": artifacts, "transcript_preview": text[:200]}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)}) 