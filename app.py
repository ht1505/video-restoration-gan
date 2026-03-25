import os
import uuid
import subprocess
import urllib.request
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "saved_models/generator.pth")
WEIGHTS_URL = os.environ.get("WEIGHTS_URL", "")

generator = None

def load_generator():
    try:
        from models import UNetGenerator
        net = UNetGenerator()
        if os.path.exists(WEIGHTS_PATH):
            net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
            print(f"[startup] Weights loaded from {WEIGHTS_PATH}")
        else:
            print(f"[startup] WARNING – no weights at {WEIGHTS_PATH}, using random weights.")
        net.eval().to(DEVICE)
        return net
    except Exception as exc:
        print(f"[startup] Could not load generator: {exc}")
        return None

@asynccontextmanager
async def lifespan(application: FastAPI):
    global generator
    if WEIGHTS_URL and not os.path.exists(WEIGHTS_PATH):
        os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
        print(f"[startup] Downloading weights from {WEIGHTS_URL}...")
        urllib.request.urlretrieve(WEIGHTS_URL, WEIGHTS_PATH)
        print("[startup] Download complete.")
    generator = load_generator()
    yield

app = FastAPI(
    title="GAN Video Restoration",
    description="Upload a degraded video and receive an AI-restored version.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>GAN Video Restoration</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;700;800&display=swap');

  :root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --border: #2a2a3a;
    --accent: #00e5ff;
    --accent2: #ff3cac;
    --text: #e8e8f0;
    --muted: #6b6b80;
  }

  * { box-sizing: border-box; margin: 0; padding: 0; }

  body {
    background: var(--bg);
    color: var(--text);
    font-family: 'Syne', sans-serif;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 48px 24px;
  }

  body::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 20%, rgba(0,229,255,0.05) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 80%, rgba(255,60,172,0.05) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }

  .container { position: relative; z-index: 1; width: 100%; max-width: 760px; }

  header { text-align: center; margin-bottom: 56px; }

  .badge {
    display: inline-block;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: var(--accent);
    border: 1px solid var(--accent);
    padding: 4px 14px;
    margin-bottom: 20px;
    opacity: 0.8;
  }

  h1 {
    font-size: clamp(2.2rem, 6vw, 3.6rem);
    font-weight: 800;
    line-height: 1.05;
    background: linear-gradient(135deg, var(--text) 40%, var(--accent) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 16px;
  }

  .subtitle {
    color: var(--muted);
    font-size: 1rem;
    line-height: 1.6;
    max-width: 480px;
    margin: 0 auto;
  }

  .capabilities {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 40px;
  }

  .cap {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 18px 16px;
    text-align: center;
  }

  .cap-icon { font-size: 1.6rem; margin-bottom: 8px; }
  .cap-label { font-size: 0.78rem; color: var(--muted); font-family: 'Space Mono', monospace; letter-spacing: 0.5px; }

  .upload-zone {
    background: var(--surface);
    border: 1px solid var(--border);
    padding: 48px 32px;
    text-align: center;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
    position: relative;
    overflow: hidden;
    margin-bottom: 24px;
  }

  .upload-zone:hover { border-color: var(--accent); background: #15151e; }
  .upload-zone.dragging { border-color: var(--accent); background: rgba(0,229,255,0.04); }

  .upload-zone::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(135deg, rgba(0,229,255,0.03) 0%, transparent 60%);
    pointer-events: none;
  }

  .upload-icon { font-size: 2.8rem; margin-bottom: 16px; }

  .upload-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 8px;
    color: var(--text);
  }

  .upload-hint {
    font-size: 0.82rem;
    color: var(--muted);
    font-family: 'Space Mono', monospace;
  }

  #file-input { display: none; }

  #file-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.82rem;
    color: var(--accent);
    margin-top: 12px;
    min-height: 20px;
  }

  .btn {
    width: 100%;
    padding: 16px;
    background: var(--accent);
    color: #000;
    border: none;
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    cursor: pointer;
    transition: opacity 0.2s, transform 0.1s;
  }

  .btn:hover { opacity: 0.9; }
  .btn:active { transform: scale(0.99); }
  .btn:disabled { opacity: 0.35; cursor: not-allowed; }

  #status {
    margin-top: 28px;
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    min-height: 28px;
    text-align: center;
  }

  .status-processing { color: var(--accent); }
  .status-error { color: var(--accent2); }
  .status-done { color: #00ff9d; }

  .progress-bar {
    width: 100%;
    height: 2px;
    background: var(--border);
    margin-top: 12px;
    overflow: hidden;
    display: none;
  }

  .progress-bar.active { display: block; }

  .progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    animation: progress 2.5s ease-in-out infinite;
    transform-origin: left;
  }

  @keyframes progress {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(200%); }
  }

  .result-box {
    margin-top: 32px;
    background: var(--surface);
    border: 1px solid #00ff9d44;
    padding: 24px;
    display: none;
  }

  .result-box.visible { display: block; }

  .result-title {
    font-size: 0.75rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #00ff9d;
    font-family: 'Space Mono', monospace;
    margin-bottom: 16px;
  }

  .download-btn {
    display: inline-block;
    padding: 12px 28px;
    background: transparent;
    border: 1px solid #00ff9d;
    color: #00ff9d;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 0.9rem;
    cursor: pointer;
    text-decoration: none;
    transition: background 0.2s;
  }

  .download-btn:hover { background: rgba(0,255,157,0.08); }

  footer {
    margin-top: 64px;
    text-align: center;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    color: var(--muted);
    opacity: 0.5;
  }

  @media (max-width: 500px) {
    .capabilities { grid-template-columns: 1fr; }
  }
</style>
</head>
<body>
<div class="container">
  <header>
    <div class="badge">Pix2Pix GAN · U-Net · PatchGAN</div>
    <h1>Video Restoration</h1>
    <p class="subtitle">Upload a degraded video. The GAN model will de-blur, de-noise, and inpaint missing regions frame by frame.</p>
  </header>

  <div class="capabilities">
    <div class="cap">
      <div class="cap-icon">🔍</div>
      <div class="cap-label">De-blurring</div>
    </div>
    <div class="cap">
      <div class="cap-icon">🧹</div>
      <div class="cap-label">De-noising</div>
    </div>
    <div class="cap">
      <div class="cap-icon">🎨</div>
      <div class="cap-label">Inpainting</div>
    </div>
  </div>

  <div class="upload-zone" id="drop-zone" onclick="document.getElementById('file-input').click()">
    <div class="upload-icon">🎞️</div>
    <div class="upload-title">Drop your video here</div>
    <div class="upload-hint">AVI · MP4 · MOV — max 50MB</div>
    <div id="file-name"></div>
    <input type="file" id="file-input" accept="video/*,.avi">
  </div>

  <button class="btn" id="restore-btn" disabled onclick="restoreVideo()">
    Run Restoration
  </button>

  <div class="progress-bar" id="progress-bar">
    <div class="progress-fill"></div>
  </div>

  <div id="status"></div>

  <div class="result-box" id="result-box">
    <div class="result-title">✓ Restoration Complete</div>
    <a class="download-btn" id="download-link" href="#" download="restored.avi">
      ↓ Download Restored Video
    </a>
  </div>

  <footer>GAN-video · Pix2Pix Architecture · Deployed on Render</footer>
</div>

<script>
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const fileNameEl = document.getElementById('file-name');
  const restoreBtn = document.getElementById('restore-btn');
  const statusEl = document.getElementById('status');
  const progressBar = document.getElementById('progress-bar');
  const resultBox = document.getElementById('result-box');
  const downloadLink = document.getElementById('download-link');

  let selectedFile = null;

  fileInput.addEventListener('change', () => {
    if (fileInput.files[0]) selectFile(fileInput.files[0]);
  });

  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('dragging'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragging'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('dragging');
    if (e.dataTransfer.files[0]) selectFile(e.dataTransfer.files[0]);
  });

  function selectFile(file) {
    selectedFile = file;
    fileNameEl.textContent = '📎 ' + file.name;
    restoreBtn.disabled = false;
    resultBox.classList.remove('visible');
    statusEl.textContent = '';
  }

  async function restoreVideo() {
    if (!selectedFile) return;

    restoreBtn.disabled = true;
    progressBar.classList.add('active');
    statusEl.className = 'status-processing';
    statusEl.textContent = '⟳ Processing frames through GAN...';
    resultBox.classList.remove('visible');

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
      const response = await fetch('/restore', { method: 'POST', body: formData });

      progressBar.classList.remove('active');

      if (!response.ok) {
        const err = await response.json().catch(() => ({ detail: 'Unknown error' }));
        statusEl.className = 'status-error';
        statusEl.textContent = '✗ Error: ' + (err.detail || response.statusText);
        restoreBtn.disabled = false;
        return;
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      downloadLink.href = url;

      statusEl.className = 'status-done';
      statusEl.textContent = '✓ Restoration complete!';
      resultBox.classList.add('visible');
      restoreBtn.disabled = false;

    } catch (err) {
      progressBar.classList.remove('active');
      statusEl.className = 'status-error';
      statusEl.textContent = '✗ Network error: ' + err.message;
      restoreBtn.disabled = false;
    }
  }
</script>
</body>
</html>"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_PAGE)

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "weights_loaded": os.path.exists(WEIGHTS_PATH),
    }

@app.post("/restore")
async def restore_video(file: UploadFile = File(...)):
    if generator is None:
        raise HTTPException(status_code=503, detail="Generator model not available. Check server logs.")

    uid = str(uuid.uuid4())
    tmp_dir = "/tmp/gan_restore"
    os.makedirs(tmp_dir, exist_ok=True)

    ext = os.path.splitext(file.filename)[1] or ".avi"
    in_path = f"{tmp_dir}/{uid}_input{ext}"
    out_dir = tmp_dir

    with open(in_path, "wb") as f:
        f.write(await file.read())

    cmd = [
        "python", "inference.py",
        "--degraded_video", in_path,
        "--weights_path", WEIGHTS_PATH,
        "--output_dir", out_dir,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

    if result.returncode != 0:
        raise HTTPException(status_code=500, detail=f"Inference failed: {result.stderr[-2000:]}")

    # Find the output file
    candidates = sorted(
        [f for f in os.listdir(out_dir) if f.endswith(".avi") and "input" not in f],
        key=lambda x: os.path.getmtime(os.path.join(out_dir, x)),
        reverse=True,
    )
    if not candidates:
        raise HTTPException(status_code=500, detail="Restored video not found after inference.")

    return FileResponse(
        os.path.join(out_dir, candidates[0]),
        media_type="video/x-msvideo",
        filename="restored.avi",
    )
