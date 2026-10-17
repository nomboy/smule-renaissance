"""Web service for Smule Renaissance Small vocal restoration.

Uses same-origin relative URLs (no absolute host), so it works behind the
sandbox preview reverse proxy, which rewrites the Host header and strips
X-Forwarded-* headers.
"""

import tempfile

import torch
import torchaudio
import uvicorn
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, Response

from main import load_and_preprocess_audio, process_audio
from model import Renaissance
from spectral_ops import STFT, iSTFT

CHECKPOINT = "smule-renaissance-small.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model on {DEVICE} with dtype {DTYPE} ...", flush=True)
_model = Renaissance().to(DEVICE).to(DTYPE)
_model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
_model.eval()
_stft = STFT(n_fft=4096, hop_length=2048, win_length=4096)
_istft = iSTFT(n_fft=4096, hop_length=2048, win_length=4096)
print("Model ready.", flush=True)

app = FastAPI()

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<base href="./" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Smule Renaissance Small - Vocal Restoration</title>
<style>
  :root { color-scheme: light dark; }
  * { box-sizing: border-box; }
  body { margin: 0; min-height: 100vh; display: flex; align-items: center;
         justify-content: center; padding: 24px;
         font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
         background: #0f1115; color: #e8eaed; }
  .card { width: 100%; max-width: 620px; background: #171a21; border: 1px solid #262b36;
          border-radius: 16px; padding: 28px; box-shadow: 0 12px 40px rgba(0,0,0,.45); }
  h1 { margin: 0 0 6px; font-size: 20px; }
  .sub { margin: 0 0 20px; font-size: 13px; line-height: 1.5; color: #9aa3b2; }
  .drop { display: flex; align-items: center; gap: 10px; padding: 22px 16px;
          border: 1.5px dashed #39404d; border-radius: 12px; cursor: pointer;
          text-align: center; justify-content: center; transition: .15s; font-size: 14px; color: #c8cdd6; }
  .drop:hover, .drop.over { border-color: #4f7cff; background: #1b2030; }
  .row { display: flex; align-items: center; gap: 14px; margin-top: 18px; }
  button { background: #4f7cff; color: #fff; border: 0; border-radius: 10px;
           padding: 11px 22px; font-size: 14px; font-weight: 600; cursor: pointer; }
  button:disabled { opacity: .45; cursor: not-allowed; }
  #status { font-size: 13px; color: #9aa3b2; }
  .result { margin-top: 22px; padding-top: 18px; border-top: 1px solid #262b36; }
  audio { width: 100%; margin-top: 8px; }
  .lbl { font-size: 12px; text-transform: uppercase; letter-spacing: .06em; color: #7c869a; }
  a.dl { display: inline-block; margin-top: 12px; font-size: 13px; color: #8fb0ff; }
  .err { color: #ff6b6b; }
</style>
</head>
<body>
  <div class="card">
    <h1>Smule Renaissance Small</h1>
    <p class="sub">Vocal restoration. Upload a degraded vocal clip; the model resamples to 48 kHz and returns the restored audio.</p>

    <label class="drop" id="drop">
      <input id="file" type="file" accept="audio/*" hidden />
      <span id="hint">Click to choose an audio file (wav, mp3, m4a ...)</span>
    </label>

    <div class="row">
      <button id="go" disabled>Restore</button>
      <span id="status"></span>
    </div>

    <div class="result" id="result" hidden></div>
  </div>

<script>
(function () {
  var fileInput = document.getElementById('file');
  var drop = document.getElementById('drop');
  var hint = document.getElementById('hint');
  var go = document.getElementById('go');
  var status = document.getElementById('status');
  var result = document.getElementById('result');
  var chosen = null;

  function pick(file) {
    if (!file) { return; }
    chosen = file;
    hint.textContent = file.name + '  (' + (file.size / 1024).toFixed(0) + ' KB)';
    go.disabled = false;
    status.textContent = '';
    status.className = '';
    result.hidden = true;
  }

  fileInput.addEventListener('change', function () {
    if (fileInput.files && fileInput.files.length) { pick(fileInput.files[0]); }
  });

  ['dragenter', 'dragover'].forEach(function (e) {
    drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.add('over'); });
  });
  ['dragleave', 'drop'].forEach(function (e) {
    drop.addEventListener(e, function (ev) { ev.preventDefault(); drop.classList.remove('over'); });
  });
  drop.addEventListener('drop', function (ev) {
    if (ev.dataTransfer && ev.dataTransfer.files && ev.dataTransfer.files.length) {
      pick(ev.dataTransfer.files[0]);
    }
  });

  go.addEventListener('click', async function () {
    if (!chosen) { return; }
    go.disabled = true;
    status.className = '';
    status.textContent = 'Processing ... (may take a moment)';
    result.hidden = true;
    try {
      var fd = new FormData();
      fd.append('file', chosen, chosen.name);
      var res = await fetch('/enhance', { method: 'POST', body: fd });
      if (!res.ok) {
        var msg = await res.text();
        throw new Error(msg || ('HTTP ' + res.status));
      }
      var blob = await res.blob();
      var url = URL.createObjectURL(blob);
      result.innerHTML = '';
      var lbl = document.createElement('div');
      lbl.className = 'lbl';
      lbl.textContent = 'Restored output';
      var audio = document.createElement('audio');
      audio.controls = true;
      audio.src = url;
      var a = document.createElement('a');
      a.className = 'dl';
      a.href = url;
      a.download = 'enhanced.wav';
      a.textContent = 'Download enhanced.wav';
      result.appendChild(lbl);
      result.appendChild(audio);
      result.appendChild(a);
      result.hidden = false;
      status.textContent = 'Done';
    } catch (err) {
      status.className = 'err';
      status.textContent = 'Failed: ' + err.message;
    } finally {
      go.disabled = false;
    }
  });
})();
</script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
def index():
    return PAGE


@app.post("/enhance")
async def enhance(file: UploadFile = File(...)):
    raw = await file.read()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(raw)
        in_path = f.name

    waveform = load_and_preprocess_audio(in_path, DEVICE, DTYPE)
    enhanced = process_audio(_model, _stft, _istft, waveform, DEVICE)
    enhanced = enhanced.cpu().to(torch.float32)

    out_path = tempfile.mktemp(suffix="_enhanced.wav")
    torchaudio.save(out_path, enhanced, 48000)
    with open(out_path, "rb") as f:
        data = f.read()
    return Response(content=data, media_type="audio/wav")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860, log_level="info")
