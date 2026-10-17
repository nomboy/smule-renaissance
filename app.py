"""Gradio web service for Smule Renaissance Small vocal restoration.

Wraps the inference pipeline defined in main.py behind a browser UI so the
model can be tested without the command line.
"""

import tempfile

import gradio as gr
import torch
import torchaudio

from main import load_and_preprocess_audio, process_audio
from model import Renaissance
from spectral_ops import STFT, iSTFT

CHECKPOINT = "smule-renaissance-small.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

print(f"Loading model on {DEVICE} with dtype {DTYPE} ...")
_model = Renaissance().to(DEVICE).to(DTYPE)
_model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
_model.eval()

_stft = STFT(n_fft=4096, hop_length=2048, win_length=4096)
_istft = iSTFT(n_fft=4096, hop_length=2048, win_length=4096)
print("Model ready.")


def enhance(input_path):
    if not input_path:
        return None
    waveform = load_and_preprocess_audio(input_path, DEVICE, DTYPE)
    enhanced = process_audio(_model, _stft, _istft, waveform, DEVICE)
    enhanced = enhanced.cpu().to(torch.float32)

    out_path = tempfile.mktemp(suffix="_enhanced.wav")
    torchaudio.save(out_path, enhanced, 48000)
    return out_path


demo = gr.Interface(
    fn=enhance,
    inputs=gr.Audio(type="filepath", label="Input (degraded vocal)"),
    outputs=gr.Audio(type="filepath", label="Output (restored vocal)"),
    title="Smule Renaissance Small - Vocal Restoration",
    description=(
        "Upload or record a degraded vocal clip. The model resamples to 48 kHz, "
        "processes it in the spectral domain, and returns the restored audio."
    ),
)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
