"""Lsir demo: Armenian speech-to-text, original Whisper vs our fine-tuned model.

Run locally / in Colab:   pip install -r requirements.txt && python app.py
On Hugging Face Spaces:   this file is the Space's entry point.

Set LSIR_MODEL to a local folder or Hugging Face repo id with the fine-tuned model.
"""
import os

import gradio as gr
import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfilt
from transformers import WhisperForConditionalGeneration, WhisperProcessor

LSIR_MODEL = os.environ.get("LSIR_MODEL", "lsir-whisper-small-hy")
BASE_MODEL = "openai/whisper-small"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_SECONDS = 30

processor = WhisperProcessor.from_pretrained(BASE_MODEL)
models = {
    "original": WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, torch_dtype=DTYPE).to(DEVICE).eval(),
    "lsir": WhisperForConditionalGeneration.from_pretrained(LSIR_MODEL, torch_dtype=DTYPE).to(DEVICE).eval(),
}

SOS_PHONE = butter(4, [300, 3400], btype="band", fs=8000, output="sos")


def phone_line(wav, snr_db=20, seed=0):
    """Same telephone simulation as the notebooks: 8 kHz, 300-3400 Hz, 8-bit mu-law, noise."""
    rng = np.random.default_rng(seed)
    y = sosfilt(SOS_PHONE, resample_poly(wav, 1, 2))
    y = 0.9 * y / (np.max(np.abs(y)) + 1e-9)
    mu = 255.0
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    c = np.round((c + 1) / 2 * mu) / mu * 2 - 1
    y = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu
    y = y + rng.normal(0, np.sqrt(np.mean(y**2) / 10 ** (snr_db / 10)), y.shape)
    return resample_poly(y, 2, 1).astype(np.float32)


def to_16k_mono(audio):
    sr, wav = audio
    wav = wav.astype(np.float32)
    if np.issubdtype(audio[1].dtype, np.integer):
        wav /= np.iinfo(audio[1].dtype).max
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        wav = resample_poly(wav, 16000 // g, sr // g).astype(np.float32)
    return wav[: MAX_SECONDS * 16000]


@torch.no_grad()
def transcribe(model, wav):
    feats = processor.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features
    ids = model.generate(feats.to(DEVICE, DTYPE), language="hy", task="transcribe", max_new_tokens=440)
    return processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def run(audio, phone):
    if audio is None:
        return "Record or upload some Armenian speech first.", "", None
    wav = to_16k_mono(audio)
    if phone:
        wav = phone_line(wav)
    return transcribe(models["original"], wav), transcribe(models["lsir"], wav), (16000, wav)


with gr.Blocks(title="Lsir: Armenian speech-to-text") as demo:
    gr.Markdown(
        "# Լսիր · Lsir\n"
        "Speech-to-text that understands **Armenian**, even over a phone line. "
        "Record or upload up to 30 seconds of Armenian speech and compare the original Whisper model "
        "with our fine-tuned one.\n\n"
        "*Prototype from a high-school hackathon. Don't rely on it for anything important.*"
    )
    with gr.Row():
        audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Armenian speech")
        with gr.Column():
            phone = gr.Checkbox(value=True, label="Simulate a phone line (8 kHz, narrow band, noise)")
            btn = gr.Button("Transcribe", variant="primary")
            heard = gr.Audio(label="What the model hears", interactive=False)
    with gr.Row():
        out_orig = gr.Textbox(label="Original whisper-small", lines=4)
        out_lsir = gr.Textbox(label="Lsir (whisper-small fine-tuned on Armenian + phone audio)", lines=4)
    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
    if os.path.isdir(examples_dir):
        gr.Examples(
            [[os.path.join(examples_dir, f), True] for f in sorted(os.listdir(examples_dir)) if f.endswith(".wav")],
            inputs=[audio, phone],
            label="Example clips from the FLEURS Armenian test set (never used in training)",
        )
    btn.click(run, inputs=[audio, phone], outputs=[out_orig, out_lsir, heard])

if __name__ == "__main__":
    demo.launch(share=bool(os.environ.get("LSIR_SHARE")))
