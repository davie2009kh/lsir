"""Lsir demo: Armenian speech-to-text over a phone line.

Compares the original BuzzASR-Armenian model with Lsir (BuzzASR + our phone-robust LoRA adapter).

Run in Colab / any GPU machine:  pip install -r requirements.txt && python app.py
On a Hugging Face ZeroGPU Space: this file is the entry point (the `spaces` package provides the GPU).

LSIR_ADAPTER = local folder or Hugging Face repo id of the LoRA adapter saved by notebook 4.
"""
import json
import os

import gradio as gr
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from scipy.signal import butter, resample_poly, sosfilt
from transformers import WhisperForConditionalGeneration, WhisperProcessor

try:                      # ZeroGPU on Hugging Face Spaces; a no-op decorator elsewhere
    import spaces
    gpu = spaces.GPU(duration=60)
except ImportError:
    def gpu(f):
        return f

BASE = "BuzzASR/armenian"
LSIR_ADAPTER = os.environ.get("LSIR_ADAPTER", "lsir-buzzasr-armenian-phone-lora-encoder")
DEVICE = "cuda" if torch.cuda.is_available() or os.environ.get("SPACES_ZERO_GPU") else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
GEN = dict(num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2, max_new_tokens=440)
MAX_SECONDS = 30

processor = WhisperProcessor.from_pretrained(BASE)
base = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=DTYPE)
gcj = json.load(open(hf_hub_download(BASE, "generation_config.json")))
for k in ["lang_to_id", "task_to_id", "language", "task", "no_timestamps_token_id"]:
    setattr(base.generation_config, k, gcj[k])          # restore BuzzASR's language table (dropped by newer transformers)
base.generation_config.is_multilingual = True
TEXT_VOCAB = gcj["eos_token_id"]          # special tokens start here; decode ordinary text tokens only
model = PeftModel.from_pretrained(base, LSIR_ADAPTER).to(DEVICE).eval()

SOS_PHONE = butter(4, [300, 3400], btype="band", fs=8000, output="sos")


def phone_line(wav, snr_db=20, seed=0):
    """Same telephone simulation as the notebooks: 8 kHz, 300-3400 Hz, 8-bit mu-law, line noise."""
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
    x = wav.astype(np.float32)
    if np.issubdtype(wav.dtype, np.integer):
        x /= np.iinfo(wav.dtype).max
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != 16000:
        g = np.gcd(sr, 16000)
        x = resample_poly(x, 16000 // g, sr // g).astype(np.float32)
    return x[: MAX_SECONDS * 16000]


@torch.no_grad()
def _transcribe(wav, use_adapter):
    feats = processor.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, DTYPE)
    if use_adapter:
        ids = model.generate(input_features=feats, **GEN)
    else:
        with model.disable_adapter():
            ids = model.generate(input_features=feats, **GEN)
    return processor.tokenizer.decode([t for t in ids[0].tolist() if t < TEXT_VOCAB]).strip()


@gpu
def run(audio, phone):
    if audio is None:
        return "Record or upload some Armenian speech first.", "", None
    wav = to_16k_mono(audio)
    if phone:
        wav = phone_line(wav)
    return _transcribe(wav, False), _transcribe(wav, True), (16000, wav)


with gr.Blocks(title="Lsir: Armenian speech-to-text over the phone") as demo:
    gr.Markdown(
        "# Լսիր · Lsir\n"
        "Armenian speech-to-text that keeps working **over a phone line**. Record or upload up to 30 seconds of "
        "Armenian speech, choose whether it should go through a simulated phone line, and compare the best free "
        "Armenian model (BuzzASR) with our phone-trained version.\n\n"
        "*High-school hackathon prototype, not a medical tool. Don't rely on it for anything important.*"
    )
    with gr.Row():
        audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Armenian speech")
        with gr.Column():
            phone = gr.Checkbox(value=True, label="Send it through a phone line (8 kHz, narrow band, μ-law, noise)")
            btn = gr.Button("Transcribe", variant="primary")
            heard = gr.Audio(label="What the model hears", interactive=False)
    with gr.Row():
        out_orig = gr.Textbox(label="BuzzASR-Armenian (original)", lines=4)
        out_lsir = gr.Textbox(label="Lsir (BuzzASR + our phone LoRA)", lines=4)
    examples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples")
    if os.path.isdir(examples_dir):
        gr.Examples([[os.path.join(examples_dir, f), True] for f in sorted(os.listdir(examples_dir)) if f.endswith(".wav")],
                    inputs=[audio, phone], label="Example clips (FLEURS Armenian test set, never used in training)")
    btn.click(run, inputs=[audio, phone], outputs=[out_orig, out_lsir, heard])

if __name__ == "__main__":
    demo.launch(share=bool(os.environ.get("LSIR_SHARE")))
