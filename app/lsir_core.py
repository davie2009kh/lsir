"""Model, phone-line simulation and error marking for the Lsir demo (same channels and settings as notebooks 4-6)."""
import html
import json
import os
import re
import shutil
import subprocess
import time

import jiwer
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from peft import PeftModel
from scipy.signal import butter, resample_poly, sosfilt
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

BASE = "BuzzASR/armenian"
HERE = os.path.dirname(os.path.abspath(__file__))
LSIR_ADAPTER = os.environ.get("LSIR_ADAPTER") or (os.path.join(HERE, "adapter") if os.path.isdir(os.path.join(HERE, "adapter"))
                                                else "davitlive5/lsir-buzzasr-armenian-phone-lora")
DEVICE = "cuda" if torch.cuda.is_available() or os.environ.get("SPACES_ZERO_GPU") else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
GEN = dict(num_beams=5, no_repeat_ngram_size=3, repetition_penalty=1.2, max_new_tokens=440)   # = notebook 6 "beam 5"
MAX_SECONDS = 30
FFMPEG = shutil.which("ffmpeg")
HAS_GSM = bool(FFMPEG) and "libgsm" in subprocess.run([FFMPEG, "-hide_banner", "-encoders"],
                                                      capture_output=True, text=True).stdout
print("ffmpeg:", FFMPEG, "| GSM encoder:", HAS_GSM)

processor = WhisperProcessor.from_pretrained(BASE)
_base = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=DTYPE)
_gcj = json.load(open(hf_hub_download(BASE, "generation_config.json")))
for k in ["lang_to_id", "task_to_id", "language", "task", "no_timestamps_token_id"]:
    setattr(_base.generation_config, k, _gcj[k])        # restore BuzzASR's language table (dropped by newer transformers)
_base.generation_config.is_multilingual = True
TEXT_VOCAB = _gcj["eos_token_id"]                       # special tokens start here; decode ordinary text tokens only
model = PeftModel.from_pretrained(_base, LSIR_ADAPTER, torch_device="cpu").to(DEVICE).eval()   # CPU first (ZeroGPU)

# ---------------------------------------------------------------- phone lines (identical to notebook 6)
SOS_PHONE = butter(4, [300, 3400], btype="band", fs=8000, output="sos")


def mu_law(y, mu=255.0):
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    c = np.round((c + 1) / 2 * mu) / mu * 2 - 1
    return np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu


def add_noise(y, snr_db, rng, kind="white"):
    n = rng.normal(0, 1, len(y))
    if kind == "pink":
        f = np.fft.rfft(n); f /= np.sqrt(np.arange(1, len(f) + 1)); n = np.fft.irfft(f, len(y))
    n *= np.sqrt(np.mean(y ** 2) / (np.mean(n ** 2) + 1e-12) / 10 ** (snr_db / 10))
    return y + n


def landline(wav, seed):
    """8 kHz, 300-3400 Hz, 8-bit mu-law (G.711), 20 dB line noise."""
    rng = np.random.default_rng(seed)
    y = sosfilt(SOS_PHONE, resample_poly(wav, 1, 2))
    y = mu_law(0.9 * y / (np.max(np.abs(y)) + 1e-9))
    return resample_poly(add_noise(y, 20, rng), 2, 1).astype(np.float32)


def mobile(wav, seed):
    """GSM 06.10 mobile codec + 15 dB pink noise + 5% packet loss: the condition Lsir never trained on."""
    rng = np.random.default_rng(seed)
    x8 = resample_poly(wav, 1, 2).astype(np.float32)
    enc = subprocess.run([FFMPEG, "-loglevel", "error", "-f", "f32le", "-ar", "8000", "-ac", "1", "-i", "pipe:0",
                          "-c:a", "libgsm", "-b:a", "13k", "-f", "gsm", "pipe:1"], input=x8.tobytes(), capture_output=True)
    dec = subprocess.run([FFMPEG, "-loglevel", "error", "-f", "gsm", "-i", "pipe:0", "-f", "f32le", "-ar", "8000",
                          "-ac", "1", "pipe:1"], input=enc.stdout, capture_output=True)
    y = np.frombuffer(dec.stdout, dtype=np.float32)
    y = np.pad(y, (0, max(0, len(x8) - len(y))))[:len(x8)]
    y = add_noise(resample_poly(y, 2, 1), 15, rng, kind="pink")
    for s in range(0, len(y), 320):
        if rng.random() < 0.05:
            y[s:s + 320] = 0.0
    return y.astype(np.float32)


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


# ---------------------------------------------------------------- both models in one decoding pass
@torch.no_grad()
def transcribe_both(wav):
    """Lsir's LoRA sits only in the audio encoder, so the decoder is identical for both models. We run the encoder
    twice (adapter off, then on) and decode both in a single batch: same output, about half the time."""
    t0 = time.time()
    feats = processor.feature_extractor(wav, sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, DTYPE)
    encoder = model.get_base_model().get_encoder()
    with model.disable_adapter():
        h_orig = encoder(feats).last_hidden_state
    h_lsir = encoder(feats).last_hidden_state
    try:
        ids = model.get_base_model().generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=torch.cat([h_orig, h_lsir])), **GEN)
        texts = [_text(s) for s in ids]
    except Exception as err:                      # fall back to two ordinary runs if this library version objects
        print("batched decoding failed, running the models one after another:", repr(err))
        with model.disable_adapter():
            a = model.generate(input_features=feats, **GEN)[0]
        texts = [_text(a), _text(model.generate(input_features=feats, **GEN)[0])]
    return texts[0], texts[1], time.time() - t0


def _text(ids):
    return processor.tokenizer.decode([t for t in ids.tolist() if t < TEXT_VOCAB]).strip()


# ---------------------------------------------------------------- marking errors against a known transcript
_norm = BasicTextNormalizer()


def norm(t):
    return re.sub(r"\s+", " ", _norm(t or "")).strip()


def marked(hyp, ref):
    """HTML of the transcript with wrong and extra words marked, and missing words shown as gaps."""
    h = norm(hyp)
    if not h:
        return '<p class="lsir-text lsir-empty">(nothing recognised)</p>', None
    if not ref:
        return f'<p class="lsir-text">{html.escape(h)}</p>', None
    r = norm(ref)
    out = jiwer.process_words(r, h)
    rw, hw = out.references[0], out.hypotheses[0]
    parts, errors = [], 0
    for c in out.alignments[0]:
        if c.type == "equal":
            parts += [html.escape(w) for w in hw[c.hyp_start_idx:c.hyp_end_idx]]
        elif c.type == "substitute":
            for k in range(c.hyp_end_idx - c.hyp_start_idx):
                parts.append(f'<mark class="lsir-sub" title="should be: {html.escape(rw[c.ref_start_idx + k])}">'
                             f'{html.escape(hw[c.hyp_start_idx + k])}</mark>')
                errors += 1
        elif c.type == "insert":
            for w in hw[c.hyp_start_idx:c.hyp_end_idx]:
                parts.append(f'<mark class="lsir-ins" title="extra word">{html.escape(w)}</mark>')
                errors += 1
        elif c.type == "delete":
            for w in rw[c.ref_start_idx:c.ref_end_idx]:
                parts.append(f'<span class="lsir-del" title="missing: {html.escape(w)}">{html.escape(w)}</span>')
                errors += 1
    return f'<p class="lsir-text">{" ".join(parts)}</p>', (errors, len(rw))
