# %% [markdown]
# # Lsir (Լսիր): notebook 3b, Meta Omnilingual ASR on Armenian, microphone vs phone line
#
# **Track:** AI + Languages · Veritas AI International High School Hackathon 2026
#
# Companion to notebook 3. Meta's **Omnilingual ASR** (2025, 1,600+ languages) needs its own Python 3.12
# environment (its fairseq2 library doesn't support Kaggle's newer Python). In notebook 3 its audio library didn't
# match the installed PyTorch, so here we install matching versions and run it on **exactly the same 932 FLEURS
# Armenian test clips**, clean and through the same phone-line simulation, scored the same way.
#
# **Fairness note.** Omnilingual was trained on a large multilingual corpus that includes FLEURS; the test split uses
# different sentences from training, but the recording style is familiar to it.

# %%
SEED = 42
OMNI_CARD = "omniASR_CTC_1B_v2"

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer", "uv"], check=True)
import io, os, re, json, time, urllib.request
from collections import Counter
import numpy as np, pandas as pd, soundfile as sf, jiwer
from scipy.signal import butter, sosfilt, resample_poly
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

OUT = "/kaggle/working/results"; os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# ## 1. Same test data and phone simulation as notebook 3

# %%
DATA_DIR = "/tmp/fleurs"; os.makedirs(DATA_DIR, exist_ok=True)
def load_fleurs(lang, split):
    path = f"{DATA_DIR}/{lang}_{split}.parquet"
    if not os.path.exists(path):
        urllib.request.urlretrieve(f"https://huggingface.co/api/datasets/google/fleurs/parquet/{lang}/{split}/0.parquet", path)
    return pd.read_parquet(path)

def decode_audio(a):
    wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
    wav = wav.mean(axis=1) if wav.ndim > 1 else wav
    return wav if sr == 16000 else resample_poly(wav, 16000, sr).astype(np.float32)

test = load_fleurs("hy_am", "test")
test["wav"] = [decode_audio(a) for a in test["audio"]]
test = test.drop(columns=["audio"]).reset_index(drop=True)

SOS_PHONE = butter(4, [300, 3400], btype="band", fs=8000, output="sos")
def phone_line(wav, snr_db=20, seed=0):
    rng = np.random.default_rng(seed)
    y = sosfilt(SOS_PHONE, resample_poly(wav, 1, 2))
    y = 0.9 * y / (np.max(np.abs(y)) + 1e-9)
    mu = 255.0
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    c = np.round((c + 1) / 2 * mu) / mu * 2 - 1
    y = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu
    y = y + rng.normal(0, np.sqrt(np.mean(y ** 2) / (10 ** (snr_db / 10))), y.shape)
    return resample_poly(y, 2, 1).astype(np.float32)

AUDIO = {"clean": test["wav"].tolist(), "phone": [phone_line(w, seed=i) for i, w in enumerate(test["wav"])]}
refs = test["raw_transcription"].tolist()
print(len(refs), "clips")

# %% [markdown]
# ## 2. Install Omnilingual ASR in its own Python 3.12 environment, with a matching audio library

# %%
def sh(*cmd):
    r = subprocess.run(list(cmd), capture_output=True, text=True)
    print(" ".join(cmd[:6]), "->", r.returncode); print(r.stdout[-1500:], r.stderr[-2500:])
    return r

sh("uv", "venv", "-q", "-p", "3.12", "/tmp/omni")
sh("uv", "pip", "install", "-q", "--python", "/tmp/omni/bin/python", "omnilingual-asr")
probe = ["/tmp/omni/bin/python", "-c", "import torch; print(torch.__version__.split('+')[0])"]
tv = subprocess.run(probe, capture_output=True, text=True).stdout.strip()
print("torch inside the Omnilingual environment:", tv)
# torchaudio must be built for exactly this torch version (that mismatch was the failure in notebook 3)
sh("uv", "pip", "install", "-q", "--python", "/tmp/omni/bin/python", f"torchaudio=={tv}")

# %% [markdown]
# ## 3. Transcribe both versions of every clip

# %%
OMNI_SCRIPT = r"""
import json, sys, numpy as np, torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
data = np.load(sys.argv[1]); n = int(data["n"])
print("cuda", torch.cuda.is_available(), flush=True)
pipe = ASRInferencePipeline(model_card=sys.argv[2])
res = {}
for ch in ["clean", "phone"]:
    audio = [{"waveform": torch.tensor(data[f"{ch}_{i}"], dtype=torch.float32), "sample_rate": 16000} for i in range(n)]
    res[ch] = [str(t) for t in pipe.transcribe(audio, lang=["hye_Armn"] * n, batch_size=8)]
    print(ch, "done", flush=True)
json.dump(res, open(sys.argv[3], "w"), ensure_ascii=False)
"""
open("/tmp/omni_run.py", "w").write(OMNI_SCRIPT)
arrays = {f"{ch}_{i}": w for ch, ws in AUDIO.items() for i, w in enumerate(ws)}
np.savez("/tmp/omni_audio.npz", n=len(refs), **arrays)
t0 = time.time()
sh("/tmp/omni/bin/python", "/tmp/omni_run.py", "/tmp/omni_audio.npz", OMNI_CARD, "/tmp/omni_out.json")
omni = json.load(open("/tmp/omni_out.json"))
print(f"done in {(time.time() - t0) / 60:.1f} min")
NAME = "omnilingual-" + OMNI_CARD.split("_", 1)[1]
pd.DataFrame({"id": test["id"], "gender": test["gender"], "ref": refs,
              f"{NAME} | clean": omni["clean"], f"{NAME} | phone": omni["phone"]}
             ).to_csv(f"{OUT}/omnilingual_transcripts.csv", index=False)

# %% [markdown]
# ## 4. Scores (same method as notebook 3)

# %%
normalizer = BasicTextNormalizer()
def norm(t):
    return re.sub(r"\s+", " ", normalizer(t or "")).strip()

R = [norm(r) for r in refs]
nref = np.array([len(r.split()) for r in R])
cref = np.array([len(r) for r in R])
rng = np.random.default_rng(SEED)
boot = [rng.integers(0, len(R), len(R)) for _ in range(1000)]
gender = test["gender"].map({0: "male", 1: "female"}).fillna("other").values
rows = []
for ch in ["clean", "phone"]:
    H = [norm(h) for h in omni[ch]]
    we = np.array([jiwer.process_words(r, h).wer * len(r.split()) if r else 0 for r, h in zip(R, H)])
    ce = np.array([jiwer.process_characters(r, h).cer * len(r) if r else 0 for r, h in zip(R, H)])
    bw = [100 * we[b].sum() / nref[b].sum() for b in boot]
    row = {"model": NAME, "audio": ch, "WER %": 100 * we.sum() / nref.sum(),
           "WER 95% CI": f"{np.percentile(bw, 2.5):.1f}-{np.percentile(bw, 97.5):.1f}",
           "CER %": 100 * ce.sum() / cref.sum()}
    for g in ["female", "male"]:
        k = gender == g
        row[f"WER % {g}"] = 100 * we[k].sum() / nref[k].sum()
    rows.append(row)
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/omnilingual_results.csv", index=False)
print(res.round(1).to_string(index=False))

for i in [42, 411]:
    print(f"--- clip {i} ---")
    print("  correct:", refs[i])
    print("  clean  :", omni["clean"][i])
    print("  phone  :", omni["phone"][i])
