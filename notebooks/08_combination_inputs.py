# %% [markdown]
# # Lsir (Լսիր): notebook 8, inputs for system combination and a confidence check
#
# **Track:** AI + Languages · Veritas AI International High School Hackathon 2026
#
# Our system-combination experiment (ROVER: word-level voting between BuzzASR, Meta Omnilingual and Meta MMS)
# cut errors by about 12% on FLEURS clean and phone audio. To check that it also holds on the **unseen GSM mobile
# codec** and on **Common Voice**, we need MMS and Omnilingual transcripts for those conditions. This notebook makes
# them, on exactly the same clips and audio conditions as notebooks 4 and 6. The voting itself runs on these saved
# transcripts (`tools/rover.py`).
#
# In `MODE = "fleurs"` we also record **how confident our model is** in each transcript (average log-probability of
# its tokens), to test whether low confidence can tell a voice assistant when to ask the caller to repeat.

# %%
MODE = "cv"            # "cv": Common Voice test subset (same 1,000 clips as notebook 6) | "fleurs": FLEURS test
N_CV = 1000            # Common Voice clips to test (all speakers aged 50+ are always included)
SEED = 42
GREEDY = dict(num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2, max_new_tokens=440)
BEAM = dict(num_beams=5, no_repeat_ngram_size=3, repetition_penalty=1.2, max_new_tokens=440)

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer", "peft", "uv"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)
import glob, io, os, re, json, time, shutil, urllib.request
from collections import Counter
import numpy as np, pandas as pd, torch, jiwer
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, resample_poly
from huggingface_hub import hf_hub_download
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from peft import PeftModel
import transformers
transformers.logging.set_verbosity_error()
DEVICE = "cuda"
OUT = "/kaggle/working/results"; os.makedirs(OUT, exist_ok=True)
FFMPEG = shutil.which("ffmpeg")

# %% [markdown]
# ## 1. Test data

# %%
def fetch(url, path):
    if not os.path.exists(path):
        urllib.request.urlretrieve(url, path)
    return pd.read_parquet(path)

def to_16k(audio_bytes):
    """Decode any audio file (Common Voice uses mp3) to 16 kHz mono float with ffmpeg."""
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"],
                       input=audio_bytes, capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()

if MODE == "cv":
    df = fetch("https://huggingface.co/api/datasets/Chillarmo/common_voice_20_armenian/parquet/default/test/0.parquet", "/tmp/cv_test.parquet")
    print("Common Voice Armenian test:", len(df), "clips |", df["client_id"].nunique(), "speakers")
    print("columns:", [c for c in df.columns if c != "audio"])
    print("age:", dict(Counter(df["age"].fillna(""))))
    print("gender:", dict(Counter(df["gender"].fillna(""))))
    older = df["age"].fillna("").isin(["fifties", "sixties", "seventies", "eighties", "nineties"])
    rest = df[~older].sample(max(0, min(N_CV, len(df)) - int(older.sum())), random_state=SEED)
    df = pd.concat([df[older], rest]).reset_index(drop=True)
    df["ref"] = df["sentence"]
else:
    df = fetch("https://huggingface.co/api/datasets/google/fleurs/parquet/hy_am/test/0.parquet", "/tmp/fleurs_test.parquet")
    df["ref"] = df["raw_transcription"]
df["wav"] = [to_16k(a["bytes"]) for a in df["audio"]]
df = df.drop(columns=["audio"])
print(len(df), "clips used,", round(df["wav"].map(len).sum() / 16000 / 3600, 2), "hours")

# %% [markdown]
# ## 2. Audio conditions (identical to notebooks 4-5)

# %%
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

def phone_line(wav, seed):
    rng = np.random.default_rng(seed)
    y = mu_law(0.9 * (lambda z: z / (np.max(np.abs(z)) + 1e-9))(sosfilt(SOS_PHONE, resample_poly(wav, 1, 2))))
    return resample_poly(add_noise(y, 20, rng), 2, 1).astype(np.float32)

def gsm_unseen(wav, seed):
    """GSM 06.10 mobile codec + 15 dB pink noise + 5% packet loss (never used in training)."""
    rng = np.random.default_rng(seed)
    x8 = resample_poly(wav, 1, 2).astype(np.float32)
    enc = subprocess.run([FFMPEG, "-loglevel", "error", "-f", "f32le", "-ar", "8000", "-ac", "1", "-i", "pipe:0",
                          "-c:a", "libgsm", "-b:a", "13k", "-f", "gsm", "pipe:1"], input=x8.tobytes(), capture_output=True)
    dec = subprocess.run([FFMPEG, "-loglevel", "error", "-f", "gsm", "-i", "pipe:0", "-f", "f32le", "-ar", "8000", "-ac", "1", "pipe:1"],
                         input=enc.stdout, capture_output=True)
    y = np.frombuffer(dec.stdout, dtype=np.float32)
    y = np.pad(y, (0, max(0, len(x8) - len(y))))[:len(x8)]
    y = add_noise(resample_poly(y, 2, 1), 15, rng, kind="pink")
    for s in range(0, len(y), 320):
        if rng.random() < 0.05:
            y[s:s + 320] = 0.0
    return y.astype(np.float32)

AUDIO = {"clean": df["wav"].tolist(),
         "phone": [phone_line(w, seed=i) for i, w in enumerate(df["wav"])],
         "unseen GSM": [gsm_unseen(w, seed=50_000 + i) for i, w in enumerate(df["wav"])]}

# %% [markdown]
# ## 3. Meta MMS-1B (Armenian adapter)

# %%
from transformers import AutoProcessor, Wav2Vec2ForCTC
outputs = {}
def save():
    keep = ["client_id", "age", "gender"] if MODE == "cv" else ["id", "gender"]
    pd.DataFrame({**{k: df[k] for k in keep}, "ref": df["ref"], **{f"{m} | {c}": v for (m, c), v in outputs.items()}}
                 ).to_csv(f"{OUT}/nb08_{MODE}_transcripts.csv", index=False)

mproc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="hye")
mms = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="hye", ignore_mismatched_sizes=True,
                                     torch_dtype=torch.float16).to(DEVICE).eval()
t0 = time.time()
for ch, wavs in AUDIO.items():
    hyps = []
    for i in range(0, len(wavs), 8):
        inp = mproc(wavs[i:i + 8], sampling_rate=16000, return_tensors="pt", padding=True)
        mask = inp.get("attention_mask")
        with torch.no_grad():
            lg = mms(inp.input_values.to(DEVICE, torch.float16), attention_mask=None if mask is None else mask.to(DEVICE)).logits
        hyps += mproc.batch_decode(lg.argmax(-1))
    outputs[("mms-1b-all", ch)] = hyps
    print(f"MMS {ch} done ({(time.time() - t0) / 60:.0f} min)", flush=True)
del mms; torch.cuda.empty_cache(); save()

# %% [markdown]
# ## 4. Meta Omnilingual ASR (its own Python 3.12 environment, as in notebook 3b)

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
# fairseq2 pins NumPy 1.26, and the newest pyarrow needs NumPy 2, so pin a pyarrow that still supports NumPy 1.x
sh("uv", "pip", "install", "-q", "--python", "/tmp/omni/bin/python", f"torchaudio=={tv}", "pyarrow>=20,<22")
sh("/tmp/omni/bin/python", "-c", "import numpy, pyarrow, torchaudio; print(numpy.__version__, pyarrow.__version__, torchaudio.__version__)")


# %%
OMNI_SCRIPT = r"""
CHANNELS = None
import json, sys, numpy as np, torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
data = np.load(sys.argv[1]); n = int(data["n"])
print("cuda", torch.cuda.is_available(), flush=True)
pipe = ASRInferencePipeline(model_card=sys.argv[2])
res = {}
for ch in CHANNELS:
    audio = [{"waveform": torch.tensor(data[f"{ch}_{i}"], dtype=torch.float32), "sample_rate": 16000} for i in range(n)]
    res[ch] = [str(t) for t in pipe.transcribe(audio, lang=["hye_Armn"] * n, batch_size=8)]
    print(ch, "done", flush=True)
json.dump(res, open(sys.argv[3], "w"), ensure_ascii=False)
"""
open("/tmp/omni_run.py", "w").write(OMNI_SCRIPT.replace("CHANNELS = None", f"CHANNELS = {list(AUDIO)!r}"))
arrays = {f"{ch}_{i}": w for ch, ws in AUDIO.items() for i, w in enumerate(ws)}
np.savez("/tmp/omni_audio.npz", n=len(df), **arrays)
t0 = time.time()
sh("/tmp/omni/bin/python", "/tmp/omni_run.py", "/tmp/omni_audio.npz", "omniASR_CTC_1B_v2", "/tmp/omni_out.json")
omni = json.load(open("/tmp/omni_out.json"))
for ch in AUDIO:
    outputs[("omnilingual-CTC_1B_v2", ch)] = omni[ch]
save()

# %% [markdown]
# ## 5. (FLEURS only) Can our model tell when it's probably wrong?
# For each transcript we compute Lsir's average token log-probability. If low confidence goes with high error, a
# voice assistant could ask the caller to repeat instead of acting on a wrong transcript.

# %%
if MODE == "fleurs":
    from peft import PeftModel
    from huggingface_hub import hf_hub_download
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    BASE = "BuzzASR/armenian"
    proc = WhisperProcessor.from_pretrained(BASE)
    base = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float16).to(DEVICE).eval()
    GCJ = json.load(open(hf_hub_download(BASE, "generation_config.json")))
    for k in ["lang_to_id", "task_to_id", "language", "task", "no_timestamps_token_id"]:
        setattr(base.generation_config, k, GCJ[k])
    base.generation_config.is_multilingual = True
    TV = GCJ["eos_token_id"]
    adapter = glob.glob("/kaggle/input/**/lsir-buzzasr-armenian-phone-lora-encoder/adapter_config.json", recursive=True)
    model = PeftModel.from_pretrained(base, os.path.dirname(adapter[0])).eval()
    conf = {}
    for ch, wavs in AUDIO.items():
        texts, scores = [], []
        for i in range(0, len(wavs), 8):
            f = proc.feature_extractor(list(wavs[i:i + 8]), sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, torch.float16)
            with torch.no_grad():
                g = model.generate(input_features=f, num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2,
                                   max_new_tokens=440, output_scores=True, return_dict_in_generate=True)
            tr = model.compute_transition_scores(g.sequences, g.scores, normalize_logits=True).float().cpu().numpy()
            seqs = g.sequences[:, -tr.shape[1]:].cpu().numpy()
            for s, t in zip(seqs, tr):
                keep = s < TV
                scores.append(float(t[keep].mean()) if keep.any() else -10.0)
                texts.append(proc.tokenizer.decode([x for x in s if x < TV]).strip())
        outputs[("lsir", ch)] = texts
        conf[ch] = scores
        print("confidence", ch, "done", flush=True)
    for ch in AUDIO:
        df[f"lsir confidence | {ch}"] = conf[ch]
    save()
    pd.DataFrame({f"lsir confidence | {ch}": conf[ch] for ch in AUDIO}).to_csv(f"{OUT}/nb08_fleurs_confidence.csv", index=False)
print(os.listdir(OUT))
