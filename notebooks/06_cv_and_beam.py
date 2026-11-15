# %% [markdown]
# # Lsir (Լսիր): notebook 6, a second test set (Common Voice) and better decoding (beam search)
#
# **Track:** AI + Languages · Veritas AI International High School Hackathon 2026
#
# Two questions a judge would ask about notebooks 3-4:
#
# * **Does it hold up on data from a different source?** FLEURS is read by a small group of speakers in good
#   conditions. **Mozilla Common Voice** Armenian is recorded by volunteers on their own phones and laptops, and it
#   records each speaker's **age**, **gender** and an anonymous **speaker ID**. That lets us check how the models do
#   for **older speakers (60+)**, the people our phone line is for, and whether the gender gap from FLEURS is real
#   or caused by a few voices. (`MODE = "cv"`)
# * **Can smarter decoding help for free?** By default the model writes the single most likely next token each
#   step (*greedy*). **Beam search** keeps the 5 best partial transcripts and picks the best complete one.
#   (`MODE = "beam"`, on the FLEURS test set)
#
# We compare the original **BuzzASR-Armenian** with **Lsir** (BuzzASR + our phone LoRA from notebook 4), on clean
# audio, the fixed phone simulation, and the unseen GSM mobile codec.
#
# **Fairness note.** BuzzASR was trained on Common Voice's *training* split. Common Voice keeps different speakers
# in its test split, so the speakers are new, but the style is familiar to BuzzASR.

# %%
MODE = "cv"            # "cv": Common Voice test set, greedy decoding | "beam": FLEURS test set, greedy vs beam search
N_CV = 1000            # Common Voice clips to test (all speakers aged 50+ are always included)
SEED = 42
GREEDY = dict(num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2, max_new_tokens=440)
BEAM = dict(num_beams=5, no_repeat_ngram_size=3, repetition_penalty=1.2, max_new_tokens=440)

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer", "peft"], check=True)
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
# ## 3. Models: original BuzzASR and Lsir (BuzzASR + our phone LoRA from notebook 4)

# %%
BASE = "BuzzASR/armenian"
proc = WhisperProcessor.from_pretrained(BASE)
base = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float16).to(DEVICE).eval()
GCJ = json.load(open(hf_hub_download(BASE, "generation_config.json")))
for k in ["lang_to_id", "task_to_id", "language", "task", "no_timestamps_token_id"]:
    setattr(base.generation_config, k, GCJ[k])      # newer transformers drop BuzzASR's language table
base.generation_config.is_multilingual = True
TEXT_VOCAB = GCJ["eos_token_id"]                     # decode ordinary text tokens only (see notebook 4)
adapter = glob.glob("/kaggle/input/**/lsir-buzzasr-armenian-phone-lora-encoder/adapter_config.json", recursive=True)
print("adapter:", adapter)
model = PeftModel.from_pretrained(base, os.path.dirname(adapter[0])).eval()

@torch.no_grad()
def transcribe(wavs, use_adapter, gen, bs=8):
    out = []
    for i in range(0, len(wavs), bs):
        f = proc.feature_extractor(list(wavs[i:i + bs]), sampling_rate=16000, return_tensors="pt").input_features
        f = f.to(DEVICE, torch.float16)
        if use_adapter:
            ids = model.generate(input_features=f, **gen)
        else:
            with model.disable_adapter():
                ids = model.generate(input_features=f, **gen)
        out += [proc.tokenizer.decode([t for t in s.tolist() if t < TEXT_VOCAB]).strip() for s in ids]
    return out

# %% [markdown]
# ## 4. Transcribe

# %%
RUNS = [("BuzzASR (original)", False, "greedy"), ("Lsir (ours)", True, "greedy")]
if MODE == "beam":
    RUNS += [("BuzzASR (original)", False, "beam 5"), ("Lsir (ours)", True, "beam 5")]
outputs = {}
t0 = time.time()
for ch, wavs in AUDIO.items():
    for name, use_adapter, dec in RUNS:
        outputs[(name, dec, ch)] = transcribe(wavs, use_adapter, GREEDY if dec == "greedy" else BEAM)
        print(f"{name:20s} {dec:7s} {ch:10s} done ({(time.time() - t0) / 60:.0f} min)", flush=True)
    keep = ["client_id", "age", "gender"] if MODE == "cv" else ["id", "gender"]
    pd.DataFrame({**{k: df[k] for k in keep}, "ref": df["ref"],
                  **{" | ".join(k): v for k, v in outputs.items()}}).to_csv(f"{OUT}/nb06_{MODE}_transcripts.csv", index=False)

# %% [markdown]
# ## 5. Results, with 95% bootstrap confidence intervals

# %%
normalizer = BasicTextNormalizer()
def norm(t):
    return re.sub(r"\s+", " ", normalizer(t or "")).strip()

R = [norm(r) for r in df["ref"]]
nref = np.array([max(len(r.split()), 1) for r in R])
rng = np.random.default_rng(SEED)
boot = [rng.integers(0, len(R), len(R)) for _ in range(1000)]
errs = {}
rows = []
for key, hyps in outputs.items():
    H = [norm(h) for h in hyps]
    e = np.array([jiwer.process_words(r, h).wer * len(r.split()) if r else 0 for r, h in zip(R, H)])
    errs[key] = e
    bw = [100 * e[b].sum() / nref[b].sum() for b in boot]
    rows.append({"model": key[0], "decoding": key[1], "audio": key[2], "WER %": 100 * e.sum() / nref.sum(),
                 "95% CI": f"{np.percentile(bw, 2.5):.1f}-{np.percentile(bw, 97.5):.1f}",
                 "CER %": 100 * jiwer.cer(R, H)})
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/nb06_{MODE}_results.csv", index=False)
pd.set_option("display.width", 220)
print(res.round(1).to_string(index=False))

def paired(a, b):
    ea, eb = errs[a], errs[b]
    d = np.array([100 * (ea[i].sum() - eb[i].sum()) / nref[i].sum() for i in boot])
    return f"{' '.join(a)} -> {' '.join(b)}: {-100*(ea.sum()-eb.sum())/nref.sum():+.1f} WER points (95% CI {-np.percentile(d,97.5):+.1f} to {-np.percentile(d,2.5):+.1f})"

print()
for ch in AUDIO:
    print(paired(("BuzzASR (original)", "greedy", ch), ("Lsir (ours)", "greedy", ch)))
    if MODE == "beam":
        for m in ["BuzzASR (original)", "Lsir (ours)"]:
            print(paired((m, "greedy", ch), (m, "beam 5", ch)))

# %% [markdown]
# ## 6. Who does it fail for? By age group and gender, counting speakers (Common Voice only)

# %%
if MODE == "cv":
    age_map = {"teens": "under 20", "twenties": "20-39", "thirties": "20-39", "fourties": "40-59", "forties": "40-59",
               "fifties": "40-59", "sixties": "60+", "seventies": "60+", "eighties": "60+", "nineties": "60+"}
    df["age group"] = df["age"].fillna("").map(age_map).fillna("not given")
    df["gender group"] = df["gender"].fillna("").map(lambda g: "female" if g.startswith("female") else "male" if g.startswith("male") else "not given")
    grp = []
    for col in ["age group", "gender group"]:
        for g, idx in df.groupby(col).groups.items():
            idx = np.array(list(idx))
            row = {"group": f"{col}: {g}", "clips": len(idx), "speakers": df.loc[idx, "client_id"].nunique()}
            for key, e in errs.items():
                row[f"{key[0]} | {key[2]}"] = 100 * e[idx].sum() / nref[idx].sum()
            grp.append(row)
    gdf = pd.DataFrame(grp)
    gdf.to_csv(f"{OUT}/nb06_cv_by_group.csv", index=False)
    print(gdf.round(1).to_string(index=False))
    print("\nGroups with few speakers are unreliable: one person's voice can move the number a lot.")

# %% [markdown]
# ## 7. Examples (unseen GSM codec): correct transcript, original, ours

# %%
o, l = outputs[("BuzzASR (original)", "greedy", "unseen GSM")], outputs[("Lsir (ours)", "greedy", "unseen GSM")]
gain = errs[("BuzzASR (original)", "greedy", "unseen GSM")] - errs[("Lsir (ours)", "greedy", "unseen GSM")]
for i in list(np.argsort(-gain)[:3]) + list(np.argsort(gain)[:2]):
    print(f"--- clip {i} (WER change {-gain[i] / nref[i] * 100:+.0f} points) ---")
    print("  correct :", df["ref"][i]); print("  original:", o[i]); print("  ours    :", l[i])
print(os.listdir(OUT))
