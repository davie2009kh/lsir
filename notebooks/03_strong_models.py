# %% [markdown]
# # Lsir (Լսիր): which free speech-to-text model should an Armenian phone line use?
# ## Notebook 3: the strongest free Armenian models, on a microphone and over the phone
#
# **Track:** AI + Languages · Veritas AI International High School Hackathon 2026
#
# Notebook 1 showed that the Whisper models most developers grab by default do badly on Armenian
# (whisper-small: over 100% word error rate). But there are newer, stronger free models built for many languages.
# A developer building a voice line for an Armenian clinic would ask: **which one should I use, and does it still
# work over a phone line?** Every published score for these models is on clean, high-quality audio.
#
# We test, on **all 932 clips** of the FLEURS Armenian **test** split (3 hours, 350 different sentences):
#
# | Model | Size | What it is |
# |---|---|---|
# | `openai/whisper-small` | 244M | the small default model many apps use |
# | `openai/whisper-large-v3-turbo` | 809M | OpenAI's newest fast multilingual model |
# | `facebook/mms-1b-all` (Armenian adapter) | 1B | Meta's Massively Multilingual Speech (1,100+ languages) |
# | `omniASR_CTC_1B_v2` | 1B | Meta's Omnilingual ASR (2025, 1,600+ languages) |
# | `BuzzASR/armenian` | 1.55B | Whisper-large-v3 fine-tuned only on Armenian by the BuzzASR project (EMNLP 2026) |
#
# **Fairness note.** MMS, Omnilingual and BuzzASR were trained partly on FLEURS's *training* split. The
# test split uses **different sentences** (we check: 0 shared), so the test is still on unseen sentences, but
# these models have heard the FLEURS recording style before, which may flatter them a little.
#
# *Runs on a free Kaggle T4 GPU. All data and models are downloaded inside the notebook.*

# %%
SMOKE = False
SEED = 42
RUN_OMNI = True          # Omnilingual needs its own Python 3.12 environment; skipped cleanly if install fails

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer"], check=True)

import io, os, re, time, json, random, urllib.request
from collections import Counter
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import jiwer
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, resample_poly
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor, Wav2Vec2ForCTC
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import transformers

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results"
os.makedirs(OUT, exist_ok=True)
print("torch", torch.__version__, "| transformers", transformers.__version__, "|",
      torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu")

# %% [markdown]
# ## 1. Data: FLEURS Armenian test split, plus a phone-line version of every clip

# %%
DATA_DIR = "/tmp/fleurs"; os.makedirs(DATA_DIR, exist_ok=True)

def load_fleurs(lang, split):
    path = f"{DATA_DIR}/{lang}_{split}.parquet"
    if not os.path.exists(path):
        urllib.request.urlretrieve(
            f"https://huggingface.co/api/datasets/google/fleurs/parquet/{lang}/{split}/0.parquet", path)
    return pd.read_parquet(path)

def decode_audio(a):
    wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = resample_poly(wav, 16000, sr).astype(np.float32)
    return wav

test = load_fleurs("hy_am", "test")
train_ids = set(load_fleurs("hy_am", "train")["id"])
print(f"test clips: {len(test)}, sentences: {test['id'].nunique()}, shared with train: {len(train_ids & set(test['id']))}")
test["wav"] = [decode_audio(a) for a in test["audio"]]
test = test.drop(columns=["audio"]).reset_index(drop=True)
if SMOKE:
    test = test.sample(16, random_state=0).reset_index(drop=True)

SOS_PHONE = butter(4, [300, 3400], btype="band", fs=8000, output="sos")

def phone_line(wav, snr_db=20, seed=0):
    """Simulate a narrow-band telephone channel: 8 kHz, 300-3400 Hz, 8-bit mu-law, background noise."""
    rng = np.random.default_rng(seed)
    y = resample_poly(wav, 1, 2)
    y = sosfilt(SOS_PHONE, y)
    y = 0.9 * y / (np.max(np.abs(y)) + 1e-9)
    mu = 255.0
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    c = np.round((c + 1) / 2 * mu) / mu * 2 - 1
    y = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu
    noise_power = np.mean(y ** 2) / (10 ** (snr_db / 10))
    y = y + rng.normal(0, np.sqrt(noise_power), y.shape)
    return resample_poly(y, 2, 1).astype(np.float32)

# identical phone versions to notebook 2 (seed = clip index)
test["wav_phone"] = [phone_line(w, seed=i) for i, w in enumerate(test["wav"])]
refs = test["raw_transcription"].tolist()
AUDIO = {"clean": test["wav"].tolist(), "phone": test["wav_phone"].tolist()}
print(f"{len(test)} clips, {sum(map(len, AUDIO['clean']))/16000/3600:.2f} hours per channel")

# %% [markdown]
# ## 2. Run every model on both versions of every clip
# Outputs are saved to disk after each model, so nothing is lost if a later step fails.

# %%
outputs, timings = {}, {}

def save_transcripts():
    pd.DataFrame({"id": test["id"], "gender": test["gender"], "seconds": test["wav"].map(len) / 16000,
                  "ref": refs, **{f"{m} | {ch}": v for (m, ch), v in outputs.items()}}
                 ).to_csv(f"{OUT}/strong_models_transcripts.csv", index=False)

@torch.no_grad()
def whisper_transcribe(model, proc, wavs, bs=8, **gen):
    out = []
    for i in range(0, len(wavs), bs):
        feats = proc.feature_extractor(list(wavs[i:i + bs]), sampling_rate=16000, return_tensors="pt").input_features
        ids = model.generate(feats.to(DEVICE, torch.float16), max_new_tokens=440, **gen)
        out += proc.batch_decode(ids, skip_special_tokens=True)
    return out

WHISPERS = [
    ("whisper-small", "openai/whisper-small", dict(language="hy", task="transcribe", num_beams=1)),
    ("whisper-large-v3-turbo", "openai/whisper-large-v3-turbo", dict(language="hy", task="transcribe", num_beams=1)),
    # settings recommended on the BuzzASR model card (language prompt is built into its generation config)
    ("BuzzASR-armenian", "BuzzASR/armenian", dict(num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2)),
]
lang_id = {}
for short, hf, gen in WHISPERS:
    t0 = time.time()
    proc = WhisperProcessor.from_pretrained(hf)
    model = WhisperForConditionalGeneration.from_pretrained(hf, torch_dtype=torch.float16).to(DEVICE).eval()
    for ch, wavs in AUDIO.items():
        outputs[(short, ch)] = whisper_transcribe(model, proc, wavs, **gen)
    # Which language does the model think it is hearing? (Whisper's built-in language detector, clean audio)
    if short != "BuzzASR-armenian":
        guesses = []
        for i in range(0, len(AUDIO["clean"]), 16):
            feats = proc.feature_extractor(AUDIO["clean"][i:i + 16], sampling_rate=16000, return_tensors="pt").input_features
            with torch.no_grad():
                ids = model.detect_language(feats.to(DEVICE, torch.float16))
            guesses += [t.strip("<|>") for t in proc.tokenizer.convert_ids_to_tokens(ids.view(-1).tolist())]
        lang_id[short] = Counter(guesses)
    timings[short] = time.time() - t0
    print(f"{short}: done in {timings[short]/60:.1f} min")
    del model; torch.cuda.empty_cache(); save_transcripts()

t0 = time.time()
mms_proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="hye")
mms = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="hye", ignore_mismatched_sizes=True,
                                     torch_dtype=torch.float16).to(DEVICE).eval()
for ch, wavs in AUDIO.items():
    hyps = []
    for i in range(0, len(wavs), 8):
        inp = mms_proc(wavs[i:i + 8], sampling_rate=16000, return_tensors="pt", padding=True)
        mask = inp.get("attention_mask")
        with torch.no_grad():
            logits = mms(inp.input_values.to(DEVICE, torch.float16),
                         attention_mask=None if mask is None else mask.to(DEVICE)).logits
        hyps += mms_proc.batch_decode(logits.argmax(-1))
    outputs[("mms-1b-all", ch)] = hyps
timings["mms-1b-all"] = time.time() - t0
del mms; torch.cuda.empty_cache(); save_transcripts()
print("timings (min):", {k: round(v / 60, 1) for k, v in timings.items()})

# %% [markdown]
# ### Meta Omnilingual ASR
# Omnilingual's code (fairseq2) only supports Python 3.10-3.12, and Kaggle runs a newer Python, so we install it
# in a separate Python 3.12 environment and run it there on exactly the same audio.

# %%
OMNI_SCRIPT = r'''
import json, sys, numpy as np, torch
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
data = np.load(sys.argv[1])          # plain float arrays, no pickle
n = int(data["n"])
card = sys.argv[2]
print("omni torch", torch.__version__, "cuda", torch.cuda.is_available(), flush=True)
pipe = ASRInferencePipeline(model_card=card)
res = {}
for ch in ["clean", "phone"]:
    audio = [{"waveform": torch.tensor(data[f"{ch}_{i}"], dtype=torch.float32), "sample_rate": 16000} for i in range(n)]
    res[ch] = [str(t) for t in pipe.transcribe(audio, lang=["hye_Armn"] * len(audio), batch_size=8)]
    print(ch, "done", flush=True)
json.dump(res, open(sys.argv[3], "w"), ensure_ascii=False)
'''
OMNI_CARD = "omniASR_CTC_1B_v2"
if RUN_OMNI:
    t0 = time.time()
    try:
        open("/tmp/omni_run.py", "w").write(OMNI_SCRIPT)
        arrays = {f"{ch}_{i}": w for ch, ws in AUDIO.items() for i, w in enumerate(ws)}
        np.savez("/tmp/omni_audio.npz", n=len(refs), **arrays)
        sh = lambda *c: subprocess.run(list(c), check=True, capture_output=True, text=True)
        sh(sys.executable, "-m", "pip", "install", "-q", "uv")
        sh("uv", "venv", "-q", "-p", "3.12", "/tmp/omni")
        sh("uv", "pip", "install", "-q", "--python", "/tmp/omni/bin/python", "omnilingual-asr")
        r = subprocess.run(["/tmp/omni/bin/python", "/tmp/omni_run.py", "/tmp/omni_audio.npz", OMNI_CARD, "/tmp/omni_out.json"],
                           capture_output=True, text=True)
        print(r.stdout[-2000:]); print(r.stderr[-3000:])
        omni = json.load(open("/tmp/omni_out.json"))
        for ch in ["clean", "phone"]:
            outputs[(f"omnilingual-{OMNI_CARD.split('_', 1)[1]}", ch)] = omni[ch]
        timings["omnilingual"] = time.time() - t0
        save_transcripts()
        print(f"Omnilingual done in {timings['omnilingual']/60:.1f} min")
    except Exception as e:
        print("Omnilingual could not run here:", repr(e)[:2000])
        if isinstance(e, subprocess.CalledProcessError):
            print(e.stdout[-2000:], e.stderr[-3000:])

# %% [markdown]
# ## 3. Results
# **WER** = share of words wrong; **CER** = share of letters wrong (fairer for Armenian, whose long words carry
# endings). 95% confidence intervals come from resampling the test clips 1,000 times (bootstrap).

# %%
normalizer = BasicTextNormalizer()
def norm(t):
    return re.sub(r"\s+", " ", normalizer(t or "")).strip()

R = [norm(r) for r in refs]
nref = np.array([len(r.split()) for r in R])
cref = np.array([len(r) for r in R])
rng = np.random.default_rng(SEED)
boot = [rng.integers(0, len(R), len(R)) for _ in range(1000)]

def is_loop(h, r):
    w = h.split()
    g = Counter(tuple(w[i:i + 3]) for i in range(max(len(w) - 2, 0)))
    return len(h) > 2 * max(len(r), 10) or (bool(g) and max(g.values()) >= 4)

per_clip, rows = {}, []
for (m, ch), hyps in outputs.items():
    H = [norm(h) for h in hyps]
    we = np.array([jiwer.process_words(r, h).wer * len(r.split()) if r else 0 for r, h in zip(R, H)])
    ce = np.array([jiwer.process_characters(r, h).cer * len(r) if r else 0 for r, h in zip(R, H)])
    loops = np.array([is_loop(h, r) for r, h in zip(R, H)])
    per_clip[(m, ch)] = (we, ce)
    bw = [100 * we[b].sum() / nref[b].sum() for b in boot]
    rows.append({"model": m, "audio": ch, "WER %": 100 * we.sum() / nref.sum(),
                 "WER 95% CI": f"{np.percentile(bw, 2.5):.1f}-{np.percentile(bw, 97.5):.1f}",
                 "CER %": 100 * ce.sum() / cref.sum(), "runaway loops %": 100 * loops.mean()})
res = pd.DataFrame(rows)
res.to_csv(f"{OUT}/strong_models_results.csv", index=False)
pd.set_option("display.width", 220)
wide = res.pivot(index="model", columns="audio", values=["WER %", "CER %"]).round(1)
wide["phone penalty (WER points)"] = (wide[("WER %", "phone")] - wide[("WER %", "clean")]).round(1)
wide = wide.sort_values(("WER %", "clean"))
print(f"FLEURS Armenian test: {len(R)} clips, {nref.sum()} words. Empty-output baseline = 100% WER.\n")
print(wide.to_string()); print()
print(res.round(1).to_string(index=False))

# %%
order = wide.index.tolist()
ax = res.pivot(index="model", columns="audio", values="WER %").loc[order][["clean", "phone"]].plot(
    kind="barh", figsize=(10, 4), color=["#4C72B0", "#DD8452"])
ax.axvline(100, color="gray", ls="--", lw=1)
ax.set_xlabel("Word error rate % on FLEURS Armenian test, 932 clips (lower is better)")
ax.set_title("Free Armenian speech-to-text models: microphone vs simulated phone line")
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f", fontsize=8, padding=2)
ax.invert_yaxis(); plt.tight_layout(); plt.savefig(f"{OUT}/strong_models_wer.png", dpi=130); plt.show()

# %% [markdown]
# ## 4. Does Whisper even recognise Armenian? (its built-in language detector)

# %%
for m, c in lang_id.items():
    tot = sum(c.values())
    print(f"{m}: detected as Armenian in {100*c.get('hy',0)/tot:.0f}% of {tot} Armenian clips; top guesses:",
          ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in c.most_common(5)))

# %% [markdown]
# ## 5. Who does it fail for? Women vs men, phone vs microphone
# A gap here would mean some callers get a worse service.

# %%
gender = test["gender"].map({0: "male", 1: "female"}).fillna("other").values
grows = []
for (m, ch), (we, ce) in per_clip.items():
    for g in ["female", "male"]:
        k = gender == g
        bw = [100 * we[b][k[b]].sum() / max(nref[b][k[b]].sum(), 1) for b in boot[:300]]
        grows.append({"model": m, "audio": ch, "gender": g, "clips": int(k.sum()),
                      "WER %": 100 * we[k].sum() / nref[k].sum(), "CER %": 100 * ce[k].sum() / cref[k].sum(),
                      "WER 95% CI": f"{np.percentile(bw, 2.5):.1f}-{np.percentile(bw, 97.5):.1f}"})
gdf = pd.DataFrame(grows)
gdf.to_csv(f"{OUT}/strong_models_by_gender.csv", index=False)
print(gdf.round(1).to_string(index=False))
print("\nNumber of different speakers is not given by FLEURS, so part of a gender gap could come from a few individual voices.")

# %% [markdown]
# ## 6. Real examples: correct transcript vs each model, over the phone

# %%
best = res[res.audio == "phone"].sort_values("WER %")["model"].iloc[0]
we_best = per_clip[(best, "phone")][0] / np.maximum(nref, 1)
for label, idx in [("typical", [int(np.argsort(we_best)[len(we_best) // 2])]), ("hardest for the best model", list(np.argsort(-we_best)[:2]))]:
    for i in idx:
        print(f"--- {label} (clip {i}, {len(AUDIO['clean'][i])/16000:.0f}s) ---")
        print(f"  {'correct':24s}: {refs[i]}")
        for m in order:
            print(f"  {m:24s}: {outputs[(m, 'phone')][i][:220]}")

json.dump({"timings_min": {k: round(v / 60, 1) for k, v in timings.items()}, "clips": len(R),
           "torch": torch.__version__, "transformers": transformers.__version__}, open(f"{OUT}/run_info_03.json", "w"), indent=1)
print(os.listdir(OUT))
