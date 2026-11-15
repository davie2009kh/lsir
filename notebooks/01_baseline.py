# %% [markdown]
# # Lsir (Լսիր): does speech recognition understand Armenian?
# ## Notebook 1 of 2: how well do today's free speech-to-text models work for Armenian?
#
# **Track:** AI + Languages · Veritas AI International High School Hackathon 2026
#
# **Who it's for.** An older person in Armenia who only speaks Armenian and calls a clinic or pharmacy
# phone line that is answered by a voice assistant. If the speech recogniser doesn't understand them,
# the assistant can't help them, and they're the people who most need the phone line to work.
#
# **The question in this notebook.** The same free model, OpenAI's Whisper, is used everywhere for
# speech-to-text. How much worse is it on Armenian than on Russian and English, *why* does it fail, and
# what happens when the audio comes through a phone line instead of a good microphone?
#
# **Data.** Google's FLEURS dataset: the same set of sentences read aloud by native speakers in 102
# languages, with human transcripts. We only use its **test** split here, so these are sentences no model
# in this notebook was fine-tuned on by us.
#
# **How to read the scores.**
# * **WER** (word error rate): the share of words the model got wrong (substituted, deleted or inserted). 0% is perfect.
#   It can go above 100% when the model adds lots of extra words.
# * **CER** (character error rate): the same idea, per letter. Armenian words are long and carry endings
#   (cases, plurals), so one wrong ending makes a whole word "wrong"; CER shows whether the model was close.
#
# *Runs on a free Kaggle T4 GPU. Everything (data and models) is downloaded inside the notebook.*

# %%
SMOKE = False          # True = tiny run to check the code works
N_PER_LANG = 300       # test clips per language for the zero-shot comparison
SEED = 42

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer"], check=True)

import io, os, re, time, json, random, unicodedata, urllib.request
from collections import Counter
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import jiwer
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, resample_poly
from transformers import (WhisperProcessor, WhisperForConditionalGeneration,
                          AutoProcessor, Wav2Vec2ForCTC)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import transformers
transformers.logging.set_verbosity_error()   # hide repetitive library warnings

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
OUT = "/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results"
os.makedirs(OUT, exist_ok=True)
if SMOKE:
    N_PER_LANG = 12
print("torch", torch.__version__, "| transformers", transformers.__version__, "| device", DEVICE,
      torch.cuda.get_device_name(0) if DEVICE == "cuda" else "")

# %% [markdown]
# ## 1. Load FLEURS test sets (Armenian, Russian, English)
#
# Russian is the second language most people in Armenia know, and English is the language these models
# were mostly trained on, so both make useful reference points.

# %%
LANGS = {"hy_am": ("Armenian", "hy"), "ru_ru": ("Russian", "ru"), "en_us": ("English", "en")}
DATA_DIR = "/tmp/fleurs"
os.makedirs(DATA_DIR, exist_ok=True)

def load_fleurs(lang, split):
    path = f"{DATA_DIR}/{lang}_{split}.parquet"
    if not os.path.exists(path):
        url = f"https://huggingface.co/api/datasets/google/fleurs/parquet/{lang}/{split}/0.parquet"
        urllib.request.urlretrieve(url, path)
    return pd.read_parquet(path)

def decode_audio(a):
    wav, sr = sf.read(io.BytesIO(a["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    if sr != 16000:
        wav = resample_poly(wav, 16000, sr).astype(np.float32)
    return wav

tests = {}
for lang in LANGS:
    df = load_fleurs(lang, "test")
    tests[lang] = df
    print(f"{LANGS[lang][0]:9s} test: {len(df):4d} clips, {df['id'].nunique():3d} different sentences, "
          f"{df['num_samples'].sum() / 16000 / 3600:.1f} hours, gender counts {dict(Counter(df['gender']))}")

print("\nColumns:", list(tests["hy_am"].columns))
print("Gender codes in FLEURS: 0 = male, 1 = female (2 = other)")

# %% [markdown]
# ### Look at the data first
# A few Armenian examples. `raw_transcription` is what the speaker read, with punctuation and capitals.

# %%
for _, r in tests["hy_am"].sample(4, random_state=1).iterrows():
    print(f"[{r['num_samples']/16000:4.1f}s] {r['raw_transcription']}")

# %% [markdown]
# ### Check for leakage between FLEURS splits
# FLEURS reads each sentence several times by different people. If a test sentence also appeared in the
# training split, a model fine-tuned on training data could just memorise it. We check the sentence IDs.

# %%
hy_train = load_fleurs("hy_am", "train")
hy_dev = load_fleurs("hy_am", "validation")
s_tr, s_dev, s_te = set(hy_train["id"]), set(hy_dev["id"]), set(tests["hy_am"]["id"])
print(f"Armenian: train {len(hy_train)} clips / {len(s_tr)} sentences, validation {len(hy_dev)} / {len(s_dev)}, test {len(tests['hy_am'])} / {len(s_te)}")
print("Sentences shared train∩test:", len(s_tr & s_te), "| train∩validation:", len(s_tr & s_dev), "| validation∩test:", len(s_dev & s_te))
norm_text = lambda t: re.sub(r"\s+", " ", t.strip().lower())
print("Identical transcripts shared train∩test:",
      len(set(map(norm_text, hy_train["transcription"])) & set(map(norm_text, tests["hy_am"]["transcription"]))))

# %% [markdown]
# ## 2. Pick the test clips
# To keep the comparison fast and fair we take the same number of random clips (`N_PER_LANG`) from each
# language's test set, with a fixed random seed.

# %%
subsets = {}
for lang, df in tests.items():
    sub = df.sample(min(N_PER_LANG, len(df)), random_state=SEED).reset_index(drop=True)
    sub["wav"] = [decode_audio(a) for a in sub["audio"]]
    subsets[lang] = sub.drop(columns=["audio"])
    print(f"{LANGS[lang][0]:9s}: {len(sub)} clips, {sum(len(w) for w in sub['wav'])/16000/60:.1f} minutes of audio")

# %% [markdown]
# ## 3. Our own question: what does a phone line do to the audio?
#
# Phone calls carry much less of the sound than a good microphone. Classic phone audio is sampled at
# 8,000 times a second (instead of 16,000), only keeps frequencies between about 300 and 3,400 Hz,
# is squeezed into 8-bit "μ-law" codes, and picks up line noise. We **simulate** that so we can test the
# models on "phone" versions of exactly the same clips. (We don't use any real phone calls: those would be
# other people's voices.)

# %%
SOS_PHONE = butter(4, [300, 3400], btype="band", fs=8000, output="sos")

def phone_line(wav, snr_db=20, seed=0):
    """Simulate a narrow-band telephone channel: 8 kHz, 300-3400 Hz, 8-bit mu-law, background noise."""
    rng = np.random.default_rng(seed)
    y = resample_poly(wav, 1, 2)                      # 16 kHz -> 8 kHz
    y = sosfilt(SOS_PHONE, y)                         # telephone band only
    y = 0.9 * y / (np.max(np.abs(y)) + 1e-9)
    mu = 255.0                                        # 8-bit mu-law companding (G.711)
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    c = np.round((c + 1) / 2 * mu) / mu * 2 - 1
    y = np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu
    noise_power = np.mean(y ** 2) / (10 ** (snr_db / 10))
    y = y + rng.normal(0, np.sqrt(noise_power), y.shape)
    return resample_poly(y, 2, 1).astype(np.float32)  # back to 16 kHz for the model

for lang, sub in subsets.items():
    sub["wav_phone"] = [phone_line(w, seed=i) for i, w in enumerate(sub["wav"])]

# Look at what the phone line removes (spectrogram of one Armenian clip)
w = subsets["hy_am"]["wav"][0]
fig, ax = plt.subplots(1, 2, figsize=(12, 3), sharey=True)
for a, (title, x) in zip(ax, [("Original (16 kHz microphone)", w), ("Simulated phone line", subsets["hy_am"]["wav_phone"][0])]):
    a.specgram(x, Fs=16000, NFFT=512, noverlap=384, cmap="magma")
    a.set_title(title); a.set_xlabel("seconds")
ax[0].set_ylabel("frequency (Hz)")
plt.tight_layout(); plt.savefig(f"{OUT}/phone_spectrogram.png", dpi=120); plt.show()

# %% [markdown]
# ## 4. Scoring helpers
#
# Before scoring we lower-case both the model output and the human transcript and remove punctuation
# (Whisper's standard `BasicTextNormalizer`, which also handles Armenian punctuation such as `։` and `՝`).
# We also measure two failure signals we noticed while reading outputs:
# * **Armenian-script share**: what fraction of the letters the model wrote are Armenian letters
#   (a model that answers in Russian or in Latin letters gets a low share);
# * **Loop / runaway**: the output is more than twice as long as the transcript or one phrase repeats 4+ times,
#   a known Whisper "hallucination" failure.

# %%
normalizer = BasicTextNormalizer()

def norm(t):
    return re.sub(r"\s+", " ", normalizer(t or "")).strip()

def armenian_share(t):
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return np.nan
    return sum("Ա" <= c <= "֏" for c in letters) / len(letters)

def is_loop(hyp, ref):
    words = hyp.split()
    if len(hyp) > 2 * max(len(ref), 10):
        return True
    grams = Counter(tuple(words[i:i + 3]) for i in range(max(len(words) - 2, 0)))
    return bool(grams) and max(grams.values()) >= 4

def score(df):
    refs, hyps = df["ref_norm"].tolist(), df["hyp_norm"].tolist()
    return pd.Series({
        "clips": len(df),
        "WER %": 100 * jiwer.wer(refs, hyps),
        "CER %": 100 * jiwer.cer(refs, hyps),
        "Armenian-script share %": 100 * df["arm_share"].mean(),
        "loops %": 100 * df["loop"].mean(),
    })

def utt_metrics(ref, hyp):
    if not ref:
        return np.nan, np.nan
    return 100 * jiwer.wer(ref, hyp), 100 * jiwer.cer(ref, hyp)

# %% [markdown]
# ## 5. The simplest baseline: say nothing
# The brief asks for a simple baseline. For speech-to-text the simplest "model" outputs nothing at all.
# It deletes every word, so its WER is exactly 100%. Any model scoring *above* 100% is doing worse than
# staying silent, because it's adding wrong words.

# %%
rows = []
for lang, sub in subsets.items():
    rows.append({"language": LANGS[lang][0], "model": "Baseline: empty output", "WER %": 100.0, "CER %": 100.0})
print(pd.DataFrame(rows).to_string(index=False))

# %% [markdown]
# ## 6. Run the models
#
# * `openai/whisper-small` (244M parameters): small enough to run cheaply on a normal server, the size a
#   small company would actually deploy for a phone line.
# * `openai/whisper-large-v3-turbo` (809M): OpenAI's newest fast large model.
# * `facebook/mms-1b-all` (1B, Armenian adapter): Meta's model built specifically for 1,000+ languages.
#   It isn't Whisper, so it's a fair "specialised" comparison. Note it was trained partly on FLEURS's
#   *training* split (different sentences from the test split), which may help it here.
#
# For Whisper we try two settings:
# * **auto**: Whisper guesses the language itself (what happens if the developer doesn't set it). It turns out to
  recognise Armenian almost always (section 8), so the two settings give nearly the same results;
# * **forced**: we tell Whisper the language in advance.

# %%
def whisper_run(model, processor, wavs, language=None, bs=16):
    hyps, detected = [], []
    for i in range(0, len(wavs), bs):
        feats = processor.feature_extractor(list(wavs[i:i + bs]), sampling_rate=16000,
                                            return_tensors="pt").input_features.to(DEVICE, DTYPE)
        kwargs = dict(task="transcribe", max_new_tokens=440, num_beams=1)
        if language:
            kwargs["language"] = language
        with torch.no_grad():
            ids = model.generate(feats, **kwargs)
        hyps += processor.batch_decode(ids, skip_special_tokens=True)
        if language is None:   # ask Whisper's own language detector what it heard
            with torch.no_grad():
                lang_ids = model.detect_language(feats)
            detected += [t.strip("<|>") for t in processor.tokenizer.convert_ids_to_tokens(lang_ids.view(-1).tolist())]
        else:
            detected += [language] * len(ids)
    return hyps, detected

def mms_run(model, processor, wavs, bs=8):
    hyps = []
    for i in range(0, len(wavs), bs):
        inp = processor(list(wavs[i:i + bs]), sampling_rate=16000, return_tensors="pt", padding=True)
        mask = inp.get("attention_mask")
        with torch.no_grad():
            logits = model(inp.input_values.to(DEVICE, DTYPE),
                           attention_mask=None if mask is None else mask.to(DEVICE)).logits
        hyps += processor.batch_decode(logits.argmax(-1))
    return hyps

all_rows = []

def record(lang, model_name, setting, channel, hyps, detected=None):
    sub = subsets[lang]
    for j, (ref, hyp) in enumerate(zip(sub["raw_transcription"], hyps)):
        r, h = norm(ref), norm(hyp)
        w, c = utt_metrics(r, h)
        all_rows.append(dict(language=LANGS[lang][0], lang=lang, model=model_name, setting=setting, channel=channel,
                             id=sub["id"][j], gender=sub["gender"][j], seconds=len(sub["wav"][j]) / 16000,
                             ref=ref, hyp=hyp, ref_norm=r, hyp_norm=h, utt_wer=w, utt_cer=c,
                             detected=(detected[j] if detected else None),
                             arm_share=armenian_share(h) if lang == "hy_am" else np.nan,
                             loop=is_loop(h, r)))

WHISPERS = ["openai/whisper-small", "openai/whisper-large-v3-turbo"]
timings = {}
for name in WHISPERS:
    t0 = time.time()
    processor = WhisperProcessor.from_pretrained(name)
    model = WhisperForConditionalGeneration.from_pretrained(name, torch_dtype=DTYPE).to(DEVICE).eval()
    short = name.split("/")[1]
    for lang, sub in subsets.items():
        code = LANGS[lang][1]
        hyps, det = whisper_run(model, processor, sub["wav"].tolist(), language=None)
        record(lang, short, "auto", "clean", hyps, det)
        hyps, det = whisper_run(model, processor, sub["wav"].tolist(), language=code)
        record(lang, short, "forced", "clean", hyps, det)
        hyps, det = whisper_run(model, processor, sub["wav_phone"].tolist(), language=code)
        record(lang, short, "forced", "phone", hyps, det)
        print(f"{short:24s} {LANGS[lang][0]:9s} done ({time.time()-t0:.0f}s)")
    timings[short] = time.time() - t0
    del model; torch.cuda.empty_cache()

t0 = time.time()
mms_proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="hye")
mms = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="hye", ignore_mismatched_sizes=True,
                                     torch_dtype=DTYPE).to(DEVICE).eval()
for channel in ["clean", "phone"]:
    col = "wav" if channel == "clean" else "wav_phone"
    record("hy_am", "mms-1b-all", "forced", channel, mms_run(mms, mms_proc, subsets["hy_am"][col].tolist()))
timings["mms-1b-all"] = time.time() - t0
del mms; torch.cuda.empty_cache()
print("timings (s):", {k: round(v) for k, v in timings.items()})

res = pd.DataFrame(all_rows)
res.to_csv(f"{OUT}/zero_shot_per_clip.csv", index=False)

# %% [markdown]
# ## 7. Results: the same model, very different quality per language

# %%
summary = (res.groupby(["language", "model", "setting", "channel"], sort=False)
              .apply(score, include_groups=False).reset_index())
summary.to_csv(f"{OUT}/zero_shot_summary.csv", index=False)
pd.set_option("display.width", 200)
print(summary.round(1).to_string(index=False))

# %%
clean_forced = summary[(summary.setting == "forced") & (summary.channel == "clean")]
pivot = clean_forced.pivot(index="model", columns="language", values="WER %")[["English", "Russian", "Armenian"]]
ax = pivot.plot(kind="bar", figsize=(9, 4), rot=0, color=["#4C72B0", "#55A868", "#C44E52"])
ax.axhline(100, color="gray", ls="--", lw=1); ax.text(-0.45, 102, "empty-output baseline (100%)", color="gray", fontsize=8)
ax.set_ylabel("Word error rate % (lower is better)")
ax.set_title("Same models, same sentences: word error rate by language (FLEURS test, language given)")
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/wer_by_language.png", dpi=120); plt.show()

# %% [markdown]
# ## 8. Does Whisper even recognise that it's hearing Armenian?
# When we let Whisper guess the language, which language does it think Armenian speech is? If it recognises
# Armenian, the problem is not language detection but *writing* Armenian correctly.

# %%
auto_hy = res[(res.lang == "hy_am") & (res.setting == "auto")]
for m, g in auto_hy.groupby("model", sort=False):
    c = Counter(g["detected"])
    print(f"{m}: guessed language for {len(g)} Armenian clips ->", ", ".join(f"{k}: {v}" for k, v in c.most_common(6)))
    print(f"   correctly detected as Armenian: {100*c.get('hy',0)/len(g):.0f}%")
for lang in ["ru_ru", "en_us"]:
    g = res[(res.lang == lang) & (res.setting == "auto") & (res.model == "whisper-small")]
    print(f"(for comparison) whisper-small detects {LANGS[lang][0]} correctly in {100*(g['detected']==LANGS[lang][1]).mean():.0f}% of clips")

# %% [markdown]
# ## 9. Why does Armenian fail? Real examples, model output next to the correct transcript

# %%
def show(model, setting, channel, n=4, worst=True, lang="hy_am"):
    g = res[(res.lang == lang) & (res.model == model) & (res.setting == setting) & (res.channel == channel)]
    g = g.sort_values("utt_wer", ascending=not worst).head(n) if worst is not None else g.sample(n, random_state=0)
    for _, r in g.iterrows():
        print(f"WER {r.utt_wer:5.0f}% | CER {r.utt_cer:5.0f}% | detected={r.detected}")
        print("   correct:", r.ref)
        print("   model  :", r.hyp[:300])

print("=== whisper-small, auto language: typical Armenian outputs ===")
show("whisper-small", "auto", "clean", worst=None)
print("\n=== whisper-small, Armenian forced: worst outputs ===")
show("whisper-small", "forced", "clean", n=3)
print("\n=== whisper-large-v3-turbo, Armenian forced: typical outputs ===")
show("whisper-large-v3-turbo", "forced", "clean", worst=None, n=3)

# %% [markdown]
# ## 10. Does it work equally well for men and women?
# FLEURS records the speaker's gender, so we can check whether the error rate differs.

# %%
hy = res[(res.lang == "hy_am") & (res.setting == "forced")].copy()
hy["gender"] = hy["gender"].map({0: "male", 1: "female"}).fillna("other")
by_gender = hy.groupby(["model", "channel", "gender"]).apply(score, include_groups=False)[["clips", "WER %", "CER %"]]
print(by_gender.round(1).to_string())

# %% [markdown]
# ## 11. Save everything for the write-up

# %%
json.dump({"n_per_lang": N_PER_LANG, "timings_s": timings, "torch": torch.__version__,
           "transformers": transformers.__version__}, open(f"{OUT}/run_info.json", "w"), indent=1)
print(os.listdir(OUT))
