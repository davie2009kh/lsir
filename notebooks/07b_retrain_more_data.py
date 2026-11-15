# %% [markdown]
# # Lsir (Լսիր): notebook 7b, retraining with more varied data (v3)
#
# **Track:** AI + Languages · Veritas AI International High School Hackathon 2026
#
# Notebook 4's phone-robust LoRA (v2) gave a small but significant gain on an unseen mobile codec. Its limit is the
# data: BuzzASR had already learned every FLEURS sentence. Here we **continue training the v2 adapter for one more
# epoch** on a bigger and more varied mix:
#
# * the FLEURS Armenian training clips (as before);
# * **3,000 Common Voice Armenian training clips**: volunteers recording on their own phones and laptops;
# * **synthetic conversational Armenian** from notebook 7a (two open TTS voices, quality-filtered).
#
# Everything else matches notebook 4: the same randomised telephone augmentation (codecs, packet loss, babble,
# echo, speed and volume changes, SpecAugment), the same 300 validation clips to pick the epoch (epoch 0 = v2, so
# v3 can't be worse than v2 on validation), and the same three test conditions on all 932 FLEURS test clips.

# %%
SMOKE = False
LORA_PART = "encoder"     # "encoder" or "all"
EPOCHS = 1                # one pass over the bigger mix, starting from the v2 adapter
CV_TRAIN = 3000           # Common Voice training clips to add
BATCH = 4                 # clips per step on the GPU
ACCUM = 4                 # gradient accumulation -> 16 clips per update
LR = 3e-5                 # gentler still: we continue from an already-trained adapter
LORA_R = 16
CLEAN_PROB = 0.5
N_VAL = 300               # more validation clips for a reliable choice of epoch
LABEL_COL = "transcription"   # FLEURS normalised text (lower case, no punctuation) = the style BuzzASR writes
SEED = 42
BASE = "BuzzASR/armenian"
GEN = dict(num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2, max_new_tokens=440)   # BuzzASR card settings

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer", "peft"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)   # Kaggle's old torchao breaks peft; we don't use it

import io, os, re, time, json, random, shutil, urllib.request
from collections import Counter
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import jiwer
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, resample_poly, fftconvolve
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration, get_linear_schedule_with_warmup
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from peft import LoraConfig, get_peft_model
import transformers, peft
transformers.logging.set_verbosity_error()   # hide repetitive library warnings

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)
DEVICE = "cuda"
OUT = "/kaggle/working/results"; os.makedirs(OUT, exist_ok=True)
ADAPTER_OUT = "/kaggle/working/lsir-v3-adapter"
if SMOKE:
    EPOCHS, N_VAL = 1, 8
print("torch", torch.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__, "|", torch.cuda.get_device_name(0))

# %% [markdown]
# ## 1. Data

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

splits = {}
for split in ["train", "validation", "test"]:
    df = load_fleurs("hy_am", split)
    df["wav"] = [decode_audio(a) for a in df["audio"]]
    splits[split] = df.drop(columns=["audio"]).reset_index(drop=True)
    print(f"{split:10s}: {len(df)} clips, {df['id'].nunique()} sentences")
print("sentences shared train∩test:", len(set(splits["train"].id) & set(splits["test"].id)))
from transformers.models.whisper.english_normalizer import BasicTextNormalizer as _BTN
_n = _BTN()
def style(t):   # BuzzASR's output style: lower case, no punctuation
    return re.sub(r"\s+", " ", _n(t or "")).strip()

fl = splits["train"].assign(label=splits["train"]["transcription"], source="FLEURS")[["wav", "label", "source"]]

def to_16k(b):
    r = subprocess.run([shutil.which("ffmpeg"), "-loglevel", "error", "-i", "pipe:0", "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"],
                       input=b, capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()

urllib.request.urlretrieve("https://huggingface.co/api/datasets/Chillarmo/common_voice_20_armenian/parquet/default/train/0.parquet", "/tmp/cv_train.parquet")
cv_all = pd.read_parquet("/tmp/cv_train.parquet")
urllib.request.urlretrieve("https://huggingface.co/api/datasets/Chillarmo/common_voice_20_armenian/parquet/default/test/0.parquet", "/tmp/cv_test.parquet")
cv_test_sents = set(pd.read_parquet("/tmp/cv_test.parquet", columns=["sentence"])["sentence"].map(style))
overlap = cv_all["sentence"].map(style).isin(cv_test_sents)
print(f"Common Voice train clips whose sentence also appears in Common Voice test: {int(overlap.sum())} of {len(cv_all)} "
      "(removed from our training so notebook 6's Common Voice test stays clean for our model)")
cv = cv_all[~overlap].sample(CV_TRAIN, random_state=SEED)
cv = pd.DataFrame({"wav": [to_16k(a["bytes"]) for a in cv["audio"]], "label": cv["sentence"].map(style), "source": "Common Voice"})

import glob as _glob
meta_path = _glob.glob("/kaggle/input/**/synth/synth_meta.csv", recursive=True)
print("synthetic metadata:", meta_path)
sm = pd.read_csv(meta_path[0]) if meta_path else pd.DataFrame(columns=["file", "voice", "text"])
sdir = os.path.dirname(meta_path[0]) if meta_path else ""
syn = pd.DataFrame({"wav": [sf.read(os.path.join(sdir, f), dtype="float32")[0] for f in sm["file"]],
                    "label": sm["text"].map(style), "source": "synthetic " + sm["voice"]})
train_df = pd.concat([fl, cv, syn], ignore_index=True)
train_df = train_df[(train_df["wav"].map(len) <= 27 * 16000) & (train_df["label"].str.len() > 0)].reset_index(drop=True)
print(train_df.groupby("source").agg(clips=("label", "size"), hours=("wav", lambda w: round(sum(map(len, w)) / 16000 / 3600, 2))))
val_df = splits["validation"].sample(N_VAL, random_state=SEED).reset_index(drop=True)
test_df = splits["test"].copy()
if SMOKE:
    train_df, test_df = train_df.sample(32, random_state=0).reset_index(drop=True), test_df.head(8).copy()

# %% [markdown]
# ## 2. The telephone-channel simulator
# `phone_line` is the fixed simulation used in notebooks 1-3 (kept identical so results are comparable).
# `random_channel` is the richer, randomised version used for **training**. `amr_unseen` is the **held-out test** condition.

# %%
FFMPEG = shutil.which("ffmpeg")
encoders = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout if FFMPEG else ""
HAS_OPUS = "libopus" in encoders
print("ffmpeg:", FFMPEG, "| opus encoder:", HAS_OPUS)

CODEC_FMT = {"libopus": "ogg", "libopencore_amrnb": "amr", "libgsm": "gsm", "g723_1": "nut"}

def codec_roundtrip(wav16, codec, bitrate):
    """Encode 16 kHz audio with a real speech codec at 8 kHz and decode it back (via ffmpeg pipes)."""
    x8 = resample_poly(wav16, 1, 2).astype(np.float32)
    fmt = CODEC_FMT[codec]
    enc = subprocess.run([FFMPEG, "-loglevel", "error", "-f", "f32le", "-ar", "8000", "-ac", "1", "-i", "pipe:0",
                          "-c:a", codec, "-b:a", bitrate, "-f", fmt, "pipe:1"], input=x8.tobytes(), capture_output=True)
    dec_in = ["-f", "gsm"] if fmt == "gsm" else []
    dec = subprocess.run([FFMPEG, "-loglevel", "error", *dec_in, "-i", "pipe:0", "-f", "f32le", "-ar", "8000", "-ac", "1", "pipe:1"],
                         input=enc.stdout, capture_output=True)
    y = np.frombuffer(dec.stdout, dtype=np.float32)
    if len(y) < len(x8) // 2:      # codec failed -> fall back to plain narrow band
        y = x8
    y = np.pad(y, (0, max(0, len(x8) - len(y))))[:len(x8)]
    return resample_poly(y, 2, 1).astype(np.float32)

# Held-out codec for the unseen test: the first one this ffmpeg can encode. It is never used in training.
UNSEEN_CODEC = next(((c, b, n) for c, b, n in [("libopencore_amrnb", "7.95k", "AMR-NB mobile codec"),
                                               ("libgsm", "13k", "GSM mobile codec"),
                                               ("g723_1", "6.3k", "G.723.1 VoIP codec")] if c in encoders), None)
print("unseen test codec:", UNSEEN_CODEC)

SOS_PHONE = butter(4, [300, 3400], btype="band", fs=8000, output="sos")

def mu_law(y, mu=255.0):
    c = np.sign(y) * np.log1p(mu * np.abs(y)) / np.log1p(mu)
    c = np.round((c + 1) / 2 * mu) / mu * 2 - 1
    return np.sign(c) * ((1 + mu) ** np.abs(c) - 1) / mu

def a_law(y, A=87.6):
    ay = np.abs(y)
    c = np.where(ay < 1 / A, A * ay / (1 + np.log(A)), (1 + np.log(np.maximum(A * ay, 1e-9))) / (1 + np.log(A))) * np.sign(y)
    c = np.round((c + 1) / 2 * 255) / 255 * 2 - 1
    ac = np.abs(c)
    return np.sign(c) * np.where(ac < 1 / (1 + np.log(A)), ac * (1 + np.log(A)) / A, np.exp(ac * (1 + np.log(A)) - 1) / A)

def add_noise(y, snr_db, rng, kind="white"):
    n = rng.normal(0, 1, len(y))
    if kind == "pink":                                   # 1/f noise via spectral shaping
        f = np.fft.rfft(n); f /= np.sqrt(np.arange(1, len(f) + 1)); n = np.fft.irfft(f, len(y))
    n *= np.sqrt(np.mean(y ** 2) / (np.mean(n ** 2) + 1e-12) / 10 ** (snr_db / 10))
    return y + n

def phone_line(wav, snr_db=20, seed=0):
    """Fixed simulation from notebooks 1-3: 8 kHz, 300-3400 Hz, 8-bit mu-law, white noise."""
    rng = np.random.default_rng(seed)
    y = sosfilt(SOS_PHONE, resample_poly(wav, 1, 2))
    y = mu_law(0.9 * y / (np.max(np.abs(y)) + 1e-9))
    y = add_noise(y, snr_db, rng)
    return resample_poly(y, 2, 1).astype(np.float32)

def packet_loss(y16, rate, rng, frame=320):
    y = y16.copy()
    for s in range(0, len(y), frame):
        if rng.random() < rate:
            y[s:s + frame] = 0.0
    return y

def reverb(y16, rng):
    rt60 = rng.uniform(0.2, 0.7)
    t = np.arange(int(rt60 * 16000)) / 16000
    rir = rng.normal(0, 1, len(t)) * np.exp(-6.9 * t / rt60); rir[0] = 1.0
    out = fftconvolve(y16, rir)[:len(y16)]
    return out / (np.max(np.abs(out)) + 1e-9) * np.max(np.abs(y16))

def speed(y16, rng):
    up, down = (10, 9) if rng.random() < 0.5 else (10, 11)    # 0.9x or 1.1x speed (pitch changes too)
    return resample_poly(y16, up, down).astype(np.float32)

def random_channel(wav, rng, babble_pool):
    """Randomised training augmentation: room -> speed -> background -> phone network -> packet loss."""
    y = wav.copy()
    if rng.random() < 0.3:
        y = reverb(y, rng)
    if rng.random() < 0.3:
        y = speed(y, rng)
    if rng.random() < 0.4:                                   # babble: 2-4 other speakers in the background
        b = np.zeros_like(y)
        for _ in range(rng.integers(2, 5)):
            o = babble_pool[rng.integers(len(babble_pool))]
            o = np.resize(o, len(y)); b += o / (np.std(o) + 1e-9)
        b *= np.std(y) / (np.std(b) + 1e-9) / 10 ** (rng.uniform(5, 20) / 20)
        y = y + b
    lo, hi = rng.uniform(200, 400), rng.uniform(3000, 3700)    # phone network: band limit + one codec
    y8 = sosfilt(butter(4, [lo, hi], btype="band", fs=8000, output="sos"), resample_poly(y, 1, 2))
    y8 = 0.9 * y8 / (np.max(np.abs(y8)) + 1e-9)
    codec = rng.choice(["mulaw", "alaw", "opus"] if HAS_OPUS else ["mulaw", "alaw"])
    if codec == "mulaw":
        y8 = mu_law(y8)
    elif codec == "alaw":
        y8 = a_law(y8)
    y = resample_poly(y8, 2, 1).astype(np.float32)
    if codec == "opus":
        y = codec_roundtrip(y, "libopus", str(rng.choice(["6k", "8k", "12k"])))
    if rng.random() < 0.6:
        y = add_noise(y, rng.uniform(10, 35), rng, kind=str(rng.choice(["white", "pink"])))
    if rng.random() < 0.3:
        y = packet_loss(y, rng.uniform(0.01, 0.08), rng)
    y = y * 10 ** (rng.uniform(-6, 6) / 20)
    return np.clip(y, -1, 1).astype(np.float32)

def amr_unseen(wav, seed):
    """Held-out test condition (never used in training): a mobile/VoIP codec + 5% packet loss + 15 dB pink noise."""
    rng = np.random.default_rng(seed)
    if UNSEEN_CODEC:
        y = codec_roundtrip(wav, UNSEEN_CODEC[0], UNSEEN_CODEC[1])
    else:   # fallback if this ffmpeg has none of them: coarse quantisation at 8 kHz
        y8 = sosfilt(SOS_PHONE, resample_poly(wav, 1, 2)); y8 = np.round(y8 / (np.max(np.abs(y8)) + 1e-9) * 15) / 15
        y = resample_poly(y8, 2, 1).astype(np.float32)
    y = add_noise(y, 15, rng, kind="pink")
    return packet_loss(y, 0.05, rng).astype(np.float32)

UNSEEN_NAME = f"unseen: {UNSEEN_CODEC[2] if UNSEEN_CODEC else 'coarse 8 kHz'} + loss + noise"
test_df["wav_phone"] = [phone_line(w, seed=i) for i, w in enumerate(test_df["wav"])]       # same as notebooks 1-3
test_df["wav_unseen"] = [amr_unseen(w, seed=50_000 + i) for i, w in enumerate(test_df["wav"])]
val_df["wav_phone"] = [phone_line(w, seed=10_000 + i) for i, w in enumerate(val_df["wav"])]
val_df["wav_unseen"] = [amr_unseen(w, seed=60_000 + i) for i, w in enumerate(val_df["wav"])]
babble_pool = train_df["wav"].tolist()

demo_rng = np.random.default_rng(0)
w = test_df["wav"].iloc[0]
fig, ax = plt.subplots(1, 4, figsize=(16, 3), sharey=True)
for a, (t, x) in zip(ax, [("clean", w), ("fixed phone (notebooks 1-3)", test_df["wav_phone"].iloc[0]),
                          ("one random training channel", random_channel(w, demo_rng, babble_pool)), (UNSEEN_NAME, test_df["wav_unseen"].iloc[0])]):
    a.specgram(x, Fs=16000, NFFT=512, noverlap=384, cmap="magma"); a.set_title(t, fontsize=9)
plt.tight_layout(); plt.savefig(f"{OUT}/channels_spectrogram.png", dpi=110); plt.show()

# %% [markdown]
# ## 3. Model, labels and LoRA
# BuzzASR replaced Whisper's tokenizer with an Armenian-specific one. With the newer transformers library on Kaggle,
# the tokenizer's *special* token numbers no longer match the model's (we checked: the model's "start" token id
# decodes as `<|zh|>`), so we build each training label from the model's own special-token ids
# (start, Armenian, transcribe, no-timestamps, text, end) and compare with the model's own output.

# %%
proc = WhisperProcessor.from_pretrained(BASE)
tok = proc.tokenizer
model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float16).to(DEVICE)
# Newer transformers versions drop BuzzASR's language table when loading, so read it from the file and put it back.
from huggingface_hub import hf_hub_download
GCJ = json.load(open(hf_hub_download(BASE, "generation_config.json")))
gc = model.generation_config
for k in ["lang_to_id", "task_to_id", "language", "task", "no_timestamps_token_id"]:
    setattr(gc, k, GCJ[k])
gc.is_multilingual = True
PREFIX = [GCJ["decoder_start_token_id"], GCJ["lang_to_id"]["<|hy|>"], GCJ["task_to_id"]["transcribe"], GCJ["no_timestamps_token_id"]]
EOS = GCJ["eos_token_id"]
# The model's special tokens (end, start, languages, ...) start right after its text vocabulary, at id = EOS.
# Newer transformers number the tokenizer's *special* tokens differently from the model, so we only use the
# tokenizer for ordinary text, take every special id from the model's own config, and decode text ids only.
TEXT_VOCAB = EOS
print("model special ids -> start", PREFIX[0], "| <|hy|>", PREFIX[1], "| transcribe", PREFIX[2], "| no-timestamps", PREFIX[3], "| end", EOS)

def label_ids(text):
    ids = tok(" " + text.strip(), add_special_tokens=False).input_ids
    assert max(ids) < TEXT_VOCAB, "text token outside the model's text vocabulary"
    return PREFIX + ids + [EOS]

def decode(seqs):
    return [tok.decode([t for t in s.tolist() if t < TEXT_VOCAB]).strip() for s in seqs]

with torch.no_grad():
    f = proc.feature_extractor(train_df["wav"][0], sampling_rate=16000, return_tensors="pt").input_features.to(DEVICE, torch.float16)
    g = model.generate(input_features=f, **GEN)
print("model's own output ids start:", g[0][:6].tolist(), "| our label starts:", label_ids("x")[:4])
print("correct :", train_df["label"][0][:120])
print("model   :", decode(g)[0][:120])
print("label   :", decode(torch.tensor([label_ids(train_df["label"][0])]))[0][:120])
print("Average tokens per Armenian word with BuzzASR's tokenizer:",
      round(np.mean([len(tok(" " + t, add_special_tokens=False).input_ids) / max(len(t.split()), 1) for t in train_df["label"][:300]]), 1),
      "(original Whisper tokenizer: 8.8, notebook 2)")

if LORA_PART == "encoder":
    targets = r".*encoder\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)"
else:
    targets = r".*(encoder|decoder)\.layers\.\d+\.(self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|out_proj)|.*encoder\.layers\.\d+\.(fc1|fc2)"
lora_cfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05, target_modules=targets, bias="none")
model.config.apply_spec_augment = True            # SpecAugment during training (Whisper's built-in masking)
model.config.mask_time_prob, model.config.mask_feature_prob, model.config.mask_feature_length = 0.05, 0.05, 10
v2 = _glob.glob("/kaggle/input/**/lsir-buzzasr-armenian-phone-lora-encoder/adapter_config.json", recursive=True)
print("starting from v2 adapter:", v2)
from peft import PeftModel
model = PeftModel.from_pretrained(model, os.path.dirname(v2[0]), is_trainable=True)   # continue training v2
for n, p in model.named_parameters():          # LoRA weights in fp32 for stable training; frozen base stays fp16
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()
inner = model.base_model.model
inner.config.use_cache = False
inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# %% [markdown]
# ## 4. Training

# %%
class ArmenianPhoneSpeech(Dataset):
    def __init__(self, df):
        self.df, self.epoch = df, 0
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        rng = np.random.default_rng(SEED * 100003 + self.epoch * 1009 + i)
        wav = row["wav"] if rng.random() < CLEAN_PROB else random_channel(row["wav"], rng, babble_pool)
        feats = proc.feature_extractor(wav, sampling_rate=16000).input_features[0]
        return feats, label_ids(row["label"])

def collate(batch):
    feats = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float16)
    L = max(len(b[1]) for b in batch)
    labels = torch.full((len(batch), L), -100, dtype=torch.long)
    for j, (_, ids) in enumerate(batch):
        labels[j, :len(ids)] = torch.tensor(ids)
    return feats, labels[:, 1:]      # the model adds <|startoftranscript|> itself

normalizer = BasicTextNormalizer()
def norm(t):
    return re.sub(r"\s+", " ", normalizer(t or "")).strip()

@torch.no_grad()
def transcribe(m, wavs, bs=8):
    m.eval()
    out = []
    for i in range(0, len(wavs), bs):
        feats = proc.feature_extractor(list(wavs[i:i + bs]), sampling_rate=16000, return_tensors="pt").input_features
        with torch.autocast("cuda", dtype=torch.float16):
            ids = m.generate(input_features=feats.to(DEVICE, torch.float16), **GEN)
        out += decode(ids)
    return out

def wer(refs, hyps):
    return 100 * jiwer.wer([norm(r) for r in refs], [norm(h) for h in hyps])

VAL_CH = {"clean": "wav", "phone": "wav_phone", "unseen": "wav_unseen"}
def validate(tag):
    inner.config.use_cache = True
    r = {ch: wer(val_df["raw_transcription"], transcribe(model, val_df[col].tolist())) for ch, col in VAL_CH.items()}
    inner.config.use_cache = False
    print(f"[{tag}] validation WER " + "  ".join(f"{k} {v:.1f}%" for k, v in r.items()), flush=True)
    return r

ds = ArmenianPhoneSpeech(train_df)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=4,
                    generator=torch.Generator().manual_seed(SEED))
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
total_updates = EPOCHS * (len(loader) // ACCUM)
sched = get_linear_schedule_with_warmup(opt, max(1, total_updates // 15), total_updates)
scaler = torch.amp.GradScaler("cuda")

# epoch 0 = the v2 adapter we start from, so "keep the best epoch" can never make it worse than v2.
# Model selection uses clean + fixed-phone validation only; the unseen condition is only watched, never optimised.
v = validate("epoch 0 = v2 adapter")
history = [dict(epoch=0, train_loss=np.nan, **{f"val_{k}": x for k, x in v.items()}, minutes=0)]
select = lambda r: (r["clean"] + r["phone"]) / 2
best = select(v)
best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items() if "lora_" in k}
t0 = time.time()
for epoch in range(1, EPOCHS + 1):
    ds.epoch = epoch
    model.train()
    losses = []
    for step, (feats, labels) in enumerate(loader):
        with torch.autocast("cuda", dtype=torch.float16):
            loss = model(input_features=feats.to(DEVICE), labels=labels.to(DEVICE)).loss / ACCUM
        scaler.scale(loss).backward()
        losses.append(loss.item() * ACCUM)
        if (step + 1) % ACCUM == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
        if step % 100 == 0:
            print(f"  epoch {epoch} step {step}/{len(loader)} loss {np.mean(losses[-100:]):.3f} ({(time.time()-t0)/60:.0f} min)", flush=True)
    v = validate(f"epoch {epoch}")
    history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), **{f"val_{k}": x for k, x in v.items()},
                        minutes=(time.time() - t0) / 60))
    if select(v) < best:
        best = select(v)
        best_state = {k: t.detach().cpu().clone() for k, t in model.state_dict().items() if "lora_" in k}
model.load_state_dict(best_state, strict=False)
best_epoch = min(history, key=lambda h: (h["val_clean"] + h["val_phone"]) / 2)["epoch"]
print("best epoch on validation:", best_epoch)
if not SMOKE:
    model.save_pretrained(ADAPTER_OUT)
    proc.save_pretrained(ADAPTER_OUT)
hist = pd.DataFrame(history); hist.to_csv(f"{OUT}/lora_{LORA_PART}_training_history.csv", index=False)
print(hist.round(2).to_string(index=False))

# %% [markdown]
# ## 5. Final test on all 932 FLEURS Armenian test clips: original BuzzASR vs ours, three audio conditions

# %%
refs = test_df["raw_transcription"].tolist()
A, B = "BuzzASR-armenian (original)", "Lsir v3: + Common Voice + synthetic speech"
CH = {"clean": "wav", "phone (fixed sim)": "wav_phone", UNSEEN_NAME: "wav_unseen"}
outputs = {}
inner.config.use_cache = True
t0 = time.time()
for ch, col in CH.items():
    with model.disable_adapter():
        outputs[(A, ch)] = transcribe(model, test_df[col].tolist())
    outputs[(B, ch)] = transcribe(model, test_df[col].tolist())
    print(ch, "done", f"({(time.time()-t0)/60:.0f} min)", flush=True)
    pd.DataFrame({"id": test_df["id"], "gender": test_df["gender"], "ref": refs,
                  **{f"{m} | {c}": v for (m, c), v in outputs.items()}}).to_csv(f"{OUT}/lora_{LORA_PART}_test_transcripts.csv", index=False)

# %%
R = [norm(r) for r in refs]
nref = np.array([len(r.split()) for r in R]); cref = np.array([len(r) for r in R])
rng = np.random.default_rng(SEED)
boot = [rng.integers(0, len(R), len(R)) for _ in range(1000)]
NUM = re.compile(r"\d+")

def numbers_kept(ref, hyp):
    """Share of the numbers in the correct transcript that appear in the model output (dates, times, counts...)."""
    want = NUM.findall(ref)
    if not want:
        return np.nan
    got = Counter(NUM.findall(hyp))
    kept = 0
    for n in want:
        if got[n] > 0:
            kept += 1; got[n] -= 1
    return kept / len(want)

per_clip, rows = {}, []
for (m, ch), hyps in outputs.items():
    H = [norm(h) for h in hyps]
    we = np.array([jiwer.process_words(r, h).wer * len(r.split()) if r else 0 for r, h in zip(R, H)])
    ce = np.array([jiwer.process_characters(r, h).cer * len(r) if r else 0 for r, h in zip(R, H)])
    nk = np.array([numbers_kept(r, h) for r, h in zip(refs, hyps)])
    per_clip[(m, ch)] = (we, ce)
    bw = [100 * we[b].sum() / nref[b].sum() for b in boot]
    rows.append({"model": m, "audio": ch, "WER %": 100 * we.sum() / nref.sum(),
                 "WER 95% CI": f"{np.percentile(bw, 2.5):.1f}-{np.percentile(bw, 97.5):.1f}",
                 "CER %": 100 * ce.sum() / cref.sum(), "numbers kept %": 100 * np.nanmean(nk),
                 "clips with numbers": int(np.isfinite(nk).sum())})
final = pd.DataFrame(rows); final.to_csv(f"{OUT}/lora_{LORA_PART}_final_results.csv", index=False)
pd.set_option("display.width", 220)
print(final.round(1).to_string(index=False)); print()
for ch in CH:
    ea, eb = per_clip[(A, ch)][0], per_clip[(B, ch)][0]
    d = np.array([100 * (ea[b].sum() - eb[b].sum()) / nref[b].sum() for b in boot])
    print(f"[{ch}] WER original -> ours: {100*ea.sum()/nref.sum():.1f}% -> {100*eb.sum()/nref.sum():.1f}% "
          f"(change {-100*(ea.sum()-eb.sum())/nref.sum():+.1f} points, 95% CI {-np.percentile(d,97.5):+.1f} to {-np.percentile(d,2.5):+.1f}); "
          f"ours better in {100*(d>0).mean():.0f}% of resamples")

# %%
pv = final.pivot(index="audio", columns="model", values="WER %").loc[list(CH)]
ax = pv[[A, B]].plot(kind="bar", figsize=(10, 4), rot=0, color=["#8C8C8C", "#C44E52"])
ax.set_ylabel("Word error rate % (lower is better)"); ax.set_xlabel("")
ax.set_title(f"BuzzASR-Armenian before and after phone-robust LoRA ({LORA_PART}), FLEURS test, 932 clips")
for c in ax.containers:
    ax.bar_label(c, fmt="%.1f", fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/lora_{LORA_PART}_wer.png", dpi=130); plt.show()

# %% [markdown]
# ## 6. Women vs men

# %%
gender = test_df["gender"].map({0: "male", 1: "female"}).fillna("other").values
g_rows = []
for (m, ch), (we, ce) in per_clip.items():
    for gname in ["female", "male"]:
        k = gender == gname
        g_rows.append({"model": m, "audio": ch, "gender": gname, "clips": int(k.sum()),
                       "WER %": 100 * we[k].sum() / nref[k].sum(), "CER %": 100 * ce[k].sum() / cref[k].sum()})
gdf = pd.DataFrame(g_rows); gdf.to_csv(f"{OUT}/lora_{LORA_PART}_by_gender.csv", index=False)
print(gdf.pivot_table(index=["audio", "gender"], columns="model", values="WER %").round(1).to_string())

# %% [markdown]
# ## 7. Examples: where our model helps most, and where it still fails (unseen mobile-codec condition)

# %%
uch = UNSEEN_NAME
gain = per_clip[(A, uch)][0] - per_clip[(B, uch)][0]
for label, idx in [("biggest improvement", np.argsort(-gain)[:3]),
                   ("still hardest for ours", np.argsort(-per_clip[(B, uch)][0] / np.maximum(nref, 1))[:3])]:
    for i in idx:
        print(f"--- {label} (clip {i}) ---")
        print("  correct  :", refs[i])
        print("  original :", outputs[(A, uch)][i][:250])
        print("  ours     :", outputs[(B, uch)][i][:250])

json.dump({"lora_part": LORA_PART, "epochs": EPOCHS, "best_epoch": int(best_epoch), "lr": LR, "lora_r": LORA_R,
           "clean_prob": CLEAN_PROB, "label_col": LABEL_COL, "n_val": N_VAL, "batch": BATCH * ACCUM, "train_clips": len(train_df), "test_clips": len(test_df),
           "has_opus": HAS_OPUS, "unseen_codec": UNSEEN_CODEC, "unseen_condition": UNSEEN_NAME,
           "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
          open(f"{OUT}/run_info_04_{LORA_PART}.json", "w"), indent=1)
print(os.listdir(OUT))
