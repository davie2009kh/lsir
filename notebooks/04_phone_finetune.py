# %% [markdown]
# # Lsir (Լսիր): making the best Armenian model work over the phone
# ## Notebook 4: phone-line fine-tuning of BuzzASR-Armenian with LoRA
#
# **Track:** AI + Languages · Veritas AI International High School Hackathon 2026
#
# Notebook 3 compared the strongest free Armenian speech-to-text models. All of them lose accuracy when the
# audio comes through a phone line, because none of them were trained on phone audio, and an Armenian-only caller
# to a clinic phone line is exactly who we're building for.
#
# **Our improvement:** take `BuzzASR/armenian` (Whisper-large-v3 already fine-tuned on Armenian) and teach it
# what Armenian sounds like over a phone line. We train on the same FLEURS Armenian **training** clips, but turn a
# random half of them into simulated phone audio (8 kHz, 300-3,400 Hz, 8-bit μ-law, random line noise).
#
# **Why LoRA?** The model has 1.55 billion parameters, too many to train fully on a free 16 GB GPU. LoRA freezes
# the original weights and trains small extra matrices inside the attention layers (about 1-2% of the size), which
# fits on a free T4 and is quick to train.
#
# **Fair test:** we pick the best epoch on FLEURS **validation** clips, and only then test once on all 932 FLEURS
# **test** clips (different sentences from training), on clean audio and on phone audio.
#
# *Runs on a free Kaggle T4 GPU. All data and models are downloaded inside the notebook.*

# %%
SMOKE = False
EPOCHS = 2
BATCH = 4                 # clips per step on the GPU
ACCUM = 4                 # gradient accumulation -> 16 clips per update
LR = 1e-4
LORA_R = 32
PHONE_PROB = 0.5
N_VAL = 120
SEED = 42
BASE = "BuzzASR/armenian"
GEN = dict(num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2, max_new_tokens=440)   # BuzzASR card settings

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer", "peft"], check=True)

import io, os, re, time, json, glob, random, urllib.request
from collections import Counter
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import jiwer
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, resample_poly
from torch.utils.data import Dataset, DataLoader
from transformers import WhisperProcessor, WhisperForConditionalGeneration, get_linear_schedule_with_warmup
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from peft import LoraConfig, get_peft_model, PeftModel
import transformers, peft
transformers.logging.set_verbosity_error()   # hide repetitive library warnings

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)
DEVICE = "cuda"
OUT = "/kaggle/working/results"; os.makedirs(OUT, exist_ok=True)
ADAPTER_OUT = "/kaggle/working/lsir-buzzasr-armenian-phone-lora"
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
train_df = splits["train"][splits["train"]["wav"].map(len) <= 30 * 16000].reset_index(drop=True)
val_df = splits["validation"].sample(N_VAL, random_state=SEED).reset_index(drop=True)
test_df = splits["test"]
if SMOKE:
    train_df, test_df = train_df.sample(32, random_state=0).reset_index(drop=True), test_df.head(8)

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

test_df["wav_phone"] = [phone_line(w, seed=i) for i, w in enumerate(test_df["wav"])]       # same as notebooks 2-3
val_df["wav_phone"] = [phone_line(w, seed=10_000 + i) for i, w in enumerate(val_df["wav"])]

# %% [markdown]
# ## 2. Model, tokenizer and LoRA

# %%
proc = WhisperProcessor.from_pretrained(BASE)
tok = proc.tokenizer
tok.set_prefix_tokens(language="hy", task="transcribe", predict_timestamps=False)
model = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float16).to(DEVICE)

# check our training labels start exactly like the model's own outputs: <|startoftranscript|><|hy|><|transcribe|><|notimestamps|>
ex = tok(train_df["raw_transcription"][0]).input_ids
print("label prefix:", tok.convert_ids_to_tokens(ex[:4]), "... end:", tok.convert_ids_to_tokens(ex[-1:]))
assert ex[0] == model.generation_config.decoder_start_token_id
assert ex[1] == model.generation_config.lang_to_id["<|hy|>"] and ex[-1] == model.generation_config.eos_token_id
print("Average tokens per Armenian word with BuzzASR's Armenian tokenizer:",
      round(np.mean([len(tok(t, add_special_tokens=False).input_ids) / max(len(t.split()), 1) for t in train_df["raw_transcription"][:300]]), 1),
      "(original Whisper: 8.8, see notebook 2)")

lora_cfg = LoraConfig(r=LORA_R, lora_alpha=2 * LORA_R, lora_dropout=0.05,
                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj"], bias="none")
model = get_peft_model(model, lora_cfg)
for n, p in model.named_parameters():          # LoRA weights in fp32 for stable training; frozen base stays fp16
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()
model.base_model.model.config.use_cache = False
model.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

# %% [markdown]
# ## 3. Training

# %%
class ArmenianSpeech(Dataset):
    def __init__(self, df, phone_prob):
        self.df, self.phone_prob, self.epoch = df, phone_prob, 0
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav = row["wav"]
        rng = random.Random(SEED * 100003 + self.epoch * 1009 + i)
        if rng.random() < self.phone_prob:
            wav = phone_line(wav, snr_db=rng.uniform(10, 30), seed=rng.randrange(1 << 30))
        feats = proc.feature_extractor(wav, sampling_rate=16000).input_features[0]
        return feats, tok(row["raw_transcription"]).input_ids

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
        out += proc.batch_decode(ids, skip_special_tokens=True)
    return out

def wer(refs, hyps):
    return 100 * jiwer.wer([norm(r) for r in refs], [norm(h) for h in hyps])

def validate(tag):
    model.base_model.model.config.use_cache = True
    vc = wer(val_df["raw_transcription"], transcribe(model, val_df["wav"].tolist()))
    vp = wer(val_df["raw_transcription"], transcribe(model, val_df["wav_phone"].tolist()))
    model.base_model.model.config.use_cache = False
    print(f"[{tag}] validation WER clean {vc:.1f}%  phone {vp:.1f}%")
    return vc, vp

ds = ArmenianSpeech(train_df, PHONE_PROB)
loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=4,
                    generator=torch.Generator().manual_seed(SEED))
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.01)
total_updates = EPOCHS * (len(loader) // ACCUM)
sched = get_linear_schedule_with_warmup(opt, max(1, total_updates // 20), total_updates)
scaler = torch.amp.GradScaler("cuda")

# epoch 0 = the original BuzzASR model (LoRA starts as a no-op), so "keep the best epoch" can never make it worse
vc, vp = validate("epoch 0 = original BuzzASR")
history = [dict(epoch=0, train_loss=np.nan, val_wer_clean=vc, val_wer_phone=vp, minutes=0)]
best = (vc + vp) / 2
best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
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
            print(f"  epoch {epoch} step {step}/{len(loader)} loss {np.mean(losses[-100:]):.3f} ({(time.time()-t0)/60:.0f} min)")
    vc, vp = validate(f"epoch {epoch}")
    history.append(dict(epoch=epoch, train_loss=float(np.mean(losses)), val_wer_clean=vc, val_wer_phone=vp,
                        minutes=(time.time() - t0) / 60))
    if (vc + vp) / 2 < best:
        best = (vc + vp) / 2
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
model.load_state_dict(best_state, strict=False)
best_epoch = min(history, key=lambda h: (h["val_wer_clean"] + h["val_wer_phone"]) / 2)["epoch"]
print("best epoch on validation:", best_epoch)
if not SMOKE:
    model.save_pretrained(ADAPTER_OUT)
    proc.save_pretrained(ADAPTER_OUT)
hist = pd.DataFrame(history); hist.to_csv(f"{OUT}/lora_training_history.csv", index=False)
print(hist.round(2).to_string(index=False))

# %% [markdown]
# ## 4. Final test on all 932 FLEURS Armenian test clips
# We compare the original BuzzASR model with our phone-tuned version on exactly the same clips.
# (If notebook 3's outputs are attached, we reuse its BuzzASR transcripts instead of recomputing them.)

# %%
refs = test_df["raw_transcription"].tolist()
outputs = {}
nb3 = glob.glob("/kaggle/input/*/results/strong_models_transcripts.csv") + glob.glob("/kaggle/input/*/*/results/strong_models_transcripts.csv")
reused = False
if nb3 and not SMOKE:
    prev = pd.read_csv(nb3[0])
    if len(prev) == len(test_df) and (prev["id"].values == test_df["id"].values).all() and "BuzzASR-armenian | clean" in prev:
        for ch in ["clean", "phone"]:
            outputs[("BuzzASR-armenian (original)", ch)] = prev[f"BuzzASR-armenian | {ch}"].fillna("").tolist()
        reused = True
        print("Reused BuzzASR transcripts from notebook 3:", nb3[0])
t0 = time.time()
model.base_model.model.config.use_cache = True
for ch, col in [("clean", "wav"), ("phone", "wav_phone")]:
    if not reused:
        with model.disable_adapter():
            outputs[("BuzzASR-armenian (original)", ch)] = transcribe(model, test_df[col].tolist())
    outputs[("Lsir: BuzzASR + phone LoRA (ours)", ch)] = transcribe(model, test_df[col].tolist())
    print(ch, "done", f"({(time.time()-t0)/60:.0f} min)")
pd.DataFrame({"id": test_df["id"], "gender": test_df["gender"], "ref": refs,
              **{f"{m} | {ch}": v for (m, ch), v in outputs.items()}}).to_csv(f"{OUT}/lora_test_transcripts.csv", index=False)

# %%
R = [norm(r) for r in refs]
nref = np.array([len(r.split()) for r in R]); cref = np.array([len(r) for r in R])
rng = np.random.default_rng(SEED)
boot = [rng.integers(0, len(R), len(R)) for _ in range(1000)]
per_clip, rows = {}, []
for (m, ch), hyps in outputs.items():
    H = [norm(h) for h in hyps]
    we = np.array([jiwer.process_words(r, h).wer * len(r.split()) if r else 0 for r, h in zip(R, H)])
    ce = np.array([jiwer.process_characters(r, h).cer * len(r) if r else 0 for r, h in zip(R, H)])
    per_clip[(m, ch)] = (we, ce)
    bw = [100 * we[b].sum() / nref[b].sum() for b in boot]
    rows.append({"model": m, "audio": ch, "WER %": 100 * we.sum() / nref.sum(),
                 "WER 95% CI": f"{np.percentile(bw, 2.5):.1f}-{np.percentile(bw, 97.5):.1f}",
                 "CER %": 100 * ce.sum() / cref.sum()})
final = pd.DataFrame(rows); final.to_csv(f"{OUT}/lora_final_results.csv", index=False)
print(final.round(1).to_string(index=False))

A, B = "BuzzASR-armenian (original)", "Lsir: BuzzASR + phone LoRA (ours)"
for ch in ["clean", "phone"]:
    ea, eb = per_clip[(A, ch)][0], per_clip[(B, ch)][0]
    d = np.array([100 * (ea[b].sum() - eb[b].sum()) / nref[b].sum() for b in boot])
    print(f"[{ch}] WER change original -> ours: {-100*(ea.sum()-eb.sum())/nref.sum():+.1f} points "
          f"(95% CI {-np.percentile(d,97.5):+.1f} to {-np.percentile(d,2.5):+.1f}); ours better in {100*(d>0).mean():.0f}% of resamples")

# %% [markdown]
# ## 5. Women vs men

# %%
gender = test_df["gender"].map({0: "male", 1: "female"}).fillna("other").values
g_rows = []
for (m, ch), (we, ce) in per_clip.items():
    for g in ["female", "male"]:
        k = gender == g
        g_rows.append({"model": m, "audio": ch, "gender": g, "clips": int(k.sum()),
                       "WER %": 100 * we[k].sum() / nref[k].sum(), "CER %": 100 * ce[k].sum() / cref[k].sum()})
gdf = pd.DataFrame(g_rows); gdf.to_csv(f"{OUT}/lora_by_gender.csv", index=False)
print(gdf.round(1).to_string(index=False))

# %% [markdown]
# ## 6. Examples where our phone-tuned model helps most, and where it still fails

# %%
gain = per_clip[(A, "phone")][0] - per_clip[(B, "phone")][0]
for label, idx in [("biggest improvement", np.argsort(-gain)[:3]), ("still worst for ours", np.argsort(-per_clip[(B, "phone")][0] / np.maximum(nref, 1))[:3])]:
    for i in idx:
        print(f"--- {label} (clip {i}) ---")
        print("  correct  :", refs[i])
        print("  original :", outputs[(A, "phone")][i][:250])
        print("  ours     :", outputs[(B, "phone")][i][:250])

json.dump({"epochs": EPOCHS, "best_epoch": int(best_epoch), "lr": LR, "lora_r": LORA_R, "phone_prob": PHONE_PROB,
           "batch": BATCH * ACCUM, "train_clips": len(train_df), "test_clips": len(test_df), "reused_nb3": reused,
           "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
          open(f"{OUT}/run_info_04.json", "w"), indent=1)
print(os.listdir(OUT))
