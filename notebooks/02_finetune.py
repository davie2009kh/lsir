# %% [markdown]
# # Lsir (Լսիր): teaching Whisper to understand Armenian, including over the phone
# ## Notebook 2 of 2: fine-tuning and the final evaluation
#
# **Track:** AI + Languages · Veritas AI International High School Hackathon 2026
#
# In notebook 1 we found that free speech-to-text models make many more mistakes on Armenian than on
# Russian or English, and even more over a phone line. Here we try to close that gap for the model size a
# small company could actually run on a phone line, `whisper-small` (244M parameters):
#
# 1. **Fine-tune** it on the Armenian training recordings in FLEURS (about 3,000 clips, roughly 10 hours).
# 2. **Fine-tune with phone-line augmentation**: the same training, but half of the clips are randomly turned
#    into simulated phone audio, so the model also learns what Armenian sounds like over a phone line.
# 3. Compare against the original `whisper-small`, the much bigger `whisper-large-v3-turbo`, and Meta's
#    Armenian-capable `mms-1b-all`, on **every** clip of the FLEURS Armenian **test** split, which training never sees.
#
# *Runs on a free Kaggle T4 GPU in about 2 hours. Data and models are downloaded inside the notebook.*

# %%
SMOKE = False
EPOCHS = 5
BATCH = 16
LR = 1e-5
PHONE_PROB = 0.5        # share of training clips turned into phone audio in the augmented run
N_VAL = 200             # validation clips used to pick the best epoch
SEED = 42

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer"], check=True)

import io, os, re, time, json, copy, random, urllib.request
from collections import Counter
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import jiwer
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, resample_poly
from torch.utils.data import Dataset, DataLoader
from transformers import (WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor, Wav2Vec2ForCTC,
                          get_linear_schedule_with_warmup)
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import transformers

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT = "/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results"
MODEL_OUT = "/kaggle/working/lsir-whisper-small-hy" if os.path.isdir("/kaggle/working") else "lsir-whisper-small-hy"
os.makedirs(OUT, exist_ok=True)
if SMOKE:
    EPOCHS, N_VAL = 1, 16
print("torch", torch.__version__, "| transformers", transformers.__version__, "|",
      torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu")

# %% [markdown]
# ## 1. Data: FLEURS Armenian (train / validation / test)
# FLEURS splits are made of **different sentences**, so the test sentences never appear in training
# (we check this below, as in notebook 1).

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
    print(f"{split:10s}: {len(df):5d} clips, {df['id'].nunique():4d} sentences, {df['num_samples'].sum()/16000/3600:.1f} h")
print("sentence overlap train∩test:", len(set(splits['train'].id) & set(splits['test'].id)),
      "| validation∩test:", len(set(splits['validation'].id) & set(splits['test'].id)))

if SMOKE:
    splits = {k: v.sample(32 if k == "train" else 16, random_state=0).reset_index(drop=True) for k, v in splits.items()}
train_df, val_df, test_df = splits["train"], splits["validation"].sample(min(N_VAL, len(splits["validation"])), random_state=SEED).reset_index(drop=True), splits["test"]

# %% [markdown]
# ## 2. The phone-line simulation (same as notebook 1)

# %%
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

# Test clips get a FIXED phone version (seed = clip index), identical for every model we compare.
test_df["wav_phone"] = [phone_line(w, seed=i) for i, w in enumerate(test_df["wav"])]

# %% [markdown]
# ## 3. Turning text into Whisper's tokens
# Whisper reads text as "tokens". Its tokenizer was built mostly from English, so Armenian words get
# chopped into many small pieces. That's one reason it struggles: the same sentence costs many more steps
# to write in Armenian. We measure how many tokens a sentence needs in each language.

# %%
processor = WhisperProcessor.from_pretrained("openai/whisper-small", language="armenian", task="transcribe")
tok = processor.tokenizer

def n_tokens(t):
    return len(tok(t, add_special_tokens=False).input_ids)

hy_tok = np.mean([n_tokens(t) / max(len(t.split()), 1) for t in train_df["raw_transcription"][:300]])
ru = load_fleurs("ru_ru", "test")["raw_transcription"][:300]; en = load_fleurs("en_us", "test")["raw_transcription"][:300]
print(f"Average tokens per word: Armenian {hy_tok:.1f} | Russian {np.mean([n_tokens(t)/max(len(t.split()),1) for t in ru]):.1f} | English {np.mean([n_tokens(t)/max(len(t.split()),1) for t in en]):.1f}")

train_df["n_tok"] = [n_tokens(t) for t in train_df["raw_transcription"]]
print("Longest Armenian training transcript:", train_df["n_tok"].max(), "tokens (Whisper's limit is 448)")
too_long = train_df["n_tok"] > 440
print("Training clips dropped because the transcript is longer than Whisper's limit:", int(too_long.sum()))
train_df = train_df[~too_long & (train_df["wav"].map(len) <= 30 * 16000)].reset_index(drop=True)

# %% [markdown]
# ## 4. Training setup
# * We train on the **written** transcript (`raw_transcription`, with capitals and Armenian punctuation),
#   so the model learns to output text a person can read.
# * Each training step the model hears 16 clips, predicts their transcripts, and adjusts its weights to make
#   the correct transcript more likely (cross-entropy loss), using the AdamW optimiser with a small learning rate.
# * After every pass over the data (an *epoch*) we measure WER on 200 **validation** clips and keep the best
#   epoch. The test split is only used once, at the very end.

# %%
class ArmenianSpeech(Dataset):
    def __init__(self, df, phone_prob=0.0, seed=0):
        self.df, self.phone_prob, self.seed = df, phone_prob, seed
        self.epoch = 0
    def __len__(self):
        return len(self.df)
    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav = row["wav"]
        rng = random.Random(self.seed * 100003 + self.epoch * 1009 + i)
        if self.phone_prob and rng.random() < self.phone_prob:
            wav = phone_line(wav, snr_db=rng.uniform(10, 30), seed=rng.randrange(1 << 30))
        feats = processor.feature_extractor(wav, sampling_rate=16000).input_features[0]
        labels = tok(row["raw_transcription"]).input_ids
        return feats, labels

def collate(batch):
    feats = torch.tensor(np.stack([b[0] for b in batch]), dtype=torch.float32)
    max_len = max(len(b[1]) for b in batch)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for j, (_, ids) in enumerate(batch):
        labels[j, :len(ids)] = torch.tensor(ids)
    # the model adds the start token itself, so remove it from the labels
    if (labels[:, 0] == tok.convert_tokens_to_ids("<|startoftranscript|>")).all():
        labels = labels[:, 1:]
    return feats, labels

normalizer = BasicTextNormalizer()
def norm(t):
    return re.sub(r"\s+", " ", normalizer(t or "")).strip()

@torch.no_grad()
def transcribe(model, wavs, bs=16, language="hy", proc=None):
    proc = proc or processor   # large-v3 models need their own processor (128 mel bins instead of 80)
    model.eval()
    out = []
    for i in range(0, len(wavs), bs):
        feats = proc.feature_extractor(list(wavs[i:i + bs]), sampling_rate=16000, return_tensors="pt").input_features
        feats = feats.to(DEVICE, next(model.parameters()).dtype)
        with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
            ids = model.generate(feats, language=language, task="transcribe", max_new_tokens=440, num_beams=1)
        out += proc.batch_decode(ids, skip_special_tokens=True)
    return out

def wer(refs, hyps):
    return 100 * jiwer.wer([norm(r) for r in refs], [norm(h) for h in hyps])

def train(run_name, phone_prob):
    seed_all(SEED)
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(DEVICE)
    model.generation_config.language, model.generation_config.task = "hy", "transcribe"
    model.generation_config.forced_decoder_ids = None
    model.config.forced_decoder_ids = None
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    ds = ArmenianSpeech(train_df, phone_prob=phone_prob, seed=SEED)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True, collate_fn=collate, num_workers=4,
                        generator=torch.Generator().manual_seed(SEED), persistent_workers=False)
    total = EPOCHS * len(loader)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=min(100, total // 10), num_training_steps=total)
    scaler = torch.amp.GradScaler("cuda", enabled=DEVICE == "cuda")

    val_wavs = val_df["wav"].tolist()
    val_phone = [phone_line(w, seed=10_000 + i) for i, w in enumerate(val_wavs)]
    history, best, best_state = [], float("inf"), None
    t0 = time.time()
    for epoch in range(EPOCHS):
        ds.epoch = epoch
        model.train(); model.config.use_cache = False
        losses = []
        for step, (feats, labels) in enumerate(loader):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.float16, enabled=DEVICE == "cuda"):
                loss = model(input_features=feats, labels=labels).loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sched.step()
            losses.append(loss.item())
        model.config.use_cache = True
        v_clean = wer(val_df["raw_transcription"], transcribe(model, val_wavs))
        v_phone = wer(val_df["raw_transcription"], transcribe(model, val_phone))
        # pick the epoch that is best on average across clean and phone audio
        score = (v_clean + v_phone) / 2
        history.append(dict(run=run_name, epoch=epoch + 1, train_loss=float(np.mean(losses)),
                            val_wer_clean=v_clean, val_wer_phone=v_phone, minutes=(time.time() - t0) / 60))
        print(f"[{run_name}] epoch {epoch+1}/{EPOCHS}  loss {np.mean(losses):.3f}  "
              f"val WER clean {v_clean:.1f}%  phone {v_phone:.1f}%  ({(time.time()-t0)/60:.0f} min)")
        if score < best:
            best, best_state = score, {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, history

# %% [markdown]
# ## 5. Fine-tune, run A: clean audio only

# %%
model_clean, hist_a = train("A: fine-tuned (clean audio)", phone_prob=0.0)
model_clean.half(); model_clean.to("cpu"); torch.cuda.empty_cache()
model_clean.save_pretrained(MODEL_OUT + "-clean", safe_serialization=True)   # save straight away

# %% [markdown]
# ## 6. Fine-tune, run B: with phone-line augmentation (our improvement for phone calls)

# %%
model_phone, hist_b = train("B: fine-tuned + phone augmentation", phone_prob=PHONE_PROB)
model_phone.half(); model_phone.to("cpu"); torch.cuda.empty_cache()
model_phone.save_pretrained(MODEL_OUT, safe_serialization=True)
processor.save_pretrained(MODEL_OUT)

hist = pd.DataFrame(hist_a + hist_b)
hist.to_csv(f"{OUT}/training_history.csv", index=False)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))
for run, g in hist.groupby("run", sort=False):
    ax[0].plot(g.epoch, g.val_wer_clean, marker="o", label=run)
    ax[1].plot(g.epoch, g.val_wer_phone, marker="o", label=run)
for a, t in zip(ax, ["Validation WER, clean audio", "Validation WER, phone audio"]):
    a.set_title(t); a.set_xlabel("epoch"); a.set_ylabel("WER %"); a.grid(alpha=.3)
ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/training_curves.png", dpi=120); plt.show()

# %% [markdown]
# ## 7. Final test: every model on all FLEURS Armenian test clips, clean and over the phone
# This is the first and only time the fine-tuned models see the test split.

# %%
refs = test_df["raw_transcription"].tolist()
clean_wavs, phone_wavs = test_df["wav"].tolist(), test_df["wav_phone"].tolist()
outputs = {}

def run_whisper(name, model, proc=None):
    model.to(DEVICE)
    for ch, wavs in [("clean", clean_wavs), ("phone", phone_wavs)]:
        t0 = time.time()
        outputs[(name, ch)] = transcribe(model, wavs, proc=proc)
        print(f"{name:40s} {ch}: WER {wer(refs, outputs[(name, ch)]):5.1f}%  ({time.time()-t0:.0f}s)")
    model.to("cpu"); torch.cuda.empty_cache()

for hf_name, label in [("openai/whisper-small", "whisper-small (original)"),
                       ("openai/whisper-large-v3-turbo", "whisper-large-v3-turbo (original, 3.3x bigger)")]:
    m = WhisperForConditionalGeneration.from_pretrained(hf_name, torch_dtype=torch.float16)
    run_whisper(label, m, proc=WhisperProcessor.from_pretrained(hf_name)); del m

run_whisper("A: whisper-small fine-tuned (clean)", model_clean)
run_whisper("B: whisper-small fine-tuned + phone aug", model_phone)

mms_proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="hye")
mms = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="hye", ignore_mismatched_sizes=True,
                                     torch_dtype=torch.float16).to(DEVICE).eval()
for ch, wavs in [("clean", clean_wavs), ("phone", phone_wavs)]:
    hyps = []
    for i in range(0, len(wavs), 8):
        inp = mms_proc(wavs[i:i + 8], sampling_rate=16000, return_tensors="pt", padding=True)
        mask = inp.get("attention_mask")
        with torch.no_grad():
            logits = mms(inp.input_values.to(DEVICE, torch.float16),
                         attention_mask=None if mask is None else mask.to(DEVICE)).logits
        hyps += mms_proc.batch_decode(logits.argmax(-1))
    outputs[("mms-1b-all (Meta, Armenian adapter)", ch)] = hyps
    print(f"{'mms-1b-all':40s} {ch}: WER {wer(refs, hyps):5.1f}%")
del mms; torch.cuda.empty_cache()

# keep the raw transcripts safe before any analysis
pd.DataFrame({"id": test_df["id"], "gender": test_df["gender"], "seconds": test_df["wav"].map(len) / 16000,
              "ref": refs, **{f"{k[0]} | {k[1]}": v for k, v in outputs.items()}}).to_csv(f"{OUT}/test_transcripts.csv", index=False)

# %% [markdown]
# ### Results table, with 95% confidence intervals
# We resample the test clips 1,000 times (a *bootstrap*) to see how much each WER could move by chance.

# %%
rng = np.random.default_rng(SEED)
R = [norm(r) for r in refs]
boot_idx = [rng.integers(0, len(R), len(R)) for _ in range(1000)]

def corpus_counts(hyps):
    """Per-clip (errors, reference words) so bootstrap is fast."""
    H = [norm(h) for h in hyps]
    errs, nref = [], []
    for r, h in zip(R, H):
        o = jiwer.process_words(r, h)
        errs.append(o.substitutions + o.deletions + o.insertions); nref.append(len(r.split()))
    return np.array(errs), np.array(nref)

rows, counts = [], {}
for (name, ch), hyps in outputs.items():
    e, n = corpus_counts(hyps)
    counts[(name, ch)] = (e, n)
    boots = [100 * e[b].sum() / n[b].sum() for b in boot_idx]
    rows.append({"model": name, "audio": ch, "WER %": 100 * e.sum() / n.sum(),
                 "95% CI": f"{np.percentile(boots, 2.5):.1f}-{np.percentile(boots, 97.5):.1f}",
                 "CER %": 100 * jiwer.cer(R, [norm(h) for h in hyps])})
rows.append({"model": "Baseline: empty output", "audio": "both", "WER %": 100.0, "95% CI": "-", "CER %": 100.0})
final = pd.DataFrame(rows)
final.to_csv(f"{OUT}/final_test_results.csv", index=False)
pd.set_option("display.width", 200)
print(f"FLEURS Armenian test set: {len(R)} clips, {sum(len(r.split()) for r in R)} words\n")
print(final.round(1).to_string(index=False))

# %%
plot_df = final[final.model != "Baseline: empty output"].pivot(index="model", columns="audio", values="WER %")
plot_df = plot_df.loc[[m for m in dict.fromkeys(k[0] for k in outputs)]]
ax = plot_df[["clean", "phone"]].plot(kind="barh", figsize=(10, 4.2), color=["#4C72B0", "#DD8452"])
ax.axvline(100, color="gray", ls="--", lw=1)
ax.set_xlabel("Word error rate % on FLEURS Armenian test (lower is better)")
ax.set_title("Armenian speech-to-text: before and after fine-tuning, microphone vs phone line")
for c in ax.containers:
    ax.bar_label(c, fmt="%.0f", fontsize=8, padding=2)
ax.invert_yaxis(); plt.tight_layout(); plt.savefig(f"{OUT}/final_wer.png", dpi=120); plt.show()

# %% [markdown]
# ### Is the improvement real or luck? (paired bootstrap)
# For each pair we resample the same test clips for both models and count how often the newer one is better.

# %%
def paired(a, b, ch):
    ea, n = counts[(a, ch)]; eb, _ = counts[(b, ch)]
    diffs = np.array([100 * (ea[i].sum() - eb[i].sum()) / n[i].sum() for i in boot_idx])
    return f"{a} -> {b} [{ch}]: WER drops by {100*(ea.sum()-eb.sum())/n.sum():.1f} points " \
           f"(95% CI {np.percentile(diffs,2.5):.1f} to {np.percentile(diffs,97.5):.1f}); better in {100*(diffs>0).mean():.0f}% of resamples"

for ch in ["clean", "phone"]:
    print(paired("whisper-small (original)", "A: whisper-small fine-tuned (clean)", ch))
    print(paired("A: whisper-small fine-tuned (clean)", "B: whisper-small fine-tuned + phone aug", ch))

# %% [markdown]
# ## 8. Who does it fail for? Men vs women, short vs long sentences

# %%
gender = test_df["gender"].map({0: "male", 1: "female"}).fillna("other").values
secs = test_df["wav"].map(len).values / 16000
length_bin = pd.cut(secs, [0, 8, 14, 100], labels=["short (<8s)", "medium (8-14s)", "long (>14s)"])
grp_rows = []
for (name, ch), (e, n) in counts.items():
    for gname, keys in [("gender", gender), ("length", np.asarray(length_bin))]:
        for g in pd.unique(keys):
            m = keys == g
            grp_rows.append({"model": name, "audio": ch, "group": f"{gname}: {g}", "clips": int(m.sum()),
                             "WER %": 100 * e[m].sum() / n[m].sum()})
groups = pd.DataFrame(grp_rows)
groups.to_csv(f"{OUT}/wer_by_group.csv", index=False)
print(groups.pivot_table(index=["group", "clips"], columns=["model", "audio"], values="WER %").round(1)
      .loc[:, (["whisper-small (original)", "B: whisper-small fine-tuned + phone aug"], slice(None))].to_string())

# %% [markdown]
# ## 9. Where does the fine-tuned model still fail?
# Real examples: the correct transcript next to what each model wrote (phone audio).

# %%
best_name = "B: whisper-small fine-tuned + phone aug"
B = [norm(h) for h in outputs[(best_name, "phone")]]
utt = np.array([100 * jiwer.wer(r, h) if r else np.nan for r, h in zip(R, B)])
examples = []
for label, idx in [("typical (median)", np.argsort(utt)[len(utt) // 2: len(utt) // 2 + 2]),
                   ("worst", np.argsort(-np.nan_to_num(utt))[:3])]:
    for i in idx:
        print(f"--- {label} | clip {i} | WER {utt[i]:.0f}% ---")
        print("  correct            :", refs[i])
        print("  whisper-small      :", outputs[("whisper-small (original)", "phone")][i][:250])
        print("  Lsir (fine-tuned B):", outputs[(best_name, "phone")][i])
        examples.append(dict(kind=label, clip=int(i), wer=float(utt[i]), correct=refs[i],
                             original=outputs[("whisper-small (original)", "phone")][i], lsir=outputs[(best_name, "phone")][i]))
pd.DataFrame(examples).to_csv(f"{OUT}/examples.csv", index=False)

# %% [markdown]
# ### Which words does it get wrong most often?
# We align each transcript with the model output and count the most frequent word swaps.
# Numbers and foreign names are a common source of errors because FLEURS writes them in different forms.

# %%
subs = Counter()
for r, h in zip(R, B):
    o = jiwer.process_words(r, h)
    rw, hw = o.references[0], o.hypotheses[0]
    for chunk in o.alignments[0]:
        if chunk.type == "substitute":
            for a, b in zip(rw[chunk.ref_start_idx:chunk.ref_end_idx], hw[chunk.hyp_start_idx:chunk.hyp_end_idx]):
                subs[(a, b)] += 1
print("Most common word swaps (correct -> model):")
for (a, b), c in subs.most_common(15):
    print(f"  {c:3d} x  {a} -> {b}")

has_digit = np.array([bool(re.search(r"\d", r)) for r in refs])
e, n = counts[(best_name, "phone")]
print(f"\nWER on sentences containing numbers: {100*e[has_digit].sum()/n[has_digit].sum():.1f}% "
      f"({has_digit.sum()} clips) vs without numbers: {100*e[~has_digit].sum()/n[~has_digit].sum():.1f}%")
latin = np.array([bool(re.search(r"[A-Za-z]", r)) for r in refs])
print(f"WER on sentences containing Latin-script words (foreign names etc.): {100*e[latin].sum()/n[latin].sum():.1f}% "
      f"({latin.sum()} clips) vs without: {100*e[~latin].sum()/n[~latin].sum():.1f}%")

# %% [markdown]
# ## 10. Save the run settings
The fine-tuned models were saved right after training, and the test transcripts right after testing.

# %%
json.dump({"epochs": EPOCHS, "batch": BATCH, "lr": LR, "phone_prob": PHONE_PROB, "train_clips": len(train_df),
           "test_clips": len(test_df), "torch": torch.__version__, "transformers": transformers.__version__},
          open(f"{OUT}/run_info.json", "w"), indent=1)
print(os.listdir(OUT), os.path.exists(MODEL_OUT))
