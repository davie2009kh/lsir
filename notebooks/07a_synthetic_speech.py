# %% [markdown]
# # Lsir (Լսիր): notebook 7a, synthetic Armenian speech for training
#
# **Track:** AI + Languages · Veritas AI International High School Hackathon 2026
#
# Our base model has already learned all of FLEURS, so more of the same sentences can't teach it much. Here we
# make **new** training speech: everyday conversational Armenian sentences read aloud by open Armenian
# text-to-speech voices.
#
# * **Voices used for training:** *Anadil Armenian TTS* (a LoRA on the VoxCPM2 TTS model, trained on 26k Armenian
#   utterances, MIT licence) and Meta's *MMS-TTS* Armenian voice (Western Armenian pronunciation, for variety).
# * **Never used for training:** the Piper voice from notebook 5, so the clinic-call test stays fair.
# * **Text:** an Armenian translation of the DailyDialog conversations, cleaned and filtered. Sentences that appear in
#   the FLEURS test or validation sets are removed (the clinic-call sentences were written by us, so they cannot appear).
# * **Quality filter:** we transcribe each synthetic clip with BuzzASR and drop clips with more than 30% letter
#   errors (broken or garbled synthesis), a common practice when training on synthetic speech.

# %%
N_ANADIL = 1000
N_MMS = 300
TIME_BUDGET_MIN = 150      # stop generating if it takes longer than this
SEED = 42

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer", "voxcpm"], check=True)
import os, re, io, json, time, random, urllib.request
import numpy as np, pandas as pd, soundfile as sf, torch, jiwer
from scipy.signal import resample_poly
from huggingface_hub import hf_hub_download
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
random.seed(SEED); np.random.seed(SEED)
OUT = "/kaggle/working/synth"; os.makedirs(OUT, exist_ok=True)
normalizer = BasicTextNormalizer()
def norm(t):
    return re.sub(r"\s+", " ", normalizer(t or "")).strip()

# %% [markdown]
# ## 1. Conversational Armenian sentences

# %%
urllib.request.urlretrieve("https://huggingface.co/api/datasets/EdUarD0110/daily_dialog_armenian/parquet/default/train/0.parquet", "/tmp/dd.parquet")
dd = pd.read_parquet("/tmp/dd.parquet")
ARM = re.compile(r"^[Ա-և\s,.:։՝՞՛՜!?«»()\-0-9]+$")

def clean(t):
    t = re.sub(r"(\s*\.\s*){2,}", ". ", str(t)).strip(' "')
    return re.sub(r"\s+", " ", t)

cands = pd.unique(pd.concat([dd["input"], dd["response"]]).map(clean))
cands = [t for t in cands if ARM.match(t) and 4 <= len(t.split()) <= 20]
# remove anything that appears in our test sets (FLEURS test/validation, the clinic sentences)
banned = set()
for split in ["test", "validation"]:
    urllib.request.urlretrieve(f"https://huggingface.co/api/datasets/google/fleurs/parquet/hy_am/{split}/0.parquet", f"/tmp/f_{split}.parquet")
    banned |= {norm(t) for t in pd.read_parquet(f"/tmp/f_{split}.parquet", columns=["raw_transcription"])["raw_transcription"]}
cands = [t for t in cands if norm(t) not in banned]
random.shuffle(cands)
print(len(cands), "clean candidate sentences; examples:")
for t in cands[:5]:
    print("  ", t)
texts_anadil, texts_mms = cands[:N_ANADIL], cands[N_ANADIL:N_ANADIL + N_MMS]

# %% [markdown]
# ## 2. Voice 1: Anadil Armenian TTS (VoxCPM2 + Armenian LoRA)

# %%
from voxcpm import VoxCPM
from voxcpm.model.voxcpm import LoRAConfig
cfg_path = hf_hub_download("Anadilorg/Anadil_Armenian_TTS", "lora_config.json")
w_path = hf_hub_download("Anadilorg/Anadil_Armenian_TTS", "lora_weights.safetensors")
lc = json.load(open(cfg_path)).get("lora_config", {})
tts = VoxCPM.from_pretrained("openbmb/VoxCPM2", load_denoiser=False,
                             lora_config=LoRAConfig(**{k: v for k, v in lc.items() if k in LoRAConfig.model_fields}),
                             lora_weights_path=w_path, device="cuda")
tts.load_lora(w_path)
sr_tts = getattr(getattr(tts, "tts_model", None), "sample_rate", 48000)
print("VoxCPM2 sample rate:", sr_tts)

rows, t0 = [], time.time()
for i, text in enumerate(texts_anadil):
    if (time.time() - t0) / 60 > TIME_BUDGET_MIN * 0.75:
        print("time budget reached at", i); break
    try:
        audio = np.asarray(tts.generate(text=f"[speaker:spk_tmp_001 language:hye] {text}",
                                        inference_timesteps=10, cfg_value=2.0), dtype=np.float32).squeeze()
    except Exception as e:
        print("failed:", repr(e)[:120]); continue
    wav = resample_poly(audio, 16000, sr_tts).astype(np.float32)
    path = f"{OUT}/anadil_{i:04d}.flac"
    sf.write(path, wav, 16000)
    rows.append(dict(file=os.path.basename(path), voice="anadil", text=text, seconds=len(wav) / 16000))
    if i % 100 == 0:
        print(f"anadil {i}/{len(texts_anadil)} ({(time.time() - t0) / 60:.0f} min)", flush=True)
del tts; torch.cuda.empty_cache()

# %% [markdown]
# ## 3. Voice 2: Meta MMS-TTS Armenian (Western Armenian pronunciation, adds variety)

# %%
from transformers import VitsModel, AutoTokenizer
mtok = AutoTokenizer.from_pretrained("facebook/mms-tts-hyw")
mtts = VitsModel.from_pretrained("facebook/mms-tts-hyw").to("cuda").eval()
t1 = time.time()
for i, text in enumerate(texts_mms):
    if (time.time() - t0) / 60 > TIME_BUDGET_MIN:
        print("time budget reached at", i); break
    with torch.no_grad():
        wav = mtts(**mtok(text, return_tensors="pt").to("cuda")).waveform[0].float().cpu().numpy()
    wav = resample_poly(wav, 16000, mtts.config.sampling_rate).astype(np.float32)
    path = f"{OUT}/mms_{i:04d}.flac"
    sf.write(path, wav, 16000)
    rows.append(dict(file=os.path.basename(path), voice="mms-hyw", text=text, seconds=len(wav) / 16000))
print(f"MMS done in {(time.time() - t1) / 60:.1f} min")
del mtts; torch.cuda.empty_cache()
meta = pd.DataFrame(rows)
print(meta.groupby("voice")["seconds"].agg(["count", "sum"]).round(0))

# %% [markdown]
# ## 4. Quality filter: keep clips that BuzzASR can mostly read back (letter error rate up to 30%)

# %%
from transformers import WhisperProcessor, WhisperForConditionalGeneration
BASE = "BuzzASR/armenian"
proc = WhisperProcessor.from_pretrained(BASE)
asr = WhisperForConditionalGeneration.from_pretrained(BASE, torch_dtype=torch.float16).to("cuda").eval()
GCJ = json.load(open(hf_hub_download(BASE, "generation_config.json")))
for k in ["lang_to_id", "task_to_id", "language", "task", "no_timestamps_token_id"]:
    setattr(asr.generation_config, k, GCJ[k])
asr.generation_config.is_multilingual = True
TV = GCJ["eos_token_id"]
hyps = []
for i in range(0, len(meta), 8):
    wavs = [sf.read(f"{OUT}/{f}", dtype="float32")[0][:30 * 16000] for f in meta["file"][i:i + 8]]
    feats = proc.feature_extractor(wavs, sampling_rate=16000, return_tensors="pt").input_features.to("cuda", torch.float16)
    with torch.no_grad():
        ids = asr.generate(input_features=feats, num_beams=1, max_new_tokens=440)
    hyps += [proc.tokenizer.decode([t for t in s.tolist() if t < TV]).strip() for s in ids]
meta["asr"] = hyps
meta["cer"] = [jiwer.cer(norm(r), norm(h)) if norm(r) else 1.0 for r, h in zip(meta["text"], meta["asr"])]
meta["keep"] = meta["cer"] <= 0.30
print(meta.groupby("voice").agg(clips=("keep", "size"), kept=("keep", "sum"), median_cer=("cer", "median")))
for v in meta["voice"].unique():
    r = meta[meta.voice == v].iloc[0]
    print(f"[{v}] text: {r.text}\n        asr : {r.asr}  (CER {r.cer:.2f})")
for f in meta.loc[~meta["keep"], "file"]:
    os.remove(f"{OUT}/{f}")
meta[meta["keep"]].to_csv(f"{OUT}/synth_meta.csv", index=False)
print("kept", int(meta["keep"].sum()), "clips,", round(meta.loc[meta["keep"], "seconds"].sum() / 3600, 2), "hours")
