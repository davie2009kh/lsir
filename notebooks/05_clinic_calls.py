# %% [markdown]
# # Lsir (Լսիր): does it understand a call to a clinic?
# ## Notebook 5: a clinic-call test set (our own data)
#
# **Track:** AI + Languages · Veritas AI International High School Hackathon 2026
#
# FLEURS (notebooks 1-4) is people reading news-style sentences. A real call to a clinic sounds different: dates,
# times, phone numbers, doses, medicine names and symptoms. Getting a **number** or a **medicine name** wrong is
# far more dangerous than getting a filler word wrong.
#
# So we wrote **40 short sentences that a caller to an Armenian clinic or pharmacy might say**, and had them
# read by a free Armenian text-to-speech voice (Piper `hy_AM-gor-medium`, Eastern Armenian, male) at a normal and
# a slow speaking pace (slow speech is closer to many older callers). We play every
# sentence clean, through the fixed phone simulation, and through the **unseen** mobile-codec condition from
# notebook 4. Then we measure word errors, **how many numbers survive**, and **how many medical terms survive**.
#
# **Honest limitations.** This is one synthetic male voice, not real callers: it is clearer and more regular than
# a real elderly caller, so real-world error rates would be higher. The sentences were written by our team (with
# an AI assistant's help) and are only 40, so treat this as a sanity check of what matters to a clinic, not a benchmark.
#
# *Runs on a free Kaggle T4 GPU. Uses the phone-tuned LoRA adapters saved by the two notebook 4 runs.*

# %%
SMOKE = False
SEED = 42
GEN = dict(num_beams=1, no_repeat_ngram_size=3, repetition_penalty=1.2, max_new_tokens=440)
PACES = {"normal pace": 1.0, "slow pace": 1.35}     # Piper length_scale: above 1 = slower speech

# %%
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "jiwer", "peft", "piper-tts"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchao"], check=False)   # Kaggle's old torchao breaks peft; we don't use it

import asyncio, glob, io, os, re, json, time, shutil
from collections import Counter
import numpy as np
import pandas as pd
import torch
import jiwer
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfilt, resample_poly
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoProcessor, Wav2Vec2ForCTC
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from peft import PeftModel
import transformers
transformers.logging.set_verbosity_error()

DEVICE = "cuda"
OUT = "/kaggle/working/results"; os.makedirs(OUT, exist_ok=True)

# %% [markdown]
# ## 1. The clinic-call sentences
# Each sentence lists the **medical terms** we check separately (numbers are found automatically).

# %%
CLINIC = [
    ("Բարև Ձեզ, ուզում եմ գրանցվել թերապևտի մոտ վաղը առավոտյան։", ["թերապևտի"]),
    ("Իմ հեռախոսահամարն է 093 45 67 89։", []),
    ("Մայրս 72 տարեկան է, և նրա ճնշումը բարձր է։", ["ճնշումը"]),
    ("Կարո՞ղ եմ հերթ վերցնել ատամնաբույժի մոտ հինգշաբթի օրը ժամը 3-ին։", ["ատամնաբույժի"]),
    ("Բժիշկը նշանակել է պարացետամոլ, 500 միլիգրամ, օրը 2 անգամ։", ["պարացետամոլ", "միլիգրամ"]),
    ("Արդեն երեք օր է ջերմություն ունեմ, և կոկորդս ցավում է։", ["ջերմություն", "կոկորդս"]),
    ("Պոլիկլինիկան աշխատո՞ւմ է շաբաթ օրը։", ["պոլիկլինիկան"]),
    ("Իմ անունը Անահիտ Սարգսյան է, ծնվել եմ 1958 թվականին։", ["սարգսյան"]),
    ("Ուզում եմ չեղարկել նոյեմբերի 18-ի այցը։", ["չեղարկել"]),
    ("Երեխայիս մոտ ալերգիա է սկսվել նոր դեղից։", ["ալերգիա"]),
    ("Ինձ անհրաժեշտ է ինսուլինի նոր դեղատոմս։", ["ինսուլինի", "դեղատոմս"]),
    ("Մենք ապրում ենք Կոմիտասի պողոտա 25, բնակարան 14։", ["կոմիտասի"]),
    ("Կարո՞ղ եք ասել, թե լաբորատորիան ժամը քանիսին է բացվում։", ["լաբորատորիան"]),
    ("Արյան անալիզի պատասխանները պատրաստ են, թե ոչ։", ["անալիզի"]),
    ("Հայրս ընկել է, և ոտքը շատ է ուռել։", ["ուռել"]),
    ("Խնդրում եմ միացնել ինձ սրտաբանի հետ։", ["սրտաբանի"]),
    ("Ամեն առավոտ խմում եմ մետֆորմին, 850 միլիգրամ։", ["մետֆորմին", "միլիգրամ"]),
    ("Շտապ օգնություն է պետք, տատիկս վատ է զգում։", ["շտապ"]),
    ("Վերջին երկու շաբաթը գլխացավերս ավելի են ուժեղացել։", ["գլխացավերս"]),
    ("Կարո՞ղ եմ գալ առանց նախնական գրանցման։", ["գրանցման"]),
    ("Իմ ապահովագրության համարն է 4471 2390։", ["ապահովագրության"]),
    ("Բժիշկ Հակոբյանը այսօր ընդունո՞ւմ է։", ["հակոբյանը"]),
    ("Ուզում եմ փոխել այցի ժամը 10:30-ից 12-ի։", []),
    ("Դեղատանը չկար ամոքսիցիլին, ի՞նչ անեմ։", ["ամոքսիցիլին", "դեղատանը"]),
    ("Առավոտյան ճնշումս 160 էր 100-ի վրա։", ["ճնշումս"]),
    ("Սիրտս արագ է խփում, և շնչահեղձ եմ լինում։", ["շնչահեղձ"]),
    ("Երեխան 2 տարեկան է, քաշը 12 կիլոգրամ է։", ["կիլոգրամ"]),
    ("Վաղը ժամը 9-ին պետք է արյուն հանձնեմ անալիզի։", ["անալիզի"]),
    ("Խնդրում եմ հետազոտության արդյունքներն ուղարկել էլեկտրոնային փոստով։", ["հետազոտության"]),
    ("Ես հղի եմ, 24 շաբաթական, և խորհրդատվություն եմ ուզում։", ["հղի", "խորհրդատվություն"]),
    ("Ես առաջին տիպի շաքարային դիաբետ ունեմ։", ["դիաբետ"]),
    ("Կարո՞ղ եք կրկնել հասցեն, ես չլսեցի։", ["հասցեն"]),
    ("Ամուսինս պառկած է հիվանդանոցի երրորդ մասնաշենքում։", ["հիվանդանոցի", "մասնաշենքում"]),
    ("Ներարկումից հետո ձեռքս կարմրել է։", ["ներարկումից"]),
    ("Որքա՞ն արժե ուլտրաձայնային հետազոտությունը։", ["ուլտրաձայնային"]),
    ("Գլուխս պտտվում է, երբ արագ վեր եմ կենում։", ["պտտվում"]),
    ("Պատվաստումն արվել է հոկտեմբերի 5-ին։", ["պատվաստումն"]),
    ("Խնդրում եմ զանգահարել ինձ 077 12 34 56 համարով։", []),
    ("Ակնոցի համար ուզում եմ գրանցվել ակնաբույժի մոտ։", ["ակնաբույժի"]),
    ("Թոռս հազում է արդեն մեկ շաբաթ, ի՞նչ դեղ տամ։", ["հազում"]),
]
if SMOKE:
    CLINIC = CLINIC[:4]
print(len(CLINIC), "sentences,", sum(len(t) for _, t in CLINIC), "medical terms,",
      sum(len(re.findall(r"\d+", s)) for s, _ in CLINIC), "numbers")

# %% [markdown]
# ## 2. Speak them with an Armenian text-to-speech voice, at two speaking paces

# %%
FFMPEG = shutil.which("ffmpeg")
os.makedirs("/tmp/tts", exist_ok=True)

import urllib.request
PIPER_URL = "https://huggingface.co/davit312/piper-TTS-Armenian/resolve/main/v3/hy_AM-gor-medium.onnx"
for ext in ["", ".json"]:
    urllib.request.urlretrieve(PIPER_URL + ext, "/tmp/tts/voice.onnx" + ext)

def speak(text, length_scale, path):
    r = subprocess.run(["piper", "-m", "/tmp/tts/voice.onnx", "-f", path, "--length_scale", str(length_scale)],
                       input=text, text=True, capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-500:])

def mp3_to_16k(path):
    r = subprocess.run([FFMPEG, "-loglevel", "error", "-i", path, "-f", "f32le", "-ar", "16000", "-ac", "1", "pipe:1"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).copy()

rows = []
for i, (sent, terms) in enumerate(CLINIC):
    for g, scale in PACES.items():
        p = f"/tmp/tts/{i:02d}_{g.split()[0]}.wav"
        for attempt in range(3):
            try:
                speak(sent, scale, p); break
            except Exception as e:
                print("retry", i, g, repr(e)[:100]); time.sleep(2)
        rows.append(dict(sid=i, voice=g, text=sent, terms=terms, wav=mp3_to_16k(p)))
clips = pd.DataFrame(rows)
print(f"{len(clips)} clips, {clips['wav'].map(len).sum()/16000/60:.1f} minutes")

# %% [markdown]
# ## 3. Same channels as notebook 4: clean, fixed phone simulation, unseen mobile codec

# %%
encoders = subprocess.run([FFMPEG, "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
UNSEEN_CODEC = next(((c, b) for c, b in [("libopencore_amrnb", "7.95k"), ("libgsm", "13k"), ("g723_1", "6.3k")] if c in encoders), None)
CODEC_FMT = {"libopencore_amrnb": "amr", "libgsm": "gsm", "g723_1": "nut"}
print("unseen codec (same rule as notebook 4):", UNSEEN_CODEC)
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

def phone_line(wav, snr_db=20, seed=0):
    rng = np.random.default_rng(seed)
    y = sosfilt(SOS_PHONE, resample_poly(wav, 1, 2))
    y = mu_law(0.9 * y / (np.max(np.abs(y)) + 1e-9))
    return resample_poly(add_noise(y, snr_db, rng), 2, 1).astype(np.float32)

def codec_roundtrip(wav16, codec, bitrate):
    x8 = resample_poly(wav16, 1, 2).astype(np.float32)
    fmt = CODEC_FMT[codec]
    enc = subprocess.run([FFMPEG, "-loglevel", "error", "-f", "f32le", "-ar", "8000", "-ac", "1", "-i", "pipe:0",
                          "-c:a", codec, "-b:a", bitrate, "-f", fmt, "pipe:1"], input=x8.tobytes(), capture_output=True)
    dec_in = ["-f", "gsm"] if fmt == "gsm" else []
    dec = subprocess.run([FFMPEG, "-loglevel", "error", *dec_in, "-i", "pipe:0", "-f", "f32le", "-ar", "8000", "-ac", "1", "pipe:1"],
                         input=enc.stdout, capture_output=True)
    y = np.frombuffer(dec.stdout, dtype=np.float32)
    y = x8 if len(y) < len(x8) // 2 else np.pad(y, (0, max(0, len(x8) - len(y))))[:len(x8)]
    return resample_poly(y, 2, 1).astype(np.float32)

def packet_loss(y, rate, rng, frame=320):
    y = y.copy()
    for s in range(0, len(y), frame):
        if rng.random() < rate:
            y[s:s + frame] = 0.0
    return y

def amr_unseen(wav, seed):
    rng = np.random.default_rng(seed)
    if UNSEEN_CODEC:
        y = codec_roundtrip(wav, *UNSEEN_CODEC)
    else:
        y8 = sosfilt(SOS_PHONE, resample_poly(wav, 1, 2)); y8 = np.round(y8 / (np.max(np.abs(y8)) + 1e-9) * 15) / 15
        y = resample_poly(y8, 2, 1).astype(np.float32)
    return packet_loss(add_noise(y, 15, rng, kind="pink"), 0.05, rng).astype(np.float32)

AUDIO = {"clean": clips["wav"].tolist(),
         "phone": [phone_line(w, seed=70_000 + i) for i, w in enumerate(clips["wav"])],
         "unseen mobile codec": [amr_unseen(w, seed=80_000 + i) for i, w in enumerate(clips["wav"])]}

# %% [markdown]
# ## 4. Run the models

# %%
@torch.no_grad()
def whisper_run(model, proc, wavs, bs=8, text_vocab=None, **gen):
    out = []
    for i in range(0, len(wavs), bs):
        f = proc.feature_extractor(list(wavs[i:i + bs]), sampling_rate=16000, return_tensors="pt").input_features
        with torch.autocast("cuda", dtype=torch.float16):
            ids = model.generate(input_features=f.to(DEVICE, torch.float16), max_new_tokens=440, **gen)
        if text_vocab:   # BuzzASR: decode ordinary text tokens only (see notebook 4)
            out += [proc.tokenizer.decode([t for t in s.tolist() if t < text_vocab]).strip() for s in ids]
        else:
            out += proc.batch_decode(ids, skip_special_tokens=True)
    return out

outputs = {}
proc = WhisperProcessor.from_pretrained("BuzzASR/armenian")
base = WhisperForConditionalGeneration.from_pretrained("BuzzASR/armenian", torch_dtype=torch.float16).to(DEVICE).eval()
from huggingface_hub import hf_hub_download
GCJ = json.load(open(hf_hub_download("BuzzASR/armenian", "generation_config.json")))
for k in ["lang_to_id", "task_to_id", "language", "task", "no_timestamps_token_id"]:   # restore BuzzASR's language table
    setattr(base.generation_config, k, GCJ[k])
base.generation_config.is_multilingual = True
TV = GCJ["eos_token_id"]
gen = {k: v for k, v in GEN.items() if k != "max_new_tokens"}
for ch, wavs in AUDIO.items():
    outputs[("BuzzASR-armenian (original)", ch)] = whisper_run(base, proc, wavs, text_vocab=TV, **gen)

adapters = sorted(glob.glob("/kaggle/input/**/lsir-buzzasr-armenian-phone-lora-*/adapter_config.json", recursive=True))
print("adapters found:", adapters)
for cfg in adapters:
    d = os.path.dirname(cfg); part = d.rsplit("-", 1)[-1]
    m = PeftModel.from_pretrained(base, d).eval()
    for ch, wavs in AUDIO.items():
        outputs[(f"Lsir: BuzzASR + phone LoRA ({part})", ch)] = whisper_run(m, proc, wavs, text_vocab=TV, **gen)
    base = m.unload()          # remove the adapter again before loading the next one
del base; torch.cuda.empty_cache()

tproc = WhisperProcessor.from_pretrained("openai/whisper-large-v3-turbo")
turbo = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo", torch_dtype=torch.float16).to(DEVICE).eval()
for ch, wavs in AUDIO.items():
    outputs[("whisper-large-v3-turbo", ch)] = whisper_run(turbo, tproc, wavs, language="hy", task="transcribe")
del turbo; torch.cuda.empty_cache()

mproc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="hye")
mms = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="hye", ignore_mismatched_sizes=True,
                                     torch_dtype=torch.float16).to(DEVICE).eval()
for ch, wavs in AUDIO.items():
    hyps = []
    for i in range(0, len(wavs), 8):
        inp = mproc(wavs[i:i + 8], sampling_rate=16000, return_tensors="pt", padding=True)
        mask = inp.get("attention_mask")
        with torch.no_grad():
            lg = mms(inp.input_values.to(DEVICE, torch.float16), attention_mask=None if mask is None else mask.to(DEVICE)).logits
        hyps += mproc.batch_decode(lg.argmax(-1))
    outputs[("mms-1b-all", ch)] = hyps
del mms; torch.cuda.empty_cache()

pd.DataFrame({"sid": clips["sid"], "voice": clips["voice"], "ref": clips["text"],
              **{f"{m} | {c}": v for (m, c), v in outputs.items()}}).to_csv(f"{OUT}/clinic_transcripts.csv", index=False)

# %% [markdown]
# ## 5. Results: words, numbers and medical terms

# %%
normalizer = BasicTextNormalizer()
def norm(t):
    return re.sub(r"\s+", " ", normalizer(t or "")).strip()
NUM = re.compile(r"\d+")

def kept(want, have):
    have = Counter(have); k = 0
    for w in want:
        if have[w] > 0:
            k += 1; have[w] -= 1
    return k, len(want)

res = []
R = [norm(t) for t in clips["text"]]
for (m, ch), hyps in outputs.items():
    H = [norm(h) for h in hyps]
    nk = [kept(NUM.findall(r), NUM.findall(h)) for r, h in zip(clips["text"], hyps)]
    tk = [kept([norm(t) for t in terms], [w for w in h.split()]) for terms, h in zip(clips["terms"], H)]
    for g in list(PACES) + ["all"]:
        k = (clips["voice"] == g).values if g != "all" else np.ones(len(clips), bool)
        res.append({"model": m, "audio": ch, "voice": g,
                    "WER %": 100 * jiwer.wer([r for r, kk in zip(R, k) if kk], [h for h, kk in zip(H, k) if kk]),
                    "numbers kept %": 100 * sum(a for (a, b), kk in zip(nk, k) if kk) / max(1, sum(b for (a, b), kk in zip(nk, k) if kk)),
                    "medical terms kept %": 100 * sum(a for (a, b), kk in zip(tk, k) if kk) / max(1, sum(b for (a, b), kk in zip(tk, k) if kk))})
res = pd.DataFrame(res); res.to_csv(f"{OUT}/clinic_results.csv", index=False)
pd.set_option("display.width", 220)
print(res[res.voice == "all"].pivot(index="model", columns="audio", values=["WER %", "numbers kept %", "medical terms kept %"]).round(1).to_string())
print()
print(res[res.voice != "all"].pivot_table(index=["model", "voice"], columns="audio", values="WER %").round(1).to_string())

# %%
allv = res[res.voice == "all"]
fig, ax = plt.subplots(1, 2, figsize=(14, 4))
for a, metric in zip(ax, ["WER %", "numbers kept %"]):
    allv.pivot(index="model", columns="audio", values=metric)[list(AUDIO)].plot(kind="barh", ax=a, legend=a is ax[0])
    a.set_title(f"Clinic-call test: {metric}"); a.set_ylabel("")
    for c in a.containers:
        a.bar_label(c, fmt="%.0f", fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/clinic_results.png", dpi=130); plt.show()

# %% [markdown]
# ## 6. Real examples over the unseen mobile codec: what would the clinic's system have heard?

# %%
show = [m for m in dict.fromkeys(k[0] for k in outputs)]
for i in [1, 4, 22, 23, 37] if not SMOKE else [0, 1]:
    for g in ["slow pace"]:
        j = clips.index[(clips.sid == i) & (clips.voice == g)][0]
        print(f"--- sentence {i} ({g}) ---")
        print(f"  {'correct':38s}: {clips.text[j]}")
        for m in show:
            print(f"  {m[:38]:38s}: {outputs[(m, 'unseen mobile codec')][j][:200]}")
print(os.listdir(OUT))
