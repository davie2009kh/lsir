# Lsir (Լսիր): Armenian speech-to-text over the phone

**Track:** AI + Languages
**Team:** {{TEAM NAMES}}

*"Lsir" (Լսիր) means "listen!" in Armenian. This project is a prototype and is not medical advice.*

## The problem
About 3 million people in Armenia speak Armenian, and many older people speak nothing else. Clinics,
pharmacies and banks are starting to answer the phone with voice assistants, and those run on speech-to-text
models trained mostly on English. If the model can't understand Armenian, and especially Armenian **over a
phone line**, the callers who most need help are the ones who get misunderstood.

## What we built
1. **A benchmark of free speech-to-text models on Armenian, on a microphone and over a phone line.** We built a
   telephone-channel simulator and tested 5 models on all 932 clips of the FLEURS Armenian test set.
2. **A phone-robust fine-tune of the best model** (BuzzASR-Armenian + LoRA, trained with telephone augmentation),
   tested on a mobile codec it never saw in training.
3. **A clinic-call test**: 40 sentences a caller to an Armenian clinic might say (appointments, phone numbers,
   doses, medicine names), spoken by two Armenian neural voices, scored on whether **numbers and medical terms** survive.
4. **A demo app**: speak or upload Armenian, optionally send it through a phone line, and compare models side by side.

## What we added
Starting point: suggested project **4.4.3 Speech recognition across accents**. We added:
- **Our own user and question:** an Armenian-only caller to a clinic phone line. No published benchmark measures
  Armenian ASR over a phone line, so we built the simulator (8 kHz narrow band, G.711 μ-law/A-law and Opus codecs,
  packet loss, babble noise, echo) and an **unseen-codec** test (GSM) to keep ourselves honest.
- **Our own data:** the clinic-call test set (sentences written by us; synthetic voices).
- **Our own analysis of why models fail:** tokenizer cost, runaway loops, and a gender gap that appears in every model.

## How it works
**Data.** [FLEURS](https://huggingface.co/datasets/google/fleurs) Armenian (`hy_am`): 3,053 training clips
(10.4 h), 395 validation, 932 test (3.0 h, 17,802 words). FLEURS splits contain **different sentences**
(we checked: 0 shared between train and test). Russian and English FLEURS test clips are used as reference points.

**Models compared.** `openai/whisper-small`, `openai/whisper-large-v3-turbo`, `facebook/mms-1b-all` (Armenian
adapter), Meta Omnilingual ASR (`omniASR_CTC_1B_v2`) and `BuzzASR/armenian` (Whisper-large-v3 fine-tuned on Armenian).

**Our fine-tune.** LoRA adapters on BuzzASR-Armenian (1.55B parameters; LoRA trains about 1.5%) on a free Kaggle T4,
with on-the-fly telephone augmentation and SpecAugment. The epoch is chosen on validation clips, and the test split
is used once at the end.

**Metrics.** Word error rate (WER) and character error rate (CER) after lower-casing and removing punctuation,
with 95% bootstrap confidence intervals (1,000 resamples of the test clips). The trivial baseline (empty output)
scores exactly 100% WER.

## Results

### 1. The default model doesn't understand Armenian (FLEURS test, 300 clips per language)
| Model | English | Russian | **Armenian** | Armenian over phone |
|---|---|---|---|---|
| whisper-small | 9.1% | 13.9% | **110.7%** | 116.2% |
| whisper-large-v3-turbo | 6.4% | 8.4% | **45.6%** | 58.6% |

Whisper-small scores worse than the empty-output baseline on Armenian, because on 8% of clips it gets stuck
repeating one syllable hundreds of times. Its tokenizer needs **8.8 tokens per Armenian word vs 1.3 for English**
(BuzzASR's Armenian tokenizer: 1.6).

### 2. Which free model works best, on a microphone and over the phone? (all 932 test clips)
| Model | Clean WER | Phone WER | Phone penalty | Clean CER |
|---|---|---|---|---|
| **BuzzASR-armenian** | **13.6%** (12.9-14.5) | **17.4%** (16.5-18.3) | +3.8 | 5.2% |
| mms-1b-all | 20.0% (19.1-21.0) | 28.3% (27.3-29.3) | +8.3 | 5.1% |
| Omnilingual CTC-1B v2 (Meta) | 15.5% (14.6-16.4) | 21.7% (20.6-22.9) | +6.2 | **4.2%** |
| whisper-large-v3-turbo | 46.0% (44.8-47.2) | 57.9% (56.6-59.2) | +11.9 | 11.7% |
| whisper-small | 108.0% (102.1-114.1) | 117.8% (112.0-124.6) | +9.8 | 43.4% |
| *Baseline: empty output* | *100%* | *100%* | | |

A model trained specifically on Armenian beats general models that are bigger or newer, and it also loses the
least accuracy over the phone. Meta's Omnilingual gets the fewest *letter* errors (CER 4.2%) but more whole-word
errors, and it is hurt more by the phone line.

### 3. Who does it fail for? Every model is worse on women's voices
| Model (phone audio) | Women (593 clips) | Men (339 clips) |
|---|---|---|
| BuzzASR-armenian | 19.0% | 14.6% |
| mms-1b-all | 31.7% | 22.2% |
| Omnilingual CTC-1B v2 | 23.9% | 17.9% |
| whisper-large-v3-turbo | 61.8% | 51.1% |

The gap appears on clean audio too (BuzzASR: 14.5% vs 12.0%). FLEURS doesn't say how many different speakers
there are, so part of the gap could come from a few individual voices. We report it rather than hide it.

### 4. Our phone-robust fine-tune
{{FINETUNE_RESULTS}}

### 5. Clinic-call test (40 sentences x 2 synthetic voices)
{{CLINIC_RESULTS}}

### Where it fails, and why we think so
- **Unusual words and names:** "Կասաբլանկան" (Casablanca) or "Կիվու" are often misheard; the model has rarely heard them.
- **Numbers:** FLEURS writes some numbers as digits and some as words, and models do both, so a correct number can
  still count as an error. Over the unseen mobile codec only about 72% of numbers survive with the original model.
- **Phone audio removes high frequencies**, so sounds like "ս", "շ", "ց" and "ծ" get confused.
- **What didn't work:** our first fine-tune (trained on punctuated transcripts, higher learning rate) made the model
  *worse* (clean 13.7% → 15.4%), because it learned to add punctuation and capitals instead of hearing better. Fine-tuning
  the decoder too was worse still. BuzzASR had already been trained on the same FLEURS sentences, so there was little new
  to learn from them.

## What we'd do next
1. **Real phone calls:** record Armenian speakers (with permission) through real mobile and landline calls, and test
   whether the simulator's ranking of models holds.
2. **More, newer Armenian audio:** BuzzASR already learned FLEURS, so real gains need new speech (e.g. Common Voice
   Armenian or recordings of older speakers) and Western Armenian for the diaspora.
3. **Investigate the gender gap** with a dataset that has speaker IDs, and test whether pitch-aware augmentation narrows it.

## How to run
| Notebook | What it does | Runtime (free Kaggle T4) |
|---|---|---|
| `notebooks/01_baseline.py` | Whisper on Armenian vs Russian vs English, phone simulation | ~40 min |
| `notebooks/02_finetune.py` | Fine-tuning whisper-small (interrupted run; partial results in `results/`) | ~2.5 h |
| `notebooks/03_strong_models.py` | 4 models on all 932 test clips, clean and phone | ~1.5 h |
| `notebooks/03b_omnilingual.py` | Meta Omnilingual ASR, same test | ~40 min |
| `notebooks/04_phone_finetune.py` | Phone-robust LoRA fine-tune of BuzzASR + 3-condition test | ~3.5 h |
| `notebooks/05_clinic_calls.py` | Clinic-call test with two Armenian neural voices | ~40 min |
| `app/app.py` | Gradio demo (works in Colab or as a Hugging Face Space) | |

Notebooks are written as `.py` files with `# %%` cells and converted with `tools/py2nb.py`; the executed versions
with all outputs are linked below. Every dataset and model is downloaded inside the notebooks, and no API keys are needed.

## Sources
- **FLEURS:** Conneau et al., 2022. https://huggingface.co/datasets/google/fleurs (CC-BY-4.0)
- **Whisper:** Radford et al., 2022. https://huggingface.co/openai/whisper-small, https://huggingface.co/openai/whisper-large-v3-turbo (MIT)
- **BuzzASR-Armenian:** Singh et al., 2026, *BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models*. https://huggingface.co/BuzzASR/armenian (MIT)
- **MMS:** Pratap et al., 2023. https://huggingface.co/facebook/mms-1b-all (CC-BY-NC-4.0)
- **Omnilingual ASR:** Meta AI, 2025. https://github.com/facebookresearch/omnilingual-asr (Apache-2.0)
- **Edge TTS voices** (hy-AM-AnahitNeural, hy-AM-HaykNeural) via the `edge-tts` package, used only to synthesise the clinic-call test.
- **Methods:** LoRA (Hu et al., 2021) via Hugging Face PEFT; SpecAugment (Park et al., 2019); the telephone-augmentation
  mix follows Dissen et al. (arXiv 2406.18928) and data-augmentation work on low-bit-rate telephony (Applied Sciences 12(3):1580, 2022).
- **Libraries:** PyTorch, Transformers, PEFT, jiwer, SciPy, pandas, matplotlib, Gradio, ffmpeg.
- **Compute:** free Kaggle T4 GPUs.
- **AI tools:** {{AI TOOLS: describe how you used Claude / other assistants}}

*This project is a prototype and is not medical advice.*
