# Lsir (Լսիր): full technical report

*The one-page summary is the [README](../README.md). This report has every table, the failure analysis and how to run the code.*

**Track:** AI + Languages  
**Team:** Alen Gevorgyan, Koryun Gaboyan, David Ghazaryan  
**Live demo:** https://huggingface.co/spaces/davitlive5/lsir-armenian-phone-asr  
**Model:** https://huggingface.co/davitlive5/lsir-buzzasr-armenian-phone-lora

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
   doses, medicine names), spoken by an Armenian text-to-speech voice at two speaking paces, scored on whether **numbers and medical terms** survive.
4. **A demo app**: speak or upload Armenian, optionally send it through a phone line, and compare models side by side. Try it live: https://huggingface.co/spaces/davitlive5/lsir-armenian-phone-asr

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

**Our fine-tune.** LoRA adapters on BuzzASR-Armenian (1.55B parameters; LoRA on the audio encoder trains about 1.5%) on a free Kaggle T4,
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

### 3. Who does it fail for? On FLEURS, every model is worse on women's voices
| Model (phone audio) | Women (593 clips) | Men (339 clips) |
|---|---|---|
| BuzzASR-armenian | 19.0% | 14.6% |
| mms-1b-all | 31.7% | 22.2% |
| Omnilingual CTC-1B v2 | 23.9% | 17.9% |
| whisper-large-v3-turbo | 61.8% | 51.1% |

The gap appears on clean audio too (BuzzASR: 14.5% vs 12.0%). FLEURS doesn't say how many different speakers
there are, so part of the gap could come from a few individual voices. Section 6 tests this with Common Voice speaker IDs: the gap does not replicate.

### 4. Our phone-robust fine-tune
LoRA on BuzzASR's audio encoder (about 1.5% of parameters, r=16) trained for 2 epochs on FLEURS Armenian training clips,
half of them sent through our randomised telephone channel. Tested on all 932 test clips:

| Test audio | Original BuzzASR | **Lsir (ours)** | Change in WER (95% CI) |
|---|---|---|---|
| Clean microphone | 13.7% | 13.8% | +0.1 (-0.2 to +0.4), no real change |
| Phone (fixed simulation) | 17.4% | 17.2% | -0.2 (-0.6 to +0.1) |
| **Unseen GSM mobile codec** (+ packet loss + noise) | 24.3% | **23.3%** | **-1.0 (-1.5 to -0.5)**, better in 100% of resamples |

| Numbers that survive transcription (177 clips with numbers) | Original | **Lsir** |
|---|---|---|
| Clean | 86.2% | **88.1%** |
| Phone | 82.8% | **86.4%** |
| Unseen mobile codec | 72.1% | **78.2%** |

The gain is modest but statistically significant on the GSM codec, which the model **never saw in training** (packet loss and noise were part of training; the codec was not), which suggests it
learned "phone audio" in general rather than our simulator. Clean audio is unchanged. Numbers (times, doses, phone
numbers), the details that matter most to a clinic, survive noticeably more often. Women's voices over the mobile codec
improve from 28.7% to 27.2% WER, so the gender gap narrows slightly but remains.

### 5. Clinic-call test (40 sentences, one synthetic voice at two speaking paces)
40 short clinic and pharmacy sentences we wrote (appointments, phone numbers, doses, medicine names), spoken by the
Piper Armenian voice at a normal and a slow pace (80 clips). Besides WER we count how many of the 29 numbers and 44
medical terms per pace come out exactly right. Both paces are pooled below.

| Audio | Model | WER | Numbers kept | Medical terms kept |
|---|---|---|---|---|
| Clean | BuzzASR | 37.2% | 44.8% | 55.7% |
| Clean | **Lsir** | **36.8%** | **53.4%** | 55.7% |
| Phone | BuzzASR | 44.1% | 43.1% | 39.8% |
| Phone | **Lsir** | **43.3%** | **50.0%** | **45.5%** |
| Unseen mobile codec | BuzzASR | 50.2% | 43.1% | 33.0% |
| Unseen mobile codec | **Lsir** | **47.2%** | **46.6%** | **42.0%** |
| Unseen mobile codec | whisper-large-v3-turbo | 173.2% | 6.9% | 11.4% |
| Unseen mobile codec | mms-1b-all | 76.0% | 10.3% | 21.6% |

Lsir keeps more numbers and medicine names than the original model in every condition, and the gap is largest over the
unseen codec (medical terms 33% -> 42%). **Caveats:** this is a small set (80 clips, one synthetic voice), so these
are directions, not significant results. WER is much higher than on FLEURS because the sentences are packed with
numbers and rare drug names, and numbers can be written several ways (digits or words). A real test needs real callers.

### 6. A second, independent test: Common Voice Armenian (1,000 test clips, volunteers on their own devices)
| Test audio | Original BuzzASR | **Lsir (ours)** | Change in WER (95% CI) |
|---|---|---|---|
| Clean | 3.7% | 4.0% | +0.3 (0.0 to +0.6) |
| Phone (fixed simulation) | 12.4% | 12.8% | +0.4 (-0.3 to +1.1) |
| **Unseen GSM mobile codec** | 18.9% | **17.8%** | **-1.1 (-1.9 to -0.3)** |

The gain on the unseen mobile codec **replicates** on a dataset from a different source (-1.1 points here, -1.0 on
FLEURS), at a small cost on clean audio. (BuzzASR was trained on Common Voice's training split, which is why its
clean score is so low. The test split has different speakers.)

**Who is missing from the data?** The Common Voice Armenian test split has **no speakers over 59**. The oldest group
is people in their 50s, with only a few speakers. The callers our tool is meant for are missing from public Armenian
speech data, so we couldn't measure how well any model works for them.

**Is the gender gap real?** With Common Voice's speaker IDs (73 female, 49 male speakers in our sample), women's
speech is transcribed slightly *better* (clean 3.7% vs 4.4%). So the large gap we saw on FLEURS is more likely
caused by a few individual FLEURS speakers than by gender itself.

### 7. System combination: three different models vote word by word (ROVER)
BuzzASR (Whisper-style), Meta Omnilingual and Meta MMS (both CTC) make *different* mistakes. We align each model's
transcript to a pivot transcript and take a majority vote for every word (ties keep the pivot's word). The pivot
is the best single model.

| FLEURS test (932 clips) | WER | Change vs BuzzASR alone (strict 95% CI) |
|---|---|---|
| BuzzASR alone, clean | 13.6% | |
| **Vote: BuzzASR + Omnilingual + MMS, clean** | **11.9%** | **-1.7 (-2.3 to -1.1)** |
| BuzzASR alone, phone | 17.4% | |
| Vote with BuzzASR as pivot, phone | 15.3% | -2.2 (-2.7 to -1.6) |
| **Vote with Lsir (ours) as pivot, phone** | **15.1%** | **-2.3 (-2.9 to -1.7)** |

Voting removes about 12-13% of errors, and our phone-trained model is the best pivot over the phone. Using the
weaker Omnilingual as the pivot did *not* help (13.3%), which is why we use the best model as the pivot. The cost is
running three models instead of one. Omnilingual and MMS were trained partly on FLEURS's training split.
*Voting on the unseen GSM codec and on Common Voice (to confirm the setup on data we didn't choose it on) is still running and will be added here.*

### 8. Beam search: letting the model consider 5 candidate transcripts instead of 1
Until now every model wrote its single most likely next token at each step ("greedy" decoding). With **beam search**
it keeps the 5 most likely partial transcripts and picks the best complete one. We set 5 beams in advance (a common
default) and did not tune it. FLEURS test, 932 clips:

| Audio | BuzzASR greedy | BuzzASR beam 5 | Lsir greedy | **Lsir beam 5** | Lsir vs BuzzASR, both beam 5 (strict 95% CI) |
|---|---|---|---|---|---|
| Clean | 13.7% | 12.5% | 13.8% | **12.3%** | -0.23 (-0.52 to +0.07) |
| Phone | 17.4% | 15.8% | 17.2% | **15.2%** | **-0.58 (-1.01 to -0.15)** |
| Unseen GSM codec | 24.3% | 22.1% | 23.3% | **20.8%** | **-1.31 (-1.78 to -0.86)** |

Beam search helps every model, and it helps ours **more** than the original (unseen codec: -2.5 vs -2.1 points),
so with beam search our phone gain becomes significant on the simulated phone line too, and clean audio is no longer
slightly worse. Altogether, Lsir with beam search makes **14% fewer errors** than the original model as normally run
over the unseen codec (24.3% → 20.8%, CI of change -4.0 to -2.9). The cost is speed: on a T4, beam search took about
4x as long (about 27 vs 7 minutes for the 932 clips, about 1.7 vs 0.45 seconds per clip).

**Voting on top of beam search** (BuzzASR or Lsir with beam 5 as pivot, plus Omnilingual and MMS):

| Audio | Lsir beam 5 alone | Vote, Lsir beam 5 pivot | Change (strict 95% CI) |
|---|---|---|---|
| Clean | 12.3% | **11.6%** | -0.66 (-1.24 to -0.07) |
| Phone | 15.2% | **14.4%** | -0.88 (-1.41 to -0.37) |

Voting still helps, but less than with greedy decoding (beam search already fixes some of the same errors). The best
phone result, 14.4%, is about 17% fewer errors than the original BuzzASR with greedy decoding (17.4%).

*Strict confidence intervals resample whole sentences (FLEURS) or whole speakers (Common Voice) instead of single
clips (`tools/cluster_bootstrap.py`); all conclusions above hold under them.*

### Where it fails, and why we think so
- **Western Armenian:** Common Voice includes some Western Armenian sentences (classical spelling like «մէջ», «կը»). Our model rewrites them in Eastern spelling, which counts as errors.
- **Unusual words and names:** "Կասաբլանկան" (Casablanca) or "Կիվու" are often misheard; the model has rarely heard them.
- **Numbers:** FLEURS writes some numbers as digits and some as words, and models do both, so a correct number can
  still count as an error. Over the unseen mobile codec only about 72% of numbers survive with the original model.
- **Phone audio removes high frequencies**, so sounds like "ս", "շ", "ց" and "ծ" get confused.
- **What didn't work:** our first fine-tune (trained on punctuated transcripts, higher learning rate) made the model
  *worse* (clean 13.7% → 15.4%), because it learned to add punctuation and capitals instead of hearing better. Fine-tuning
  the decoder too was worse still. BuzzASR had already been trained on the same FLEURS sentences, so there was little new
  to learn from them.

**A note on fairness:** our first fine-tune was changed into v2 after we looked at its test outputs. To make sure we
didn't simply tune to the test set, we froze v2 and then tested it on a second dataset (Common Voice, section 6),
where the same gain appeared.

## What we'd do next
1. **Real phone calls:** record Armenian speakers (with permission) through real mobile and landline calls, and test
   whether the simulator's ranking of models holds.
2. **More, newer Armenian audio:** BuzzASR already learned FLEURS, so real gains need new speech (e.g. Common Voice
   Armenian or recordings of older speakers) and Western Armenian for the diaspora.
3. **Older speakers:** public Armenian speech data has almost nobody over 60. Recording older volunteers (with permission) is the most important missing piece.

## How to run
Every notebook below ran on a free Kaggle T4 GPU, and its **executed version with all outputs is public on Kaggle**.
The source of each is in `notebooks/` as a `.py` file with `# %%` cells (converted with `tools/py2nb.py`). Every
dataset and model is downloaded inside the notebook, and no API keys are needed.

| Notebook (source) | Executed, with outputs | What it does | Runtime |
|---|---|---|---|
| `notebooks/01_baseline.py` | [Kaggle](https://www.kaggle.com/code/davit2009/lsir-01-armenian-asr-baseline) | Whisper on Armenian vs Russian vs English, phone simulation | ~40 min |
| `notebooks/02_finetune.py` | partial results in `results/` | Fine-tuning whisper-small (interrupted run) | ~2.5 h |
| `notebooks/03_strong_models.py` | [Kaggle](https://www.kaggle.com/code/davit2009/lsir-03-armenian-asr-strong-models) | 4 models on all 932 test clips, clean and phone | ~1.5 h |
| `notebooks/03b_omnilingual.py` | [Kaggle](https://www.kaggle.com/code/davit2009/lsir-03b-omnilingual-armenian) | Meta Omnilingual ASR, same test | ~40 min |
| `notebooks/04_phone_finetune.py` | [Kaggle](https://www.kaggle.com/code/davit2009/lsir-04-phone-lora-encoder) | Phone-robust LoRA fine-tune of BuzzASR + 3-condition test | ~3.5 h |
| `notebooks/05_clinic_calls.py` | [Kaggle](https://www.kaggle.com/code/davit2009/lsir-05-clinic-calls) | Clinic-call test with a synthetic Armenian voice (Piper) | ~40 min |
| `notebooks/06_cv_and_beam.py` | [Common Voice](https://www.kaggle.com/code/davit2009/lsir-06-cv), [beam search](https://www.kaggle.com/code/davit2009/lsir-06-beam) | Second test set with age/gender groups; greedy vs beam search | ~1.5 h, ~3.5 h |
| `notebooks/07a_synthetic_speech.py` | [Kaggle](https://www.kaggle.com/code/davit2009/lsir-07a-synthetic-armenian-speech) | Synthetic Armenian training speech (Anadil, MMS-TTS), filtered by ASR | ~1.5 h |
| `tools/rover.py`, `tools/cluster_bootstrap.py` | run locally on the saved transcripts in `results/` | Word-level voting between models; sentence/speaker-level significance tests | seconds |
| `app/app.py` | [Live demo](https://huggingface.co/spaces/davitlive5/lsir-armenian-phone-asr) | Gradio demo, the same code as the Hugging Face Space | |

## Sources
- **FLEURS:** Conneau et al., 2022. https://huggingface.co/datasets/google/fleurs (CC-BY-4.0)
- **Whisper:** Radford et al., 2022. https://huggingface.co/openai/whisper-small, https://huggingface.co/openai/whisper-large-v3-turbo (MIT)
- **BuzzASR-Armenian:** Singh et al., 2026, *BuzzASR: A Swarm of 100+ Monolingual Speech Recognition Models*. https://huggingface.co/BuzzASR/armenian (MIT)
- **MMS:** Pratap et al., 2023. https://huggingface.co/facebook/mms-1b-all (CC-BY-NC-4.0)
- **Omnilingual ASR:** Meta AI, 2025. https://github.com/facebookresearch/omnilingual-asr (Apache-2.0)
- **Piper Armenian voice** `hy_AM-gor-medium` (davit312/piper-TTS-Armenian, GPL-2.0), used only to synthesise the clinic-call test. https://huggingface.co/davit312/piper-TTS-Armenian
- **Common Voice 20 Armenian** (Mozilla, CC0-1.0), via the mirror https://huggingface.co/datasets/Chillarmo/common_voice_20_armenian
- **Synthetic training speech:** Anadil Armenian TTS (https://huggingface.co/Anadilorg/Anadil_Armenian_TTS, MIT) on
  VoxCPM2 (https://huggingface.co/openbmb/VoxCPM2, Apache-2.0), and https://huggingface.co/facebook/mms-tts-hyw
  (CC-BY-NC-4.0); sentences from https://huggingface.co/datasets/EdUarD0110/daily_dialog_armenian (no licence stated)
- **ROVER** voting: Fiscus, 1997, *A post-processing system to yield reduced word error rates (ROVER)*, IEEE ASRU.
- **Methods:** LoRA (Hu et al., 2021) via Hugging Face PEFT; SpecAugment (Park et al., 2019); the telephone-augmentation
  mix follows Dissen et al. (arXiv 2406.18928) and data-augmentation work on low-bit-rate telephony (Applied Sciences 12(3):1580, 2022).
- **Libraries:** PyTorch, Transformers, PEFT, jiwer, SciPy, pandas, matplotlib, Gradio, ffmpeg.
- **Compute:** free Kaggle T4 GPUs.
- **AI tools:** We used Claude (Anthropic) as an AI coding assistant. It wrote and debugged most of the code and drafted the documents; we chose the problem, the user and the tests, checked every result, and can explain each step.

*This project is a prototype and is not medical advice.*
