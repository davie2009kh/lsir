# Lsir (Լսիր): Armenian speech-to-text that works over the phone

**Track:** Languages  
**Team:** Alen Gevorgyan, Koryun Gaboyan, David Ghazaryan  
**Live demo:** https://huggingface.co/spaces/davitlive5/lsir-armenian-phone-asr · **Model:** https://huggingface.co/davitlive5/lsir-buzzasr-armenian-phone-lora · **Full report:** [docs/REPORT.md](docs/REPORT.md)  
**Code:** this repository; every notebook's executed version with outputs is public on Kaggle (links in [docs/REPORT.md#how-to-run](docs/REPORT.md#how-to-run))

*This project is a prototype and is not medical advice.*

## The problem
Many older people in Armenia speak only Armenian. Clinics and pharmacies are starting to answer the phone with voice
assistants, but those run on speech-to-text models built mostly for English, and nobody has measured how well they
understand Armenian **over a phone line**. The callers who most need help are the ones most likely to be misunderstood.

## What we built
A caller's phone audio goes in and Armenian text comes out. Our demo lets anyone speak or upload Armenian, optionally
send it through a simulated phone line, and see the original model and ours side by side.

## What we added
Starting point: suggested project 4.4.3 *Speech recognition across accents*. We added our own user (an Armenian-only
caller to a clinic line) and our own question: does Armenian speech recognition survive a phone call? To answer it we
built a telephone simulator (narrow band, landline/VoIP codecs, packet loss, background chatter, echo), trained a
phone-robust model, and wrote our own 40-sentence clinic-call test (appointments, phone numbers, doses, medicine names).

## How it works
**Data:** FLEURS Armenian (3,053 training clips, 10.4 h; 932 test clips with sentences never seen in training) and
1,000 Common Voice Armenian test clips as a second, independent check. **Model:** we compared 5 free models and picked
the best, BuzzASR-Armenian (Whisper-large-v3 trained on Armenian). We added a small LoRA adapter to its audio
"ears" (about 1.5% of the weights) and trained it on a free Kaggle GPU, with half the clips sent through random phone
lines. To keep ourselves honest, we test on a mobile codec (GSM) that is **never used in training**.

## Results (word error rate, lower is better)
| Test audio (FLEURS, 932 clips) | whisper-small | BuzzASR | **Lsir (ours)** | **Lsir + 3-model vote** |
|---|---|---|---|---|
| Clean | 108% | 12.5% | 12.3% | **11.6%** |
| Phone line | 118% | 15.8% | **15.2%** | **14.4%** |
| **Unseen mobile codec** | — | 22.1% | **20.8%** | {{ROVER_GSM}} |

*BuzzASR and Lsir use beam search (5 beams); whisper-small is run as most apps run it.* The usual baseline,
whisper-small, is worse than writing nothing (100%). Against the original BuzzASR, our model is significantly better
over the phone (-0.6 points, 95% CI -1.0 to -0.15) and on the codec it never trained on (-1.3, CI -1.8 to -0.9;
strict test that resamples whole sentences), and equal on clean audio. The codec gain **replicates on Common Voice**
(18.9% → 17.8%). Over that codec, numbers survive more often (72% → 78%), and on our clinic test more medicine names
come through (33% → 42%). Letting three different models vote word by word removes another 5-6% of errors. We chose
the voting setup on the FLEURS test, so we treat it as exploratory until it is confirmed on Common Voice.

**Where it fails:** on clean audio our model is no better (slightly worse on Common Voice: 3.7% → 4.0%, greedy decoding). Western
Armenian spelling, rare names and sounds lost on phone lines ("ս", "շ", "ց") cause most errors. Our phone audio is
simulated and our clinic voice is synthetic. Our first fine-tune made the model worse, because it learned punctuation
instead of sounds. The FLEURS gender gap disappeared on Common Voice, and public data has no Armenian speakers over 59.

## What we'd do next
1. Test on real phone calls from Armenian speakers, recorded with permission.
2. Record older volunteers, the people this is for, who are missing from public data.

## Sources
FLEURS (Google, CC-BY-4.0); Common Voice 20 Armenian (Mozilla, CC0); BuzzASR/armenian (MIT); OpenAI Whisper (MIT);
Meta MMS and Omnilingual ASR; Piper Armenian voice (davit312/piper-TTS-Armenian) for the clinic test; PEFT/LoRA;
ROVER (Fiscus, 1997). Full list with links: [docs/REPORT.md#sources](docs/REPORT.md#sources).
**AI tools:** We used Claude (Anthropic) as an AI coding assistant. It wrote and debugged most of the code and drafted the documents; we chose the problem, the user and the tests, checked every result, and can explain each step.
