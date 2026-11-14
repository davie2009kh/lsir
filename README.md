# Lsir (Լսիր): speech-to-text that understands Armenian, even over the phone

**Track:** AI + Languages
**Team:** {{TEAM NAMES}}

*"Lsir" (Լսիր) means "listen!" in Armenian.*

## The problem
About 3 million people in Armenia, and millions more in the diaspora, speak Armenian, but free
speech-to-text models are trained mostly on English. Clinics, pharmacies and banks are starting to answer their
phones with voice assistants. When the speech recogniser doesn't understand Armenian, the people who lose out are
older callers who speak only Armenian, the people who most need the phone line to work.

## What we built
Lsir takes a recording of Armenian speech (from a microphone or a phone line) and returns the Armenian text.
It's a version of OpenAI's `whisper-small` that we fine-tuned on Armenian, including simulated phone audio.
Our demo shows the original Whisper and Lsir side by side, with a switch that passes the audio through a phone
line first.

{{DEMO EXAMPLE}}

## What we added
We started from suggested project **4.4.3 Speech recognition across accents** and added:
1. **Our own user and setting:** an Armenian-only caller to a clinic or pharmacy phone line. So we built a
   **phone-line simulation** (8 kHz, 300–3,400 Hz, 8-bit μ-law, line noise) and tested every model on
   phone versions of the same clips, a question the brief doesn't ask.
2. **A fix that we measured:** fine-tuning Whisper on Armenian, and a second version trained with phone-line
   augmentation.
3. **Our own question about *why* it fails:** Whisper's tokenizer needs {{TOK_HY}} tokens per Armenian word vs
   {{TOK_EN}} for English, and Whisper often fails to recognise Armenian as a language at all.

## How it works
**Data:** [FLEURS](https://huggingface.co/datasets/google/fleurs) (Google), Armenian (`hy_am`): 3,053 training
clips (10.4 h), 395 validation and 932 test clips (3.0 h) read by native speakers. FLEURS splits contain
**different sentences** (we checked: 0 sentences shared between training and test). Russian and English FLEURS test
clips are used only as reference points.

**Models:**
- Baseline: `openai/whisper-small` as-is (and the trivial "empty output" baseline, which scores exactly 100% WER).
- Comparisons: `openai/whisper-large-v3-turbo` (3.3× bigger) and Meta's `facebook/mms-1b-all` (Armenian adapter).
- Ours: `whisper-small` fine-tuned for {{EPOCHS}} epochs on the FLEURS Armenian training clips with a plain PyTorch
  loop (AdamW, learning rate 1e-5, batch 16, fp16), picking the best epoch on validation clips. Version B
  turns 50% of training clips into simulated phone audio.

**Metric:** word error rate (WER) and character error rate (CER) after lower-casing and removing punctuation.

## Results
{{RESULTS TABLE}}

**Where it fails.** {{FAILURES}}

**Fairness check.** {{GENDER}}

## What we'd do next
{{NEXT}}

## Sources
- FLEURS: Conneau et al., 2022, *FLEURS: Few-shot Learning Evaluation of Universal Representations of Speech*.
  https://huggingface.co/datasets/google/fleurs (CC-BY-4.0)
- Whisper: Radford et al., 2022. https://huggingface.co/openai/whisper-small,
  https://huggingface.co/openai/whisper-large-v3-turbo (MIT)
- MMS: Pratap et al., 2023. https://huggingface.co/facebook/mms-1b-all (CC-BY-NC-4.0)
- Libraries: PyTorch, Hugging Face Transformers, jiwer, SciPy, pandas, Gradio. Hugging Face's
  "Fine-Tune Whisper" blog post for the label-preparation step.
- Compute: free Kaggle T4 GPU.
- AI tools: {{AI TOOLS}}

*This project is a prototype. It is not a medical, legal or financial tool and shouldn't be relied on for anything important.*
