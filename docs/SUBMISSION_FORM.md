# Submission form answers (copy-paste)

Submission form deadline: **Sun Nov 15, 11:59 PM ET = Mon Nov 16, 08:59 Yerevan.** Submit a first version
early; you can resubmit as often as you like.

| Form question | Answer |
|---|---|
| Project title | Lsir (Լսիր): Armenian speech-to-text that works over the phone |
| Building as | Solo / Team: *fill in* |
| Team members | *Each member's name, email and school* |
| Track | **AI + Languages** |
| Starting point | 4.4.3 Speech recognition across accents (extended with our own user, data and question) |
| One-sentence pitch | Lsir helps an Armenian-only caller to a clinic or pharmacy phone line by taking their phone audio and returning accurate Armenian text, using a speech model we trained to cope with phone-line sound. |
| Code link | *GitHub repo link (public), check it in a logged-out window* |
| Write-up link | The GitHub README (same repo) |
| Demo video link | *YouTube unlisted / Google Drive link* |
| What we added | (1) **Our own user and question:** Armenian over a phone line, which no published benchmark measures. We built a telephone-channel simulator (narrow band, G.711/Opus codecs, packet loss, babble noise, echo) and compared 5 free models on 932 test clips, clean and over the phone. (2) **Our own model:** phone-robust LoRA fine-tuning of the best Armenian model (BuzzASR), tested on a mobile codec never used in training. (3) **Our own data:** a 40-sentence clinic-call test set (appointments, phone numbers, doses, medicine names) spoken by a synthetic Armenian voice (Piper) at two speaking paces, scored on whether numbers and medical terms survive. (4) A gender check: every model makes more errors on women's voices. |
| Datasets, models and AI tools used | FLEURS (Google, CC-BY-4.0); openai/whisper-small and whisper-large-v3-turbo; facebook/mms-1b-all; Meta Omnilingual ASR (omniASR_CTC_1B_v2); BuzzASR/armenian; PEFT/LoRA; Piper Armenian TTS voice (davit312/piper-TTS-Armenian) for the synthetic clinic test; Kaggle free T4 GPUs. AI tools: Claude (Anthropic) as a coding assistant for writing and debugging code and drafting documents. *Adjust to describe exactly how you used it.* |
| New to AI? | *Your answer* |
