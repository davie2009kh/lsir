# Lsir: 2-minute demo video script

Format: a screen recording with your voice over it (OBS, Loom or the built-in recorder). Upload to YouTube as
**unlisted** or to Google Drive as "anyone with the link can view", then check the link in a logged-out window.
Numbers in {{braces}} get filled in from the final results table in the README.

---

### 0:00-0:15 · The problem (show: title slide or the README header)
> "Many older people in Armenia speak only Armenian. *(If you have a real family example, say it here; only say
> what's true.)* More and more clinics and pharmacies answer the phone with a voice
> assistant, and those assistants run on speech-to-text models trained mostly on English. We asked: can today's
> free models understand Armenian, and do they still work over a phone line?"

### 0:15-0:45 · What we found (show: the `wer_by_language.png` and `strong_models_wer.png` charts)
> "On the same FLEURS sentences, Whisper-small makes 9% word errors in English, 14% in Russian, and 111% in
> Armenian. That's worse than writing nothing, because it gets stuck in loops. Its tokenizer needs 8.8 pieces
> for every Armenian word, against 1.3 for English. The best free Armenian model, BuzzASR, gets 13.6% on a
> good microphone, but over a phone line that becomes 17.4%, and 24% over a mobile codec."

### 0:45-1:15 · The demo (show: the Gradio app)
1. Play or upload an Armenian clip (e.g. a clinic sentence) with the **phone line** switch on.
2. Point at the two outputs: *"The original model writes ... Ours writes ..."*
3. Show one clip where a **number** (a time or phone number) is wrong in the original and right in ours, if there is one.

### 1:15-1:45 · How it works (show: `channels_spectrogram.png`, then `lora_encoder_wer.png`)
> "We built a phone-line simulator: narrow band, real codecs like G.711 and Opus, packet loss, background voices
> and echo. We used it to train small LoRA adapters on BuzzASR with the free FLEURS Armenian training data, only
> about 1.5% of the model, on a free Kaggle GPU. To be fair, we tested on a mobile codec the model never saw
> in training. There, word errors dropped from 24.3% to 23.3%, a small but statistically significant gain, and
> the share of numbers that survive went from 72% to 78%. Clean audio stayed the same."

### 1:45-2:00 · Limits and next step (show: README "What we'd do next")
> "Our phone audio is simulated and FLEURS is read speech, not real calls. And every model we tested makes more
> mistakes on women's voices, which we report but can't fully explain. Next, we'd test with real clinic calls,
> recorded with permission. This is a prototype, not a medical tool."

---

**Tips**
- Rehearse once with a timer; 2:00 is a hard limit.
- Keep the app already open and the example clip ready before you press record.
- If the live app is slow, record the demo part separately and cut it in.
