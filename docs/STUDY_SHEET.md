# Lsir: study sheet for the team (read before the finals)

The rules say the team must understand every part of the project and be able to explain it. This sheet covers
every idea used, in plain words, then the questions judges are most likely to ask, with good answers.

## 1. The one-sentence pitch
Lsir helps an Armenian-only caller to a clinic or pharmacy phone line be understood by a voice assistant. It takes
their phone audio and returns the Armenian text, using a speech model we trained to cope with phone-line sound.

## 2. Core ideas, in plain words

**Speech-to-text (ASR).** A model hears audio and writes what was said.

**Whisper.** OpenAI's free ASR model. It turns audio into a *log-mel spectrogram* (a picture of which frequencies
are loud at each moment), an **encoder** reads that picture, and a **decoder** writes the text one *token* at a time.

**Tokens.** Pieces of words. Whisper's original tokenizer was built mostly from English, so one Armenian word
costs about **8.8 tokens**, vs **1.3** for English. More steps means more chances to go wrong and more loops.

**BuzzASR-Armenian.** Whisper-large-v3 (1.55 billion parameters) that the BuzzASR project fine-tuned only on
Armenian, with a new Armenian tokenizer. It's the best free Armenian model we found.

**MMS and Omnilingual (Meta).** Different design (wav2vec2 + CTC): they predict one letter per short slice of
audio instead of writing token by token. Fast, and they can't get stuck in loops.

**WER (word error rate).** (substituted + deleted + inserted words) ÷ words in the correct transcript.
0% is perfect. It can go **above 100%** when the model invents many extra words, which is why whisper-small scored
111% on Armenian: worse than outputting nothing (which scores exactly 100%).

**CER (character error rate).** The same, per letter. Fairer for Armenian, whose long words carry endings: one
wrong ending makes a whole word "wrong" in WER but is only one or two letters in CER.

**Normalisation.** Before scoring we lower-case both texts and remove punctuation, so "Բարև։" and "բարև" count
as the same word.

**Train / validation / test split.** The model learns from *train*, we choose settings (like which epoch to keep)
with *validation*, and we report results on *test* exactly once. FLEURS splits use **different sentences**
(we checked: 0 shared), so the test is on sentences the model never trained on.

**Fine-tuning.** Continuing to train an existing model on new data (here: Armenian audio, some of it made to
sound like a phone call).

**LoRA.** Instead of changing all 1.55B weights (impossible on a free 16 GB GPU), we freeze them and add small
trainable matrices inside some layers. Only about 1-2% of the parameters are trained. It's fast, cheap and hard
to damage: if training doesn't help, the original model is still there underneath.

**Encoder-only LoRA vs all.** A phone line changes the *sound*, not the language, so we tried adapting only the
encoder (the "ears") and compared it with adapting encoder and decoder.

**Epoch.** One pass over all training clips. We check validation after each epoch and keep the best one.
"Epoch 0" is the original model, so our choice can never be worse than the starting point on validation.

**Data augmentation.** Making new training examples by changing existing ones. Ours imitates phone calls:
- *8 kHz, narrow band*: phones carry only roughly 300-3,400 Hz, so high sounds like "s", "sh" and "ts" get muffled
- *Codecs*: G.711 μ-law and A-law (landline/VoIP), Opus (WhatsApp-style). A codec compresses audio and loses detail
- *Packet loss*: 20 ms gaps, as on a bad mobile connection
- *Noise*: white or pink hiss, and **babble** (other people talking in the background)
- *Reverberation* (room echo), *speed changes* (±10%), *volume changes*
- *SpecAugment*: hiding small time and frequency stripes of the spectrogram so the model can't rely on any one detail

**The unseen test.** We test on a mobile codec (GSM, the classic mobile-phone codec) that is **never used in
training**. If our model improves there too, it learned "phone audio" in general, not just our simulator.

**Bootstrap confidence interval.** We resample the 932 test clips 1,000 times with replacement and recompute WER.
The middle 95% of those values is the 95% confidence interval: how much the score could move by chance.
A **paired** bootstrap resamples the same clips for both models to check that "ours is better" isn't luck.

**ROVER (system combination).** Three different models transcribe the same audio, we line their words up, and
each word position takes the majority vote (ties keep our model's word). Models built differently (Whisper-style vs
CTC) make *different* mistakes, so voting cancels many of them.

**Baseline.** The simplest thing to compare against. Ours: the empty output (exactly 100% WER), whisper-small
as most apps would use it, and the original BuzzASR before our training.

## 3. Our main findings (fill in the final numbers from the README)
1. Off-the-shelf whisper-small is unusable for Armenian (111% WER vs 9% for English on the same sentences).
2. The best free Armenian model (BuzzASR) is far better, but it still loses accuracy over a phone line.
3. Our phone-trained LoRA reduces errors on phone audio, including a codec it never saw, while keeping clean
   audio about the same.
4. On FLEURS every model makes more mistakes on women's voices, but with Common Voice speaker IDs the gap disappears,
   so it's probably a few individual speakers. Public data has no Armenian speakers over 59.
5. Letting three different models vote word by word (ROVER) gave the best result: FLEURS clean 13.6% -> 11.9%,
   phone 17.4% -> 15.1% (about 12-13% fewer errors), with no extra training.

## 4. Questions judges will probably ask

**"How do you know it works?"** We tested on 932 FLEURS test clips with sentences never used in training, under
three conditions (clean, phone, unseen mobile codec), with 95% confidence intervals and a paired bootstrap against
the original model. Plus a separate clinic-call test with numbers and medicine names.

**"Why not just use the biggest model?"** We compared them. Bigger general models (whisper-large-v3-turbo) are
still much worse than a model trained specifically for Armenian. Size isn't enough; language-specific data matters.

**"Isn't BuzzASR doing all the work?"** BuzzASR solved clean Armenian. Nobody had measured or fixed Armenian *over
the phone*, which is what a clinic line actually receives. Our contributions: the phone-line evaluation, the
augmentation pipeline, the phone-robust LoRA, the unseen-codec test, the clinic-call test and the gender check.

**"Your phone audio is simulated. Would it work on real calls?"** It's the main limitation. We reduced the risk by
mixing many channel types and testing on a codec we never trained on. The next step is testing on real calls,
recorded with permission.

**"What would cause it to fail?"** Very noisy calls, strong regional accents or Western Armenian (FLEURS is Eastern
Armenian), people talking over each other, names and numbers (we measure these separately), and very long calls
(the model listens in 30-second chunks).

**"Are the strong models' scores fair?"** MMS, Omnilingual and BuzzASR were trained partly on FLEURS's training
split. The test sentences are different, but the recording style is familiar to them, which may flatter all three
a little. We say so in the write-up.

**"Why does voting help if every model is weaker alone?"** Because their mistakes don't overlap much: a CTC model
(MMS/Omnilingual) spells sounds letter by letter and never loops, while our Whisper-style model knows Armenian words
better. Where one slips, the other two usually agree on the right word. The cost is running three models.

**"Why are women's voices worse?"** On FLEURS every model was worse for women, but FLEURS doesn't say how many
different speakers there are. On Common Voice, which has speaker IDs (73 women, 49 men in our sample), women were
transcribed slightly *better*. So the FLEURS gap is most likely caused by a few individual speakers, not by gender
itself. That's why we tested on a second dataset before drawing a conclusion.

**"Does it work for older people?"** We couldn't measure it: the Common Voice Armenian test set has no speakers over
59. That's a real gap in public data, and recording older volunteers (with permission) is our top next step.

**"Who would use this?"** A clinic, pharmacy or bank voice line in Armenia, so that older Armenian-only callers can
book appointments or ask about prescriptions without being misunderstood.

**"Is it a medical tool?"** No. It's a prototype and not medical advice. A real deployment would need testing with
real callers and a human fallback for anything urgent.

**"What did you use AI tools for?"** We used Claude (an AI coding assistant) to help write and debug code and to
draft documents. We chose the problem and the tests, checked the results, and can explain every step. (Adjust this
answer to what you actually did.)
