# Results not produced by a completed notebook run

`nb02_whisper_small_partial.csv`: notebook 2 (whisper-small fine-tune) ran interactively on Kaggle on 2026-11-14 and
was interrupted after epoch 4 of 5 of run A (clean audio) when the laptop running the browser session went to sleep.
Numbers were copied from the notebook's printed output. They are **validation** scores (200 FLEURS Armenian
validation clips), not test scores. Run B (phone augmentation) did not run.

`nb04_v1_all/lora_all_training_history.csv`: notebook 4 v1 with `LORA_PART="all"` (encoder+decoder LoRA, r=32,
lr 2e-4, trained on punctuated `raw_transcription`). Validation got worse at every epoch, so the run kept
epoch 0 (= the unchanged original model); the run was stopped before its test phase, which would only have compared
two identical models. Numbers copied from the Kaggle run log (100 validation clips).

`nb04_v1_encoder/`: notebook 4 v1 with `LORA_PART="encoder"`, complete outputs (test on all 932 clips).

`nb04_v2_encoder/`: notebook 4 **final** version (encoder-only LoRA r=16, lr 5e-5, trained on FLEURS's
normalised `transcription`, epoch chosen on 300 validation clips). Complete outputs, all 932 test clips.
