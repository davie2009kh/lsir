"""Lsir demo: Armenian speech-to-text that keeps working over a phone line.

Compares the original BuzzASR-Armenian model with Lsir (BuzzASR + our phone-robust LoRA adapter), both with beam search.
Run anywhere with a GPU:  pip install -r requirements.txt && python app.py   (on a ZeroGPU Space this is the entry point)
"""
import inspect
import json
import os

import gradio as gr

try:                      # ZeroGPU on Hugging Face Spaces; a no-op decorator elsewhere
    import spaces
    gpu = spaces.GPU(duration=15)   # ~1-4 s needed; a small reservation lets logged-out visitors run more often
except ImportError:
    def gpu(f):
        return f

from lsir_core import HAS_GSM, landline, marked, mobile, to_16k_mono, transcribe_both

HERE = os.path.dirname(os.path.abspath(__file__))
EX_DIR = os.path.join(HERE, "examples")
EXAMPLES = json.load(open(os.path.join(EX_DIR, "examples.json"))) if os.path.exists(os.path.join(EX_DIR, "examples.json")) else []
LANDLINE, MOBILE, MIC = "Landline or internet call", "Mobile network (GSM)", "Microphone, no phone line"
CHANNELS = [MOBILE, LANDLINE, MIC] if HAS_GSM else [LANDLINE, MIC]

HERO = """
<div id="lsir-hero">
  <p class="word" lang="hy">Լսիր</p>
  <p class="gloss">“Listen!” in Armenian</p>
  <div class="pitch">
    <h1>Armenian speech-to-text that still understands you over a bad phone line</h1>
    <p>Clinics and pharmacies in Armenia are starting to answer calls with voice assistants, and many older callers
    speak only Armenian. A phone line throws away most of the sound of a voice, so even the best free Armenian model
    starts mishearing words. We trained it to hear through the line.</p>
    <svg class="band" viewBox="0 0 320 62" role="img" aria-label="A voice spans up to 8,000 hertz; a phone line keeps only 300 to 3,400 hertz">
      <text x="12" y="12" class="kept-label">What a phone line keeps</text>
      <text x="150" y="12">What it throws away</text>
      <rect class="full" x="0" y="18" width="320" height="16" rx="3"/>
      <rect class="kept" x="12" y="18" width="124" height="16" rx="3"/>
      <text x="0" y="50">0 Hz</text><text x="124" y="50">3,400</text><text x="286" y="50">8,000</text>
    </svg>
  </div>
</div>
"""

ABOUT = """
<p>We tested both models on 932 sentences from the FLEURS Armenian test set that were never used in training. Share
of words wrong (lower is better), both with beam search:</p>
<table>
<tr><th>Audio</th><th>BuzzASR</th><th>Lsir</th></tr>
<tr><td>Microphone</td><td>12.5%</td><td class="lsir-us">12.3%</td></tr>
<tr><td>Landline or internet call</td><td>15.8%</td><td class="lsir-us">15.2%</td></tr>
<tr><td>Mobile network (GSM), never used in training</td><td>22.1%</td><td class="lsir-us">20.8%</td></tr>
</table>
<p>Lsir is BuzzASR-Armenian (Whisper-large-v3 trained on Armenian) plus a small LoRA adapter on its audio encoder,
trained on a free Kaggle GPU with half the clips sent through randomised phone lines. The gain on the mobile codec it
never saw also holds on a second dataset (Common Voice Armenian, 1,000 clips). The three example clips here were
picked to show the difference; on average the gain is smaller.</p>
<p>Code, notebooks with outputs and the full write-up: <a href="https://github.com/davie2009kh/lsir">github.com/davie2009kh/lsir</a>.
Model: <a href="https://huggingface.co/davitlive5/lsir-buzzasr-armenian-phone-lora">davitlive5/lsir-buzzasr-armenian-phone-lora</a>.</p>
"""

FOOT = """<p class="lsir-foot">If a result box shows “Error”, Hugging Face's free GPU time for logged-out visitors has run
out for today: log in to Hugging Face (free) and try again.</p><p class="lsir-foot">A high-school hackathon prototype (Veritas AI 2026) by Alen Gevorgyan, Koryun Gaboyan and
David Ghazaryan. It is not a medical tool. Example clips: FLEURS (Google, CC-BY 4.0). Base model: BuzzASR/armenian (MIT).</p>"""

LEGEND = ('<p class="lsir-legend"><mark>Highlighted</mark>: wrong or extra word (hover to see the right one). '
          '<span class="lsir-del">Struck through</span>: word the model missed.</p>')


def card(kind, name, sub, body, score):
    s = f'<p class="lsir-score">{score[0]} of {score[1]} words wrong</p>' if score else ""
    return (f'<div class="lsir-card {kind}"><div class="lsir-card-head"><h3>{name}</h3><span>{sub}</span></div>'
            f'{body}{s}</div>')


ORIG = ("orig", "BuzzASR", "the best free Armenian model")
OURS = ("lsir", "Lsir", "ours: trained to hear through phone lines")
WAIT = '<p class="lsir-text lsir-wait">The transcript will appear here.</p>'


def empty_cards(msg=WAIT):
    return card(*ORIG, msg, None), card(*OURS, msg, None)


@gpu
def run(audio, channel, ref, row):
    if audio is None:
        msg = '<p class="lsir-text lsir-wait">Pick an example, or record or upload Armenian speech first.</p>'
        return (*empty_cards(msg), None, "")
    wav = to_16k_mono(audio)
    row = int(row) if row is not None and row >= 0 else 0     # same seeds as the notebooks for the example clips
    if channel == LANDLINE:
        wav = landline(wav, seed=row)
    elif channel == MOBILE:
        wav = mobile(wav, seed=50_000 + row)
    orig, lsir, secs = transcribe_both(wav)
    (b1, s1), (b2, s2) = marked(orig, ref), marked(lsir, ref)
    extra = ""
    if ref:
        extra = LEGEND + f'<div class="lsir-ref"><b>Correct transcript:</b> {ref}</div>'
    timing = f'<p class="lsir-time">Both models transcribed in {secs:.1f} s (beam search with 5 candidates).</p>'
    return card(*ORIG, b1, s1), card(*OURS, b2, s2), (16000, wav), extra + timing


theme = gr.themes.Base(
    primary_hue=gr.themes.colors.rose, neutral_hue=gr.themes.colors.stone,
    font=[gr.themes.GoogleFont("Noto Sans Armenian"), gr.themes.GoogleFont("Noto Sans"), "system-ui", "sans-serif"],
).set(
    body_background_fill="#f4ecea", body_background_fill_dark="#1d1a1f",
    block_background_fill="#ffffff", block_background_fill_dark="#27232a",
    block_border_color="#e2d4cf", block_border_color_dark="#3b3439",
    body_text_color="#24222a", body_text_color_dark="#f2ece9",
    color_accent="#8c1d3a", color_accent_soft="#f3dde2", color_accent_soft_dark="#43242d",
    button_primary_background_fill="#8c1d3a", button_primary_background_fill_dark="#e8708c",
    block_radius="12px", block_shadow="none",
)
CSS = open(os.path.join(HERE, "style.css")).read()
STYLE = dict(theme=theme, css=CSS)
BLOCKS_STYLE = STYLE if "css" in inspect.signature(gr.Blocks.__init__).parameters else {}   # Gradio 6 moved these
LAUNCH_STYLE = {} if BLOCKS_STYLE else STYLE

with gr.Blocks(title="Lsir: Armenian speech-to-text over the phone", **BLOCKS_STYLE) as demo:
    ref, row = gr.State(""), gr.State(-1)
    gr.HTML(HERO)

    gr.HTML('<div class="lsir-step-title"><span class="n">1</span><h2>Choose some Armenian speech</h2>'
            '<p>an example sentence, or your own voice (up to 30 seconds)</p></div>')
    chips = []
    with gr.Row(elem_id="lsir-chips"):
        for e in EXAMPLES:
            chips.append(gr.Button(f"«{e['label']}…» ({e['seconds']:.0f} s)", elem_classes="lsir-chip", size="sm", scale=0, min_width=0))
    audio = gr.Audio(sources=["microphone", "upload"], type="numpy", label="Armenian speech", show_label=False)

    gr.HTML('<div class="lsir-step-title"><span class="n">2</span><h2>Send it down a phone line</h2>'
            '<p>the mobile codec is the one Lsir never heard in training</p></div>')
    with gr.Row(equal_height=True):
        channel = gr.Radio(CHANNELS, value=CHANNELS[0], show_label=False, elem_id="lsir-channel", scale=1)
        go = gr.Button("Transcribe", elem_id="lsir-go", scale=0, min_width=200)

    with gr.Row(equal_height=True):
        out_orig = gr.HTML(empty_cards()[0])
        out_lsir = gr.HTML(empty_cards()[1])
    notes = gr.HTML("")
    heard = gr.Audio(label="What the models heard (after the phone line)", interactive=False)

    with gr.Accordion("How we tested it", open=False, elem_id="lsir-about"):
        gr.HTML(ABOUT)
    gr.HTML(FOOT)

    for i, (b, e) in enumerate(zip(chips, EXAMPLES)):
        b.click(lambda e=e: (os.path.join(EX_DIR, e["file"]), e["raw_transcription"], e["row"]),
                outputs=[audio, ref, row])
    for ev in (audio.upload, audio.stop_recording):
        ev(lambda: ("", -1), outputs=[ref, row])
    go.click(run, inputs=[audio, channel, ref, row], outputs=[out_orig, out_lsir, heard, notes])

if __name__ == "__main__":
    demo.launch(allowed_paths=[EX_DIR], **LAUNCH_STYLE)
