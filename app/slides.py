"""Five presentation slides for the finals, as plain HTML + CSS (no JavaScript, so they work inside gr.HTML).
Slides scroll sideways with snap; the arrows are in-page links. tools: build a standalone copy with build_standalone()."""

FONT_LINK = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Noto+Sans:wght@400;600;800'
             '&family=Noto+Sans+Armenian:wght@400;700&family=Noto+Serif+Armenian:wght@300;900&display=swap">')


def nav(i, n=6):
    prev = f'<a href="#sl{i-1}" aria-label="Previous slide">&#8592;</a>' if i > 1 else '<span></span>'
    nxt = f'<a href="#sl{i+1}" aria-label="Next slide">&#8594;</a>' if i < n else '<span></span>'
    return f'<div class="dk-nav">{prev}<span class="dk-count">{i} / {n}</span>{nxt}</div>'


SLIDES = f"""
<div id="lsir-deck">
 <div class="dk-track">

  <section class="dk-slide dk-title" id="sl1">
    <p class="dk-word" lang="hy">Լսիր</p>
    <div class="dk-title-text">
      <p class="dk-gloss">“Listen!” in Armenian</p>
      <h2>Armenian speech-to-text that still understands you over a phone line</h2>
      <p class="dk-team">Alen Gevorgyan, Koryun Gaboyan, David Ghazaryan<br><span>Veritas AI Hackathon 2026, Languages track</span></p>
    </div>
    {nav(1)}
  </section>

  <section class="dk-slide dk-problem" id="sl2">
    <h2>The voice assistant can’t hear her</h2>
    <p class="dk-lede">Clinics in Armenia are starting to answer calls with voice assistants. Many older callers speak only
    Armenian. The speech model most apps use (Whisper-small) gets this many words wrong:</p>
    <div class="dk-vs">
      <div><p class="dk-big ink">9%</p><p class="dk-cap">English</p></div>
      <div><p class="dk-big pom">110%</p><p class="dk-cap">Armenian: worse than writing nothing</p></div>
    </div>
    <p class="dk-kicker">And nobody had tested Armenian speech recognition over a phone line.</p>
    {nav(2)}
  </section>

  <section class="dk-slide dk-how" id="sl3">
    <h2>What we built</h2>
    <ol class="dk-flow">
      <li><b>Phone-line simulator</b><span>Turns clean speech into a call: narrow band, codecs, packet loss, noise</span></li>
      <li><b>5 free models compared</b><span>932 test recordings, clean and over the phone. Best: BuzzASR-Armenian</span></li>
      <li><b>Our adapter</b><span>1.5% of the model’s weights, trained on phone audio on a free GPU</span></li>
      <li><b>Live demo</b><span>Anyone can try it, on this page</span></li>
    </ol>
    <p class="dk-note">Fair test: one mobile codec (GSM) was kept out of training entirely.</p>
    {nav(3)}
  </section>

  <section class="dk-slide dk-tech" id="sl4">
    <h2>Under the hood</h2>
    <div class="dk-pipe" role="img" aria-label="Phone audio goes into the encoder with our LoRA adapters, then the frozen decoder, then beam search, then Armenian text">
      <div class="st"><b>Phone audio</b><span>log-mel spectrogram, 30 s windows</span></div><i>&#8594;</i>
      <div class="st hot"><b>Encoder: listens</b><span>32 layers + our LoRA adapters (trained)</span></div><i>&#8594;</i>
      <div class="st"><b>Decoder: writes</b><span>32 layers, frozen</span></div><i>&#8594;</i>
      <div class="st"><b>Beam search</b><span>5 candidates</span></div><i>&#8594;</i>
      <div class="st"><b>Armenian text</b><span lang="hy">Բարև ձեզ…</span></div>
    </div>
    <div class="dk-spec">
      <div><h3>Model</h3><p>BuzzASR-Armenian: OpenAI Whisper-large-v3 fine-tuned on Armenian. Encoder-decoder transformer, 1.55 billion parameters.</p></div>
      <div><h3>Fine-tuning</h3><p>LoRA (Hugging Face PEFT) on the encoder only: attention + feed-forward layers. Rank 16, alpha 32, dropout 0.05: about 1.5% of the weights.</p></div>
      <div><h3>Training</h3><p>3,053 FLEURS Armenian clips, 2 epochs, learning rate 5e-5, 16 clips per update, free Kaggle T4. Half the clips sent through a random phone line.</p></div>
      <div><h3>Decoding</h3><p>Whisper’s own decoder with beam search (5 beams), no repeated 3-word phrases, repetition penalty 1.2.</p></div>
    </div>
    <p class="dk-note">Why only the encoder? A phone line damages the sound, not the language. Adapting the decoder too made results worse.</p>
    {nav(4)}
  </section>

  <section class="dk-slide dk-results" id="sl5">
    <h2>On a phone codec it never heard, errors drop</h2>
    <div class="dk-bars" role="img" aria-label="Word error rate on the never-seen mobile codec: original 22.1 percent, Lsir 20.8 percent">
      <div class="dk-bar"><span class="lbl">Original BuzzASR</span><span class="fill orig" style="width:{22.1/25*100:.1f}%"></span><span class="val">22.1%</span></div>
      <div class="dk-bar"><span class="lbl">Lsir (ours)</span><span class="fill lsir" style="width:{20.8/25*100:.1f}%"></span><span class="val">20.8%</span></div>
      <p class="dk-axis">Word error rate, 932 FLEURS test recordings, both with beam search. Lower is better.</p>
    </div>
    <div class="dk-facts">
      <p><b>18.9% → 17.8%</b>The gain repeats on a second dataset (Common Voice)</p>
      <p><b>72% → 78%</b>Numbers (doses, phone numbers) that survive the call</p>
      <p><b>33% → 42%</b>Medicine names right on our clinic-call test</p>
    </div>
    {nav(5)}
  </section>

  <section class="dk-slide dk-next" id="sl6">
    <h2>Honest about what’s missing</h2>
    <div class="dk-cols">
      <div><h3>Didn’t work</h3><p>Three models voting word by word: helped on one dataset, failed on the second, so we dropped it.
      More training data: no gain.</p></div>
      <div><h3>Limits</h3><p>Phone audio is simulated. Public Armenian speech data has nobody over 59, the people we build for.</p></div>
      <div><h3>Next</h3><p>Test on real calls and record older volunteers, with permission.</p></div>
    </div>
    <p class="dk-end"><span lang="hy">Լսիր</span> Technology that listens to everyone.</p>
    <p class="dk-small">Model, demo and every notebook are public. Built with Claude (Anthropic) as a coding assistant; we checked every result. Prototype, not medical advice.</p>
    {nav(6)}
  </section>

 </div>
</div>
"""

CSS = """
#lsir-deck { --t:#f4ecea; --i:#24222a; --m:#6b6370; --p:#8c1d3a; --l:#e2d4cf; --bo:#2e6db5; --bl:#b8344f;
  border-radius: 14px; overflow: hidden; container-type: inline-size; background: var(--t); color: var(--i);
  font-family: "Noto Sans", "Noto Sans Armenian", system-ui, sans-serif; }
#lsir-deck * { box-sizing: border-box; }
#lsir-deck .dk-track { display: flex; overflow-x: auto; scroll-snap-type: x mandatory; scroll-behavior: smooth; scrollbar-width: none; }
#lsir-deck .dk-track::-webkit-scrollbar { display: none; }
#lsir-deck .dk-slide { flex: 0 0 100%; scroll-snap-align: start; position: relative; aspect-ratio: 16 / 9;
  padding: 4% 6% 9%; display: flex; flex-direction: column; justify-content: center; background: var(--t); overflow: hidden; scroll-margin-top: 60px; }
#lsir-deck h2 { font-size: clamp(25px, 4.42cqw, 70px); line-height: 1.12; font-weight: 800; margin: 0 0 .6em; letter-spacing: -0.01em; color: var(--i); }
#lsir-deck p { margin: 0; }
#lsir-deck .dk-nav { position: absolute; left: 6%; right: 6%; bottom: 3%; display: flex; justify-content: space-between; align-items: center; }
#lsir-deck .dk-nav a { text-decoration: none; color: var(--p); font-size: clamp(20px, 2.60cqw, 39px); font-weight: 800;
  width: 2em; height: 2em; display: grid; place-items: center; border-radius: 50%; border: 2px solid var(--l); background: #fff; }
#lsir-deck .dk-nav a:hover { border-color: var(--p); }
#lsir-deck .dk-count { color: var(--m); font-size: clamp(13px, 1.43cqw, 20px); }

/* 1 title: the word is the hero, pomegranate field */
#lsir-deck .dk-title { background: var(--p); color: #fff; flex-direction: row; align-items: center; gap: 5%; }
#lsir-deck .dk-word { font-family: "Noto Serif Armenian", serif; font-weight: 900; font-size: clamp(92px, 20.80cqw, 338px); line-height: .9; color: #fff; }
#lsir-deck .dk-title h2 { color: #fff; font-size: clamp(23px, 3.77cqw, 59px); }
#lsir-deck .dk-gloss { color: #f3c9d3; font-size: clamp(16px, 1.95cqw, 28px); margin-bottom: .6em; }
#lsir-deck .dk-team { color: #fff !important; font-size: clamp(14px, 1.82cqw, 27px); font-weight: 600; line-height: 1.5; }
#lsir-deck .dk-team span { font-weight: 400; color: #f3c9d3; }
#lsir-deck .dk-title .dk-nav a { background: transparent; border-color: rgba(255,255,255,.4); color: #fff; }
#lsir-deck .dk-title .dk-count { color: #f3c9d3; }

/* 2 problem */
#lsir-deck .dk-lede { font-size: clamp(16px, 2.02cqw, 31px); line-height: 1.45; max-width: 60ch; color: var(--i); }
#lsir-deck .dk-vs { display: flex; gap: 8%; margin: 3% 0 2.5%; align-items: flex-end; }
#lsir-deck .dk-big { font-size: clamp(62px, 13.00cqw, 221px); font-weight: 800; line-height: .95; letter-spacing: -0.03em; }
#lsir-deck .dk-big.ink { color: var(--m); }
#lsir-deck .dk-big.pom { color: var(--p); }
#lsir-deck .dk-cap { font-size: clamp(14px, 1.69cqw, 26px); color: var(--m); margin-top: .3em; }
#lsir-deck .dk-kicker { font-size: clamp(17px, 2.21cqw, 33px); font-weight: 700; border-left: 5px solid var(--p); padding-left: .7em; }

/* 3 how: a numbered flow, because it is a real sequence */
#lsir-deck .dk-flow { list-style: none; counter-reset: s; margin: 0; padding: 0; display: grid; grid-template-columns: repeat(4, 1fr); gap: 2.4%; }
#lsir-deck .dk-flow li { counter-increment: s; position: relative; padding-top: 1.1em; border-top: 4px solid var(--l); }
#lsir-deck .dk-flow li:nth-child(3) { border-top-color: var(--p); }
#lsir-deck .dk-flow li::before { content: counter(s); position: absolute; top: -0.85em; left: 0; width: 1.6em; height: 1.6em; border-radius: 50%;
  background: var(--i); color: #fff; display: grid; place-items: center; font-weight: 800; font-size: clamp(13px, 1.56cqw, 23px); }
#lsir-deck .dk-flow li:nth-child(3)::before { background: var(--p); }
#lsir-deck .dk-flow b { display: block; font-size: clamp(16px, 2.15cqw, 33px); margin: .5em 0 .35em; }
#lsir-deck .dk-flow span { display: block; color: var(--m); font-size: clamp(13px, 1.56cqw, 24px); line-height: 1.4; }
#lsir-deck .dk-note { margin-top: 4%; font-size: clamp(14px, 1.76cqw, 27px); font-weight: 600; color: var(--p); }

/* 4 tech */
#lsir-deck .dk-tech h2 { margin-bottom: .5em; }
#lsir-deck .dk-pipe { display: flex; align-items: stretch; gap: .8%; margin-bottom: 3.2%; }
#lsir-deck .dk-pipe .st { flex: 1; background: #fff; border: 2px solid var(--l); border-radius: 10px; padding: 1.1% 1.2%; }
#lsir-deck .dk-pipe .st.hot { border-color: var(--p); background: #f3dde2; flex: 1.25; }
#lsir-deck .dk-pipe b { display: block; font-size: clamp(12px, 1.69cqw, 27px); }
#lsir-deck .dk-pipe .st.hot b { color: var(--p); }
#lsir-deck .dk-pipe span { display: block; color: var(--m); font-size: clamp(12px, 1.30cqw, 20px); line-height: 1.3; margin-top: .25em; }
#lsir-deck .dk-pipe i { font-style: normal; color: var(--p); font-weight: 800; align-self: center; font-size: clamp(13px, 1.95cqw, 31px); }
#lsir-deck .dk-spec { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2.6%; }
#lsir-deck .dk-spec h3 { font-size: clamp(14px, 1.89cqw, 29px); margin: 0 0 .35em; color: var(--p); }
#lsir-deck .dk-spec p { font-size: clamp(13px, 1.40cqw, 22px); line-height: 1.42; }
#lsir-deck .dk-tech .dk-note { margin-top: 3%; }
@media (max-width: 700px) { #lsir-deck .dk-pipe { flex-direction: column; } #lsir-deck .dk-pipe i { transform: rotate(90deg); } #lsir-deck .dk-spec { grid-template-columns: 1fr; } }

/* 5 results */
#lsir-deck .dk-results h2 { margin-bottom: .45em; }
#lsir-deck .dk-bars { display: grid; gap: .9em; margin-bottom: 4%; }
#lsir-deck .dk-bar { display: grid; grid-template-columns: 22% 1fr auto; align-items: center; gap: 2%; }
#lsir-deck .dk-bar .lbl { font-size: clamp(14px, 1.89cqw, 28px); font-weight: 600; }
#lsir-deck .dk-bar .fill { height: clamp(25px, 4.68cqw, 72px); border-radius: 0 6px 6px 0; display: block; }
#lsir-deck .dk-bar .fill.orig { background: var(--bo); }
#lsir-deck .dk-bar .fill.lsir { background: var(--bl); }
#lsir-deck .dk-bar .val { font-size: clamp(20px, 3.12cqw, 49px); font-weight: 800; min-width: 3.4em; }
#lsir-deck .dk-axis { color: var(--m); font-size: clamp(12px, 1.37cqw, 20px); }
#lsir-deck .dk-facts { display: grid; grid-template-columns: repeat(3, 1fr); gap: 3%; }
#lsir-deck .dk-facts p { font-size: clamp(13px, 1.56cqw, 24px); color: var(--m); line-height: 1.4; }
#lsir-deck .dk-facts b { display: block; color: var(--i); font-size: clamp(20px, 2.99cqw, 46px); font-weight: 800; margin-bottom: .15em; }
#lsir-deck .dk-facts p:first-child b { color: var(--p); }

/* 5 next */
#lsir-deck .dk-cols { display: grid; grid-template-columns: repeat(3, 1fr); gap: 4%; }
#lsir-deck .dk-cols h3 { font-size: clamp(17px, 2.21cqw, 33px); margin: 0 0 .4em; color: var(--p); }
#lsir-deck .dk-cols p { font-size: clamp(13px, 1.69cqw, 26px); line-height: 1.45; }
#lsir-deck .dk-end { margin-top: 5%; font-size: clamp(20px, 3.25cqw, 52px); font-weight: 800; }
#lsir-deck .dk-end span { font-family: "Noto Serif Armenian", serif; font-weight: 900; color: var(--p); margin-right: .3em; }
#lsir-deck .dk-small { margin-top: 1.2em; color: var(--m); font-size: clamp(11px, 1.23cqw, 18px); }

@media (max-width: 700px) {
  #lsir-deck .dk-slide { aspect-ratio: auto; min-height: 420px; padding: 28px 20px 70px; }
  #lsir-deck .dk-title { flex-direction: column; align-items: flex-start; }
  #lsir-deck .dk-flow, #lsir-deck .dk-facts, #lsir-deck .dk-cols { grid-template-columns: 1fr; gap: 18px; }
  #lsir-deck .dk-nav { left: 20px; right: 20px; bottom: 16px; }
}
@media (prefers-reduced-motion: reduce) { #lsir-deck .dk-track { scroll-behavior: auto; } }
"""


def build_standalone(path):
    """Full-screen copy for screen sharing: arrow keys / space move between slides."""
    js = """<script>
const t=document.querySelector('.dk-track');
document.addEventListener('keydown',e=>{const w=t.clientWidth;
 if(['ArrowRight','PageDown',' '].includes(e.key)){t.scrollBy({left:w});e.preventDefault();}
 if(['ArrowLeft','PageUp'].includes(e.key)){t.scrollBy({left:-w});e.preventDefault();}});
document.querySelectorAll('.dk-nav a').forEach(a=>a.addEventListener('click',e=>{e.preventDefault();
 document.querySelector(a.getAttribute('href')).scrollIntoView({behavior:'smooth',inline:'start'});}));
</script>"""
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Lsir slides</title>{FONT_LINK}<style>html,body{{margin:0;height:100%;background:#24222a}}
body{{display:grid;place-items:center}} #lsir-deck{{width:min(100vw,177.78vh);border-radius:0}}{CSS}</style></head>
<body>{SLIDES}{js}</body></html>"""
    open(path, "w").write(page)


if __name__ == "__main__":
    import sys
    build_standalone(sys.argv[1] if len(sys.argv) > 1 else "slides.html")
