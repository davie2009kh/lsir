"""Shared pieces for the scene generators: fonts, colours, and splitting text into animatable spans."""
import html
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FONTS = """
    @font-face { font-family: "Lsir Serif"; src: url("assets/fonts/NotoSerifArmenian.ttf"); font-weight: 100 900; }
    @font-face { font-family: "Lsir Arm"; src: url("assets/fonts/NotoSansArmenian.ttf"); font-weight: 100 900; }
    @font-face { font-family: "Lsir Latin"; src: url("assets/fonts/NotoSans.ttf"); font-weight: 100 900; }"""

BASE = """
    #root { position: absolute; inset: 0; background: #f4ecea; color: #24222a; overflow: hidden;
            font-family: "Lsir Latin", "Lsir Arm", sans-serif; }
    .wd { display: inline-block; white-space: pre; }
    .lt { display: inline-block; white-space: pre; }
    .nw { white-space: nowrap; }
    em { font-style: normal; }"""

INK, MUTED, LINE, TUFF = "#24222a", "#6b6370", "#e2d4cf", "#f4ecea"
POM, ARARAT, MARK = "#8c1d3a", "#36577a", "#ffe28a"
BAR_ORIG, BAR_LSIR = "#2e6db5", "#b8344f"        # validated with the dataviz palette checker on #f4ecea


def words(text, prefix):
    """Wrap each word in a span (id prefix-i) so it can be animated on its own."""
    return " ".join(f'<span id="{prefix}-{i}" class="wd">{html.escape(w)}</span>' for i, w in enumerate(text.split()))


def letters(text, prefix):
    """One span per letter (id prefix-i, counting every character), grouped per word so lines break between words."""
    out, i = [], 0
    for wi, w in enumerate(text.split(" ")):
        spans = []
        for c in w:
            spans.append(f'<span id="{prefix}-{i}" class="lt">{html.escape(c)}</span>'); i += 1
        out.append(f'<span class="nw">{"".join(spans)}</span>')
        i += 1                                                      # the space keeps its index slot
    return " ".join(out)


def scene(cid, css, body, script):
    return f"""<template>
  <style>{FONTS}{BASE}
{css}
  </style>
  <div id="root" data-composition-id="{cid}" data-width="1920" data-height="1080">
{body}
  </div>
  <script>
  const C = /*CUES*/{{}}/*END*/;
  const tl = gsap.timeline({{ paused: true }});
{script}
  tl.to("#root", {{ opacity: 1, duration: 0.01 }}, C.dur - 0.01);
  window.__timelines["{cid}"] = tl;
  </script>
</template>
"""


def write(cid, text):
    (ROOT / "compositions" / f"{cid}.html").write_text(text)
    print(f"{cid}.html written ({len(text)} chars)")
