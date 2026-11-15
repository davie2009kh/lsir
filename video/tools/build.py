"""Single source of timing for the Lsir film.

Reads narration.json, measures each voiceover clip in assets/audio/vo/<id>.mp3 when it exists (otherwise estimates
~2.6 words/s), lays the voice lines and the two real speech clips out in order, then writes:
  - index.html                 root composition: scene hosts, audio, captions host
  - compositions/captions.html every caption group, timed to the voice
  - the CUES block inside each scene file (times relative to that scene's start)
  - timings.json               everything above, for checking
Run: python3 tools/build.py
"""
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
N = json.loads((ROOT / "narration.json").read_text())["lines"]
LINES = {l["id"]: l for l in N}
WPS = 2.6
CLEAN_EXCERPT = 3.6          # seconds of the clean clip we play
MOBILE_LEN = 7.56            # full degraded clip
END_HOLD = 6.0


def dur(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return float(out)


def line_len(lid):
    f = ROOT / "assets/audio/vo" / f"{lid}.mp3"
    if f.exists():
        return dur(f), True
    return round(len(LINES[lid]["text"].split()) / WPS + 0.3, 2), False


# ---------------------------------------------------------------- the order of everything that makes sound
# (event id, kind, gap after the previous event's end, scene it belongs to)
SEQ = [
    ("L1", "vo", 0.5, "hook"), ("clean", "clip", 0.25, "hook"), ("L2", "vo", 0.3, "hook"),
    ("mobile", "clip", 0.25, "hook"), ("L3", "vo", 0.15, "hook"),
    ("L4", "vo", 0.9, "problem"), ("L5", "vo", 0.45, "problem"),
    ("L6", "vo", 0.9, "baseline"),
    ("L7", "vo", 0.9, "demo"), ("L8", "vo", 0.35, "demo"), ("L9", "vo", 0.6, "demo"),
    ("L10", "vo", 1.0, "how"), ("L11", "vo", 0.5, "how"),
    ("L12", "vo", 2.6, "next"), ("L13", "vo", 2.4, "next"),
]
SCENES = ["hook", "problem", "baseline", "demo", "how", "next"]

events, t, have_vo = {}, 0.0, {}
for eid, kind, gap, scene in SEQ:
    start = t + gap if events else gap
    if kind == "vo":
        length, real = line_len(eid)
        have_vo[eid] = real
    else:
        length = CLEAN_EXCERPT if eid == "clean" else MOBILE_LEN
    events[eid] = {"start": round(start, 3), "end": round(start + length, 3), "scene": scene, "kind": kind}
    t = start + length

scene_t = {}
for i, s in enumerate(SCENES):
    first = min(e["start"] for e in events.values() if e["scene"] == s)
    scene_t[s] = {"start": 0.0 if i == 0 else round(first - 0.45, 3)}
for i, s in enumerate(SCENES):
    scene_t[s]["end"] = scene_t[SCENES[i + 1]]["start"] if i + 1 < len(SCENES) else round(t + END_HOLD, 3)
TOTAL = scene_t[SCENES[-1]]["end"]


def word_time(lid, word, nth=1):
    """Estimated start of a word inside a line: time spread by character count across the clip."""
    words = LINES[lid]["text"].split()
    e = events[lid]
    total = sum(len(w) + 1 for w in words)
    acc, seen = 0, 0
    for w in words:
        if re.sub(r"[^\w.]", "", w).rstrip(".").lower() == word.lower():
            seen += 1
            if seen == nth:
                return round(e["start"] + (e["end"] - e["start"]) * acc / total, 3)
        acc += len(w) + 1
    raise KeyError(f"{word} not in {lid}")


ABS = {
    "hook": {"L1": events["L1"]["start"], "clean": events["clean"]["start"], "clean_end": events["clean"]["end"],
             "L2": events["L2"]["start"], "mobile": events["mobile"]["start"], "mobile_end": events["mobile"]["end"],
             "L3": events["L3"]["start"], "eleven": word_time("L3", "eleven")},
    "problem": {"L4": events["L4"]["start"], "when": word_time("L4", "When"), "L5": events["L5"]["start"],
                "narrow": word_time("L5", "narrow"), "nobody": word_time("L5", "Nobody")},
    "baseline": {"L6": events["L6"]["start"], "whisper": word_time("L6", "Whisper"),
                 "worse": word_time("L6", "worse"), "L6_end": events["L6"]["end"]},
    "demo": {"L7": events["L7"]["start"], "L8": events["L8"]["start"], "send": word_time("L8", "send"),
             "compare": word_time("L8", "compare"), "L9": events["L9"]["start"], "eleven": word_time("L9", "eleven"),
             "five": word_time("L9", "five"), "both": word_time("L9", "Both")},
    "how": {"L10": events["L10"]["start"], "adapter": word_time("L10", "adapter"),
            "percent": word_time("L10", "percent"), "clips": word_time("L10", "clips"),
            "L11": events["L11"]["start"], "from": word_time("L11", "from"), "to": word_time("L11", "to", 1),
            "second": word_time("L11", "second")},
    "next": {"L12": events["L12"]["start"], "older": word_time("L12", "older"),
             "public": word_time("L12", "Public"), "L13": events["L13"]["start"]},
}
CUES = {s: {k: round(v - scene_t[s]["start"], 3) for k, v in c.items()} | {"dur": round(scene_t[s]["end"] - scene_t[s]["start"], 3)}
        for s, c in ABS.items()}

# ---------------------------------------------------------------- write cues into scene files
for s in SCENES:
    f = ROOT / "compositions" / f"{s}.html"
    if f.exists():
        src = f.read_text()
        src2 = re.sub(r"/\*CUES\*/.*?/\*END\*/", "/*CUES*/" + json.dumps(CUES[s]) + "/*END*/", src, flags=re.S)
        f.write_text(src2)

# ---------------------------------------------------------------- captions
def groups(text, max_words=5):
    out, cur = [], []
    for w in text.split():
        cur.append(w)
        if len(cur) >= max_words or re.search(r"[.,:;?!]$", w):
            out.append(cur); cur = []
    if cur:
        out.append(cur)
    # merge a dangling single word into the previous group
    merged = []
    for g in out:
        if merged and len(g) == 1 and len(merged[-1]) < 6:
            merged[-1] += g
        else:
            merged.append(g)
    return merged


cap_items, cap_js = [], []
for l in N:
    e = events[l["id"]]
    gs = groups(l["text"])
    chars = [sum(len(w) + 1 for w in g) for g in gs]
    span, acc = e["end"] - e["start"], 0
    for gi, (g, c) in enumerate(zip(gs, chars)):
        st = e["start"] + span * acc / sum(chars)
        en = e["start"] + span * (acc + c) / sum(chars)
        acc += c
        cid = f"cap-{l['id'].lower()}-{gi}"
        text = " ".join(g)
        cap_items.append(f'    <p id="{cid}" class="clip cap" data-start="{st:.3f}" data-duration="{en - st:.3f}"><span>{text}</span></p>')
        cap_js.append(f'  tl.fromTo("#{cid}", {{opacity: 0}}, {{opacity: 1, duration: 0.12, ease: "power2.out"}}, {st:.3f});\n'
                      f'  tl.to("#{cid}", {{opacity: 0, duration: 0.1, ease: "power2.in"}}, {en - 0.1:.3f});')

CAPTIONS = f"""<template>
  <style>
    @font-face {{ font-family: "Lsir Sans"; src: url("assets/fonts/NotoSans.ttf"); font-weight: 100 900; }}
    #root {{ position: absolute; inset: 0; pointer-events: none; }}
    .cap {{ position: absolute; left: 0; right: 0; bottom: 96px; margin: 0; padding: 0 240px; text-align: center;
           font-family: "Lsir Sans", sans-serif; font-weight: 600; font-size: 40px; line-height: 1.2; opacity: 0; }}
    .cap span {{ display: inline-block; padding: 10px 24px; border-radius: 10px; color: #24222a;
                background: rgba(255, 255, 255, 0.92); box-shadow: 0 6px 18px rgba(36, 34, 42, 0.10); }}
  </style>
  <div id="root" data-composition-id="captions" data-width="1920" data-height="1080">
{chr(10).join(cap_items)}
  </div>
  <script>
  const tl = gsap.timeline({{ paused: true }});
{chr(10).join(cap_js)}
  window.__timelines["captions"] = tl;
  </script>
</template>
"""
(ROOT / "compositions" / "captions.html").write_text(CAPTIONS)

# ---------------------------------------------------------------- root composition
hosts = []
for i, s in enumerate(SCENES):
    st, en = scene_t[s]["start"], scene_t[s]["end"]
    hosts.append(f'      <div id="{s}" data-composition-id="{s}" data-composition-src="compositions/{s}.html" '
                 f'data-start="{st:.3f}" data-duration="{en - st:.3f}" data-track-index="1" data-track-kind="graphics" '
                 f'data-width="1920" data-height="1080" style="z-index: 1"></div>')
audio = [
    f'      <audio id="sfx-clean" class="clip" src="assets/audio/clip555_clean.wav" data-start="{events["clean"]["start"]:.3f}" '
    f'data-duration="{CLEAN_EXCERPT:.3f}" data-volume="1" data-fade-out="0.4" data-track-index="3"></audio>',
    f'      <audio id="sfx-mobile" class="clip" src="assets/audio/clip555_mobile.wav" data-start="{events["mobile"]["start"]:.3f}" '
    f'data-duration="{MOBILE_LEN:.3f}" data-volume="1" data-track-index="3"></audio>',
]
for l in N:
    if have_vo.get(l["id"]):
        e = events[l["id"]]
        audio.append(f'      <audio id="vo-{l["id"].lower()}" class="clip" src="assets/audio/vo/{l["id"]}.mp3" '
                     f'data-start="{e["start"]:.3f}" data-duration="{e["end"] - e["start"]:.3f}" data-volume="1" data-track-index="2"></audio>')

INDEX = f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=1920, height=1080" />
    <script src="https://cdn.jsdelivr.net/npm/gsap@3.15.0/dist/gsap.min.js" integrity="sha384-XmJ9SoHtVOHoQUcKvFAzVXwdkKo1Ie3bhmSoIAkcdsHGaIrVJIkmozyq0FJeb/Ly" crossorigin="anonymous"></script>
    <style>
      * {{ margin: 0; padding: 0; box-sizing: border-box; }}
      html, body {{ margin: 0; width: 1920px; height: 1080px; overflow: hidden; background: #f4ecea; }}
      #root {{ width: 100%; height: 100%; position: relative; }}
      #ground {{ position: absolute; inset: 0; background: #f4ecea; }}
    </style>
  </head>
  <body>
    <!-- generated by tools/build.py from narration.json: edit there, then re-run -->
    <div id="root" data-composition-id="main" data-start="0" data-duration="{TOTAL:.3f}" data-width="1920" data-height="1080">
      <div id="ground"></div>
{chr(10).join(hosts)}
      <div id="captions" data-composition-id="captions" data-composition-src="compositions/captions.html" data-start="0"
           data-duration="{TOTAL:.3f}" data-track-index="4" data-track-kind="captions" data-width="1920" data-height="1080"
           style="z-index: 5"></div>
{chr(10).join(audio)}
    </div>
    <script>
      const tl = gsap.timeline({{ paused: true }});
      window.__timelines["main"] = tl;
    </script>
  </body>
</html>
"""
(ROOT / "index.html").write_text(INDEX)
json.dump({"total": TOTAL, "events": events, "scenes": scene_t, "cues": CUES, "voice_measured": have_vo},
          open(ROOT / "timings.json", "w"), indent=1)
print(f"total {TOTAL:.1f} s | voice measured for {sum(have_vo.values())}/{len(have_vo)} lines")
for s in SCENES:
    print(f"  {s:9s} {scene_t[s]['start']:6.1f} -> {scene_t[s]['end']:6.1f}")
