"""Hook: the real sentence, clean then over the mobile line, and the original model's real transcript."""
import html, json, re
from common import ROOT, scene, words, write, ARARAT, MUTED, INK, MARK

env = json.loads((ROOT / "assets/waveforms.json").read_text())
res = json.loads((ROOT / "assets/clip555_result.json").read_text())
body_html = re.search(r'<p class="lsir-text">(.*?)</p>', res["orig_html"], re.S).group(1)
toks = re.findall(r'<mark class="lsir-(sub|ins)"[^>]*>(.*?)</mark>|<span class="lsir-del"[^>]*>(.*?)</span>|([^\s<]+)', body_html)
spans = []
for i, (mk, mt, dt, pl) in enumerate(toks):
    kind, t = ("bad", mt) if mk else (("missed", dt) if dt else ("ok", pl))
    inner = '<i class="hl"></i>' if kind == "bad" else ('<i class="st"></i>' if kind == "missed" else "")
    spans.append(f'<span id="hk-w{i}" class="w {kind}">{inner}<b>{html.escape(html.unescape(t))}</b></span>')
bars = "\n".join(f'      <i class="bar" style="--c:{max(c, .04):.3f};--m:{max(m, .04):.3f};--d:{abs(i - 59.5) / 59.5:.3f}"></i>'
                 for i, (c, m) in enumerate(zip(env["clean"], env["mobile"])))

css = f"""
    #hk-l2 .wd, #hk-who, #hk-score {{ opacity: 0; }}
    #hk-text .w {{ opacity: 0; filter: blur(6px); }}
    #hk-label {{ position: absolute; left: 120px; top: 64px; height: 64px; width: 1680px; font-size: 46px; font-weight: 600; color: {MUTED}; }}
    #hk-l1, #hk-l2 {{ position: absolute; left: 0; top: 0; white-space: nowrap; }}
    #hk-l2 b {{ color: {INK}; font-weight: 800; }}
    #hk-wave {{ position: absolute; left: 120px; top: 136px; width: 1680px; height: 260px; display: flex; align-items: center; gap: 4px; }}
    #hk-wave .bar {{ display: block; flex: 0 0 10px; height: 260px; border-radius: 5px; background: {INK}; }}
    #hk-head {{ position: absolute; left: 120px; top: 126px; width: 5px; height: 280px; border-radius: 3px; background: #8c1d3a; opacity: 0; }}
    #hk-who {{ position: absolute; left: 120px; top: 432px; font-size: 38px; font-weight: 700; color: {ARARAT}; }}
    #hk-text {{ position: absolute; left: 120px; top: 482px; width: 1220px; font-family: "Lsir Arm", sans-serif; font-size: 54px; line-height: 1.4; font-weight: 500; }}
    #hk-text .w {{ position: relative; display: inline-block; margin-right: 0.26em; }}
    #hk-text .w b {{ position: relative; font-weight: 500; z-index: 1; }}
    #hk-text .hl {{ position: absolute; left: -6px; right: -6px; top: 10%; bottom: 4%; border-radius: 6px; background: {MARK}; transform: scaleX(0); transform-origin: 0 50%; }}
    #hk-text .st {{ position: absolute; left: -2px; right: -2px; top: 54%; height: 5px; border-radius: 3px; background: {MUTED}; transform: scaleX(0); transform-origin: 0 50%; z-index: 2; }}
    #hk-text .missed b {{ color: {MUTED}; }}
    #hk-score {{ position: absolute; left: 1400px; top: 470px; width: 420px; color: {ARARAT}; }}
    #hk-num {{ font-size: 250px; font-weight: 800; line-height: 0.95; letter-spacing: -0.03em; font-variant-numeric: tabular-nums; }}
    #hk-num span {{ font-size: 110px; font-weight: 700; letter-spacing: 0; }}
    #hk-sub {{ font-size: 50px; font-weight: 700; margin-top: 10px; }}"""

body = f"""    <div id="hk-all">
    <div id="hk-label"><span id="hk-l1">{words("Clean recording, one Armenian sentence", "hk-a")}</span><span id="hk-l2">{words("The same sentence after a", "hk-b")} <b>{words("mobile phone line", "hk-c")}</b></span></div>
    <div id="hk-wave">
{bars}
    </div>
    <div id="hk-head"></div>
    <div id="hk-who">BuzzASR, the best free Armenian model, hears:</div>
    <div id="hk-text">{" ".join(spans)}</div>
    <div id="hk-score"><div id="hk-num"><em id="hk-n">0</em><span>/16</span></div><div id="hk-sub">words wrong</div></div>
    </div>"""

script = """
  const bars = gsap.utils.toArray("#hk-wave .bar");
  const ws = gsap.utils.toArray("#hk-text .w");
  const v = (b, k) => parseFloat(b.style.getPropertyValue(k));
  // label, per word blur-in
  // the waveform draws itself out from the centre: the real loudness of the clip
  bars.forEach((b) => tl.fromTo(b, { scaleY: v(b, "--c") * 0.3 }, { scaleY: v(b, "--c"), duration: 0.7, ease: "expo.out" }, 0.15 + v(b, "--d") * 0.45));
  // clean clip: a playhead sweeps the part we hear
  tl.set("#hk-head", { x: 0, opacity: 0 }, 0);
  tl.to("#hk-head", { opacity: 1, duration: 0.15 }, C.clean - 0.15);
  tl.to("#hk-head", { x: 1680 * (C.clean_end - C.clean) / 7.56, duration: C.clean_end - C.clean, ease: "none" }, C.clean);
  tl.to("#hk-head", { opacity: 0, duration: 0.15 }, C.clean_end);
  // the phone line: label swaps (blur swap), bars collapse to the degraded envelope and lose colour
  tl.to("#hk-l1 .wd", { opacity: 0, filter: "blur(8px)", duration: 0.15, stagger: 0.03 }, C.L2);
  tl.fromTo("#hk-l2 .wd", { opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.16, stagger: 0.07 }, C.L2 + 0.2);
  bars.forEach((b, i) => tl.to(b, { scaleY: v(b, "--m"), backgroundColor: "#9a8f95", duration: 0.55, ease: "power3.inOut" }, C.L2 + 0.9 + i * 0.005));
  // degraded clip plays; the transcript arrives word by word, in time with the audio
  tl.set("#hk-head", { x: 0 }, C.mobile - 0.2);
  tl.to("#hk-head", { opacity: 1, duration: 0.15 }, C.mobile - 0.15);
  tl.to("#hk-head", { x: 1680, duration: C.mobile_end - C.mobile, ease: "none" }, C.mobile);
  tl.to("#hk-head", { opacity: 0, duration: 0.15 }, C.mobile_end);
  tl.fromTo("#hk-who", { opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.18 }, C.mobile + 0.2);
  ws.forEach((w, i) => tl.to(w, { opacity: 1, filter: "blur(0px)", duration: 0.14 },
                             C.mobile + 0.5 + i * (C.mobile_end - C.mobile - 0.7) / ws.length));
  // the verdict: the demo's highlighter sweeps each wrong word, missed words get struck, the count runs up
  let k = 0;
  ws.forEach((w) => {
    const hl = w.querySelector(".hl"), st = w.querySelector(".st");
    if (hl) tl.to(hl, { scaleX: 1, duration: 0.22, ease: "power2.out" }, C.eleven - 0.6 + (k++) * 0.07);
    if (st) tl.to(st, { scaleX: 1, duration: 0.2, ease: "power2.out" }, C.eleven - 0.6 + (k++) * 0.07);
  });
  tl.fromTo("#hk-score", { opacity: 0, filter: "blur(10px)", scale: 0.92, transformOrigin: "0% 50%" },
            { opacity: 1, filter: "blur(0px)", scale: 1, duration: 0.5, ease: "expo.out" }, C.eleven - 0.3);
  const n = { v: 0 }, el = document.querySelector("#hk-n");
  tl.fromTo(n, { v: 0 }, { v: 11, duration: 0.9, ease: "power2.out", onUpdate: () => { el.textContent = Math.round(n.v); } }, C.eleven - 0.3);
  // whip out to the left into the next scene
  tl.to("#hk-all", { x: -260, filter: "blur(14px)", opacity: 0, duration: 0.3, ease: "power3.in" }, C.dur - 0.3);"""

write("hook", scene("hook", css, body, script))
