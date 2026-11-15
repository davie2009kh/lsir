"""Baseline: 108% of words wrong for whisper-small, worse than writing nothing."""
from common import scene, words, write, MUTED, INK, LINE, POM

W = 1680; X = lambda v: W * v / 120          # track: 0-120% across 1680 px
css = f"""
    #bl-setup .wd, #bl-setup2 .wd, #bl-who .wd, #bl-num, #bl-unit .wd, #bl-mark, #bl-markl {{ opacity: 0; }}
    #bl-all {{ position: absolute; inset: 0; }}
    #bl-setup {{ position: absolute; left: 120px; top: 90px; font-size: 66px; font-weight: 800; line-height: 1.1; }}
    #bl-setup2 {{ position: absolute; left: 120px; top: 178px; font-size: 42px; font-weight: 600; color: {MUTED}; }}
    #bl-who {{ position: absolute; left: 120px; top: 90px; font-size: 58px; font-weight: 800; }}
    #bl-num {{ position: absolute; left: 100px; top: 168px; font-size: 440px; font-weight: 900; line-height: 1; letter-spacing: -0.04em;
              font-variant-numeric: tabular-nums; }}
    #bl-unit {{ position: absolute; left: 128px; top: 632px; width: 1600px; font-size: 52px; font-weight: 700; color: {MUTED}; line-height: 1.15; }}
    #bl-track {{ position: absolute; left: 120px; top: 330px; width: {W}px; height: 130px; border-radius: 10px; background: {LINE}; }}
    #bl-fill {{ position: absolute; left: 0; top: 0; bottom: 0; width: {X(108):.1f}px; border-radius: 10px; background: {INK}; transform-origin: 0 50%; }}
    #bl-mark {{ position: absolute; left: {120 + X(100) - 3:.1f}px; top: 744px; width: 6px; height: 104px; border-radius: 3px; background: {POM}; }}
    #bl-markl {{ position: absolute; left: {120 + X(100) - 640:.1f}px; top: 856px; width: 620px; text-align: right; font-size: 46px; font-weight: 700; color: {POM}; }}"""
body = f"""    <div id="bl-all">
    <div id="bl-setup">{words("5 free speech models", "bl-s")}</div>
    <div id="bl-setup2">{words("932 Armenian test recordings, sentences never used in training", "bl-s2")}</div>
    <div id="bl-who">{words("whisper-small, the default most apps use", "bl-w")}</div>
    <div id="bl-num"><em id="bl-n">0</em>%</div>
    <div id="bl-unit">{words("of words wrong (word error rate)", "bl-u")}</div>
    <div id="bl-track"><i id="bl-fill"></i></div>
    <div id="bl-mark"></div>
    <div id="bl-markl">writing nothing: 100%</div>
    </div>"""
script = """
  // the previous scene's strip slides down and becomes the error bar (morph)
  tl.fromTo("#bl-track", { y: 0, scaleY: 1, transformOrigin: "50% 0%" }, { y: 440, scaleY: 0.55, duration: 0.8, ease: "power3.inOut" }, 0.05);
  tl.fromTo("#bl-setup .wd", { opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.15, stagger: 0.08 }, C.L6);
  tl.to("#bl-setup2 .wd", { opacity: 1, duration: 0.12, stagger: 0.05 }, C.L6 + 0.5);
  // blur swap to the one model that matters here
  tl.to(["#bl-setup", "#bl-setup2"], { opacity: 0, filter: "blur(8px)", duration: 0.18, ease: "power2.in" }, C.whisper - 0.25);
  tl.fromTo("#bl-who .wd", { opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.15, stagger: 0.07 }, C.whisper);
  tl.fromTo("#bl-num", { opacity: 0, scale: 0.9, transformOrigin: "0% 60%" }, { opacity: 1, scale: 1, duration: 0.5, ease: "expo.out" }, C.whisper + 0.3);
  tl.to("#bl-unit .wd", { opacity: 1, duration: 0.12, stagger: 0.06 }, C.whisper + 0.6);
  tl.fromTo(["#bl-mark", "#bl-markl"], { opacity: 0 }, { opacity: 1, duration: 0.3 }, C.whisper + 0.9);
  // the count and the bar run together; passing 100% (0.95 s in, on this ease) the marker takes a punch
  const n = { v: 0 }, el = document.querySelector("#bl-n"), t0 = C.worse - 0.4;
  tl.fromTo(n, { v: 0 }, { v: 108, duration: 1.3, ease: "power2.out", onUpdate: () => { el.textContent = Math.round(n.v); } }, t0);
  tl.fromTo("#bl-fill", { scaleX: 0 }, { scaleX: 1, duration: 1.3, ease: "power2.out" }, t0);
  tl.to("#bl-mark", { scaleY: 1.45, duration: 0.12, ease: "power2.out" }, t0 + 0.95);
  tl.to("#bl-mark", { scaleY: 1, duration: 0.35, ease: "elastic.out(1, 0.5)" }, t0 + 1.07);
  tl.fromTo("#bl-markl", { scale: 1, transformOrigin: "100% 50%" }, { scale: 1.12, duration: 0.12, ease: "power2.out", immediateRender: false }, t0 + 0.95);
  tl.to("#bl-markl", { scale: 1, duration: 0.3, ease: "power2.out" }, t0 + 1.07);
  // out: a quick blur before the colour flood of the next scene
  tl.to("#bl-all", { opacity: 0, filter: "blur(10px)", duration: 0.2, ease: "power2.in" }, C.dur - 0.22);"""
write("baseline", scene("baseline", css, body, script))
