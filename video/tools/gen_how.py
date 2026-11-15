"""How it works: the model block grows, the adapter clips onto 'listens', phone-line clips feed it; then the result."""
from common import scene, words, write, MUTED, INK, LINE, POM, BAR_ORIG, BAR_LSIR

BW = 1220; bar = lambda v: BW * v / 25
tiles = "".join(f'<i class="hw-tile{" ph" if k % 2 else ""}">' + "<b></b>" * 7 + "</i>" for k in range(10))
css = f"""
    #hw-chart, #hw-tab, #hw-pct, #hw-feedl, #hw-belt, #hw-cs, #hw-cv, .hw-row .val {{ opacity: 0; }}
    #hw-all, #hw-diagram, #hw-chart {{ position: absolute; inset: 0; }}
    #hw-name {{ position: absolute; left: 120px; top: 70px; font-size: 64px; font-weight: 800; }}
    #hw-name .sub {{ display: block; font-size: 40px; font-weight: 600; color: {MUTED}; margin-top: 6px; }}
    #hw-block {{ position: absolute; left: 120px; top: 270px; width: 1680px; height: 320px; background: #fff; border-radius: 20px;
                box-shadow: 0 24px 60px rgba(36, 34, 42, 0.13); }}
    .hw-half {{ position: absolute; top: 0; width: 840px; height: 320px; display: flex; flex-direction: column; justify-content: center;
               padding-left: 70px; font-size: 96px; font-weight: 800; letter-spacing: -0.01em; }}
    .hw-half span {{ font-size: 44px; font-weight: 600; color: {MUTED}; margin-top: 4px; letter-spacing: 0; }}
    #hw-dec {{ left: 840px; border-left: 3px solid {LINE}; }}
    #hw-tab {{ position: absolute; left: 170px; top: 222px; width: 420px; height: 96px; border-radius: 14px; background: {POM};
              color: #fff; font-size: 50px; font-weight: 800; display: flex; align-items: center; justify-content: center; z-index: 3; }}
    #hw-pct {{ position: absolute; left: 620px; top: 238px; font-size: 50px; font-weight: 800; color: {POM}; z-index: 3; }}
    #hw-belt {{ position: absolute; left: 0; top: 650px; width: 1920px; height: 130px; overflow: hidden; }}
    #hw-feed {{ position: absolute; left: 0; top: 0; display: flex; gap: 26px; }}
    .hw-tile {{ display: flex; align-items: center; gap: 9px; width: 200px; height: 120px; padding: 0 24px; border-radius: 14px; background: #fff; }}
    .hw-tile b {{ display: block; width: 13px; height: 72px; border-radius: 7px; background: {INK}; }}
    .hw-tile b:nth-child(2n) {{ height: 40px; }} .hw-tile b:nth-child(3n) {{ height: 92px; }}
    .hw-tile.ph {{ background: {LINE}; }}
    .hw-tile.ph b {{ background: #9a8f95; height: 22px; }} .hw-tile.ph b:nth-child(3n) {{ height: 34px; }}
    #hw-feedl {{ position: absolute; left: 120px; top: 800px; font-size: 46px; font-weight: 600; color: {MUTED}; }}
    #hw-feedl b {{ color: {INK}; font-weight: 800; }}
    #hw-ct {{ position: absolute; left: 120px; top: 80px; font-size: 66px; font-weight: 800; }}
    #hw-cs {{ position: absolute; left: 120px; top: 170px; font-size: 42px; font-weight: 600; color: {MUTED}; }}
    .hw-row {{ position: absolute; left: 120px; height: 120px; display: flex; align-items: center; }}
    .hw-row .lab {{ width: 340px; font-size: 64px; font-weight: 800; }}
    .hw-row .bar {{ height: 120px; border-radius: 0 6px 6px 0; transform-origin: 0 50%; }}
    .hw-row .val {{ margin-left: 30px; font-size: 76px; font-weight: 900; font-variant-numeric: tabular-nums; }}
    #hw-r1 {{ top: 300px; }} #hw-r2 {{ top: 450px; }}
    #hw-r1 .bar {{ width: {bar(22.1):.1f}px; background: {BAR_ORIG}; }}
    #hw-r2 .bar {{ width: {bar(20.8):.1f}px; background: {BAR_LSIR}; }}
    #hw-r2 .lab {{ color: {POM}; }}
    #hw-axis {{ position: absolute; left: 460px; top: 600px; width: {BW}px; border-top: 3px solid #cdbcb6; }}
    #hw-axis span {{ position: absolute; top: 10px; font-size: 34px; font-weight: 600; color: {MUTED}; }}
    #hw-cv {{ position: absolute; left: 120px; top: 720px; width: 1680px; font-size: 52px; font-weight: 800; line-height: 1.2; }}
    #hw-cv span {{ color: {MUTED}; font-weight: 600; }}"""
body = f"""    <div id="hw-all">
    <div id="hw-diagram">
      <div id="hw-name">{words("BuzzASR-Armenian", "hw-n")}<span class="sub">{words("Whisper-large-v3 trained on Armenian, 1.55 billion weights", "hw-ns")}</span></div>
      <div id="hw-block"><div id="hw-enc" class="hw-half">Listens<span>the audio encoder</span></div>
        <div id="hw-dec" class="hw-half">Writes<span>the text decoder</span></div></div>
      <div id="hw-tab">Lsir adapter</div>
      <div id="hw-pct">1.5% of the weights</div>
      <div id="hw-belt" data-layout-allow-overflow><div id="hw-feed">{tiles}</div></div>
      <div id="hw-feedl"><b>Training clips:</b> {words("half sent through random phone lines", "hw-f")}</div>
    </div>
    <div id="hw-chart">
      <div id="hw-ct">{words("Over the mobile codec Lsir never trained on", "hw-c")}</div>
      <div id="hw-cs">Share of words wrong, 932 test recordings, both with beam search</div>
      <div id="hw-r1" class="hw-row"><span class="lab">BuzzASR</span><i class="bar"></i><span class="val"><em id="hw-v1">0.0</em>%</span></div>
      <div id="hw-r2" class="hw-row"><span class="lab">Lsir</span><i class="bar"></i><span class="val"><em id="hw-v2">0.0</em>%</span></div>
      <div id="hw-axis" data-layout-allow-overflow><span style="left: -10px">0%</span><span style="left: {BW - 70}px">25%</span></div>
      <div id="hw-cv">A second dataset agrees <span>(Common Voice, 1,000 clips): 18.9% to 17.8%</span></div>
    </div>
    </div>"""
script = """
  // whip in from the right (matches the demo's exit)
  tl.fromTo("#hw-all", { x: 260, filter: "blur(14px)", opacity: 0 }, { x: 0, filter: "blur(0px)", opacity: 1, duration: 0.4, ease: "expo.out" }, 0);
  tl.fromTo("#hw-name .wd", { opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.15, stagger: 0.06 }, 0.15);
  tl.fromTo("#hw-block", { scaleX: 0.3, opacity: 0, transformOrigin: "0% 50%" }, { scaleX: 1, opacity: 1, duration: 0.8, ease: "expo.out" }, 0.35);
  tl.fromTo(".hw-half", { opacity: 0, filter: "blur(8px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.2, stagger: 0.25 }, 0.8);
  // the adapter drops onto "Listens" with a small spring
  tl.fromTo("#hw-tab", { opacity: 0, y: -160 }, { opacity: 1, y: 0, duration: 0.6, ease: "back.out(1.4)" }, C.adapter - 0.1);
  tl.fromTo("#hw-pct", { opacity: 0, x: -20, filter: "blur(6px)" }, { opacity: 1, x: 0, filter: "blur(0px)", duration: 0.3, ease: "expo.out" }, C.percent - 0.3);
  // training clips ride a conveyor into the encoder, every other one squeezed by a phone line
  tl.set("#hw-belt", { opacity: 1 }, C.clips - 1.6);
  tl.fromTo("#hw-feed", { x: -2300 }, { x: 140, duration: 3.2, ease: "none" }, C.clips - 1.6);
  tl.fromTo("#hw-feedl", { opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.2 }, C.clips - 0.4);
  tl.to(".hw-tile", { y: -330, scale: 0.4, opacity: 0, duration: 0.45, ease: "power3.in", stagger: 0.05 }, C.clips + 1.6);
  // blur swap to the result
  tl.to("#hw-diagram", { opacity: 0, filter: "blur(10px)", duration: 0.2, ease: "power2.in" }, C.L11 - 0.3);
  tl.set("#hw-chart", { opacity: 1 }, C.L11 - 0.1);
  tl.fromTo("#hw-ct .wd", { opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.15, stagger: 0.06 }, C.L11 - 0.1);
  tl.to("#hw-cs", { opacity: 1, duration: 0.25 }, C.L11 + 0.4);
  tl.fromTo([".hw-row .lab", "#hw-axis"], { opacity: 0 }, { opacity: 1, duration: 0.3 }, C.L11 + 0.2);
  // bars grow with their values counting up
  const v1 = { v: 0 }, v2 = { v: 0 }, e1 = document.querySelector("#hw-v1"), e2 = document.querySelector("#hw-v2");
  tl.fromTo("#hw-r1 .bar", { scaleX: 0 }, { scaleX: 1, duration: 0.9, ease: "expo.out" }, C.from - 0.2);
  tl.to("#hw-r1 .val", { opacity: 1, duration: 0.15 }, C.from - 0.2);
  tl.to(v1, { v: 22.1, duration: 0.9, ease: "expo.out", onUpdate: () => { e1.textContent = v1.v.toFixed(1); } }, C.from - 0.2);
  tl.fromTo("#hw-r2 .bar", { scaleX: 0 }, { scaleX: 1, duration: 0.9, ease: "expo.out" }, C.to - 0.2);
  tl.to("#hw-r2 .val", { opacity: 1, duration: 0.15 }, C.to - 0.2);
  tl.to(v2, { v: 20.8, duration: 0.9, ease: "expo.out", onUpdate: () => { e2.textContent = v2.v.toFixed(1); } }, C.to - 0.2);
  tl.fromTo("#hw-cv", { opacity: 0, filter: "blur(8px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.25 }, C.second - 1.2);
  // blur swap out
  tl.to("#hw-all", { opacity: 0, filter: "blur(10px)", duration: 0.2, ease: "power2.in" }, C.dur - 0.22);"""
write("how", scene("how", css, body, script))
