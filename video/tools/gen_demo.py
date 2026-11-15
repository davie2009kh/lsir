"""Demo: colour-flood reveal of the name, an iris match-cut onto the live demo, the real run, the results."""
from common import scene, letters, words, write, POM, INK

# camera on a screenshot drawn 1920 px wide: centre image point (cx, cy) at zoom z
B = dict(z=1.35, cx=990, cy=470)          # framing for the "pick, line, transcribe" shot
css = f"""
    #dm-b, #dm-c, #dm-cursor, #dm-both, #dm-l-0, #dm-l-1, #dm-l-2, #dm-l-3 {{ opacity: 0; }}
    .dm-shot {{ position: absolute; inset: 0; }}
    .dm-cam {{ position: absolute; left: 0; top: 0; width: 1920px; height: 1200px; }}
    .dm-cam img {{ display: block; width: 1920px; height: 1200px; }}
    #dm-a {{ clip-path: circle(0px at 668px 284px); }}
    #dm-flood {{ position: absolute; inset: 0; background: {POM}; clip-path: circle(0% at 50% 50%); }}
    #dm-word {{ position: absolute; left: 407px; top: 127px; font-family: "Lsir Serif", serif; font-size: 212px; line-height: 1.2;
               color: #ffffff; white-space: nowrap; letter-spacing: -0.02em; font-weight: 900; }}
    #dm-gloss {{ position: absolute; left: 470px; top: 790px; font-size: 64px; font-weight: 600; color: #f3dde2; }}
    #dm-cursor {{ position: absolute; left: 0; top: 0; width: 64px; height: 64px; z-index: 10; }}
    #dm-cursor svg {{ width: 64px; height: 64px; overflow: visible; }}
    #dm-cursor path {{ fill: #ffffff; stroke: {INK}; stroke-width: 1.6; stroke-linejoin: round; }}
    #dm-both {{ position: absolute; left: 0; right: 0; top: 90px; text-align: center; z-index: 11; }}
    #dm-both span {{ display: inline-block; font-size: 60px; font-weight: 800; color: {POM}; background: #ffffff; padding: 18px 40px;
                    border-radius: 14px; box-shadow: 0 24px 60px rgba(36, 34, 42, 0.16); }}"""
body = f"""    <div id="dm-all">
    <div id="dm-c" class="dm-shot"><div id="dm-cam-c" class="dm-cam"><img src="assets/img/s3_results.png" alt="Both models' transcripts with wrong words highlighted" /></div></div>
    <div id="dm-b" class="dm-shot"><div id="dm-cam-b" class="dm-cam"><img src="assets/img/s2_example_loaded.png" alt="Example sentence loaded, mobile line selected" /></div></div>
    <div id="dm-flood"></div>
    <div id="dm-word" lang="hy">{letters("Լսիր", "dm-l")}</div>
    <div id="dm-gloss">{words("“Listen!” in Armenian", "dm-g")}</div>
    <div id="dm-a" class="dm-shot"><div id="dm-cam-a" class="dm-cam"><img src="assets/img/s1_hero.png" alt="Lsir demo, top of the page" /></div></div>
    <div id="dm-cursor"><svg viewBox="0 0 24 24" aria-hidden="true"><path d="M5 3 L5 19 L9 15 L12 22 L15 20.5 L11.5 14 L18 14 Z" /></svg></div>
    <div id="dm-both"><span>Both models answered in 1.6 s</span></div>
    </div>"""
script = f"""
  const cam = (z, cx, cy) => ({{ scale: z, x: 960 - z * cx, y: 540 - z * cy, transformOrigin: "0 0" }});
  const B = {{ z: {B['z']}, cx: {B['cx']}, cy: {B['cy']} }};
  const onB = (ix, iy) => [960 + B.z * (ix - B.cx), 540 + B.z * (iy - B.cy)];   // image point -> screen in shot B
  // 1. colour flood from the centre, then the name prints letter by letter while its weight swells from thin to black
  tl.to("#dm-flood", {{ clipPath: "circle(80% at 50% 50%)", duration: 0.5, ease: "power2.inOut" }}, 0);
  tl.fromTo("#dm-word", {{ x: 292, y: 176, scale: 1.9, fontWeight: 150, transformOrigin: "50% 50%" }},
            {{ fontWeight: 900, duration: 1.1, ease: "power2.out" }}, 0.3);
  [0, 1, 2, 3].forEach((i) => tl.fromTo(`#dm-l-${{i}}`, {{ opacity: 0.3, filter: "blur(6px)" }},
                                        {{ opacity: 1, filter: "blur(0px)", duration: 0.18, immediateRender: false }}, 0.3 + i * 0.1));
  tl.fromTo("#dm-gloss .wd", {{ opacity: 0, filter: "blur(6px)" }}, {{ opacity: 1, filter: "blur(0px)", duration: 0.15, stagger: 0.08 }}, 0.9);
  // 2. carry: the name shrinks onto the spot where the live demo prints it, then an iris opens the demo around it
  const tc = Math.max(1.7, C.L8 - 1.6);
  tl.to("#dm-gloss", {{ opacity: 0, filter: "blur(8px)", duration: 0.15 }}, tc - 0.1);
  tl.to("#dm-word", {{ x: 0, y: 0, scale: 1, duration: 0.6, ease: "power3.inOut" }}, tc);
  tl.fromTo("#dm-cam-a", cam(1.0, 960, 540), {{ ...cam(1.0, 960, 540), duration: 0.01 }}, tc);
  tl.to("#dm-a", {{ clipPath: "circle(2300px at 668px 284px)", duration: 0.55, ease: "power3.in" }}, tc + 0.55);
  tl.set(["#dm-flood", "#dm-word"], {{ opacity: 0 }}, tc + 1.15);            // covered by the demo by now
  tl.to("#dm-cam-a", {{ ...cam(1.12, 900, 470), duration: 1.2, ease: "power2.inOut" }}, tc + 1.1);
  // 3. vertical whip down the page to the controls
  const tb = C.L8 + 0.1;
  tl.to("#dm-a", {{ y: -320, filter: "blur(16px)", opacity: 0, duration: 0.3, ease: "power3.in" }}, tb - 0.3);
  tl.fromTo("#dm-b", {{ y: 320, filter: "blur(16px)", opacity: 0 }}, {{ y: 0, filter: "blur(0px)", opacity: 1, duration: 0.4, ease: "expo.out" }}, tb);
  tl.fromTo("#dm-cam-b", cam(B.z, B.cx, B.cy), {{ ...cam(B.z, B.cx, B.cy), duration: 0.01 }}, tb);
  // the pointer: pick the sentence, choose the mobile line, press Transcribe
  const P = {{ chip: onB(580, 218), mobile: onB(531, 659), go: onB(1406, 659) }};
  const tip = (p) => ({{ x: p[0] - 12, y: p[1] - 7 }});
  tl.fromTo("#dm-cursor", {{ opacity: 0, x: 1700, y: 1000 }}, {{ opacity: 1, duration: 0.15 }}, tb + 0.2);
  tl.to("#dm-cursor", {{ ...tip(P.chip), duration: 0.6, ease: "expo.out" }}, tb + 0.2);
  const press = (t) => {{
    tl.to("#dm-cursor", {{ scale: 0.82, duration: 0.07, ease: "power2.in", transformOrigin: "18% 12%" }}, t);
    tl.to("#dm-cursor", {{ scale: 1, duration: 0.16, ease: "power2.out" }}, t + 0.07);
  }};
  press(tb + 0.85);
  tl.to("#dm-cursor", {{ ...tip(P.mobile), duration: 0.55, ease: "power3.inOut" }}, C.send);
  press(C.send + 0.6);
  tl.to("#dm-cursor", {{ ...tip(P.go), duration: 0.6, ease: "power3.inOut" }}, C.compare - 0.65);
  press(C.compare);
  // 4. the click on Transcribe: the results arrive slightly large and settle (match cut on the click's rebound)
  const tr = Math.min(C.compare + 0.25, C.L9 - 0.2);
  tl.to("#dm-cursor", {{ opacity: 0, duration: 0.1 }}, tr);
  tl.set("#dm-b", {{ opacity: 0 }}, tr);
  tl.set("#dm-c", {{ opacity: 1 }}, tr);
  tl.fromTo("#dm-cam-c", cam(1.12, 960, 640), {{ ...cam(1.0, 960, 640), duration: 0.5, ease: "expo.out" }}, tr);
  tl.to("#dm-cam-c", {{ ...cam(1.75, 672, 742), duration: 0.9, ease: "power3.inOut" }}, C.eleven - 0.6);
  tl.to("#dm-cam-c", {{ ...cam(1.75, 1248, 722), duration: 0.9, ease: "power3.inOut" }}, C.five - 0.55);
  tl.to("#dm-cam-c", {{ ...cam(1.18, 960, 700), duration: 0.9, ease: "power3.inOut" }}, C.both - 0.2);
  tl.fromTo("#dm-both", {{ opacity: 0, scale: 0.85, filter: "blur(8px)" }},
            {{ opacity: 1, scale: 1, filter: "blur(0px)", duration: 0.45, ease: "back.out(1.4)" }}, C.both + 0.5);
  // 5. horizontal whip out
  tl.to("#dm-all", {{ x: -260, filter: "blur(14px)", opacity: 0, duration: 0.3, ease: "power3.in" }}, C.dur - 0.3);"""
write("demo", scene("demo", css, body, script))
