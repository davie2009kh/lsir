"""Next steps, then the end card on a pomegranate flood."""
from common import scene, words, letters, write, MUTED, POM

css = f"""
    #nx-a .wd, #nx-b .wd, #nx-tag, #nx-word .lt, #nx-pitch .wd, #nx-team, #nx-links, #nx-fact .wd {{ opacity: 0; }}
    #nx-steps, #nx-end {{ position: absolute; inset: 0; }}
    .nx-line {{ position: absolute; left: 120px; font-size: 130px; font-weight: 800; letter-spacing: -0.025em; line-height: 1; }}
    #nx-a {{ top: 150px; }} #nx-b {{ top: 330px; }}
    #nx-tag {{ position: absolute; left: 124px; top: 90px; font-size: 46px; font-weight: 800; color: {POM}; }}
    #nx-fact {{ position: absolute; left: 120px; top: 560px; width: 1680px; font-size: 62px; font-weight: 700; line-height: 1.2; }}
    #nx-flood {{ position: absolute; inset: 0; background: {POM}; clip-path: circle(0% at 12% 88%); }}
    #nx-word {{ position: absolute; left: 110px; top: 30px; font-family: "Lsir Serif", serif; font-size: 350px; line-height: 1.2;
               color: #fff; letter-spacing: -0.02em; font-weight: 900; }}
    #nx-pitch {{ position: absolute; left: 120px; top: 545px; width: 1680px; font-size: 58px; font-weight: 800; line-height: 1.15; color: #fff; }}
    #nx-team {{ position: absolute; left: 120px; top: 726px; font-size: 38px; font-weight: 600; color: #f3dde2; }}
    #nx-links {{ position: absolute; left: 120px; top: 796px; font-size: 42px; font-weight: 700; color: #fff; line-height: 1.45; }}
    #nx-links b {{ color: #f3dde2; font-weight: 600; margin-right: 18px; }}"""
body = f"""    <div id="nx-steps">
      <div id="nx-tag">Next</div>
      <div id="nx-a" class="nx-line">{words("Real phone calls", "nx-a")}</div>
      <div id="nx-b" class="nx-line">{words("Older voices", "nx-b")}</div>
      <div id="nx-fact">{words("Public Armenian speech data has no speakers over 59.", "nx-f")}</div>
    </div>
    <div id="nx-end">
      <div id="nx-flood"></div>
      <div id="nx-word" lang="hy">{letters("Լսիր", "nx-l")}</div>
      <div id="nx-pitch">{words("Armenian speech-to-text that keeps working over the phone", "nx-p")}</div>
      <div id="nx-team">Alen Gevorgyan, Koryun Gaboyan, David Ghazaryan. Veritas AI High School Hackathon 2026</div>
      <div id="nx-links"><b>Try it</b>huggingface.co/spaces/davitlive5/lsir-armenian-phone-asr<br /><b>Code</b>github.com/davie2009kh/lsir</div>
    </div>"""
script = """
  // blur swap in; each line lands per word while its weight swells from light to heavy
  tl.fromTo("#nx-tag", { opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.18 }, C.L12 - 0.2);
  tl.fromTo("#nx-a .wd", { opacity: 0, filter: "blur(8px)", fontWeight: 300 }, { opacity: 1, filter: "blur(0px)", fontWeight: 800, duration: 0.5, ease: "power2.out", stagger: 0.09 }, C.L12 + 0.1);
  tl.fromTo("#nx-b .wd", { opacity: 0, filter: "blur(8px)", fontWeight: 300 }, { opacity: 1, filter: "blur(0px)", fontWeight: 800, duration: 0.5, ease: "power2.out", stagger: 0.09 }, C.older - 0.1);
  tl.set("#nx-fact .wd", { opacity: 1, color: "#d8cbc6" }, C.public - 0.1);
  tl.to("#nx-fact .wd", { color: "#24222a", duration: 0.12, stagger: 0.08 }, C.public);
  // colour flood from the lower left; the name prints letter by letter as its weight swells
  tl.to("#nx-flood", { clipPath: "circle(150% at 12% 88%)", duration: 0.6, ease: "power2.inOut" }, C.L13 - 0.45);
  tl.to("#nx-steps", { opacity: 0, duration: 0.2 }, C.L13 - 0.1);
  [0, 1, 2, 3].forEach((i) => tl.fromTo(`#nx-l-${i}`, { opacity: 0.3, filter: "blur(8px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.2, immediateRender: false }, C.L13 + i * 0.1));
  tl.fromTo("#nx-word", { fontWeight: 150 }, { fontWeight: 900, duration: 1.1, ease: "power2.out" }, C.L13);
  tl.fromTo("#nx-pitch .wd", { opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.15, stagger: 0.06 }, C.L13 + 0.9);
  tl.fromTo("#nx-team", { opacity: 0 }, { opacity: 1, duration: 0.3 }, C.L13 + 1.6);
  tl.fromTo("#nx-links", { opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.3 }, C.L13 + 1.9);"""
write("next", scene("next", css, body, script))
