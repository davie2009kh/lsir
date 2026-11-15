"""Problem: a caller's first sentence types into a call; then the phone band wipes away what the line drops."""
from common import scene, words, letters, write, MUTED, INK, LINE, POM

HY = "Բարև Ձեզ, ուզում եմ գրանցվել բժշկի մոտ։"
W, KX, KW = 1680, 1680 * 300 / 8000, 1680 * 3100 / 8000      # band: 0-8 kHz across 1680 px; phone keeps 300-3,400 Hz
css = f"""
    #pb-band, #pb-scr, #pb-en .wd, #pb-k-0, #pb-keptl .wd, #pb-lostl .wd, .pb-ax, #pb-none .wd, #pb-bt .wd, #pb-hy .lt {{ opacity: 0; }}
    #pb-all, #pb-call, #pb-band {{ position: absolute; inset: 0; }}
    #pb-tag {{ position: absolute; left: 120px; top: 110px; font-size: 46px; font-weight: 600; color: {MUTED}; }}
    #pb-bubble {{ position: absolute; left: 120px; top: 200px; width: 1140px; min-height: 330px; padding: 52px 60px; background: #fff;
                 border-radius: 36px 36px 36px 8px; box-shadow: 0 24px 60px rgba(36, 34, 42, 0.13); }}
    #pb-hy {{ font-family: "Lsir Arm", sans-serif; font-size: 84px; line-height: 1.25; font-weight: 700; margin: 0; }}
    #pb-en {{ font-size: 44px; color: {MUTED}; margin: 26px 0 0; font-weight: 500; }}
    #pb-phone {{ position: absolute; left: 1390px; top: 110px; width: 410px; height: 740px; border: 14px solid {INK};
                border-radius: 64px; background: #fff; }}
    #pb-phone .spk {{ position: absolute; left: 50%; top: 30px; width: 110px; height: 12px; margin-left: -55px; border-radius: 6px; background: {INK}; }}
    #pb-scr {{ position: absolute; left: 38px; right: 38px; top: 110px; font-size: 44px; font-weight: 800; line-height: 1.2; }}
    #pb-scr span {{ display: block; font-size: 32px; font-weight: 600; color: {MUTED}; margin-top: 12px; }}
    #pb-dots {{ position: absolute; left: 38px; bottom: 70px; display: flex; gap: 18px; }}
    #pb-dots i {{ display: block; width: 30px; height: 30px; border-radius: 15px; background: {POM}; opacity: 0.25; }}
    #pb-bt {{ position: absolute; left: 120px; top: 120px; font-size: 52px; font-weight: 700; }}
    #pb-strip {{ position: absolute; left: 120px; top: 330px; width: {W}px; height: 130px; border-radius: 10px; background: {POM}; overflow: hidden; }}
    #pb-cutl, #pb-cutr {{ position: absolute; top: 0; height: 130px; background: {LINE}; }}
    #pb-cutl {{ left: 0; width: {KX:.1f}px; transform-origin: 0 50%; }}
    #pb-cutr {{ left: {KX + KW:.1f}px; right: 0; transform-origin: 100% 50%; }}
    #pb-keptl {{ position: absolute; left: {120 + KX:.1f}px; top: 250px; font-size: 48px; font-weight: 800; color: {POM}; }}
    #pb-lostl {{ position: absolute; left: {120 + KX + KW + 40:.1f}px; top: 250px; font-size: 48px; font-weight: 700; color: {MUTED}; }}
    .pb-ax {{ position: absolute; top: 480px; font-size: 38px; font-weight: 600; color: {MUTED}; }}
    #pb-none {{ position: absolute; left: 120px; top: 600px; width: 1680px; font-size: 70px; font-weight: 800; line-height: 1.18; }}"""
body = f"""    <div id="pb-all">
    <div id="pb-call">
      <div id="pb-tag">{words("A first sentence to a clinic phone line", "pb-t")}</div>
      <div id="pb-bubble"><p id="pb-hy" lang="hy">{letters(HY, "pb-l")}</p>
        <p id="pb-en">{words("Hello, I would like to book a doctor's appointment.", "pb-e")}</p></div>
      <div id="pb-phone"><i class="spk"></i><div id="pb-scr">Clinic line<span>Voice assistant answering</span></div>
        <div id="pb-dots"><i></i><i></i><i></i></div></div>
    </div>
    <div id="pb-band">
      <div id="pb-bt">{words("The sound of a voice, 0 to 8,000 Hz", "pb-b")}</div>
      <div id="pb-keptl">{words("What a phone line keeps", "pb-k")}</div>
      <div id="pb-lostl">{words("What it throws away", "pb-o")}</div>
      <div id="pb-strip"><i id="pb-cutl"></i><i id="pb-cutr"></i></div>
      <div class="pb-ax" style="left: 120px">0 Hz</div>
      <div class="pb-ax" style="left: {120 + KX + KW - 80:.0f}px">3,400 Hz</div>
      <div class="pb-ax" style="left: 1640px">8,000 Hz</div>
      <div id="pb-none">{words("No published test had measured Armenian speech recognition over the phone.", "pb-n")}</div>
    </div>
    </div>"""
script = """
  // whip in from the right (matches the hook's exit)
  tl.fromTo("#pb-all", { x: 260, filter: "blur(14px)", opacity: 0 }, { x: 0, filter: "blur(0px)", opacity: 1, duration: 0.4, ease: "expo.out" }, 0);
  tl.fromTo("#pb-tag .wd", { opacity: 0, filter: "blur(6px)" }, { opacity: 1, filter: "blur(0px)", duration: 0.15, stagger: 0.06 }, 0.3);
  tl.fromTo("#pb-bubble", { scale: 0.9, transformOrigin: "0% 100%" }, { scale: 1, duration: 0.6, ease: "expo.out" }, 0.25);
  // the caller's sentence types itself in, letter by letter
  const ls = gsap.utils.toArray("#pb-hy .lt");
  ls.forEach((l, i) => tl.to(l, { opacity: 1, duration: 0.05 }, 0.7 + i * 0.045));
  tl.to("#pb-en .wd", { opacity: 1, duration: 0.15, stagger: 0.07 }, 0.7 + ls.length * 0.045 + 0.15);
  // the clinic line rings, then the voice assistant picks up
  tl.fromTo("#pb-phone", { x: 120, opacity: 0, rotation: 0 }, { x: 0, opacity: 1, duration: 0.6, ease: "expo.out" }, 0.4);
  [0, 1, 2, 3, 4, 5].forEach((k) => tl.to("#pb-phone", { rotation: k % 2 ? -3 : 3, duration: 0.07, ease: "sine.inOut" }, C.when - 0.6 + k * 0.07));
  tl.to("#pb-phone", { rotation: 0, duration: 0.1 }, C.when - 0.6 + 6 * 0.07);
  tl.to("#pb-scr", { opacity: 1, duration: 0.2 }, C.when);
  [0, 1, 2].forEach((k) => tl.to(`#pb-dots i:nth-child(${k + 1})`, { opacity: 1, duration: 0.15 }, C.when + 0.4 + k * 0.2));
  // blur swap to the phone band
  tl.to("#pb-call", { opacity: 0, filter: "blur(10px)", duration: 0.2, ease: "power2.in" }, C.L5 - 0.25);
  tl.set("#pb-band", { opacity: 1 }, C.L5 - 0.05);
  tl.fromTo("#pb-strip", { scaleX: 0, transformOrigin: "0% 50%" }, { scaleX: 1, duration: 0.7, ease: "expo.out" }, C.L5);
  tl.to("#pb-bt .wd", { opacity: 1, duration: 0.15, stagger: 0.06 }, C.L5 + 0.1);
  tl.to(".pb-ax", { opacity: 1, duration: 0.3, stagger: 0.08 }, C.L5 + 0.4);
  // the line throws away everything outside 300-3,400 Hz: two wipes close in on the kept band
  tl.fromTo("#pb-cutr", { scaleX: 0 }, { scaleX: 1, duration: 0.8, ease: "power3.inOut" }, C.narrow - 0.1);
  tl.fromTo("#pb-cutl", { scaleX: 0 }, { scaleX: 1, duration: 0.5, ease: "power3.inOut" }, C.narrow + 0.2);
  tl.to("#pb-keptl .wd", { opacity: 1, duration: 0.15, stagger: 0.06 }, C.narrow + 0.5);
  tl.to("#pb-lostl .wd", { opacity: 1, duration: 0.15, stagger: 0.06 }, C.narrow + 0.8);
  // the claim: each word goes from light grey to ink
  tl.set("#pb-none .wd", { opacity: 1, color: "#d8cbc6" }, C.nobody);
  tl.to("#pb-none .wd", { color: "#24222a", duration: 0.12, stagger: 0.08 }, C.nobody + 0.02);
  // hand-off: everything leaves except the strip, which turns neutral; the next scene takes it over
  tl.to(["#pb-bt", "#pb-keptl", "#pb-lostl", ".pb-ax", "#pb-none"], { opacity: 0, filter: "blur(8px)", duration: 0.2, ease: "power2.in" }, C.dur - 0.5);
  tl.to(["#pb-strip", "#pb-cutl", "#pb-cutr"], { backgroundColor: "#e2d4cf", duration: 0.3 }, C.dur - 0.45);"""
write("problem", scene("problem", css, body, script))
