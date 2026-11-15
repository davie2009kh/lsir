# Lsir demo film: spec

GOAL:    a judge who watches once (sound on or off) knows: Armenian speech recognition breaks over a phone line,
         and Lsir hears it better, shown on a real sentence in the real live demo.
OUTPUT:  HyperFrames, 1920 x 1080, 30 fps, MP4. Hard cap 2:00, target about 1:50.
TIME:    set by tools/build.py from the voice lines (narration.json); about 105-110 s, no music.

STORYLINE (picked, no stops asked): "One sentence down the line". The worked example is FLEURS test clip 555,
         heard clean, then after the GSM mobile line, then transcribed live by both models.
         Others considered: "A clinic call in real time" (a staged booking call; needs a real recorded caller we
         do not have) and "The number that lands" (open on 108%, all charts; loses the human sentence).

LOOK:    the live demo's own system, so film and product read as one thing.
         ground  #f4ecea (pale pink tuff), flat, no texture
         ink     #24222a for every word that is read; muted #6b6370 for labels
         accent  pomegranate #8c1d3a: the word Lsir, our result, the end card. Ararat blue #36577a: the original model.
         chart fills (validated with the dataviz validator on #f4ecea): original #2e6db5, Lsir #b8344f
         highlighter #ffe28a behind wrong words, exactly as in the demo
         display: Noto Serif Armenian 900 (the word Լսիր only), 300-420 px
         body: Noto Sans / Noto Sans Armenian, 600-700, 44-72 px for lines, 30-36 px labels, tracking 0
         medium: real product screenshots (2x captures of the live Space) on the tuff ground, 12 px radius,
         shadow 0 18px 40px rgba(36,34,42,0.12)
OBJECTS: the waveform is the real loudness envelope of clip 555 (120 bars); the phone band is 0-8,000 Hz with
         300-3,400 Hz kept, as in the demo; the model is one block split into "listens" and "writes".
MOTION:  enter power3.out 0.5 s; camera moves power2.inOut 1.0-1.4 s; exits 0.25 s power2.in.
         One hero move per shot. Text never moves while it is read.
SHOTS:   hook      waveform grows, clean clip plays; it collapses to the mobile envelope, degraded clip plays;
                   the original model's real transcript types in, wrong words get highlighted, "11 of 16 words wrong"
         problem   a clinic greeting in Armenian sits in a call bubble; the phone band greys out what is lost
         baseline  108% counts up past a 100% "writing nothing" marker (whisper-small, 932 test recordings)
         demo      the live demo: hero, pick the sentence, mobile line, Transcribe; push in on each result card
         how       one block grows: listens / writes, the adapter clips onto "listens", phone-line clips feed it;
                   then two bars 22.1% vs 20.8%, plus the Common Voice replication
         next      real calls, older voices; end card: Լսիր, team, links
SOUND:   the voiceover (ElevenLabs, added later; captions carry it until then) and the two real speech clips.
         No music, no synthesised effects.
BANNED:  dark void; glow; gradients; three cards in a row; KPI grids; everything centred; fade-and-slide-up on every
         element; a logo docked in a corner; text that moves while read; numbers not in the write-up;
         middle-dot meta strings; all-caps labels.
QA:      frames at each scene's midpoint and at every number's landing; captions inside 80% title-safe; nothing
         behind the caption band (bottom 200 px) except ground.
