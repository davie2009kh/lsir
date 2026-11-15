# Review of v1 (captions, no voice)

Checked: `npx hyperframes check` passes (0 errors, 0 warnings after fixes); frames from the rendered MP4 at
3, 22, 37, 50, 60, 63.5, 67, 88 and 108 s; audio measured (not listened to): -16.5 LUFS integrated, true peak -1 dBTP.

Fixed during review:
- hook transcript wrapped into the "11 of 16" line: reflowed (waveform and text moved up, 48 px type)
- consecutive caption groups overlapped for 0.12 s: groups now end exactly where the next begins
- labels under the 28 px floor (phone screen sub-label, chart axis): raised to 28 px
- two lines left the screen before they could be read: longer pauses before L12 and L13, shorter fact line
- playhead had no stable starting state: one `set`, then `to` tweens

Defaults avoided on purpose (SPEC BANNED): dark void, glow, gradients, three cards, docked logo, all-caps labels.
Known limits: no voice yet (captions carry it); no music (none licensed); the demo footage is still screenshots
of a real run of the live Space, moved by a camera, not a screen recording.

# Review of v3 (more motion, fuller frames)

Asked for: the opening clean clip was silent; more motion, less empty space, better type transitions.
- Silent clip: the FLEURS recording itself peaks at -39 dB; the video copy gets +36 dB of plain gain (the demo's own
  example file is untouched so the live results stay identical). Measured in the render: -20.6 dB mean in its window.
- Motion: per-word blur-ins, light-to-ink sweeps, letter-by-letter typing, variable-weight swells (150 -> 900),
  count-ups (11/16, 108%, 22.1%, 20.8%), highlighter wipes; transitions: whip (hook->problem, demo->how), morph
  (the phone band becomes the error bar), colour flood + iris match-cut (the name lands on the demo's own word,
  measured to 1 px), vertical whip down the page, blur swaps, closing flood.
- Space: type and objects enlarged to fill the 1920x1080 frame (hero numbers 250-440 px, diagram 1680 px wide).
- Check: passes, 0 errors, 0 warnings. Rendered with software rendering (no GPU). -16.7 LUFS, true peak -1 dBTP.
