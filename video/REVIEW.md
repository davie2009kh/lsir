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
