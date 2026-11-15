"""Generate the voiceover with ElevenLabs, one clip per narration line, cached by text.

The API key is read from ~/.config/elevenlabs/key (chmod 600), never from this repo or the command line.
Usage:  python3 tools/voice.py [--voice VOICE_ID] [--model eleven_multilingual_v2] [--only L3,L7]
Then:   python3 tools/build.py   (re-times the whole film from the real clip lengths)
"""
import argparse
import hashlib
import json
import os
import stat
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KEY_FILE = Path.home() / ".config/elevenlabs/key"
OUT = ROOT / "assets/audio/vo"
DEFAULT_VOICE = "21m00Tcm4TlvDq8ikWAM"      # "Rachel", a calm premade ElevenLabs narrator voice


def read_key():
    if not KEY_FILE.exists():
        raise SystemExit(f"No key file. Save it with:  ! mkdir -p ~/.config/elevenlabs && read -rs K && "
                         f"printf %s \"$K\" > {KEY_FILE} && chmod 600 {KEY_FILE}")
    mode = stat.S_IMODE(os.stat(KEY_FILE).st_mode)
    if mode & 0o077:
        raise SystemExit(f"{KEY_FILE} is readable by others; run: chmod 600 {KEY_FILE}")
    return KEY_FILE.read_text().strip()


def tts(key, voice, model, text):
    body = json.dumps({"text": text, "model_id": model,
                       "voice_settings": {"stability": 0.55, "similarity_boost": 0.75, "style": 0.15,
                                          "use_speaker_boost": True}}).encode()
    req = urllib.request.Request(f"https://api.elevenlabs.io/v1/text-to-speech/{voice}?output_format=mp3_44100_128",
                                 data=body, method="POST",
                                 headers={"xi-api-key": key, "Content-Type": "application/json", "Accept": "audio/mpeg"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default=DEFAULT_VOICE)
    ap.add_argument("--model", default="eleven_multilingual_v2")
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    key = read_key()
    OUT.mkdir(parents=True, exist_ok=True)
    lines = json.loads((ROOT / "narration.json").read_text())["lines"]
    only = set(filter(None, a.only.split(",")))
    for l in lines:
        if only and l["id"] not in only:
            continue
        text = l.get("tts", l["text"])
        tag = hashlib.sha1(f"{a.voice}|{a.model}|{text}".encode()).hexdigest()[:12]
        mp3, stamp = OUT / f"{l['id']}.mp3", OUT / f"{l['id']}.tag"
        if mp3.exists() and stamp.exists() and stamp.read_text() == tag:
            print(f"{l['id']}: cached")
            continue
        mp3.write_bytes(tts(key, a.voice, a.model, text))
        stamp.write_text(tag)
        print(f"{l['id']}: generated ({len(text)} characters)")


if __name__ == "__main__":
    main()
