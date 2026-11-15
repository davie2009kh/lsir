# Extract 3 FLEURS Armenian test clips (CC-BY-4.0) as example audio for the Lsir demo.
import io, json, urllib.request
import pandas as pd, soundfile as sf
ROWS = [555, 473, 706]
urllib.request.urlretrieve("https://huggingface.co/api/datasets/google/fleurs/parquet/hy_am/test/0.parquet", "/tmp/t.parquet")
df = pd.read_parquet("/tmp/t.parquet")
meta = []
for r in ROWS:
    row = df.iloc[r]
    wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
    name = f"fleurs_test_{r}.wav"
    sf.write(f"/kaggle/working/{name}", wav, sr)
    meta.append({"row": r, "id": int(row["id"]), "file": name, "sr": sr, "seconds": round(len(wav) / sr, 2),
                 "raw_transcription": row["raw_transcription"], "transcription": row["transcription"]})
    print(meta[-1])
json.dump(meta, open("/kaggle/working/examples.json", "w"), ensure_ascii=False, indent=1)
