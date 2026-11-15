"""Whisper's BasicTextNormalizer (lower-case, drop bracketed text and punctuation/symbols), with a copy of the same
logic as a fallback so the offline tools run without installing transformers."""
import re
import unicodedata

try:
    from transformers.models.whisper.english_normalizer import BasicTextNormalizer
except ImportError:
    class BasicTextNormalizer:
        def __call__(self, s):
            s = s.lower()
            s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)
            s = re.sub(r"\(([^)]+?)\)", "", s)
            s = "".join(" " if unicodedata.category(c)[0] in "MSP" else c for c in unicodedata.normalize("NFKC", s))
            return re.sub(r"\s+", " ", s.lower())
