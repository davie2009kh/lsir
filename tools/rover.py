"""Simple ROVER-style system combination: word-level majority voting between ASR systems.

The first system is the "pivot". Every other system's words are aligned to the pivot's words (edit-distance
alignment). For each pivot word, each system votes for the word it aligned there (or for "nothing" if it deleted
it). The most-voted option wins, and ties keep the pivot's word. Extra words inserted by other systems are ignored.
"""
import re
from collections import Counter

import jiwer
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

_norm = BasicTextNormalizer()


def norm(t):
    return re.sub(r"\s+", " ", _norm(t if isinstance(t, str) else "")).strip()


def align_to_pivot(pivot_words, other):
    """For each pivot word, return the word `other` aligned to it ('' if deleted)."""
    slots = [""] * len(pivot_words)
    if not pivot_words:
        return slots
    if not other.strip():
        return slots
    out = jiwer.process_words(" ".join(pivot_words), other)
    hyp = out.hypotheses[0]
    for chunk in out.alignments[0]:
        if chunk.type in ("equal", "substitute"):
            for k in range(chunk.ref_end_idx - chunk.ref_start_idx):
                slots[chunk.ref_start_idx + k] = hyp[chunk.hyp_start_idx + k]
    return slots


def rover(hyps):
    """hyps: list of normalised hypothesis strings; hyps[0] is the pivot."""
    pivot = hyps[0].split()
    columns = [pivot] + [align_to_pivot(pivot, h) for h in hyps[1:]]
    result = []
    for i, p in enumerate(pivot):
        votes = Counter(col[i] for col in columns)
        best, n = votes.most_common(1)[0]
        if n == votes[p]:          # tie with the pivot -> keep the pivot word
            best = p
        if best:
            result.append(best)
    return " ".join(result)
