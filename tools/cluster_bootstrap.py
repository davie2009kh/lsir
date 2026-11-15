"""Stricter significance check: paired bootstrap that resamples whole clusters (sentences or speakers).

Clips that share a speaker or a sentence are correlated, so resampling single clips can make confidence
intervals too narrow. Here we resample whole clusters instead: FLEURS sentence ids, Common Voice speaker ids.

Usage: python tools/cluster_bootstrap.py <transcripts.csv> <cluster column> <model A column> <model B column> [...pairs]
"""
import re
import sys

import jiwer
import numpy as np
import pandas as pd
from transformers.models.whisper.english_normalizer import BasicTextNormalizer

_norm = BasicTextNormalizer()


def norm(t):
    return re.sub(r"\s+", " ", _norm(t if isinstance(t, str) else "")).strip()


def errors(refs, hyps):
    e = np.array([jiwer.process_words(r, h).wer * len(r.split()) if r else 0.0 for r, h in zip(refs, hyps)])
    n = np.array([max(len(r.split()), 1) for r in refs])
    return e, n


def cluster_paired(df, cluster, col_a, col_b, n_boot=2000, seed=42):
    refs = [norm(r) for r in df["ref"]]
    ea, n = errors(refs, [norm(h) for h in df[col_a]])
    eb, _ = errors(refs, [norm(h) for h in df[col_b]])
    groups = df.groupby(cluster).indices
    keys = list(groups)
    ga = np.array([ea[groups[k]].sum() for k in keys])
    gb = np.array([eb[groups[k]].sum() for k in keys])
    gn = np.array([n[groups[k]].sum() for k in keys])
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        s = rng.integers(0, len(keys), len(keys))
        diffs.append(100 * (gb[s].sum() - ga[s].sum()) / gn[s].sum())
    diffs = np.array(diffs)
    return {"clusters": len(keys), "WER A": 100 * ea.sum() / n.sum(), "WER B": 100 * eb.sum() / n.sum(),
            "change": 100 * (eb.sum() - ea.sum()) / n.sum(),
            "95% CI": (np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)),
            "B better in %": 100 * (diffs < 0).mean()}


if __name__ == "__main__":
    path, cluster, *cols = sys.argv[1:]
    df = pd.read_csv(path)
    for a, b in zip(cols[::2], cols[1::2]):
        r = cluster_paired(df, cluster, a, b)
        print(f"{b}\n  vs {a}\n  clusters={r['clusters']}  WER {r['WER A']:.1f} -> {r['WER B']:.1f}  "
              f"change {r['change']:+.2f} (95% CI {r['95% CI'][0]:+.2f} to {r['95% CI'][1]:+.2f}), B better in {r['B better in %']:.0f}%")
