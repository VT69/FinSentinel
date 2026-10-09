"""
pipeline/sentiment.py — headline cleaning and FinBERT/VADER scoring.

Fixes (docs/ISSUES.md):
  P0-3  FinBERT score is computed by LABEL NAME via model.config.id2label.
        ProsusAI/finbert's order is {0: positive, 1: negative, 2: neutral};
        the original notebook computed probs[2] - probs[0] = P(neutral) - P(positive).
  P1-11 cleaning keeps digits, %, $ and basic punctuation ("$450M" used to become "m"),
        and non-Latin headlines are flagged/dropped instead of being reduced to
        stray English fragments.

transformers / torch / vaderSentiment are imported lazily so this module can be
imported (and unit-tested) without the heavy NLP stack.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Mapping, Sequence

import numpy as np

_URL = re.compile(r"http\S+|www\.\S+")
_KEEP = re.compile(r"[^\w\s$%.,:;!?'\"()&+\-/]")


def latin_share(text: str) -> float:
    letters = [c for c in str(text) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(unicodedata.name(c, "").startswith("LATIN") for c in letters) / len(letters)


def clean_text(text) -> str:
    if not isinstance(text, str):
        return ""
    text = _URL.sub("", text)
    text = _KEEP.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def is_english_candidate(text: str, min_latin_share: float = 0.9, min_len: int = 10) -> bool:
    """Cheap language gate: FinBERT and VADER are English-only models."""
    return len(text) > min_len and latin_share(text) >= min_latin_share


def finbert_polarity(probs: np.ndarray, id2label: Mapping[int, str]) -> np.ndarray:
    """P(positive) − P(negative), resolved by label NAME (never by position)."""
    probs = np.atleast_2d(np.asarray(probs, dtype="float64"))
    label_to_idx = {str(v).lower(): int(k) for k, v in id2label.items()}
    try:
        pos, neg = label_to_idx["positive"], label_to_idx["negative"]
    except KeyError:
        raise ValueError(f"model labels {sorted(label_to_idx)} lack 'positive'/'negative'") from None
    return probs[:, pos] - probs[:, neg]


def score_finbert(texts: Sequence[str], model_name: str = "ProsusAI/finbert", batch_size: int = 32) -> np.ndarray:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            enc = tok(list(texts[i:i + batch_size]), return_tensors="pt", truncation=True, padding=True, max_length=512)
            probs = torch.softmax(model(**enc).logits, dim=1).numpy()
            out.append(finbert_polarity(probs, model.config.id2label))
    return np.concatenate(out) if out else np.array([])


def score_vader(texts: Sequence[str]) -> np.ndarray:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    v = SentimentIntensityAnalyzer()
    return np.array([v.polarity_scores(t)["compound"] for t in texts])
