import numpy as np
import pytest

from pipeline import sentiment as s

PROSUS = {0: "positive", 1: "negative", 2: "neutral"}   # ProsusAI/finbert id2label


def test_finbert_polarity_uses_label_names():
    probs = np.array([[0.1, 0.8, 0.1]])                 # clearly negative headline
    assert s.finbert_polarity(probs, PROSUS)[0] == pytest.approx(-0.7)
    # the original notebook computed probs[2] - probs[0] = 0.0 for this headline
    assert probs[0, 2] - probs[0, 0] == pytest.approx(0.0)


def test_finbert_polarity_is_order_independent():
    probs = np.array([[0.8, 0.1, 0.1]])
    shuffled = {0: "NEUTRAL", 1: "Positive", 2: "negative"}
    assert s.finbert_polarity(probs, shuffled)[0] == pytest.approx(0.0)


def test_missing_labels_raise():
    with pytest.raises(ValueError):
        s.finbert_polarity(np.ones((1, 2)) / 2, {0: "up", 1: "down"})


def test_clean_text_keeps_numbers_and_symbols():
    t = s.clean_text("Bitcoin Price Decline Forces $450M in Long Liquidations https://x.y/z")
    assert "$450M" in t and "http" not in t


def test_language_gate():
    assert s.is_english_candidate("Nifty 50, Sensex today: What to expect")
    assert not s.is_english_candidate("शेयर बाजार में बड़ी गिरावट आज")
