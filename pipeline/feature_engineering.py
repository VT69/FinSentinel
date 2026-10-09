"""
pipeline/feature_engineering.py — features for the (archived) sentiment-augmented
Random Forest experiment. See docs/MODEL_REPORT.md for why it was archived:
sentiment only exists for ~3 months (2024-10 → 2025-01), i.e. ~13 usable rows.

Fixes vs the original:
  * rows are dropped only for NaNs in the columns actually used (the original
    `df.dropna()` also dropped the 59 warm-up rows of the unused `vol_60`, taking
    75 aligned rows down to ~11)
  * the target is documented correctly: std of the NEXT `target_horizon` log returns
"""
import numpy as np

FEATURES = [
    "finbert_score", "vader_score",
    "finbert_surprise", "vader_surprise",
    "abs_return", "sq_return",
    "sent_x_vol5", "sent_x_vol22", "sent_x_absret",
]
TARGET = "log_vol_target"


def build_features(df, target_horizon=5):
    """
    df must contain: date, return (daily LOG return), finbert_score, vader_score.
    Returns (X, y, FEATURES).
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Volatility memory (trailing windows, known at the close of day t)
    df["vol_5"] = df["return"].rolling(5).std()
    df["vol_22"] = df["return"].rolling(22).std()

    # Target: std of r[t+1 .. t+target_horizon] — NOT "next-day" volatility
    df["vol_target"] = df["return"].rolling(target_horizon).std().shift(-target_horizon)
    df[TARGET] = np.log(df["vol_target"] + 1e-6)

    # Price dynamics
    df["abs_return"] = np.abs(df["return"])
    df["sq_return"] = df["return"] ** 2

    # Sentiment surprise: today's tone minus its trailing 5-day mean
    df["finbert_surprise"] = df["finbert_score"] - df["finbert_score"].rolling(5).mean()
    df["vader_surprise"] = df["vader_score"] - df["vader_score"].rolling(5).mean()

    # Sentiment interactions
    df["sent_x_vol5"] = df["finbert_score"] * df["vol_5"]
    df["sent_x_vol22"] = df["finbert_score"] * df["vol_22"]
    df["sent_x_absret"] = df["finbert_score"] * df["abs_return"]

    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    return df[FEATURES], df[TARGET], FEATURES
