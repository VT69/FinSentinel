"""
analysis/common.py — shared loaders and faithful re-implementations of the
repo's own transformations, so the analysis scripts reproduce exactly what the
original notebooks/pipeline did (not an idealised version).

Every function below cites the original code it mirrors.
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "analysis" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 42
AUDITED_COMMIT = "e1eb270"   # the commit audited in docs/; later commits fix or delete the files below


def legacy_bytes(repo_path: str) -> bytes:
    """Content of `repo_path` as it was at the audited commit (needs a git checkout with history)."""
    try:
        return subprocess.run(["git", "show", f"{AUDITED_COMMIT}:{repo_path}"], cwd=ROOT,
                              check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"Cannot read {repo_path}@{AUDITED_COMMIT}: run inside a full git clone "
                           f"(git fetch --unshallow if needed). {exc}") from exc


# ── Loaders ──────────────────────────────────────────────────────────────────

def load_prices(asset: str) -> pd.DataFrame:
    """data/raw/{btc,nifty}_prices.csv as committed (written by notebooks/01_data_collection.ipynb cell 3)."""
    df = pd.read_csv(RAW / f"{asset.lower()}_prices.csv", parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def load_text() -> pd.DataFrame:
    return pd.read_csv(RAW / "text_data.csv")


def load_trends(asset: str) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{asset.lower()}_google_trends.csv")


# ── notebooks/02_preprocessing.ipynb cell 3-4 ───────────────────────────────

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_text(text_df: pd.DataFrame) -> pd.DataFrame:
    df = text_df.copy()
    df["clean_text"] = df["text"].apply(clean_text)
    return df[df["clean_text"].str.len() > 10]


# ── VADER (notebooks/03_sentiment_analysis.ipynb cell 5) ─────────────────────

def vader_scores(texts: pd.Series) -> pd.Series:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    v = SentimentIntensityAnalyzer()
    return texts.apply(lambda t: v.polarity_scores(t)["compound"])


# ── notebooks/04_time_alignment.ipynb cells 3,5-8 ────────────────────────────

def align_btc(prices: pd.DataFrame, sent: pd.DataFrame) -> pd.DataFrame:
    """Reproduces btc_sentiment_aligned.csv. `sent` must have timestamp, asset,
    finbert_score, vader_score columns."""
    p = prices.copy()
    p["return"] = np.log(p["close"]).diff()          # nb04 cell 3 overwrites pct return with log return
    p = p.dropna()
    s = sent.copy()
    s["timestamp"] = pd.to_datetime(s["timestamp"], utc=True)
    daily = (s.groupby([s["timestamp"].dt.date, "asset"])
               .agg({"finbert_score": "mean", "vader_score": "mean"})
               .reset_index().rename(columns={"timestamp": "date"}))
    daily["date"] = pd.to_datetime(daily["date"])
    p["date"] = pd.to_datetime(p["date"]).dt.tz_localize(None)
    m = pd.merge(p, daily[daily["asset"] == "BTC"], on="date", how="inner")
    for lag in [1, 2, 3, 5]:
        m[f"finbert_lag_{lag}"] = m["finbert_score"].shift(lag)
        m[f"vader_lag_{lag}"] = m["vader_score"].shift(lag)
    return m.dropna()


# ── pipeline/feature_engineering.py:3-56 (verbatim logic, print removed) ────

PIPELINE_FEATURES = [
    "finbert_score", "vader_score",
    "finbert_surprise", "vader_surprise",
    "abs_return", "sq_return",
    "sent_x_vol5", "sent_x_vol22", "sent_x_absret",
]


def build_features(df: pd.DataFrame, target_horizon: int = 5):
    df = df.sort_values("date").reset_index(drop=True)
    df["vol_5"] = df["return"].rolling(5).std()
    df["vol_22"] = df["return"].rolling(22).std()
    df["vol_60"] = df["return"].rolling(60).std()
    df["vol_target"] = df["return"].rolling(5).std().shift(-target_horizon)
    df["log_vol_target"] = np.log(df["vol_target"] + 1e-6)
    df["abs_return"] = np.abs(df["return"])
    df["sq_return"] = df["return"] ** 2
    df["finbert_surprise"] = df["finbert_score"] - df["finbert_score"].rolling(5).mean()
    df["vader_surprise"] = df["vader_score"] - df["vader_score"].rolling(5).mean()
    df["sent_x_vol5"] = df["finbert_score"] * df["vol_5"]
    df["sent_x_vol22"] = df["finbert_score"] * df["vol_22"]
    df["sent_x_absret"] = df["finbert_score"] * df["abs_return"]
    df = df.dropna().reset_index(drop=True)
    return df[PIPELINE_FEATURES], df["log_vol_target"], PIPELINE_FEATURES, df


# ── Price-only feature frame over the FULL history (for honest re-evaluation) ─

def price_feature_frame(prices: pd.DataFrame, horizon: int = 5) -> pd.DataFrame:
    """Same price features + target definition as pipeline/feature_engineering.py,
    but computed over the full committed price history (sentiment does not exist
    before 2024-10-03, so it cannot be included here)."""
    df = prices[["date", "close"]].copy()
    df["return"] = np.log(df["close"]).diff()
    df["abs_return"] = df["return"].abs()
    df["sq_return"] = df["return"] ** 2
    df["vol_5"] = df["return"].rolling(5).std()
    df["vol_22"] = df["return"].rolling(22).std()
    df["vol_60"] = df["return"].rolling(60).std()
    df["vol_target"] = df["return"].rolling(5).std().shift(-horizon)
    df["log_vol_target"] = np.log(df["vol_target"] + 1e-6)
    return df.dropna().reset_index(drop=True)


PRICE_FEATURES = ["abs_return", "sq_return", "vol_5", "vol_22", "vol_60"]
