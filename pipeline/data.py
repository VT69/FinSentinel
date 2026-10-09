"""pipeline/data.py — repo paths and validated loaders for the committed price files."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"
DASHBOARD_DATA = ROOT / "dashboard" / "data"
FIGURES = ROOT / "reports" / "figures"
ASSETS = ("BTC", "NIFTY")

REQUIRED_PRICE_COLUMNS = {"date", "open", "high", "low", "close", "volume"}


def price_path(asset: str) -> Path:
    asset = asset.upper()
    if asset not in ASSETS:
        raise ValueError(f"Unknown asset {asset!r}; expected one of {ASSETS}")
    return RAW / f"{asset.lower()}_prices.csv"


def load_prices(asset: str) -> pd.DataFrame:
    """Load data/raw/<asset>_prices.csv, validate it, return it sorted by date."""
    path = price_path(asset)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run notebooks/01_data_collection.ipynb or "
                                "`python data_pipeline/fetch_all.py --source yfinance`.")
    df = pd.read_csv(path, parse_dates=["date"])
    missing = REQUIRED_PRICE_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns {sorted(missing)}")
    df = df.sort_values("date").reset_index(drop=True)
    if df["date"].duplicated().any():
        raise ValueError(f"{path.name} has duplicate dates")
    if (df["close"] <= 0).any() or df["close"].isna().any():
        raise ValueError(f"{path.name} has non-positive or missing closes")
    return df


def load_close(asset: str) -> pd.Series:
    df = load_prices(asset)
    return df.set_index("date")["close"].rename(asset.upper())
