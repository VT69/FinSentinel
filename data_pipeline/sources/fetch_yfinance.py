"""
fetch_yfinance.py — Download OHLCV price data via yfinance.

Tickers pulled
--------------
Crypto:       BTC-USD, ETH-USD
Equity:       ^NSEI, ^GSPC, ^DJI, ^IXIC, ^FTSE, ^N225
Volatility:   ^VIX, ^VXN
Commodities:  GC=F, CL=F
FX:           USDINR=X, DX-Y.NYB
Bonds:        ^TNX, ^TYX, ^IRX

Output columns per CSV
----------------------
open, high, low, close, volume, adj_close,
log_return, vol_7d, vol_14d, vol_30d, vol_60d

Rolling vols are annualised with √365 for crypto (trades every day) and √252 otherwise.
Index: date (YYYY-MM-DD, no timezone).
NaNs preserved — no forward-fill.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _ROOT / "data" / "raw" / "yfinance"

# ---------------------------------------------------------------------------
# Tickers
# ---------------------------------------------------------------------------
TICKERS = [
    # Crypto
    "BTC-USD", "ETH-USD",
    # Equity indices
    "^NSEI", "^GSPC", "^DJI", "^IXIC", "^FTSE", "^N225",
    # Volatility
    "^VIX", "^VXN",
    # Commodities
    "GC=F", "CL=F",
    # FX
    "USDINR=X", "DX-Y.NYB",
    # Bonds / yields
    "^TNX", "^TYX", "^IRX",
]

START_DATE = "2010-01-01"
CRYPTO = {"BTC-USD", "ETH-USD"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ticker_to_filename(ticker: str) -> str:
    """
    Convert a ticker symbol to a safe filename.
    ^ and = are replaced with _, - is kept.
    e.g. '^NSEI' → '_NSEI.csv', 'GC=F' → 'GC_F.csv'
    """
    clean = re.sub(r"[\^=]", "_", ticker)
    return f"{clean}.csv"


def _compute_features(df: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """Add log_return and annualised rolling volatility columns."""
    df = df.copy()
    df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    for window in [7, 14, 30, 60]:
        df[f"vol_{window}d"] = (
            df["log_return"].rolling(window).std() * np.sqrt(periods_per_year)
        )
    return df


def _download_ticker(ticker: str, log) -> pd.DataFrame | None:
    """
    Download a single ticker from yfinance and return a cleaned DataFrame.
    Returns None if the download fails or returns empty data.
    """
    try:
        raw = yf.download(
            ticker,
            start=START_DATE,
            auto_adjust=False,
            progress=False,
            multi_level_index=False,
        )
    except Exception as exc:
        log.error("yfinance download failed for %s: %s", ticker, exc)
        return None

    if raw is None or raw.empty:
        log.warning("No data returned for ticker: %s", ticker)
        return None

    # Flatten column names (yfinance may return MultiIndex)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join(c).strip().lower() for c in raw.columns]
    else:
        raw.columns = [c.lower().replace(" ", "_") for c in raw.columns]

    # Standardise column names
    col_map = {
        "open": "open", "high": "high", "low": "low",
        "close": "close", "volume": "volume",
        "adj close": "adj_close", "adj_close": "adj_close",
        "adjclose": "adj_close",
    }
    raw = raw.rename(columns={c: col_map.get(c, c) for c in raw.columns})

    # Ensure required columns exist
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in raw.columns:
            raw[col] = np.nan
    if "adj_close" not in raw.columns:
        raw["adj_close"] = raw["close"]

    # Keep only needed columns before feature engineering
    raw = raw[["open", "high", "low", "close", "volume", "adj_close"]]

    # Strip timezone from index, keep date only
    raw.index = pd.to_datetime(raw.index).tz_localize(None).normalize()
    raw.index.name = "date"
    raw.index = raw.index.strftime("%Y-%m-%d")

    # Add derived columns
    raw = _compute_features(raw, 365 if ticker in CRYPTO else 252)

    return raw


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_yfinance(log=None):
    """
    Download all configured tickers and save each as a CSV.

    Args:
        log: A logger instance (from utils.logger.get_logger).
             If None, a basic print-based stub is used.

    Returns:
        List of manifest record dicts (one per successful ticker).
    """
    if log is None:
        import logging
        log = logging.getLogger("pipeline.yfinance")

    from utils.cache import is_cache_valid
    from utils.manifest import build_record

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    log.info("=== yfinance: starting downloads for %d tickers ===", len(TICKERS))

    for ticker in TICKERS:
        filename = _ticker_to_filename(ticker)
        out_path = _RAW_DIR / filename

        if is_cache_valid(out_path):
            log.info("CACHE HIT  — %s  (%s)", ticker, filename)
            # Still build a manifest record from the cached file
            try:
                df_cached = pd.read_csv(out_path, index_col="date")
                records.append(build_record("yfinance", out_path, df_cached))
            except Exception as exc:
                log.warning("Could not read cached file %s: %s", out_path, exc)
            continue

        log.info("Downloading  — %s", ticker)
        df = _download_ticker(ticker, log)

        if df is None:
            log.error("FAILED     — %s  (no data, skipping)", ticker)
            continue

        try:
            df.to_csv(out_path)
            log.info("SAVED      — %s  → %s  (%d rows)", ticker, filename, len(df))
            records.append(build_record("yfinance", out_path, df))
        except Exception as exc:
            log.error("Could not save %s: %s", out_path, exc)

    log.info("=== yfinance: complete — %d/%d tickers saved ===", len(records), len(TICKERS))
    return records


# ---------------------------------------------------------------------------
# Standalone run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.logger import get_logger
    fetch_yfinance(log=get_logger("yfinance"))
