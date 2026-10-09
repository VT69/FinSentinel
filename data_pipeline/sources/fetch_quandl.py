"""
fetch_quandl.py — Pull datasets from Nasdaq Data Link (formerly Quandl).

Requires:
    QUANDL_API_KEY environment variable
    (free registration at data.nasdaq.com)

⚠ AVAILABILITY NOTE ⚠
Many Quandl datasets have been deprecated or moved behind paywalls:
  - WIKI/PRICES (US equity prices) was discontinued June 2018.
  - ODA (IMF commodity prices) may still be accessible.
  - FRED duplicate series are skipped to avoid redundancy.

Strategy
--------
The fetcher probes each dataset, logs what's available, saves what it
can, and gracefully logs failures for unavailable datasets.

Output
------
data/raw/quandl/{dataset_code}.csv
"""

import os
import re
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _ROOT / "data" / "raw" / "quandl"

START_DATE = "2010-01-01"

# Datasets to attempt (in order of priority)
DATASETS = [
    {
        "code":        "WIKI/PRICES",
        "description": "US equity prices (likely discontinued — probe only)",
        "expected_unavailable": True,
    },
    {
        "code":        "ODA/PCOALAU_USD",
        "description": "IMF Coal price (USD/mt)",
        "expected_unavailable": False,
    },
    {
        "code":        "ODA/PNGAS_USD",
        "description": "IMF Natural Gas price (USD/MMBtu)",
        "expected_unavailable": False,
    },
    {
        "code":        "ODA/POILWTI_USD",
        "description": "IMF WTI Crude Oil price (USD/bbl)",
        "expected_unavailable": False,
    },
    {
        "code":        "ODA/PGOLD_USD",
        "description": "IMF Gold price (USD/troy oz)",
        "expected_unavailable": False,
    },
    {
        "code":        "ODA/PCOFFOTM_USD",
        "description": "IMF Coffee price (USD/kg)",
        "expected_unavailable": False,
    },
    {
        "code":        "ODA/PSILVER_USD",
        "description": "IMF Silver price (USD/troy oz)",
        "expected_unavailable": False,
    },
]


def _code_to_filename(code: str) -> str:
    """Convert 'WIKI/PRICES' → 'WIKI_PRICES.csv'."""
    return re.sub(r"[/\\]", "_", code) + ".csv"


def _pull_dataset(ndl, code: str, log) -> pd.DataFrame | None:
    """
    Attempt to pull a Nasdaq Data Link dataset.
    Returns a cleaned DataFrame or None on failure.
    """
    try:
        df = ndl.get(code, start_date=START_DATE)

        if df is None or df.empty:
            log.warning("Dataset %s returned empty data", code)
            return None

        # Normalise index
        df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
        df.index = df.index.strftime("%Y-%m-%d")
        df.index.name = "date"
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        return df

    except Exception as exc:
        err_str = str(exc).lower()
        if "not found" in err_str or "404" in err_str or "premium" in err_str or "forbidden" in err_str:
            log.warning("Dataset %s unavailable (discontinued/premium): %s", code, exc)
        else:
            log.error("Dataset %s failed: %s", code, exc)
        return None


def fetch_quandl(log=None):
    """
    Pull all configured Nasdaq Data Link datasets.

    Returns:
        List of manifest record dicts.
    """
    if log is None:
        import logging
        log = logging.getLogger("pipeline.quandl")

    from utils.cache import is_cache_valid
    from utils.manifest import build_record

    api_key = os.getenv("QUANDL_API_KEY", "").strip()
    if not api_key:
        log.error("QUANDL_API_KEY not set — skipping Quandl/Nasdaq Data Link source.")
        return []

    try:
        import nasdaqdatalink   # successor of the deprecated `quandl` package (same get() API)
        nasdaqdatalink.ApiConfig.api_key = api_key
    except ImportError:
        log.error("nasdaq-data-link not installed. Run: pip install nasdaq-data-link")
        return []

    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    records = []

    log.info("=== Quandl: attempting %d datasets ===", len(DATASETS))

    for ds in DATASETS:
        code = ds["code"]
        filename = _code_to_filename(code)
        out_path = _RAW_DIR / filename

        if ds.get("expected_unavailable"):
            log.info("PROBING    — %s (%s)", code, ds["description"])
        else:
            log.info("Downloading  — %s (%s)", code, ds["description"])

        if is_cache_valid(out_path):
            log.info("CACHE HIT  — %s", code)
            try:
                df_cached = pd.read_csv(out_path, index_col="date")
                records.append(build_record("quandl", out_path, df_cached))
            except Exception as exc:
                log.warning("Could not read cached file %s: %s", out_path, exc)
            continue

        df = _pull_dataset(nasdaqdatalink, code, log)

        if df is None:
            if ds.get("expected_unavailable"):
                log.info("CONFIRMED  — %s is unavailable (as expected)", code)
            else:
                log.error("FAILED     — %s (no data saved)", code)
            continue

        try:
            df.to_csv(out_path)
            log.info("SAVED      — %s  (%d rows, %d cols)", code, len(df), len(df.columns))
            records.append(build_record("quandl", out_path, df,
                                        extra={"description": ds["description"]}))
        except Exception as exc:
            log.error("Could not save %s: %s", out_path, exc)

    log.info("=== Quandl: complete — %d datasets saved ===", len(records))
    return records


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.logger import get_logger
    from dotenv import load_dotenv
    load_dotenv()
    fetch_quandl(log=get_logger("quandl"))
