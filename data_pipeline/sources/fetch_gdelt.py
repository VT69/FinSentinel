"""
fetch_gdelt.py — Pull daily GDELT GKG aggregates via Google BigQuery.

Requires:
    GOOGLE_APPLICATION_CREDENTIALS env var pointing to a GCP service
    account JSON key file with BigQuery read access.

What it pulls
-------------
From `gdelt-bq.gdeltv2.gkg` (available from ~2015-02-19 onward):

  date                    — YYYY-MM-DD
  event_count             — total article/record count for that day
  avg_tone                — mean of V2Tone field (first element, comma-split)
  negative_share          — fraction of records with tone < 0
  conflict_theme_count    — records whose Themes contain CRISISLEX/TERROR/CONFLICT
  economic_theme_count    — records whose Themes contain ECON
  source_diversity        — count of distinct SourceCommonName values

Processing
----------
Queries are run month-by-month (to avoid BigQuery timeouts).
Each month is cached as a Parquet file under data/raw/gdelt/cache/.
All months are then concatenated into gdelt_gkg_daily.csv.

Graceful degradation
--------------------
If credentials are absent or invalid, a placeholder CSV with correct
column structure (all NaN) is written so downstream code never breaks.
"""

import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[2]
_RAW_DIR = _ROOT / "data" / "raw" / "gdelt"
_CACHE_DIR = _RAW_DIR / "cache"
_OUT_FILE = _RAW_DIR / "gdelt_gkg_daily.csv"

GKG_START = date(2015, 1, 1)

PLACEHOLDER_COLS = [
    "event_count", "avg_tone", "negative_share",
    "conflict_theme_count", "economic_theme_count", "source_diversity",
]

_MONTH_QUERY = """
SELECT
  DATE(CAST(SUBSTR(CAST(DATE AS STRING), 1, 4) AS INT64),
       CAST(SUBSTR(CAST(DATE AS STRING), 5, 2) AS INT64),
       CAST(SUBSTR(CAST(DATE AS STRING), 7, 2) AS INT64))   AS date,
  COUNT(*)                                                   AS event_count,
  AVG(CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64))  AS avg_tone,
  COUNTIF(CAST(SPLIT(V2Tone, ',')[SAFE_OFFSET(0)] AS FLOAT64) < 0)
    / CAST(COUNT(*) AS FLOAT64)                             AS negative_share,
  COUNTIF(Themes LIKE '%CRISISLEX%'
       OR Themes LIKE '%TERROR%'
       OR Themes LIKE '%CONFLICT%')                         AS conflict_theme_count,
  COUNTIF(Themes LIKE '%ECON%')                             AS economic_theme_count,
  COUNT(DISTINCT SourceCommonName)                          AS source_diversity
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  DATE >= @start_date
  AND DATE < @end_date
  AND V2Tone IS NOT NULL
GROUP BY
  date
ORDER BY
  date
"""


def _write_placeholder(log) -> pd.DataFrame:
    """Create and save an empty placeholder CSV with correct structure."""
    log.warning("Writing placeholder empty CSV for GDELT (credentials unavailable).")
    df = pd.DataFrame(columns=["date"] + PLACEHOLDER_COLS)
    df = df.set_index("date")
    _RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(_OUT_FILE)
    return df


def _iter_months(start: date, end: date):
    """Yield (month_start, month_end) pairs covering [start, end)."""
    cur = date(start.year, start.month, 1)
    while cur <= end:
        if cur.month == 12:
            nxt = date(cur.year + 1, 1, 1)
        else:
            nxt = date(cur.year, cur.month + 1, 1)
        yield cur, nxt
        cur = nxt


def fetch_gdelt(log=None):
    """
    Pull GDELT GKG daily aggregates via BigQuery.

    Returns:
        List of manifest record dicts.
    """
    if log is None:
        import logging
        log = logging.getLogger("pipeline.gdelt")

    from utils.manifest import build_record

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()

    # ── Graceful degradation when credentials are absent ────────────────────
    if not cred_path or not Path(cred_path).exists():
        log.error(
            "GOOGLE_APPLICATION_CREDENTIALS not set or file not found ('%s'). "
            "GDELT pull skipped — placeholder file written.",
            cred_path,
        )
        df = _write_placeholder(log)
        return [build_record("gdelt", _OUT_FILE, df,
                             extra={"limitation": "credentials unavailable, placeholder file"})]

    try:
        from google.cloud import bigquery
    except ImportError:
        log.error("google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery")
        df = _write_placeholder(log)
        return [build_record("gdelt", _OUT_FILE, df,
                             extra={"limitation": "google-cloud-bigquery not installed"})]

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    try:
        client = bigquery.Client.from_service_account_json(cred_path)
    except Exception as exc:
        log.error("BigQuery client init failed: %s", exc)
        df = _write_placeholder(log)
        return [build_record("gdelt", _OUT_FILE, df,
                             extra={"limitation": f"BigQuery auth failed: {exc}"})]

    today = date.today()
    all_months: list[pd.DataFrame] = []

    log.info("=== GDELT: querying month-by-month from %s to %s ===", GKG_START, today)

    for month_start, month_end in _iter_months(GKG_START, today):
        month_key = month_start.strftime("%Y_%m")
        parquet_path = _CACHE_DIR / f"{month_key}.parquet"

        # Use parquet cache if valid — but always re-query the current, still-incomplete month
        is_current_month = (month_start.year, month_start.month) == (today.year, today.month)
        if parquet_path.exists() and not is_current_month:
            try:
                df_month = pd.read_parquet(parquet_path)
                all_months.append(df_month)
                log.debug("CACHE HIT  — GDELT month %s", month_key)
                continue
            except Exception:
                pass  # re-query if cache is corrupt

        log.info("Querying GDELT — %s", month_key)
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("start_date", "DATE", str(month_start)),
                bigquery.ScalarQueryParameter("end_date", "DATE", str(month_end)),
            ]
        )

        try:
            df_month = client.query(_MONTH_QUERY, job_config=job_config).to_dataframe()
            if df_month.empty:
                log.warning("No GDELT data for month %s", month_key)
                continue
            df_month.to_parquet(parquet_path, index=False)
            all_months.append(df_month)
            log.info("SAVED parquet — %s (%d rows)", month_key, len(df_month))
        except Exception as exc:
            log.error("BigQuery query failed for month %s: %s", month_key, exc)

    if not all_months:
        log.error("No GDELT data retrieved — writing placeholder.")
        df = _write_placeholder(log)
        return [build_record("gdelt", _OUT_FILE, df,
                             extra={"limitation": "no data returned from BigQuery"})]

    # Combine all months
    combined = pd.concat(all_months, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"]).dt.strftime("%Y-%m-%d")
    combined = combined.set_index("date").sort_index()

    combined.to_csv(_OUT_FILE)
    log.info("=== GDELT: complete — %d daily rows saved to gdelt_gkg_daily.csv ===",
             len(combined))

    from utils.manifest import build_record
    return [build_record("gdelt", _OUT_FILE, combined)]


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from utils.logger import get_logger
    from dotenv import load_dotenv
    load_dotenv()
    fetch_gdelt(log=get_logger("gdelt"))
