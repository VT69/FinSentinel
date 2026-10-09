# README_DATA.md — Financial Data Pipeline

This document describes the data pipeline for the **Financial Sentiment & Market Dynamics Research Project**.
The pipeline ingests data from 8 sources, saves clean daily-indexed CSVs to `data/raw/`, and produces a
machine-readable manifest (`manifest.json`) and execution log (`pipeline.log`).

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r data_pipeline/requirements.txt

# 2. Configure API keys
cp .env.example .env
#    then edit .env and fill in your keys

# 3. Run the full pipeline
cd data_pipeline
python fetch_all.py

# 4. Run a single source (e.g. yfinance only — no API key needed)
python fetch_all.py --source yfinance

# 5. Run multiple specific sources
python fetch_all.py --source fred,gdelt,trends
```

---

## Output Structure

```
data/raw/
├── yfinance/               # OHLCV + derived features per ticker
│   ├── BTC-USD.csv
│   ├── _NSEI.csv           # ^ replaced with _
│   └── ...
├── fred/
│   ├── VIXCLS.csv          # individual series
│   ├── fred_combined.csv   # all series merged
│   └── ...
├── gdelt/
│   ├── cache/              # parquet files, one per calendar month
│   └── gdelt_gkg_daily.csv
├── google_trends/
│   ├── raw_weekly/         # one CSV per search term
│   └── daily_resampled.csv # all terms, forward-filled daily
├── alpha_vantage/          # GDP, Fed Funds, CPI, Treasury 10Y
├── reddit/
│   ├── raw_posts.csv
│   └── daily_agg.csv
├── newsapi/
│   ├── raw_articles.csv
│   └── daily_agg.csv
├── quandl/                 # IMF commodity prices (ODA series)
├── manifest.json           # data quality stats for all files
└── pipeline.log            # timestamped run log
```

---

## Data Sources

### Source 1 — yfinance (no API key)

| Field | Value |
|---|---|
| Library | `yfinance` |
| API Key | Not required |
| Date range | 2010-01-01 → present |
| Frequency | Daily |

**Tickers:**

| Category | Tickers |
|---|---|
| Crypto | BTC-USD, ETH-USD |
| Equity Indices | ^NSEI, ^GSPC, ^DJI, ^IXIC, ^FTSE, ^N225 |
| Volatility | ^VIX, ^VXN |
| Commodities | GC=F (Gold), CL=F (Crude Oil) |
| FX | USDINR=X, DX-Y.NYB (Dollar Index) |
| Bonds / Yields | ^TNX (10Y), ^TYX (30Y), ^IRX (3M) |

**Columns per CSV:** `open, high, low, close, volume, adj_close, log_return, vol_7d, vol_14d, vol_30d, vol_60d`

Rolling volatility is annualised with √365 for crypto (BTC-USD, ETH-USD) and √252 for everything else. `log_return = ln(close / close_prev)`.

---

### Source 2 — FRED (Federal Reserve Economic Data)

| Field | Value |
|---|---|
| Library | `fredapi` |
| API Key | `FRED_API_KEY` — free at [fred.stlouisfed.org](https://fred.stlouisfed.org/docs/api/api_key.html) |
| Date range | 2010-01-01 → present |
| Frequency | Varies (daily, monthly, quarterly) — saved as-is |

**Series pulled:**

| Series ID | Description |
|---|---|
| VIXCLS | VIX daily close |
| BAMLH0A0HYM2 | US HY credit spread |
| BAMLC0A0CM | US IG credit spread |
| T10Y2Y | 10Y–2Y yield spread |
| T10Y3M | 10Y–3M yield spread |
| TEDRATE | TED spread |
| DPCREDIT | Discount window borrowing |
| DPRIME | Prime rate |
| UMCSENT | UMich Consumer Sentiment |
| USREC | NBER recession indicator |
| M2SL | M2 money supply |
| CPIAUCSL | CPI all items |
| UNRATE | Unemployment rate |
| DCOILWTICO | WTI crude oil (daily) |
| GOLDAMGBD228NLBM | Gold price (London PM fix, daily) |

---

### Source 3 — GDELT GKG via BigQuery

| Field | Value |
|---|---|
| Library | `google-cloud-bigquery` |
| API Key | GCP service account JSON → `GOOGLE_APPLICATION_CREDENTIALS` |
| Date range | 2015-01-01 → present (GKG v2 starts ~2015-02-19) |
| Frequency | Daily aggregates |
| Table | `gdelt-bq.gdeltv2.gkg` |

**Columns:** `event_count, avg_tone, negative_share, conflict_theme_count, economic_theme_count, source_diversity`

**Implementation details:**
- Queries are run **month-by-month** to avoid BigQuery timeouts.
- Each month is cached as a Parquet file (in `data/raw/gdelt/cache/`) before combining.
- Without credentials an **empty placeholder CSV** with correct column headers is written so downstream code never breaks.

> ⚠ BigQuery charges ~$5/TB scanned. Parquet caching means you only pay once per month of GDELT data.

---

### Source 4 — Google Trends

| Field | Value |
|---|---|
| Library | `pytrends` |
| API Key | Not required (unofficial API) |
| Date range | 2010-01-01 → present |
| Raw frequency | Weekly (Google Trends limitation) |
| Output frequency | Daily (forward-filled) |

**Terms:** 14 terms across Finance/Fear, Crypto, and India/NIFTY categories.

**Notes:** 3-second delays between requests to respect rate limits. Raw weekly and daily-resampled files both saved.

---

### Source 5 — Alpha Vantage

| Field | Value |
|---|---|
| Library | `requests` |
| API Key | `ALPHA_VANTAGE_KEY` — free at [alphavantage.co](https://www.alphavantage.co/support/#api-key) |
| Free quota | 25 calls/day, 5 calls/min |
| Rate limiting | 12-second sleep between calls |

**Indicators pulled (4 total):**

| File | Description |
|---|---|
| `real_gdp.csv` | Real GDP (quarterly) |
| `federal_funds_rate.csv` | Federal Funds Rate (daily) |
| `cpi.csv` | Consumer Price Index (monthly) |
| `treasury_yield_10y.csv` | 10Y Treasury yield (daily) |

On quota hit (HTTP 429 or API-level rate message), the pipeline saves completed indicators and logs the remainder as skipped.

---

### Source 6 — Reddit (PRAW)

| Field | Value |
|---|---|
| Library | `praw` |
| API Key | `REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USER_AGENT` |
| Registration | [reddit.com/prefs/apps](https://www.reddit.com/prefs/apps) (free "script" app) |

**Subreddits:** r/wallstreetbets, r/investing, r/stocks, r/bitcoin, r/cryptocurrency, r/IndiaInvestments

**⚠ Known Limitation:** The Reddit API returns only the **most recent ~1000 posts** per subreddit. Historical data beyond ~2–4 weeks is not accessible via the free public API. Pushshift (the historical Reddit archive) is no longer reliably free. This source is included for **recent sentiment signals** and the **live pipeline** — not for historical backtesting.

**Output:**
- `raw_posts.csv` — individual posts with date, subreddit, score, upvote_ratio, num_comments
- `daily_agg.csv` — per-subreddit daily post_count, mean_score, mean_upvote_ratio, mean_num_comments

---

### Source 7 — NewsAPI

| Field | Value |
|---|---|
| Library | `requests` |
| API Key | `NEWSAPI_KEY` — free at [newsapi.org](https://newsapi.org/register) |
| Free tier | 100 req/day, **1 month article history** |

**Queries:** `'stock market', 'financial markets', 'bitcoin', 'NIFTY', 'market crash', 'recession', 'inflation'`

**⚠ Known Limitation:** The free developer plan only provides articles from the **last 30 days**. This source is designed for the **live pipeline** (periodic updates), not historical research. The architecture is built now so that switching to a paid plan immediately unlocks fuller history.

---

### Source 8 — Quandl / Nasdaq Data Link

| Field | Value |
|---|---|
| Library | `nasdaqdatalink` (pip: `nasdaq-data-link`) |
| API Key | `QUANDL_API_KEY` — free at [data.nasdaq.com](https://data.nasdaq.com/sign-up) |

**Datasets attempted:**

| Code | Description | Status |
|---|---|---|
| WIKI/PRICES | US equity prices | ⚠ Discontinued June 2018 |
| ODA/PCOALAU_USD | IMF Coal price | Available |
| ODA/PNGAS_USD | IMF Natural Gas price | Available |
| ODA/POILWTI_USD | IMF WTI Crude Oil | Available |
| ODA/PGOLD_USD | IMF Gold price | Available |
| ODA/PCOFFOTM_USD | IMF Coffee price | Available |
| ODA/PSILVER_USD | IMF Silver price | Available |

---

## Engineering Design

### Caching (24-hour)
Every file is checked before re-download. If the file exists and is less than 24 hours old, the download is skipped. This prevents burning API quotas on frequent re-runs.

### Missing Data Policy
NaNs are **preserved as-is** in all saved CSVs. No forward-fill or interpolation is applied during saving. The manifest records `% missing` per column so data quality is always visible.

### Manifest (`manifest.json`)
Auto-generated after every pipeline run. Records:
- Source name, file path, date range
- Column list, row count
- % missing per column
- Pull timestamp (UTC)

### Logging (`pipeline.log`)
Every source logs its progress. If a source fails, the error is logged and the pipeline continues with the next source — it never crashes the whole run.

### Adding New Sources
1. Create `data_pipeline/sources/fetch_{sourcename}.py`
2. Implement `fetch_{sourcename}(log=None) → list[dict]`
3. Add a tuple to `SOURCE_REGISTRY` in `fetch_all.py`
4. Add dependencies to `data_pipeline/requirements.txt`
5. Add API key to `.env.example`

---

## Reproducibility

All scripts are deterministic given the same date range. For academic paper reproducibility:
- Pin library versions in `requirements.txt`
- Record the `pull_timestamp` from `manifest.json` for each dataset
- Archive the `data/raw/` directory alongside your analysis notebooks
