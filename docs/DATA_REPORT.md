# FinSentinel — Data Report

> **Status after fixes.** This report profiles the data as committed at `e1eb270`. Since then, the two duplicated rows in `nifty_google_trends.csv` were removed (194 → 192 rows). Everything else in `data/raw` is unchanged.

> Every statistic in this document was produced in this session by
> `analysis/data_profile.py` (outputs in `analysis/outputs/`: `data_profile_stdout.txt`, `profile_summary.json`, CSVs and PNGs),
> `analysis/model_eval.py` and `analysis/verify_dashboard_numbers.py`.
> Re-run: `pip install -r analysis/requirements.txt && python analysis/data_profile.py`.
> Anything that could not be computed is marked **NOT VERIFIED**, with the reason.

---

## 0. What "the dataset" is in this project (read this first)

There is **no single training dataset** in the repo. There are four groups of data:

| Group | Committed? | Used by | Profiled here? |
|---|---|---|---|
| A. Daily prices `data/raw/btc_prices.csv`, `nifty_prices.csv` | ✔ | everything | ✔ full |
| B. Headlines `data/raw/text_data.csv` | ✔ | sentiment track (Track A) | ✔ full |
| C. Google Trends `data/raw/*_google_trends.csv` (monthly) | ✔ | VSI notebooks 09/10 | ✔ full |
| D. Everything in `data/processed/` (aligned tables, GDELT events aggregate, `gmsi_exogenous.csv`, `*_vsi_full.csv`) | ✘ (gitignored, `.gitignore:31`) | RF training, GMSI, all "Paper 1" figures | **Partly reconstructed** (§6). GMSI and events are **NOT VERIFIED**: the source GDELT files are not in the repo, and `data.gdeltproject.org` returned HTTP 403 from this sandbox. |

The **ML training table** (`btc_sentiment_aligned.csv` → `build_features`) is rebuilt in §6 from groups A and B, with one substitution: FinBERT scores could not be recomputed (huggingface.co is blocked here), so VADER is used as a labelled stand-in **only to reproduce row counts**. No FinBERT statistic is reported anywhere.

---

## 1. Provenance

| File | Source | Collection method | Period (verified) | License / terms | Confidence |
|---|---|---|---|---|---|
| `btc_prices.csv` | Yahoo Finance via `yfinance` (`notebooks/01_data_collection.ipynb` cell 3) | `yf.download("BTC-USD", start="2015-01-01", end="2025-12-31")`, reset index, `return = close.pct_change()`, `dropna()` | 2015-01-02 → 2025-12-30, 4,016 rows | Yahoo's terms allow personal, non-commercial use only. yfinance is an unofficial scraper with no SLA. Redistributing the CSV in a public repo is a grey area. | high |
| `nifty_prices.csv` | same, ticker `^NSEI` | same, `start="2010-01-01"` | 2010-01-05 → 2025-12-30, 3,927 rows | same; the index data belongs to NSE Indices Ltd | high |
| `text_data.csv` | GDELT 2.0 **DOC API** (`api.gdeltproject.org/api/v2/doc/doc`), article **titles only** | per-day keyword queries (`notebooks/01a_text_data_collection.ipynb` cell 4). The committed file was produced by a run that is **not** in the notebook: the notebook's own run covers 2015-01-01→02 and returned empty (cell 5 output). | 2024-10-03 01:15 UTC → 2025-01-01 16:30 UTC, 1,500 rows | GDELT is open and free to use with attribution; the underlying headlines belong to their publishers. `source` (publisher) is empty in the committed file. | high (source); **unknown** (exact query run) |
| `btc_google_trends.csv`, `nifty_google_trends.csv` | Google Trends | **unknown.** No code in the repo writes these two files. `notebooks/01_data_collection.ipynb` cell 6 writes `btc_trends.csv`/`nifty_trends.csv` with a column named `trend`, and the committed files have `google_trend` with `YYYY-MM` dates. Probably a manual CSV export. | BTC 2015-01 → 2025-12 (132 months); NIFTY 2010-01 → 2025-12 (194 rows, **2 duplicated**) | Google Trends terms of service | medium |
| `manifest.json`, `pipeline.log` | `data_pipeline/fetch_all.py`, run 2026-03-21 | yfinance only; FRED and the others were never run successfully (log L1-80) | — | — | high |

**Interview line:** "Prices come from Yahoo via yfinance and headlines from the GDELT DOC API. Both are free, but Yahoo's terms make this research-only. I don't have a reproducible script for the exact headline pull or for the Trends export. That's a provenance gap I'd fix by moving those pulls into `data_pipeline/` with a manifest entry."

---

## 2. File format and loading

From `analysis/outputs/file_metadata.csv`:

| File | Bytes | Lines | Encoding | Delimiter | Header |
|---|---|---|---|---|---|
| btc_prices.csv | 423,823 | 4,017 | ASCII | `,` | `date,open,high,low,close,volume,return` |
| nifty_prices.csv | 379,817 | 3,928 | ASCII | `,` | same |
| text_data.csv | 241,848 | 1,501 | **UTF-8 with non-ASCII** (Devanagari, Hangul, …) | `,` (quoted fields) | `timestamp,text,source,asset,channel` |
| btc_google_trends.csv | 1,439 | 133 | ASCII | `,` | `date,google_trend` |
| nifty_google_trends.csv | 2,146 | 195 | ASCII | `,` | `date,google_trend` |

All files are loaded with plain `pd.read_csv(..., parse_dates=[...])` (e.g. `notebooks/04_time_alignment.ipynb` cells 2 and 4). There is no schema validation and no dtype specification.

---

## 3. Structure and column inventory

### 3a. Shapes and memory (`df.info()` saved as `analysis/outputs/info_*.txt`)

| Dataset | Rows × Cols | Memory (deep) |
|---|---|---|
| btc_prices | 4,016 × 7 | 219.8 KiB |
| nifty_prices | 3,927 × 7 | 214.9 KiB |
| text_data | 1,500 × 5 | 550.8 KiB |
| btc_google_trends | 132 × 2 | 8.4 KiB |
| nifty_google_trends | 194 × 2 | 12.3 KiB |

### 3b. Column inventory (full CSVs: `analysis/outputs/inventory_*.csv`)

**btc_prices / nifty_prices**

| Column | dtype | Non-null | Null % | n_unique BTC / NIFTY | Cardinality ratio BTC / NIFTY | Meaning (confidence) |
|---|---|---|---|---|---|---|
| date | datetime64 | all | 0 | 4,016 / 3,927 | 1.000 / 1.000 | bar date (high) |
| open | float64 | all | 0 | 4,013 / 3,897 | 0.9993 / 0.9924 | day open, USD / index points (high) |
| high | float64 | all | 0 | 4,013 / 3,896 | 0.9993 / 0.9921 | intraday high (high) |
| low | float64 | all | 0 | 4,014 / 3,894 | 0.9995 / 0.9916 | intraday low (high) |
| close | float64 | all | 0 | 4,013 / 3,887 | 0.9993 / 0.9898 | close (high) |
| volume | int64 | all | 0 | 4,016 / **2,206** | 1.000 / **0.5618** | traded volume; **NIFTY has 774 zeros** (§4c) (medium) |
| return | float64 | all | 0 | 4,016 / 3,924 | 1.000 / 0.9992 | **simple** return `pct_change` (high). Matches recomputation to 3.1e-16 (BTC) and 1.0e-16 (NIFTY). |

**text_data**

| Column | dtype | Non-null | Null % | n_unique | Ratio | Meaning |
|---|---|---|---|---|---|---|
| timestamp | str | 1,500 | 0 | 1,133 | 0.7553 | GDELT seendate, UTC |
| text | str | 1,499 | 0.07 | 1,390 | 0.9267 | headline title |
| source | float64 | **0** | **100.0** | 0 | 0 | publisher, **always empty** |
| asset | str | 1,500 | 0 | 2 | 0.0013 | BTC / NIFTY query |
| channel | str | 1,500 | 0 | 1 | 0.0007 | constant `news_gdelt` (zero information) |

**google_trends**: `date` is a string `YYYY-MM` (not parsed), and `google_trend` is int64 0–100.

---

## 4. Numeric columns

Full table: `analysis/outputs/numeric_profile_*.csv`. The key rows are reproduced below.

### 4a. BTC prices

| stat | open | high | low | close | volume | return |
|---|---|---|---|---|---|---|
| count | 4016 | 4016 | 4016 | 4016 | 4016 | 4016 |
| mean | 27,518 | 28,072.6 | 26,932.6 | 27,538.9 | 2.22e10 | 0.00203 |
| median | 11,352 | 11,570.4 | 11,015.3 | 11,358.4 | 1.78e10 | 0.00125 |
| mode (count) | 233.42 (2) | 244.25 (2) | 429.08 (2) | 236.15 (2) | 7.86e6 (1) | −0.3717 (1) |
| std | 31,747.8 | 32,290.5 | 31,170 | 31,758.7 | 2.29e10 | 0.03522 |
| variance | 1.008e9 | 1.043e9 | 9.716e8 | 1.009e9 | 5.25e20 | 0.001241 |
| min / max | 176.90 / 124,752 | 211.73 / 126,198 | 171.51 / 123,196 | 178.10 / 124,753 | 7.86e6 / 3.51e11 | −0.3717 / 0.2525 |
| range | 124,575 | 125,986 | 123,025 | 124,574 | 3.51e11 | 0.6242 |
| Q1 / Q2 / Q3 | 3,671.5 / 11,352 / 42,807.8 | 3,754.6 / 11,570.4 / 43,575.5 | 3,619.8 / 11,015.3 / 41,906.5 | 3,673.2 / 11,358.4 / 42,840.4 | 2.22e9 / 1.78e10 / 3.37e10 | −0.01214 / 0.00125 / 0.01607 |
| IQR | 39,136.3 | 39,820.9 | 38,286.8 | 39,167.2 | 3.15e10 | 0.02821 |
| P1 / P5 | 226.6 / 256.3 | 230.6 / 261.7 | 222.7 / 251.7 | 226.7 / 256.3 | 1.51e7 / 2.66e7 | −0.1006 / −0.0528 |
| P95 / P99 | 101,769 / 116,816 | 104,025 / 117,874 | 99,766.5 / 115,248 | 101,752 / 116,820 | 6.60e10 / 9.45e10 | 0.0556 / 0.1042 |
| skew | 1.306 | 1.293 | 1.321 | 1.305 | 2.013 | −0.120 |
| excess kurtosis | 0.726 | 0.677 | 0.779 | 0.721 | 12.79 | **7.98** |
| CV | 1.154 | 1.150 | 1.157 | 1.153 | 1.032 | 17.35 |
| zeros / negatives | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 1 / 1,896 |
| outliers IQR×1.5 | 204 | 209 | 205 | 202 | 80 | **379** |
| outliers \|z\|>3 | 4 | 3 | 5 | 4 | 53 | **70** |

### 4b. NIFTY prices

| stat | open | high | low | close | volume | return |
|---|---|---|---|---|---|---|
| mean | 11,843.6 | 11,900.2 | 11,770.4 | 11,836.6 | 240,987 | 0.000462 |
| median | 10,101 | 10,155.7 | 10,049.1 | 10,113.7 | 213,500 | 0.000588 |
| mode (count) | 4,878.6 (2) | 5,016.25 (2) | 4,924.3 (2) | 5,274.85 (3) | **0 (774)** | 0 (4) |
| std | 6,228.7 | 6,250.9 | 6,201.8 | 6,227.4 | 202,284 | 0.010398 |
| variance | 3.880e7 | 3.907e7 | 3.846e7 | 3.878e7 | 4.09e10 | 0.000108 |
| min / max | 4,623.15 / 26,325.8 | 4,623.15 / 26,325.8 | 4,531.15 / 26,172.4 | 4,544.2 / 26,216.1 | 0 / 1,811,000 | −0.1298 / 0.0876 |
| range | 21,702.7 | 21,702.7 | 21,641.3 | 21,671.9 | 1.81e6 | 0.2174 |
| Q1 / Q2 / Q3 | 6,207.6 / 10,101 / 17,005.3 | 6,258.1 / 10,155.7 / 17,112.2 | 6,179.9 / 10,049.1 / 16,847 | 6,216.2 / 10,113.7 / 16,983.4 | 131,700 / 213,500 / 312,150 | −0.00474 / 0.00059 / 0.00610 |
| IQR | 10,797.7 | 10,854.1 | 10,667.1 | 10,767.2 | 180,450 | 0.01085 |
| P1 / P5 | 4,841.5 / 5,148.5 | 4,879.3 / 5,179.5 | 4,787.6 / 5,101.5 | 4,840.1 / 5,139.5 | 0 / 0 | −0.0263 / −0.0156 |
| P95 / P99 | 24,642 / 25,866.3 | 24,764.1 / 25,946.7 | 24,535.5 / 25,786.6 | 24,653.2 / 25,866.4 | 642,600 / 873,090 | 0.0162 / 0.0261 |
| skew | 0.824 | 0.824 | 0.826 | 0.825 | 1.477 | **−0.688** |
| excess kurtosis | −0.533 | −0.533 | −0.526 | −0.531 | 4.283 | **11.80** |
| CV | 0.526 | 0.525 | 0.527 | 0.526 | 0.839 | 22.50 |
| zeros / negatives | 0/0 | 0/0 | 0/0 | 0/0 | **774** / 0 | 4 / 1,834 |
| outliers IQR / \|z\|>3 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | 285 / 49 | 156 / 47 |

### 4c. Plausibility checks (`profile_summary.json → plausibility`)

| Check | BTC | NIFTY |
|---|---|---|
| future dates (> 2026-10-09) | 0 | 0 |
| duplicate dates | 0 | 0 |
| non-positive prices | 0 | 0 |
| `high < low` / close outside [low, high] / open outside [low, high] | 0 / 0 / 0 | 0 / 0 / 0 |
| **volume == 0** | 0 | **774 rows**: 249 in 2010, 242 in 2011, 242 in 2012, 13 in 2013, then scattered (1–13/yr) up to 2024-07-03. **Physically meaningless for an index that traded**, so it is a data-vendor gap. No model in the repo uses `volume`, so it is harmless today, but it must not be fed to a model as-is. |
| day gaps | all 1 day, 0 missing calendar days (BTC trades 24/7) ✔ | gaps of 1/2/3/4/5/6 days = 2,967/120/735/95/8/1, i.e. weekends and Indian market holidays ✔ |
| largest \|log return\| | **2020-03-12: 0.4647** (COVID crash, real), 2015-01-14: 0.2376, 2017-12-07: 0.2251 | 2020-03-23: 0.1390, 2020-03-12: 0.0867 |
| first-row `return` | 0.00249: the first row is the *second* trading day, because `dropna()` in nb01 removed day 1 | 0.00873 |

**Distributions are plausible for what they measure.** Prices are strictly positive with huge right skew because BTC went from about \$178 to about \$125k (the CV of 1.15 reflects the trend, not dispersion). Returns are fat-tailed: excess kurtosis is 7.98 for BTC and 11.80 for NIFTY, against 0 for a Gaussian. NIFTY returns are negatively skewed (−0.688): crashes are larger than rallies.

**Why the two outlier counts disagree (379 vs 70 for BTC returns):** the IQR rule uses the middle 50% of the data. With fat tails the IQR is narrow (0.0282), so 1.5×IQR flags many ordinary big days. The z-rule uses the std (0.0352), which the extreme days themselves inflate (masking), so it flags fewer. **Neither should be used to delete rows here.** In volatility modelling the "outliers" *are* the signal, and dropping them would remove exactly the events the project studies. The repo correctly never removes them.

---

## 5. Categorical columns and text

### 5a. Categorical profile (`profile_summary.json → categorical.text_data`)

| Column | Value counts | Mode | Entropy (bits) | Rare (<1%) | Messy variants | NA-like strings / true NaN |
|---|---|---|---|---|---|---|
| asset | NIFTY 750 (50%), BTC 750 (50%) | BTC (tie) | 1.0 | 0 | none | 0 / 0 |
| channel | news_gdelt 1,500 (100%) | news_gdelt | 0.0 | 0 | none | 0 / 0 |
| source | NaN 1,500 (100%) | — | — | — | — | 0 / 1,500 |

**Train/test unseen-category risk:** none in practice, because no categorical column is used as a model feature (`asset` is only a filter).

### 5b. Headline text (the real "categorical" problem)

| Metric | Value |
|---|---|
| Rows / distinct UTC days | 1,500 / 91 |
| Headline length (chars) | mean 87.9, median 79, min 21, max 253 |
| **Dominant script** (`unicodedata`) | Latin 1,261 · **Devanagari 150** · **Hangul 52** · Gujarati 14 · Cyrillic 7 · Tamil 5 · CJK 3 · Kannada 2 · Malayalam 2 · Hebrew 2 · Telugu 1 · Arabic 1 |
| By asset | BTC: Latin 687, Hangul 52, Cyrillic 6, CJK 2, Hebrew 2, Arabic 1 · NIFTY: Latin 574, Devanagari 150, Gujarati 14, Tamil 5, … |
| Exact duplicate (timestamp, text, asset) rows | 65 |
| Duplicate headline text (any time) | 109 |
| After `clean_text` + `len > 10` (nb02) | **1,377 rows** (matches nb02's printed 1,377 ✔); 123 dropped |
| Dropped rows by script | Hangul 52, Devanagari 50, Cyrillic 7, Tamil 5, CJK 3, Hebrew 2, Latin 1, Telugu 1, Malayalam 1, Arabic 1 |
| **Kept non-Latin rows** | **117.** The ASCII-only regex `[^a-zA-Z\s]` strips the native script and keeps only embedded English tokens, e.g. `"Stock Market : शेयर बाजार में बड़ी गिरावट ..."` → `"stock market"`. FinBERT then scores that fragment (`notebooks/03` cell 2 output row 4). |
| Near-duplicates after cleaning | 192 |
| Headlines/day | BTC mean 7.54 (max 32), 82 days with any headline; NIFTY mean 7.59 (max 23), 87 days |

**Implication:** FinBERT and VADER are English models. About 16% of headlines (239 non-Latin rows) are either dropped or reduced to garbled English fragments. The dropped rows are not random: they are non-English sources, mostly Indian-language (NIFTY) and Korean (BTC). So the sentiment signal is biased toward English-language outlets.

### 5c. Messy variants
No casing or whitespace variants were found in `asset` or `channel`. The messy-variant problem in this repo is in the **text**: duplicates (65 exact, 192 near) and mixed scripts.

### 5d. Google Trends

| | BTC | NIFTY |
|---|---|---|
| rows | 132 (monthly) | 194 (monthly) |
| mean / median / std | 22.71 / 21 / 16.91 | 36.65 / 26 / 25.78 |
| min / max | 2 / 100 | 8 / 100 |
| skew / excess kurtosis | 1.141 / 2.464 | 0.698 / −0.797 |
| **duplicates** | 0 | **2 exact duplicate rows (2025-11, 2025-12 appear twice)** |

Trends values are *relative to the peak month of the queried window* (that's how Google normalises them), so a series pulled over a different window is on a different scale. The notebooks also merge monthly values onto daily rows (`notebooks/09` cell 4: merge on date + `ffill`). Only the 1st of each month matches, and every other day is forward-filled.

---

## 6. The ML training table (reconstructed) and the target

### 6a. Reconstruction path (`analysis/common.py` mirrors each original step)

`text_data.csv` → `clean_text` (nb02) → VADER (nb03; **FinBERT substituted by VADER**) → daily mean per UTC date and asset, inner join with BTC prices, lags, `dropna` (nb04) → `build_features` (`pipeline/feature_engineering.py`) → 80/20 chronological split (`run_pipeline.py:15-17`).

| Stage | Rows |
|---|---|
| headlines after cleaning | 1,377 |
| BTC aligned table (nb04) | **77** (the original run printed **75**, `notebooks/05` cell 3 output) |
| after `build_features(...).dropna()` | **13** |
| train / test | **10 / 3** (train dates 2024-12-14 → 2024-12-24; test 2024-12-25 → 2024-12-27) |

**Why 75 → 13:** `vol_60 = rolling(60).std()` (`pipeline/feature_engineering.py:17`) is computed and then *not used as a feature* (it is commented out on L46). But `df.dropna()` (L40) still deletes the 59 rows where it is NaN. The `shift(-5)` target (L20) deletes the last 5. That leaves 77 − 59 − 5 = 13 rows.

**Discrepancy (flagged):** my reconstruction has 77 aligned rows versus the original 75. The committed `nifty_prices.csv` starts in 2010, but the price file used at the time started on 2023-01-02 with `+05:30` timestamps (see the `notebooks/04` cell 2 output). So the raw price files were re-downloaded after the RF was trained. The committed pickle has at most 8 distinct rows in any bootstrap sample (MODEL_REPORT §1), which is consistent with an 8-row (or slightly larger) training set. **Exact original training rows: NOT VERIFIED.**

### 6b. Target (training table)

`log_vol_target = log(std(r[t+1..t+5]) + 1e-6)`, where r is the daily log return.

| | value |
|---|---|
| n | 13 |
| mean / std | −3.8507 / 0.3294 |
| min / median / max | −4.3990 / −3.7462 / −3.5053 |
| in volatility space | 0.01229 … 0.03004 (daily σ) |

**Baseline (DummyRegressor = train mean), scored like `pipeline/train_rf_model.py:20-24`:** RMSE 0.009553 and MAE 0.009414 in volatility space on the 3 test rows. MODEL_REPORT §2 shows the RF gives 0.009647 and 0.009509, i.e. **worse than or equal to the trivial baseline**.

### 6c. Target on the full price history (what a usable dataset looks like)

Same target definition, computed over all committed prices (`price_feature_frame`):

| | BTC `vol_target` | BTC `log_vol_target` | NIFTY `vol_target` | NIFTY `log_vol_target` |
|---|---|---|---|---|
| n | 3,951 | 3,951 | 3,862 | 3,862 |
| mean / median | 0.02858 / 0.02381 | −3.775 / −3.738 | 0.00866 / 0.00743 | −4.904 / −4.903 |
| std | 0.02017 | 0.684 | 0.00582 | 0.546 |
| skew | 2.362 | −0.291 | 4.254 | 0.100 |
| excess kurtosis | 12.33 | 0.166 | 36.35 | 0.769 |
| Jarque-Bera (p) | 28,689 (≈0) | 60.2 (≈0) | 224,217 (≈0) | 101.6 (≈0) |

**Normality / transform:** raw volatility is strongly right-skewed and heavy-tailed. **The log transform is justified**: it cuts BTC skew from 2.36 to −0.29 and kurtosis from 12.3 to 0.17. It is still formally non-normal (JB p≈0 at n≈4,000, which rejects even tiny deviations), but it is close enough for squared-error loss to behave. Plots: `analysis/outputs/target_distribution_{btc,nifty}.png`.

**Note for the interview:** this is a **regression** task, so "class balance" does not apply. The analogue is that the target is heavy-tailed: calm periods dominate, and squared error in volatility space is dominated by a few turbulent weeks.

---

## 7. Relationships (full price history; sentiment is unavailable before 2024-10-03)

Features are those in `pipeline/feature_engineering.py` that don't need sentiment. Full matrices: `analysis/outputs/corr_{pearson,spearman}_{btc,nifty}.csv`. Heatmaps: `corr_heatmap_{btc,nifty}.png`.

**Top correlated feature pairs (|Pearson|), i.e. multicollinearity**

| BTC | r | NIFTY | r |
|---|---|---|---|
| abs_return ~ sq_return | 0.771 | vol_22 ~ vol_60 | 0.794 |
| vol_22 ~ vol_60 | 0.769 | abs_return ~ sq_return | 0.762 |
| vol_5 ~ vol_22 | 0.693 | vol_5 ~ vol_22 | 0.726 |

`abs_return` and `sq_return` have **Spearman = 1.000**: one is a monotone transform of the other, so for a tree model `sq_return` adds **zero** information.

**VIF** (with constant): BTC abs_return 2.93, sq_return 2.48, vol_5 2.30, vol_22 3.45, vol_60 2.45. NIFTY 2.73 / 2.42 / 2.61 / 4.15 / 2.75. Everything is below 5, so collinearity is moderate. That doesn't matter for an RF, but it would inflate coefficient variance in a linear model.

**Feature ↔ target ranking** (`feature_target_ranking_*.csv`; MI = `mutual_info_regression`, random_state 42)

| BTC | Pearson | Spearman | MI | NIFTY | Pearson | Spearman | MI |
|---|---|---|---|---|---|---|---|
| vol_22 | 0.456 | 0.470 | 0.297 | vol_60 | 0.376 | 0.390 | 0.234 |
| vol_60 | 0.411 | 0.404 | 0.275 | vol_22 | 0.437 | 0.454 | 0.207 |
| vol_5 | 0.422 | 0.429 | 0.157 | vol_5 | 0.383 | 0.343 | 0.121 |
| sq_return | 0.174 | 0.265 | 0.065 | abs_return | 0.253 | 0.180 | 0.033 |
| abs_return | 0.278 | 0.265 | 0.056 | sq_return | 0.197 | 0.180 | 0.030 |

Domain sense: **volatility clusters**, so recent realised vol (especially the 1-month window) is the best predictor of next-week vol. A single day's return is a noisy proxy. This is exactly the HAR-RV intuition (Corsi, 2009).

**Sentiment correlations:** the only verifiable ones are the printed outputs in `notebooks/05` (n = 75 BTC / 54 NIFTY). Example: BTC corr(finbert_score, vader_score) = **−0.352** (cell 4 output). Two sentiment models scoring the same headlines should correlate *positively*, and this negative value is the empirical fingerprint of the FinBERT label bug (ISSUES P0-3). Recomputing it: **NOT VERIFIED** (FinBERT is not runnable here).

**Duplicates:**
- prices: 0 exact duplicate rows, 0 duplicate dates
- BTC trends: 0
- **NIFTY trends: 2 exact duplicates**
- text: 65 exact duplicates (on timestamp + text + asset), 109 duplicate headline texts, 192 near-duplicates after cleaning

---

## 8. Data quality and integrity

### 8a. Missing-data pattern (`analysis/outputs/missingness_matrix.png`)

The project joins on a daily calendar (2015-01-01 → 2025-12-31). Share of days missing:

| Column | % missing | Mechanism |
|---|---|---|
| btc_close | 0.05 | 2 edge days (series starts 01-02, ends 12-30): structural |
| nifty_close | 32.63 | **Structured**: weekday distribution of missing days = Mon 34, Tue 29, Wed 31, Thu 29, Fri 41, **Sat 573, Sun 574**, so weekends plus exchange holidays. Not random. |
| btc_headlines / nifty_headlines | 97.96 / 97.83 | **Structured**: text only exists for 2024-10-03 → 2025-01-01. The missing indicators co-occur (corr 0.922). |
| *_trend_monthly | 96.71 | by construction (monthly data on a daily grid) |

### 8b. How the code handles missing values, and whether that's defensible

| Where | Handling | Defensible? |
|---|---|---|
| `notebooks/01` cell 3 | `dropna()` after `pct_change` | ✔ drops only the first row |
| `notebooks/04` cell 7 | **inner join** price × sentiment | ⚠ silently restricts the dataset to the 82/87 days with headlines |
| `pipeline/feature_engineering.py:40` | `dropna()` on everything, including the **unused** `vol_60` | ✘ throws away 59 of 75 rows for a feature that isn't used |
| `scripts/reconstruct_gmsi.py:65` | `fillna(0)` on z-scores before averaging | ⚠ "0 = neutral" is reasonable for a z-score, but because sentiment exists for ~2% of days, the index is effectively a 3-component index diluted by 3 zeros on ~98% of days, and a 6-component index on the other ~2%. **Its scale changes over time.** |
| `scripts/construct_vsi_full.py:26-28` | `rolling(..., min_periods=1).std()` | ✘ the first rows get std from 1–2 observations (NaN or ~0) and these are kept |
| `scripts/08_…:713` | GMSI reindex + `ffill` | ✔ (past-only) |
| `notebooks/09` cell 4 | monthly Trends merged on exact date + `ffill` | ⚠ the month value is known only after the month ends, but it is ffilled from day 1, which is a look-ahead of up to ~30 days |

### 8c. Leakage audit (critical)

| # | Question | Finding | Evidence |
|---|---|---|---|
| L1 | Does any feature encode the target? (RF pipeline) | **No.** The target uses r[t+1..t+5]; every feature uses data at or before t. | `pipeline/feature_engineering.py:15-38` vs L20 (`.shift(-5)` on a 5-day window) |
| L2 | Same for notebook 06 | **Partial overlap.** Target `rolling(5).std().shift(-1)` = std of r[t−3..t+1], so **4 of its 5 days are already observed at t** and also feed `vol_5` and `abs_return`. The "next-day volatility" it forecasts is mostly known. | `notebooks/06` cell 2 |
| L3 | Scaler/encoder/imputer fit before the split? | RF pipeline: **no scaler exists**. Rolling features are past-only. ✔ `notebooks/07` fits `y_scaler` on all of y before splitting (cell 4), which would be leakage, but the cell crashed. | `notebooks/07` cells 4-6 |
| L4 | GMSI v0 | `StandardScaler().fit_transform(X)` on the **full 2015-2025 sample** (look-ahead in normalisation) | `notebooks/12` cells 10-11 |
| L5 | GMSI "exogenous" | expanding z-scores (past-only) ✔. But `sentiment_surprise` uses `rolling(30, min_periods=1)` (past-only ✔). | `scripts/reconstruct_gmsi.py:4-7, 37, 41-56` |
| L6 | Regime thresholds | **Full-sample quantiles** `gmsi['pure_gmsi'].quantile(0.20/0.80)` = look-ahead. This contradicts the dashboard's "expanding quantile" claim (`app.py:1275`). | `scripts/regime_analysis.py:28-29` |
| L7 | Shock threshold (real-data regime script) | full-sample `quantile(0.95)` = look-ahead | `scripts/regime_analysis.py:106` |
| L8 | Conditional expectations | `pd.qcut` over the full sample. That's fine for a *descriptive* in-sample association, but it is **not a forecast**, and the dashboard's word "predicts" overstates it. | `scripts/validate_gmsi.py:42` |
| L9 | VSI | VSI **contains `vol_z`** and is then correlated with vol. That's a definitional (mechanical) leak, and the repo acknowledges it. | `scripts/construct_vsi_full.py:101,130-135`; `notebooks/13` cell 17 (0.92 / 0.96) |
| L10 | Temporal alignment of news vs NIFTY close | Headlines are bucketed by **UTC date**; NIFTY closes 15:30 IST = 10:00 UTC. About 14 h of "same-day" headlines come **after** the close. For *same-day* sentiment-return correlations (nb05) that is reverse causality: news reacting to the move. For forecasting t+1..t+5 it is fine. | `notebooks/04` cell 5 (`timestamp.dt.date`) |
| L11 | Trends | ffilled monthly values are visible from day 1 of the month (see 8b) | `notebooks/09` cell 4 |

### 8d. Train/val/test split

| | RF pipeline | Notebook 06 |
|---|---|---|
| Method | `train_test_split(test_size=0.2, shuffle=False)` (chronological holdout) | same |
| Ratio | 80/20 → **10/3 rows** (reconstructed) | 80/20 of 15 rows → 12/3 |
| Stratified | n/a (regression) | n/a |
| Seed | none needed (no shuffle) | none |
| Validation set | **none** (no tuning was done, so arguably none was needed) | none |
| Reproducible | the split is deterministic, but the **input file is not committed**, so the split is not reproducible from the repo | same |
| Gap / purge | **none.** The last 5 training targets use returns that fall in the test period (overlap leakage across the boundary). With 3 test rows this is moot. `analysis/model_eval.py` uses `TimeSeriesSplit(gap=5)`. | none |

> ⚠ **There is effectively no test set**: 3 rows can't estimate anything. The real "evaluation" of the research claims (GMSI) is in-sample correlation plus a placebo test, and that test is miscalibrated (MODEL_REPORT §6).

---

## 9. The data behind the dashboard's "real data" pages

`analysis/verify_dashboard_numbers.py` re-runs `scripts/08_market_dynamics_analysis.py`'s synthetic branch and compares it with `dashboard/app.py`:

| Dashboard constant | Dashboard value | Synthetic (seed 42) output |
|---|---|---|
| `NIFTY_REGIME_STATS` (app.py:393-397) | low 0.144/0.352/0.148 · med 0.151/0.350/0.117 · high 0.162/0.349/0.083 | **identical** to 3 dp |
| `SHOCK_DECAY.btc` (app.py:402) | 0.0119, 0.0138, 0.0115, 0.0119, 0.0097 | 0.0118, 0.0138, 0.0114, 0.0118, 0.0096 (the dashboard says "approximate values" read off the figure) |
| `SHOCK_DECAY.nifty` (app.py:403) | 0.0100, 0.0100, 0.0091, 0.0088, 0.0091 | 0.0100, 0.0099, 0.0091, 0.0088, 0.0090 |
| "MFI peak 0.76, COVID crash, Mar 2020" (app.py:636, 996-999) | 0.76 | synthetic BTC MFI max after 2017 = **0.753 on 2020-05-01**; global max 1.0 on 2016-05-19 |
| Date range of the synthetic data | "2016 → 2024 · ~2000 days" (app.py:596) | 2016-02-11 → 2023-08-31, 1,971 rows |

The same functions run on the **real** committed prices (2016-01-01 → 2023-08-31) give:

| | BTC real | NIFTY real |
|---|---|---|
| mean fwd \|r\| at t+1,3,7,14,21 after top-5% shocks | 0.0396, 0.0350, 0.0341, 0.0371, 0.0358 (**peaks at t+1: no "t+3 secondary wave"**) | 0.0163, 0.0156, 0.0140, 0.0124, 0.0098 |
| MFI max (date) | 0.986 (2017-01-06) | 1.000 (2016-02-16) |
| MFI max in Mar 2020 | 0.654 | 0.911 |
| max 30d annualised vol (date) | 1.645 (2020-04-06) | 0.766 (2020-04-23) |
| cumulative log return | 4.098 | 0.885 |

The committed `fig1_mfi_btc.png` shows a cumulative log return peaking around 0.48 and a 30-day volatility never above about 0.5. Real BTC returned 4.10 log points over the same period, and its volatility reached 1.645. The figure is not BTC.

---

## 10. Data limitations

1. **Sample size.**
   - Sentiment covers 91 days (82 BTC days with headlines). That is about 3 months, inside a single regime (the Q4-2024 BTC rally). No volatility model with 9 features can be fit on that.
   - The price history (about 4,000 days) is adequate for univariate volatility models.
   - GMSI covers about 4,000 days, but its effective sample size is much smaller because of autocorrelation (MODEL_REPORT §6: variance inflation of about 6.9× even under a mild AR(1) assumption).
2. **Representativeness.**
   - Headlines are a keyword-query sample from GDELT, at most 250 per query per day, titles only, and English-biased after cleaning.
   - BTC-USD is a single venue aggregate on Yahoo. NIFTY is an index, not tradable volume.
3. **Survivorship and selection.** The two assets were chosen after the fact. GDELT coverage expands over time, so `event_count` has a trend unrelated to stress, which only partly cancels in an expanding z-score.
4. **Bias risks.**
   - Language bias in the sentiment input.
   - A FinBERT score that measures "neutral − positive" rather than sentiment (P0-3).
   - The GMSI conflict, economic, and political counts are always zero (P1-6).
5. **Do NOT use this data or these models for:**
   - any trading or risk decision
   - assets other than BTC-USD and NIFTY 50
   - intraday horizons
   - periods before 2015 (GDELT GKG v2) or outside 2024-10 → 2025-01 for anything involving sentiment
   - non-English news
   - any claim about causality
