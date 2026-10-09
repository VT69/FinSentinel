# FinSentinel — Issue Audit

Severity scale:

- **P0 BROKEN**: crashes, or produces wrong or misattributed results
- **P1 CORRECTNESS**: leakage, invalid statistics, wrong metric, skew, unseeded randomness
- **P2 DEPLOYMENT**: blocks or degrades Streamlit hosting
- **P3 QUALITY**: structure, duplication, error handling, paths, tests

**Risk of fixing** is the chance that a fix breaks something else or changes a published number.

Evidence files live in `analysis/outputs/` (see DATA_REPORT and MODEL_REPORT).

---

## 🔐 SEC-1: A real-looking API key is committed (do this first, outside the code)

| | |
|---|---|
| Where | `.env.example:16` contains `FRED_API_KEY=` followed by a 32-hex-character value, not a placeholder. Every other key in the file is `your_..._here`. |
| Why it matters | The file is public on GitHub, so the key must be treated as compromised. Removing the line does **not** remove it from git history (`git log -p .env.example`, commit `193ea3e`). |
| Fix | 1. **Revoke/rotate the key at fred.stlouisfed.org.** This step is required; editing the repo alone does not make the key safe. 2. Replace the value with `your_fred_api_key_here`. 3. Optionally purge it from history (`git filter-repo`). That rewrites history, so it is your call. |
| Risk | none for the code; a history rewrite affects forks and clones |

---

## P0: BROKEN or wrong results

| ID | Where | What's wrong | Why it matters | Proposed fix | Risk |
|---|---|---|---|---|---|
| **P0-1** | `dashboard/app.py:393-404` (constants), `:974, 1010, 1025, 1080, 1111` ("Real Data" headers), `:1290` ("Complete Results Summary — Real Data"); figures `reports/figures/fig{1,3,4,5,6}_*.png` = `dashboard/assets/` copies | MFI, shock-propagation, and regime numbers and figures come from **`scripts/08_market_dynamics_analysis.py`'s synthetic GARCH branch** (seed 42) but are labelled as real BTC/NIFTY data. Proven in `analysis/outputs/verify_dashboard_numbers_stdout.txt`: NIFTY regime stats identical to 3 dp; shock decay within ±0.0001; "MFI 0.76 COVID peak" = synthetic 0.753 on 2020-05-01. | This is the most serious issue in the repo. "Findings" 3 and 4 in the README (AC₁ paradox, BTC t+3 wave) are artefacts of a random seed. An interviewer who opens the figure will see BTC with a cumulative log return of about 0.25, when real BTC did about 4.10. | Regenerate MFI and shock figures from **real** prices (code exists; DATA_REPORT §9 already has the real values), **remove or relabel** the regime/AC₁ results that need GMSI (not in the repo), and rewrite the README/dashboard claims. **Design choice → see Phase 5.** | Medium: changes every published Paper-2 number |
| **P0-2** | `scripts/08_market_dynamics_analysis.py:72-112, 855-875` | Silent fallback to synthetic data when input files are missing. The expected files (`btc_features.csv`, `nifty_features.csv`, `gmsi_daily.csv`) are **produced by no code in the repo**, so the script always runs synthetic. | It manufactures "results" without failing. This is the root cause of P0-1. | Fail loudly (`FileNotFoundError` with instructions). Keep synthetic data only behind an explicit `--demo` flag, and watermark demo figures. | Low |
| **P0-3** | `notebooks/03_sentiment_analysis.ipynb` cell 4 (`return probs[2] - probs[0]`) | ProsusAI FinBERT label order is `{0: positive, 1: negative, 2: neutral}` (upstream `finbert/finbert.py:607-608`). The code computes **P(neutral) − P(positive)**, not P(pos) − P(neg). | Every FinBERT-derived number (`finbert_score`, VSI, `sentiment_core` in GMSI, the RF features) measures "neutrality minus positivity". Fingerprint: corr(FinBERT, VADER) = −0.352 on BTC (`notebooks/05` cell 4). | Use `model.config.id2label` to index by name: `p[label2id['positive']] - p[label2id['negative']]`. Re-score `text_with_sentiment.csv`. | Medium: must be re-run locally (huggingface.co is blocked in this sandbox); changes sentiment results |
| **P0-4** | `models/rf_btc.pkl`; trained by `pipeline/train_rf_model.py:8-17` on roughly 8–10 rows | All 600 trees have depth 0, all importances are 0.0, and the output is a constant −3.7299 (σ = 0.024) for any input. | The project's only "ML model" is a mean predictor. Presenting it as a forecaster is indefensible. | Delete it, or replace it with a model trained on the full price history (MODEL_REPORT §7b). **Design choice → Phase 5.** | Low |
| **P0-5** | `dashboard/app.py:944-954` | "Interactive Null Distribution — Simulated (same method)" is `np.random.normal(REAL_CORR_BTC*0.1, 0.022, 500)`: a fabricated histogram, **not** a permutation null. | Presented as a reproduction of a statistical test. | Remove it, or compute a real (circular-shift) null from data. | Low |
| **P0-6** | `dashboard/app.py:635, 710, 740-741, 890, 930-931, 1309-1310`; `README.md:52` | Placebo p-values "<2%", "~0.01–0.02", "~0.03–0.05" and "500 permutations" come from no code. `scripts/validate_gmsi.py:78` uses **1000** shuffles and printed p = 0.000 (BTC) / 0.003 (NIFTY) (`reports/exogenous_vsi_v1_results`). A calibrated null gives **0.046 / 0.147** (MODEL_REPORT §E). | Numbers that are made up, or inconsistent with the code, are the first thing a reviewer checks. | Report the calibrated p-values with method, or drop significance claims. | Low (text) |
| **P0-7** | `notebooks/01_data_collection.ipynb` cells 10-11 (`nifty_gdelt`, `btc_yahoo`, `nifty_yahoo`, `btc_fear_greed` undefined), `01a` cell 9 (`KeyError: 'timestamp'`, saved output), `01b` cell 5 (`KeyError: 'year'`), `07` cell 4 (`ValueError: 0 sample(s)`) | Notebooks do not run top to bottom. Several committed outputs are exceptions. | "Can you re-run your pipeline?" The honest answer today is no. | Mark them as archived exploratory notebooks, or fix the four cells. Move the working logic into scripts. | Low |
| **P0-8** | `run_pipeline.py:10`, `scripts/{reconstruct_gmsi,validate_gmsi,regime_analysis,sanity_check_gmsi,construct_vsi_full,data_check}.py` | All depend on `data/processed/*`, which is gitignored and not committed. Verified: `FileNotFoundError: data/processed/btc_sentiment_aligned.csv` and `…/gmsi_exogenous.csv`. GDELT event inputs are also absent. | None of the research results can be reproduced from the repo. | Commit the small processed artefacts (`gmsi_exogenous.csv` is about 4k rows), or add a documented `make data` path. | Low |

---

## P1: CORRECTNESS

| ID | Where | What's wrong | Why it matters | Fix | Risk |
|---|---|---|---|---|---|
| **P1-1** | `scripts/validate_gmsi.py:77-96`, `notebooks/13` cell 17, `notebooks/12` cell 22 | The placebo test uses an **i.i.d. shuffle** on autocorrelated series (GMSI ACF ≈ 0.82; forward vol ACF = 0.928). False-positive rate is **57.5% at φ=0.9** (MODEL_REPORT §D). | The headline "statistically significant" claim (Finding 2) is not supported. | Circular-shift or block permutation; or Newey-West/HAC standard errors; or a stationary bootstrap. Report the calibrated p. | Medium: the BTC claim becomes borderline and NIFTY is not significant |
| **P1-2** | `dashboard/app.py:870-871` ("monotonically decreasing … holds for both BTC and NIFTY, at both 7-day and 14-day horizons") | The dashboard's own values are not monotone. BTC 7d: Q3 0.0278 < Q4 0.0281 < Q5 0.0294; NIFTY 7d: Q2 0.0078 < Q3 0.0080 < Q4 0.0082 (`app.py:380, 384`). | A U-shape, not a decreasing trend. Easy to catch live. | Describe it as "Q1 is highest; Q2–Q5 roughly flat". | Low |
| **P1-3** | `scripts/regime_analysis.py:28-29` (full-sample 20/80% quantiles), `:106` (full-sample 95% shock threshold), `notebooks/12` cells 10-11 (StandardScaler on all of 2015-2025), `notebooks/09` cell 4 (monthly Trends ffilled from day 1), `notebooks/07` cell 4 (`y_scaler` on all y) | **Look-ahead** in normalisation and regime thresholds. The dashboard claims "expanding quantile, no look-ahead" (`app.py:1274-1275`). | In-sample thresholds are fine for description but not for "predicts" language. Results that rely on them can't be presented as forecasts. | Use expanding quantiles (as in `08_…py:377-401`), or say "in-sample descriptive". | Low |
| **P1-4** | `notebooks/06` cell 2: `rolling(5).std().shift(-1)` | The "next-day" target is std of r[t−3..t+1]. **4 of its 5 days are known at time t** and also feed `vol_5` and `abs_return`. | Partial target leakage; inflates any skill. | Use the target from `feature_engineering.py:20` (`shift(-5)`). | Low |
| **P1-5** | `pipeline/feature_engineering.py:17, 40, 46` and comment `:19` | `vol_60` is computed and unused, but `dropna()` still removes its 59 NaN rows, which **takes 75 rows down to about 11**. The comment says "next-day volatility" for a t+1..t+5 target. | This is a direct cause of P0-4. | Compute only the features that are used, or `dropna(subset=FEATURES+[target])`. Fix the comment. | Low |
| **P1-6** | `notebooks/11` cell 15 (`df["EventRootCode"].isin(["14","18",…])`) | `pd.read_csv` parses `EventRootCode` as int (or mixed, per the DtypeWarning in cell 16 output), so string membership never matches. `conflict_events`, `economic_events`, and `political_events` are **all 0** (cell 20 `describe`: mean 0.0). | The GMSI v0 (`notebooks/12` cells 9-11) averages 3 zero-variance columns into the index, diluting it by 50%. The README's "conflict themes" component doesn't exist in practice. | `read_csv(dtype={'EventRootCode': str})` and zero-pad, or compare ints. Rebuild the events aggregate. | Medium: needs the GDELT files |
| **P1-7** | `scripts/reconstruct_gmsi.py:61-70`; claims in `README.md:163-169` and `app.py:1247-1250` | Code: equal-weight mean of 6 expanding z-scores with `fillna(0)`. Docs: "PCA loading weights" and "expanding min-max → [0,1]". **No PCA anywhere in the repo** (`grep -rn PCA`). Sentiment exists on about 2% of days, so the index changes composition over time. | Methodology as described is not methodology as implemented. | Make the docs match the code, or implement PCA on an expanding window and drop components with no coverage. | Low (docs) / Medium (code) |
| **P1-8** | `app.py:1258-1260` "Consolidation: if max class prob ≥ 0.65 → FinBERT … else 0.6×FinBERT + 0.4×VADER. Validated: 82.7% accuracy on 300-article held-out set vs 64.3% VADER" | **No code** implements the 0.65 rule or any labelled evaluation (`grep -rn "82.7\|0.65\|held-out"` finds nothing relevant). | Unsupported performance claim. | Remove it, or build a labelled evaluation (e.g. the Financial PhraseBank). | Low |
| **P1-9** | `README.md:264`, `app.py:773` "MFI validated vs VIX (FRED)" | FRED was never pulled (`data/raw/pipeline.log`: only yfinance ran), and no code compares MFI with VIX. | Unsupported claim. | Remove it, or implement it (FRED key permitting). | Low |
| **P1-10** | `scripts/08_…py:424` (`sub["abs_return"].autocorr(lag=1)` on a regime-masked frame), `scripts/regime_analysis.py:76-77` (`acf` on a `dropna`'d regime slice) | Lag-1 autocorrelation computed on **non-contiguous** days: the last day of one low-stress spell is paired with the first day of the next one, weeks later. | The "AC₁ by regime" statistic, the basis of the "paradox", is mis-specified even on real data. | Compute AC1 only on within-spell consecutive pairs, or use a regime-interacted AR regression. | Low |
| **P1-11** | `notebooks/02` cell 3 `re.sub(r"[^a-zA-Z\s]", "", text)` | Deletes digits, `$`, `%`, and all non-Latin script. 117 non-Latin headlines survive as English fragments; 123 are dropped (DATA_REPORT §5b). | Biased, corrupted model input ("$450M" → "m"). | Keep digits and punctuation for FinBERT; language-detect and drop or translate non-English. | Medium |
| **P1-12** | `fetch_yfinance.py:78`, `08_…py:152-154, 884-887`, `notebooks/09` cell 16 vs `notebooks/13` cell 3 | BTC annualised with √252 in some places and √365 in others. | Volatility levels aren't comparable across files. | One helper, `annualise(asset)`. | Low |
| **P1-13** | `run_pipeline.py:15-17` | No purge gap between train and test for a 5-day-ahead target. | Overlapping label windows leak across the split. | `TimeSeriesSplit(gap=5)` (as used in `analysis/model_eval.py`). | Low |
| **P1-14** | `notebooks/09` cell 34, `notebooks/10` cell 13 | Unseeded bootstraps. | Reported bootstrap sds aren't reproducible. | `random_state=` / `np.random.default_rng(seed)`. | Low |
| **P1-15** | `scripts/construct_vsi_full.py:26-28` | `rolling(w, min_periods=1).std()`: the first rows use 1–2 observations. | Noisy early z-scores. | `min_periods=w`. | Low |
| **P1-16** | `08_…py:344-372` | "Half-life" comes from a log-linear fit on **5 points** of a non-monotone curve (synthetic BTC gives 57.9 days with R² = 0.60). | Not a meaningful estimate. | Fit an AR/HAR decay on the full series, or drop it. | Low |

### Train/inference skew: both paths side by side

| Step | TRAINING (`run_pipeline.py` → `pipeline/*`) | INFERENCE (`dashboard/app.py`) |
|---|---|---|
| Input | `data/processed/btc_sentiment_aligned.csv` (prices + daily FinBERT/VADER) | none: no file is read except PNGs (`app.py:29-33`) |
| Text preprocessing | `clean_text` (nb02) | none |
| Sentiment model | FinBERT `probs[2]-probs[0]` + VADER | none |
| Features | 9 features (`feature_engineering.py:42-48`) | none |
| Scaling | none | none |
| Model | `RandomForestRegressor` → `models/rf_btc.pkl` | **never loaded** |
| Output | log(5-day σ) | hard-coded constants (`app.py:379-404`) + seeded simulation (`app.py:409-447`) |

**Verdict:** there is no skew to measure because **there is no inference path**. If one were added, the features would have to be rebuilt exactly as in training, including the FinBERT mapping, the UTC-date bucketing, and ≥ 60 days of history (because of the `vol_60`/`dropna` quirk). Today none of that exists.

The same problem exists *inside training*: `notebooks/06` (12 features, 1-day target) and `pipeline/` (9 features, 5-day target) are **two different models with different targets**, both called "the RF".

---

## P2: DEPLOYMENT (Streamlit Community Cloud)

| ID | Where | What's wrong | Evidence | Fix | Risk |
|---|---|---|---|---|---|
| **P2-1** | `dashboard/app.py:668-675` (overview), `:1167-1172` (regime) | `fig.add_vrect(row=…, col=…)` in a loop: 620 calls on a 2-row subplot with 1,971-point traces. Each call re-scans the figure, so cost is quadratic. **Overview (the landing page) takes 185.2 s; regime takes 31.4 s.** Nothing is cached, so it repeats on every visit. | `AppTest` timings (ARCHITECTURE §3); isolated test: building the figure took 177.9 s, `to_json` 0.0 s | Build a `shapes=[…]` list once and `update_layout(shapes=…)`, and/or `@st.cache_data` the figure. Better still, delete the synthetic chart (P0-1). | Low |
| **P2-2** | `dashboard/runtime.txt` | Community Cloud **does not read `runtime.txt`**; the Python version is chosen in the deploy dialog's *Advanced settings*. Source: Streamlit forum staff replies (not official docs). | — | Remove the file and document the setting in DEPLOY.md. | None |
| **P2-3** | `dashboard/requirements.txt:2-6` | Floor-only pins (`>=`). Today they resolve to pandas 2.3.3, numpy 2.4.6, **plotly 7.1.0**, scipy 1.17.1, Pillow 11.3.0 (checked in a clean Python 3.11.17 venv). The file says `plotly>=5.18`, and a major-version jump happened silently. | `uv pip install -r` in a clean venv | Pin exact versions that are tested together (`==`). Drop `scipy`: it is only imported, never used for computation (`app.py:13`). | Low |
| **P2-4** | `requirements.txt` (root) | Unpinned. It includes `tensorflow`, `torch`, `transformers`, `snscrape` (unmaintained), and `django-pipeline` (a Django asset library, unrelated). It is missing `scipy`, `statsmodels`, `streamlit`, `bs4`, `tqdm`, `python-dotenv`. Python version per notebook kernels is 3.12.1 vs README "3.11". Whether it resolves: **NOT VERIFIED** (multi-GB install not attempted). | — | Split into `requirements-dashboard.txt` (pinned, small) and `requirements-research.txt`. | Low |
| **P2-5** | `app.py:411, 945` | `np.random.seed()` mutates the global RNG inside a multi-user server process. | — | Use `np.random.default_rng(seed)` locally. | Low |
| **P2-6** | `app.py:16` | `warnings.filterwarnings("ignore")` hides deprecation warnings (e.g. `use_container_width`, deprecated in later Streamlit releases) until something breaks. | — | Remove it. | Low |
| **P2-7** | — | No `.streamlit/config.toml`. The dark theme is forced with CSS `!important` overrides (`app.py:36-372`), which flash on load and break when Streamlit changes its DOM test-ids. | — | Set `[theme] base="dark"` etc. in `config.toml`; keep minimal CSS. | Low |
| **P2-8** | sizes | Model 232 KB, assets 1.4 MB, repo data about 1 MB. **Well within hosting limits; no LFS needed.** | `du -sh` | — | — |

**Does the app crash?** Not in this sandbox: all 7 pages ran with 0 exceptions under Streamlit 1.40.0 / Python 3.11.17, and `/_stcore/health` returned `ok`. The observable failure is the **3-minute first paint** of the landing page (P2-1), which looks like a hung app. **Live Streamlit Cloud behaviour: NOT VERIFIED** (no access to the deployment or its logs).

---

## P3: QUALITY

| ID | Where | Issue | Fix |
|---|---|---|---|
| P3-1 | repo-wide | **No tests, no CI, no linting.** | Add `pytest` smoke tests for features/target, plus `AppTest` for the dashboard. |
| P3-2 | ARCHITECTURE §7c | CWD-dependent relative paths in three incompatible conventions (repo root / `scripts/` / `notebooks/`); absolute Windows paths in `reports/*_results`. | Use `Path(__file__).resolve()` roots everywhere. |
| P3-3 | `pipeline/dataset_builder.py` | Dead code that references columns which don't exist (`log_vol`, `sent_vol`). | Delete it. |
| P3-4 | `src/*.py` (5 empty files), `docs/methodology.md` (empty), `Untitled.ipynb` (0 cells) | Placeholders. | Delete them. |
| P3-5 | `notebooks/08_market_dynamics_analysis.ipynb` | Verbatim copy of `scripts/08_…py`, never executed. | Delete it, or import the script. |
| P3-6 | `reports/figures/*` ≡ `dashboard/assets/*` (byte-identical) | Two copies; they will drift. | Generate into one place. |
| P3-7 | `reports/exogenous_vsi_v1_results`, `reports/regimes_vsi_results` | No file extension; the first is a subset of the second; image paths point to `C:/Users/Dell/.gemini/antigravity/brain/…`. These read as AI-agent walkthroughs. | Rename to `.md`, dedupe, use relative paths, and make sure you can personally defend every sentence. |
| P3-8 | `README.md:112-113` lists `06_volatility_forecasting.ipynb`, `07_model_evaluation.ipynb` (don't exist); `:133` "Source CSVs (gitignored)" (they're committed); `:17` "Real-time analysis" (static); `dashboard/README.md:37` says the dashboard **uses synthetic data**, contradicting the main README | Documentation that contradicts itself and the code. | Rewrite after the fixes. |
| P3-9 | `nifty_google_trends.csv` rows 192-195 | Duplicate 2025-11/2025-12 rows. | `drop_duplicates`. |
| P3-10 | `data_pipeline/sources/fetch_gdelt.py:153-160` | The parquet cache never expires, so the current (partial) month is frozen forever. | Skip the cache for the current month. |
| P3-11 | `data_pipeline/sources/fetch_quandl.py:129` | Uses the deprecated `quandl` package; README_DATA says `nasdaqdatalink`. | Switch to `nasdaq-data-link`. |
| P3-12 | `notebooks/01a` cell 4 (`except: pass`), many `except Exception` that log and continue | Errors swallowed silently. | Narrow the exceptions and surface failures. |
| P3-13 | `pipeline/evaluate.py:14` | `plt.show()` in pipeline code (blocks or no-ops headless). | Save to a file. |
| P3-14 | `dashboard/app.py:1322` | Code and a stray comment on the same line, no trailing newline. | Cosmetic. |
| P3-15 | git history | 10 commits titled "Jan 2 Updated"; 19,067 lines added in one "Added Dashboard" commit. | Descriptive, atomic commits from here on. |
| P3-16 | `dashboard/app.py` (1,322 lines) | One file with 335 lines of CSS, data, and 7 pages. | Split into `pages/`, `theme.css`, and a `data.py` that loads results from files. |
