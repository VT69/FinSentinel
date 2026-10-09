# FinSentinel — Model Report

> All numbers come from `analysis/model_eval.py` (`analysis/outputs/model_eval_stdout.txt`, `model_eval_summary.json`, `model_eval_walk_forward.csv`, `model_eval_{btc,nifty}.png`, `model_eval_placebo_validity.csv`) and `analysis/verify_dashboard_numbers.py`, run in this session.
> Environment: Python 3.12.3, scikit-learn 1.8.0 (the artifact's own version), numpy 2.5.3, pandas 3.0.6, scipy 1.18.1.

The repo contains **one trained ML model**: the BTC Random Forest. It also has two abandoned ones (a 12-feature RF in nb06, and a CNN in nb07 that crashed) and a set of **statistical tests** that carry the actual research claims (GMSI conditional expectations, the placebo test, MFI/shock analysis). Both are covered here, because an interviewer will probe both.

---

## 1. Headline findings (read before anything else)

| # | Finding | Evidence |
|---|---|---|
| 1 | The committed model `models/rf_btc.pkl` is a **constant**. All 600 trees have depth 0, all 9 importances are 0.0, and every input yields −3.7299327 (σ = 0.02399). | §A of `model_eval_stdout.txt` |
| 2 | Re-running the pipeline reproduces this. 10 training rows < 2×`min_samples_leaf` = 20, so no split is possible, and the RF is **no better than `DummyRegressor(mean)`** (RMSE 0.009647 vs 0.009553). | §B |
| 3 | The notebook version (`notebooks/06` cells 3, 6, 7) trained on **15 rows**. It reported RMSE 0.005628 / MAE 0.005529 on **3 test rows**, with **all importances = 0.0**. Those metrics come from a constant predictor and carry no information. | notebook outputs |
| 4 | Done properly (4,000 days of price data, walk-forward), the RF with the repo's hyperparameters **loses to a 3-variable linear HAR model** on BTC and NIFTY, and on BTC it is below zero R². | §C |
| 5 | The repo's placebo test (i.i.d. shuffle) is **miscalibrated** for autocorrelated series. With an *independent* fake GMSI (AR coefficient 0.9) it declares significance **57.5%** of the time at α=0.05. | §D |
| 6 | Under a (still optimistic) AR(1) null, the published GMSI→volatility correlations have **p ≈ 0.046 (BTC) and p ≈ 0.147 (NIFTY)**. The dashboard shows "<2%" and "~0.03–0.05". | §E |
| 7 | Every MFI, shock-propagation, and "AC₁ paradox" number on the dashboard is the output of **`np.random.seed(42)` synthetic data**. | `verify_dashboard_numbers_stdout.txt` |

---

## 2. Preprocessing pipeline, in order (RF track)

| # | Step | Code | Rationale | Assessment |
|---|---|---|---|---|
| 1 | Lowercase, strip URLs, delete non-`[a-zA-Z\s]`, keep len > 10 | `notebooks/02` cells 3-4 | normalise text for the models | Lowercasing suits FinBERT (it's built on bert-base-uncased). Deleting every non-ASCII letter and digit is harmful: `"$450M"` → `"m"`, `"10 key things"` → `"key things"`, and Hindi/Korean become English fragments (DATA_REPORT §5b). |
| 2 | FinBERT score = `probs[2] − probs[0]` | `notebooks/03` cell 4 | "positive − negative" | **Wrong index mapping.** Upstream ProsusAI defines `label_dict = {0:'positive', 1:'negative', 2:'neutral'}` (`finbert/finbert.py:607-608` in github.com/ProsusAI/finBERT, fetched this session). So the code computes **P(neutral) − P(positive)**. Fingerprints: corr(finbert, vader) = −0.352 on BTC (`notebooks/05` cell 4); "Bitcoin Price Decline Forces $450M in Long Liquidations" → 0.0106 and "stock market" → 0.896 (`notebooks/04` cell 4). Checking the live HF `config.json`: **NOT VERIFIED** (huggingface.co blocked from this sandbox). |
| 3 | VADER compound | `notebooks/03` cell 5 | lexicon baseline | fine; VADER was built for social media, not finance |
| 4 | Daily mean per UTC date per asset | `notebooks/04` cell 5 | match daily bars | equal-weights each headline; UTC date ≠ NIFTY session (DATA_REPORT §8c L10) |
| 5 | Log return `diff(log close)` | `notebooks/04` cell 3 | additive, ≈ % change for small moves | ✔ |
| 6 | Inner join prices × sentiment, lags 1/2/3/5, `dropna` | `notebooks/04` cells 7-8 | lead-lag features | Lag features are built but **not used** by `build_features`, and their `dropna` costs 5 rows. |
| 7 | `build_features` | `pipeline/feature_engineering.py:3-56` | §3 | `dropna` after computing the unused `vol_60` costs 59 rows |
| 8 | Chronological 80/20 | `run_pipeline.py:15-17` | no future in train | ✔ in principle; no purge gap |
| — | Scaling / imputation / encoding | none | trees are scale-invariant | ✔ correct for RF |

---

## 3. Feature engineering (`pipeline/feature_engineering.py`)

r_t = daily log return. All windows are trailing, so they only use data at or before t.

| Feature | Formula | Line | Why it exists | Used by the model? |
|---|---|---|---|---|
| `finbert_score` | mean over the day's headlines of P(neutral) − P(positive) (*intended*: P(pos) − P(neg)) | input | news tone | ✔ |
| `vader_score` | mean VADER compound | input | lexicon tone | ✔ |
| `finbert_surprise` | finbert_t − mean(finbert_{t−4..t}) | L28-30 | the *change* in tone matters more than its level | ✔ |
| `vader_surprise` | same for VADER | L31-33 | same | ✔ |
| `abs_return` | \|r_t\| | L24 | today's move (a one-day vol proxy) | ✔ |
| `sq_return` | r_t² | L25 | ARCH-style variance proxy. **Redundant with `abs_return`** for trees (Spearman = 1.000). | ✔ |
| `vol_5`, `vol_22` | std(r_{t−4..t}), std(r_{t−21..t}) | L15-16 | volatility clustering | only via interactions |
| `vol_60` | std(r_{t−59..t}) | L17 | quarter-scale vol | ✘ (commented out L46), but **its NaNs still drive `dropna`** |
| `sent_x_vol5`, `sent_x_vol22` | finbert × vol | L36-37 | "news matters more in volatile markets" | ✔ |
| `sent_x_absret` | finbert × \|r_t\| | L38 | same idea at one day | ✔ |
| **target** `log_vol_target` | log(std(r_{t+1..t+5}) + 1e-6) | L20-21 | log stabilises a skewed, positive target (BTC skew 2.36 → −0.29, DATA_REPORT §6c) | — |

Note: raw vol features were removed from the model (L46 comment), presumably to force the model to use sentiment. This deliberately throws away the strongest known predictor of future volatility, which is past volatility (DATA_REPORT §7: vol_22 has the highest correlation with the target). That makes the sentiment model weaker, not more honest.

---

## 4. Algorithms and how they work

### 4a. Random Forest Regressor: the configured model

Hyperparameters as configured (`pipeline/train_rf_model.py:8-15`, confirmed from the pickle): `n_estimators=600, max_depth=8, min_samples_leaf=10, max_features="sqrt", random_state=42, n_jobs=-1`. Defaults left in place: `bootstrap=True`, `criterion="squared_error"`, `min_samples_split=2`, `max_samples=None`. **No hyperparameter search exists anywhere in the repo.**

**How it works (interview depth):**
1. Each of the 600 trees is trained on a **bootstrap sample**: n rows drawn with replacement, so about 63% of distinct rows appear in each.
2. At every node, the tree considers only a random subset of features (`sqrt`: ⌊√9⌋ = 3 of 9) and picks the threshold that most reduces **squared error**, i.e. the within-child variance of y.
3. Splitting stops at `max_depth=8`, or when a child would have fewer than `min_samples_leaf=10` rows. A leaf predicts the **mean y** of its training rows.
4. The forest's prediction is the **average of the 600 tree predictions**.
5. Bagging plus feature subsampling **decorrelates** the trees, so averaging reduces variance without raising bias much. That is why forests resist overfitting better than a single deep tree.
6. What it still can't do:
   - **extrapolate.** Predictions are bounded by the range of training targets, which hurts for unprecedented volatility spikes.
   - exploit smooth linear structure efficiently. That's why HAR-OLS beats it in §C.
   - learn anything when n < 2×`min_samples_leaf`. Every tree is then a single leaf, which is what happened here.
7. Impurity importance (`feature_importances_`) is biased toward high-cardinality continuous features and is computed on training data. That's why permutation importance on held-out data is reported in §8.

### 4b. CNN (`notebooks/07`, never ran)
`Conv1D(64,3) → Conv1D(32,3) → GlobalAvgPool → Dense(1)` on 30-day windows, Huber loss with δ=0.01, Adam, early stopping. It crashed in cell 4 because NIFTY had 54 rows, all removed by `rolling(60)` + `dropna`. Had it run, `y_scaler` was fit on the full target before the split (leakage).

### 4c. Statistical machinery behind the research claims

| Method | Where | What it does |
|---|---|---|
| Spearman ρ | `validate_gmsi.py:36` | Pearson correlation of ranks; robust to monotone transforms and outliers |
| Quintile conditional means | `validate_gmsi.py:42-44` | E[fwd vol \| GMSI quintile] over the full sample (descriptive) |
| Placebo / permutation | `validate_gmsi.py:77-96` | shuffle GMSI i.i.d. 1,000×, recompute ρ, two-sided empirical p. **Assumes exchangeability, which fails for autocorrelated series (§D).** |
| KS / Mann-Whitney | `validate_gmsi.py:70-71`, `regime_analysis.py:173-174` | distribution and median shift tests; also assume independent observations |
| Granger, quantile regression | `notebooks/09` cells 36-48 | n ≈ 60; stress-regime subsets of 11 and 7 rows (cell 44 prints "Not enough data") |
| MFI | `08_…py:203-278` | mean of expanding-min-max-normalised rolling AC1(\|r\|), CoV(vol_7d), and P(\|r\| > 2σ₃₀) |
| Shock propagation | `08_…py:283-372` | mean \|r_{t+h}\| after days where \|r_t\| > expanding Q95; log-linear "half-life" fit on **5 points** |

---

## 5. Why a Random Forest? The honest comparison

| Alternative | Pro | Con | Chosen? Why |
|---|---|---|---|
| **HAR-RV (OLS on log vol over 1/5/22 days)** | the standard volatility benchmark; 4 parameters; interpretable; extrapolates | linear in logs | **Not tried in the repo.** §C shows it **beats the RF on both assets.** It should have been the baseline. |
| GARCH(1,1) / EGARCH | the textbook conditional-variance model; uses all returns | needs the `arch` library; univariate unless exogenous regressors are added | not tried. The repo even *simulates* GARCH (`app.py:414-420`) but never fits one. |
| Logistic / linear regression on sentiment | interpretable, cheap | same small-n problem | not tried |
| XGBoost / LightGBM | usually beats RF on tabular data | more hyperparameters; same n=10 failure | not tried |
| Neural nets (CNN/LSTM) | sequence modelling | data-hungry; n ≈ 54 here | tried (nb07), crashed |
| **Random Forest** | non-linear interactions, little tuning, scale-free | can't extrapolate; useless at n=10 | **chosen.** Likely reason: "sentiment × volatility interactions are non-linear". No comparison was ever run. |

**Honest answer for the interview:** "I picked RF because it handles non-linear interactions without much tuning. The real problem was data, not algorithm: three months of headlines give about ten usable rows. When I re-ran the comparison on price data alone, a three-variable HAR regression beat the forest. In hindsight I should have started with HAR as the baseline and asked whether sentiment adds anything on top of it."

---

## 6. Validation strategy

| Aspect | Repo | Sound? |
|---|---|---|
| Scheme | single chronological 80/20 holdout (`run_pipeline.py:15-17`) | the direction is right (no shuffling), but there is one split and no purge gap |
| Size | 10 train / 3 test (reconstructed); nb06: 12 / 3 | ✘ cannot estimate error |
| k-fold / CV | none | ✘ |
| Hyperparameter tuning | none (and so no validation set needed) | — |
| Research claims | in-sample correlation + i.i.d. placebo | ✘ the null is wrong (below) |

### §D. Is the placebo test valid? (No.)
Setup: the **real** BTC forward-7-day volatility series (n = 4,009, lag-1 autocorrelation **0.928**), built exactly as in `construct_vsi_full.py:26` + `validate_gmsi.py:23`, correlated with a **fake GMSI generated independently** as AR(1) noise. 200 simulations per φ, 1,000 permutations each, two-sided p as in `validate_gmsi.py:95`.

| Fake-GMSI AR coefficient φ | False-positive rate, **repo's i.i.d. shuffle** | False-positive rate, circular-shift permutation |
|---|---|---|
| 0.0 | 0.070 | 0.070 |
| 0.5 | 0.225 | 0.070 |
| 0.9 | **0.575** | 0.060 |
| 0.98 | **0.740** | 0.055 |

**Why:** a permutation test is valid only if, under H₀, every reordering is equally likely (exchangeability). Shuffling a persistent series destroys its autocorrelation, so the null distribution is far too narrow: its sd is about 1/√n = 0.0158. When **both** series are persistent, spurious correlations of ±0.05–0.10 are routine. A circular shift preserves the autocorrelation and stays close to the nominal 5% (5.5–7% here, within Monte-Carlo error of 200 sims).

### §E. Calibrated null for the published numbers
`reports/figures/gmsi_sanity_checks.png` shows GMSI lag-1 ACF ≈ 0.82, decaying slowly (still ≈ 0.2 at lag 30). Simulating an independent AR(1) GMSI with φ=0.82 (2,000 draws) against the real forward vol:

| | BTC | NIFTY |
|---|---|---|
| published Spearman ρ (`validate_gmsi.py` output; `app.py:389-390`) | −0.0837 | −0.0580 |
| null sd, i.i.d. theory | 0.0158 | 0.0160 |
| null sd, AR(1) φ=0.82 | **0.0414** | **0.0404** |
| variance inflation | 6.86× | 6.41× |
| p (i.i.d. normal approx: what the repo's test approximates) | 1.2e-7 | 2.8e-4 |
| **p (AR(1) φ=0.82 null)** | **0.046** | **0.147** |
| dashboard claim | "< 2%" (`app.py:635`, `740`) | "< 0.05", "~0.03–0.05" (`app.py:741, 930`) |

Because the real GMSI ACF decays **more slowly** than 0.82^k, the true null is even wider, so the true p-values are **larger** than these. The numbers above are the most favourable case for the claim. Conclusion: BTC is borderline at best, and NIFTY is not significant. The dashboard's p-values match neither the code's output (1,000 shuffles → p = 0.000 / 0.003 per `reports/exogenous_vsi_v1_results`) nor a valid test.

---

## 7. Re-run results vs claims

### 7a. RF track

| Source | Claim | Reproduced (this session) | Match? |
|---|---|---|---|
| `notebooks/06` cell 6 | RMSE 0.005628, MAE 0.005529 | Raw input not committed. Reconstructed pipeline: RF 0.009647 / 0.009509 vs Dummy 0.009553 / 0.009414 (3 test rows) | **✘ different data**; both are constant predictors |
| `notebooks/06` cell 7 | importances all 0.0 | all 0.0 | ✔ (the notebook itself shows the model learned nothing) |
| `models/rf_btc.pkl` | — | constant −3.7299327 | — |

### 7b. Walk-forward on the full price history (what the model *can* do)
`TimeSeriesSplit(n_splits=5, gap=5)`, price features only (sentiment doesn't exist before 2024-10), same target, same RF hyperparameters. Mean over 5 folds:

| BTC | test RMSE (vol) | test MAE (vol) | test R² (log) | test QLIKE | train R² (log) |
|---|---|---|---|---|---|
| HAR OLS log(vol5, vol22, vol60) | **0.01878** | **0.01270** | **0.0474** | **1.1496** | 0.2824 |
| RF (repo params) | 0.01912 | 0.01316 | −0.0265 | 1.2210 | 0.5085 |
| dummy mean | 0.02124 | 0.01501 | −0.3847 | 1.9660 | 0.0000 |
| persistence (log vol_5) | 0.02156 | 0.01532 | −0.3674 | 1.7989 | −0.0383 |

| NIFTY | test RMSE | test MAE | test R² (log) | test QLIKE | train R² (log) |
|---|---|---|---|---|---|
| HAR OLS | **0.00488** | **0.00311** | **0.1001** | **0.5479** | 0.2068 |
| RF (repo params) | 0.00519 | 0.00322 | 0.0620 | 0.6839 | 0.4128 |
| dummy mean | 0.00558 | 0.00363 | −0.1862 | 0.8905 | 0.0000 |
| persistence | 0.00574 | 0.00396 | −0.4390 | 1.2695 | −0.3420 |

Per-fold test R² (log) for the RF: BTC −0.115, 0.110, −0.094, −0.044, 0.010; NIFTY 0.046, 0.110, 0.153, 0.121, −0.121. **Fold variance is as large as the mean**, so the differences between RF and HAR on any single fold aren't meaningful. The averages consistently favour HAR.

**How much does the model beat trivial?** BTC RF vs dummy: RMSE 0.01912 vs 0.02124 (−10%), QLIKE 1.221 vs 1.966. HAR does better still. *"Beats the dummy mean"* is a low bar here, because the dummy ignores volatility clustering entirely. The meaningful baseline is HAR.

---

## 8. Metrics: which and why

| Metric | Definition | Why | Weakness |
|---|---|---|---|
| RMSE in vol space (repo) | √mean((σ̂ − σ)²) | in the target's units | dominated by the few turbulent weeks |
| MAE in vol space (repo) | mean\|σ̂ − σ\| | robust | same scale issue |
| R² on log vol (added) | 1 − SSE/SST | share of variance explained vs the test-set mean | negative = worse than predicting the test mean |
| **QLIKE** (added) | mean(σ²/σ̂² − log(σ²/σ̂²) − 1) | **the loss that matters for volatility forecasts**: robust to noise in the realised-vol proxy (Patton 2011), and punishes under-prediction more, which is the costly error in risk | less intuitive |

**Business meaning:**
- A **false positive** (over-forecasting volatility) means over-hedging and lower position sizes: the cost is giving up return.
- A **false negative** (under-forecasting before a spike) means under-hedging into a crash: margin calls and VaR breaches.

QLIKE's asymmetry matches that cost structure. For risk management, under-prediction in Q5 (turbulent) weeks is the metric to watch, and that is exactly where the model fails (§10).

---

## 9. Feature importance (last walk-forward fold)

| BTC feature | impurity | permutation Δ MSE (mean ± sd, 20 repeats) |
|---|---|---|
| vol_22 | 0.395 | **+0.0229 ± 0.0060** |
| vol_5 | 0.259 | −0.0069 ± 0.0032 |
| vol_60 | 0.213 | −0.0037 ± 0.0026 |
| abs_return | 0.068 | −0.0018 ± 0.0010 |
| sq_return | 0.065 | −0.0019 ± 0.0012 |

| NIFTY feature | impurity | permutation |
|---|---|---|
| vol_22 | 0.449 | −0.0118 ± 0.0051 |
| vol_60 | 0.240 | **+0.0080 ± 0.0027** |
| vol_5 | 0.196 | −0.0059 ± 0.0014 |
| abs_return | 0.063 | −0.0002 ± 0.0006 |
| sq_return | 0.052 | −0.0002 ± 0.0005 |

**Sanity read:**
- Impurity importance says "monthly vol matters most", which makes domain sense.
- Permutation importance shows that, out of sample, **only one feature carries real information per asset**. Permuting the others *improves* MSE (negative Δ), which means the forest overfits them.
- The two methods disagree because impurity is measured on training data (the RF's train R² is 0.51 against −0.03 on test), and correlated features (vol_5/22/60) share importance. Permuting one leaves the others as proxies.

---

## 10. Overfitting check and error analysis

**Train/test gap:** BTC RF train R² 0.509 vs test −0.026; NIFTY 0.413 vs 0.062. HAR shows 0.282 → 0.047 and 0.207 → 0.100. **The RF overfits**: `max_depth=8` with `min_samples_leaf=10` memorises noise in about 3,000 rows of a low signal-to-noise target.

**Learning curve (last fold, training window = most recent N rows):**

| n_train | BTC train / test R² | NIFTY train / test R² |
|---|---|---|
| 250 | 0.408 / −0.045 | 0.407 / −0.123 |
| 500 | 0.402 / −0.043 | 0.453 / −0.124 |
| 1,000 | 0.438 / 0.019 | 0.496 / −0.182 |
| 2,000 | 0.415 / 0.043 | 0.448 / −0.113 |
| 3,288 / 3,214 | 0.441 / 0.010 | 0.399 / −0.121 |

The gap does not close with more data. This is a capacity and signal problem, not a data-volume one. Regularise harder (shallower trees, larger leaves) or use a linear model.

**Error analysis (last fold, by quintile of the *actual* future vol):**

| BTC | n | MAE (vol) | bias (pred − actual, log) |
|---|---|---|---|
| Q1 calm | 132 | 0.00902 | **+0.652** |
| Q3 | 132 | 0.00344 | −0.019 |
| Q5 turbulent | 132 | **0.01704** | **−0.540** |

NIFTY follows the same pattern: +0.777 in Q1, −0.460 in Q5.

The model **shrinks toward the mean**: it over-predicts calm weeks and under-predicts turbulent ones, and in risk terms the under-prediction is the costly error.

**Worst misses:**
- BTC: 2024-08-03/04 (actual 0.074 vs predicted 0.020, the week of the 2024-08-05 global sell-off and yen carry-trade unwind), 2025-02-28/03-01, 2024-03-15.
- NIFTY: **2024-05-28 → 2024-06-03** (actual about 0.039 vs predicted about 0.006, the week of the Indian general-election result on 2024-06-04).

These are **scheduled or exogenous events that past volatility cannot see**, which is precisely the gap a *working* news signal should fill. It is the strongest argument for the project's original idea, and the clearest way to frame it in an interview.

---

## 11. The MFI / shock / regime results (Paper 2)

`analysis/verify_dashboard_numbers.py` reproduces `dashboard/app.py:393-404` from the synthetic branch of `scripts/08_market_dynamics_analysis.py`:
- NIFTY regime stats are identical to 3 dp.
- Shock-decay values match within ±0.0001.
- The "MFI 0.76 COVID peak" is the synthetic value 0.753 on 2020-05-01.

The committed `fig1/3/4/5/6` PNGs regenerate visually identical (pixel shapes differ by 1–2 px from font rendering).

On real prices the same code gives a BTC shock curve that **peaks at t+1** (0.0396), with **no t+3 secondary wave**, and an MFI that peaks in 2017, not in March 2020 (DATA_REPORT §9). **The "AC₁ paradox", the "BTC t+3 retail-lag wave", and the "MFI COVID peak" are properties of a random-number generator.**

The one real-data regime result is in `reports/regimes_vsi_results`, from `scripts/regime_analysis.py`: NIFTY 7-day-vol AR(1) falls from 0.94 (low stress) to 0.76 (high stress). It uses **full-sample** quantiles (look-ahead) and computes AR(1) on **non-contiguous** regime slices, which the code comment at `regime_analysis.py:71-75` admits "breaks time continuity". Re-verifying it: **NOT VERIFIED** (needs `gmsi_exogenous.csv`).
