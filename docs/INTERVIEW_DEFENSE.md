# FinSentinel — Interview Defense Pack

All numbers below come from `python run_pipeline.py` (`dashboard/data/metrics.json`, `gmsi_calibration.json`,
`models/har_*.json`) or from the audit scripts in `analysis/`. If you are asked for a number that is not here, say
you'd have to check. Don't estimate.

Numbers worth memorising:

| Fact | Value |
|---|---|
| Data | BTC 4,016 days (2015-01-02 → 2025-12-30); NIFTY 3,927 days (2010-01-05 → 2025-12-30) |
| Modelling rows after warm-up | BTC 3,951, NIFTY 3,862 |
| Target | std of the next 5 daily log returns, modelled as log |
| Served model | HAR: log σ̂ = a + b₅·log σ₅ + b₂₂·log σ₂₂ + b₆₀·log σ₆₀ |
| BTC coefficients | −1.063, 0.202, 0.352, 0.198 (sum 0.75) |
| NIFTY coefficients | −1.499, 0.116, 0.445, 0.153 (sum 0.71) |
| Out-of-sample R² (log σ), HAR / RF / mean | BTC 0.047 / −0.026 / −0.385 · NIFTY 0.100 / 0.062 / −0.186 |
| QLIKE, HAR / RF / mean | BTC 1.150 / 1.221 / 1.966 · NIFTY 0.548 / 0.684 / 0.890 |
| RF train vs test R² | BTC 0.509 vs −0.026 · NIFTY 0.413 vs 0.062 (overfits) |
| Shocks | 158 (BTC) / 141 (NIFTY). Post-shock \|r\| is 1.72× normal at t+1 and 1.48× at t+21 for BTC; 1.98× and 1.29× for NIFTY |
| GMSI | Spearman with forward 7-day vol −0.084 (BTC), −0.058 (NIFTY); calibrated p ≈ 0.05 and ≈ 0.15 |
| Placebo flaw | i.i.d. shuffle flags an *independent* AR(0.9) series as significant 55–58% of the time (nominal 5%) |
| Old artifact | `rf_btc.pkl`: 600 single-leaf trees, all importances 0, constant σ = 0.0240 |

---

## 1. Two-minute pitch (spoken)

> "FinSentinel asks a practical risk question: **how volatile will Bitcoin and the NIFTY 50 be over the next week,
> and do news-based stress signals help predict that?**
>
> I built a reproducible pipeline on about ten years of daily prices: 4,000 days for BTC and 3,900 for NIFTY.
> The target is the standard deviation of the next five daily returns. Because volatility is heavily right-skewed, I
> model it in logs.
>
> I compared four models in a purged walk-forward setup: five expanding training windows, with a five-day gap
> so no training label overlaps the test period. The winner was a HAR model: a regression on last week's,
> last month's and last quarter's realised volatility. It beat a Random Forest, a persistence forecast and the
> historical mean on every metric, including QLIKE, which is the standard loss for volatility forecasts. The
> honest caveat is that it only explains about 5% of log-volatility variance for BTC and 10% for NIFTY. Its
> worst misses are event weeks like March 2020, which no backward-looking model can see.
>
> The original idea was that news sentiment and a GDELT-based global stress index would fill that gap. When I
> audited my own work, I found three serious problems:
> - the sentiment data only covered three months;
> - my FinBERT score used the wrong label order;
> - my placebo significance test was invalid for autocorrelated series.
>
> With a correct test, the stress-index relationship is borderline for BTC and not significant for NIFTY. I also
> found that some dashboard figures had come from a synthetic-data fallback. I removed them and recomputed
> everything on real data.
>
> The result is a Streamlit app that serves real forecasts with input validation, 40-odd tests and CI. The biggest
> lesson for me was that a strong baseline and a correct null hypothesis matter more than model complexity."

## 2. Thirty-second version

> "I forecast next-week volatility for Bitcoin and the NIFTY 50 from ten years of daily data. A simple HAR model
> on past realised volatility beat a Random Forest in purged walk-forward tests. Then I tested whether a
> news-based stress index adds signal. Once I corrected the significance test for autocorrelation, it was at
> best borderline. It's deployed as a Streamlit app with live forecasts, tests and CI. The most valuable part was
> auditing my own pipeline and fixing what was wrong."

---

## 3. Anticipated questions with model answers

### 3a. Data (12)

**D1. Why this data?**
BTC and NIFTY are a deliberate contrast: a 24/7, retail-heavy crypto market and an institutional equity index
with trading hours. Daily prices from Yahoo are free and long enough (10+ years) to cover several regimes,
including 2017–18, March 2020 and 2022. GDELT was chosen for stress because it is free, global and goes back to
2015.

**D2. How big is it?**
BTC has 4,016 daily rows and NIFTY 3,927. After the 60-day warm-up and the 5-day forward target, 3,951 and 3,862
rows are modelled. The headline dataset is tiny: 1,500 titles over 91 days, which is why sentiment never became a
model input.

**D3. Where exactly does it come from, and can you use it?**
Prices come from Yahoo Finance via `yfinance`, which is an unofficial scraper; Yahoo's terms allow personal or
research use only. Headlines come from the GDELT DOC API. One provenance gap I'd admit: the exact headline query
run and the Google Trends export aren't scripted in the repo.

**D4. Missing values?**
BTC has none, because it trades every day. NIFTY is missing on weekends and holidays, which is structural, not
random. I don't forward-fill prices across those gaps. Each asset is modelled on its own trading calendar, so
returns are computed between actual trading days. Rows are dropped only where the features actually used are
undefined (the 60-day warm-up). The original code also dropped rows because of an *unused* feature, which cut
75 rows to 11.

**D5. Outliers?**
Returns are fat-tailed: excess kurtosis is about 8 for BTC and 12 for NIFTY. The IQR rule flags 379 BTC return
days and the 3-sigma rule flags 70. I kept all of them, because in volatility modelling the extreme days *are*
the signal. Instead I handle them through the loss: I model log volatility, and I also report QLIKE, which is
robust to noise in the volatility proxy.

**D6. Class imbalance?**
It's a regression problem, so there are no classes. The analogue is a skewed target: calm weeks dominate, so a
model can look good on RMSE while missing the few turbulent weeks. That's why I report results by quintile of
actual volatility. The model over-predicts calm weeks by about 0.6 log points and under-predicts turbulent ones
by about 0.55.

**D7. How do you know there's no leakage?**
Three ways.
- Every feature is a trailing window known at day t's close, and the target uses returns t+1 to t+5.
- A unit test changes only *future* prices and asserts the features at t are unchanged (`tests/test_volatility.py`).
- Walk-forward folds have a 5-day gap so no training label overlaps the test period.

I also found and fixed two leaks in the old notebooks: a target that was 80% already known, and a scaler fitted
on the full sample.

**D8. What's in the target exactly?**
The sample standard deviation of the next five daily log returns. That is daily units, not annualised, and it is
modelled as log(σ + 1e-6). For display I annualise with √365 for BTC (it trades every day) and √252 for NIFTY.

**D9. Why log returns?**
They add across time, they're symmetric for up and down moves, and for daily moves they're nearly equal to
percentage returns. One inconsistency I found and fixed: the raw files stored simple returns, while every model
used log returns.

**D10. What's wrong with the sentiment data?**
It covers 91 days, and after the 60-day warm-up only about 13 usable training rows remain. 16% of headlines are
non-English, and the old cleaner either dropped them or reduced them to English fragments. The FinBERT score also
had the wrong label order. So I report sentiment as untested, not as a negative result.

**D11. What would you collect more of?**
Multi-year headline history first, from a licensed news archive or GDELT GKG over 2015–2025. Then intraday data
for proper realised volatility (5-minute returns), which is a much less noisy target than daily squared returns.
Option-implied volatility (Deribit for BTC, India VIX for NIFTY) is the market's own forecast and the natural
benchmark.

**D12. Representativeness and bias?**
Two assets, one data vendor, daily frequency. Headlines are English-biased. GDELT coverage grows over time, so
raw event counts trend upward for reasons unrelated to stress; expanding z-scores only partly remove that. I
wouldn't apply any of this to other assets or to intraday horizons without re-validating.

### 3b. Modelling (12)

**M1. Why HAR?**
It's the standard volatility benchmark (Corsi, 2009). It captures volatility clustering at three horizons with
four parameters, and on my data it beat everything else out of sample. With a low signal-to-noise target, the
simple model wins.

**M2. How does HAR work?**
It's an ordinary least-squares regression of future log volatility on past log volatility measured over three
windows (5, 22, 60 days). Each window stands for traders with different horizons.
- The coefficients sum to 0.75 for BTC, below 1, so forecasts mean-revert toward the long-run level.
- The 22-day term has the biggest weight (0.35 for BTC, 0.45 for NIFTY): the monthly component matters most.

**M3. How does a Random Forest work, and why did it lose?**
It averages many decision trees. Each tree is fit on a bootstrap resample, and at each split it considers only a
random subset of features. A tree splits to reduce squared error, and a leaf predicts the mean target of its rows.
Averaging decorrelated trees reduces variance.

It lost because the signal is weak and mostly linear in logs. With depth 8 it memorised noise: training R² was
0.51 against −0.03 on test for BTC. Trees also can't extrapolate beyond the training range, which hurts in
unprecedented spikes.

**M4. Why not XGBoost?**
Same issue as the RF: more capacity than this signal supports. I'd expect a heavily regularised boosted model to
roughly match HAR, not beat it. If I had richer features (implied volatility, intraday data, news), boosting
would be the next thing to try, with the same purged walk-forward evaluation.

**M5. Why not logistic regression or a neural net?**
Logistic regression would need a classification framing ("high-volatility week yes/no"), which discards
information. A neural net needs far more data than 4,000 highly autocorrelated daily rows; there are only about
800 non-overlapping 5-day windows. The original CNN attempt had 54 rows and crashed.

**M6. Why not GARCH?**
GARCH is the classic alternative, and I should have run it as a baseline. HAR is a close cousin and often
performs similarly for multi-day horizons. GARCH would also give a full conditional distribution, not just a
point forecast. It's on my list.

**M7. How did you tune it?**
HAR has no hyperparameters beyond the windows. I kept the conventional 5/22 and swapped the daily term for 60
days, and I didn't search over windows, to avoid overfitting the validation folds. The RF hyperparameters were
never tuned either. That's fair to say, because the RF overfit even at these settings, so tuning wouldn't change
the conclusion.

**M8. What's your validation scheme?**
Expanding-window walk-forward with 5 folds, using `TimeSeriesSplit(gap=5)`. Every model is retrained on all data
before each test block and evaluated on the next block, with a 5-day purge so no overlapping target crosses the
boundary. The final model is refit on all data.

**M9. What's your baseline?**
Two: the historical mean of log volatility, and a persistence forecast (next week's volatility = this week's).
Persistence is surprisingly bad (R² −0.37 BTC) because one week of returns is a very noisy volatility estimate.

**M10. Is it overfitting?**
HAR has a train/test gap (0.28 → 0.05 for BTC), but mostly because the test folds include regime shifts. Its
per-fold test R² ranges from −0.06 to 0.16. The RF overfits clearly. Its learning curve stays flat: the gap
doesn't close between 250 and 3,300 training rows, so this is a signal problem, not a data-size problem.

**M11. What happened to the sentiment Random Forest?**
It was trained on about 10 rows with `min_samples_leaf=10`, and a split needs at least 20, so every tree was a
single leaf. The pickle predicted a constant: 2.4% daily volatility, whatever the input. Every importance was
exactly zero, which the original notebook output even shows. I deleted it, and the training code now refuses to
train below that threshold instead of silently producing a constant.

**M12. How is FinBERT used, and what was the bug?**
FinBERT is BERT fine-tuned on financial text. It outputs probabilities for positive, negative and neutral. The
ProsusAI model's label order is {0: positive, 1: negative, 2: neutral}, but the code assumed the order was
negative, neutral, positive. So `probs[2] − probs[0]` was "neutral minus positive", which is why it correlated
−0.35 with VADER. The fix reads the labels by name from `model.config.id2label`, and it has a unit test.

### 3c. Metrics (6)

**K1. Why these metrics?**
- RMSE and MAE on volatility are interpretable in the target's units.
- R² on log volatility measures share of variance explained versus the test mean.
- QLIKE is the loss that ranks volatility forecasts consistently even when the "true" volatility is measured
  with noise (Patton, 2011). It's the one I'd optimise.

**K2. Which metric matters most here?**
QLIKE, plus under-prediction in turbulent weeks. Risk systems lose money when they under-estimate volatility.

**K3. Is R² = 0.05 any good?**
For next-week volatility from daily data, small R² is normal: realised volatility from five daily returns is a
very noisy measurement. The meaningful comparison is relative. HAR cuts QLIKE from 1.97 to 1.15 versus the
historical mean for BTC, and from 0.89 to 0.55 for NIFTY. I wouldn't sell it as more than "beats naive
baselines".

**K4. What does a false positive cost here?**
Over-forecasting volatility means over-hedging and smaller positions: you give up return. A false negative
(under-forecasting before a spike) means under-hedged positions into a crash, margin calls, and VaR breaches.
The second is usually more expensive, which is why QLIKE's asymmetry fits the problem.

**K5. Why not accuracy or F1?**
It's regression. If the business wanted a "high-volatility alert", I'd threshold the forecast and report
precision and recall at that threshold, plus the cost-weighted error.

**K6. You report exp(log forecast). Is that biased?**
Yes. exp of a log-scale forecast is a conditional *median*, which is below the conditional mean for a
right-skewed variable. For risk use I'd add a smearing correction or forecast quantiles. The model card in the
app says this.

### 3d. Engineering and deployment (9)

**E1. How is the model served?**
The model is four numbers in a JSON file, loaded once per server with `st.cache_resource`. The forecast page
computes features with the *same* function used in training (`pipeline/volatility.price_features`) and returns
daily and annualised volatility. There's no pickle, so there's no scikit-learn version to match. The JSON also
stores the feature contract (windows, horizon), and loading fails if the code's contract differs.

**E2. Latency?**
A forecast is a few rolling standard deviations on at most a few thousand numbers: milliseconds. Page renders
took 0.6–4 s in a real browser, mostly Streamlit and Plotly overhead. The old landing page took 185 s because of
620 quadratic Plotly `add_vrect` calls. I measured that before fixing it.

**E3. What happens on bad input?**
`parse_closes` and `validate_closes` reject:
- non-numbers, with the position of the bad value
- fewer than 61 prices
- non-positive, NaN or infinite values
- more than 10,000 prices

Each is raised as an `InputError` with a plain-English message and shown in the UI. Suspicious jumps (> 50% in a
day) produce a warning. These paths are covered by AppTest and the browser test.

**E4. How would you scale it?**
It's stateless and cheap, so the bottleneck is Streamlit itself. For an API I'd wrap `forecast()` in FastAPI, keep
the JSON model in memory, and scale horizontally. Batch scoring for many assets is a vectorised rolling-window
computation.

**E5. How do you monitor drift?**
Track realised versus forecast volatility as each 5-day window closes:
- rolling QLIKE and bias by volatility quintile
- the coefficient stability of refits
- input distribution checks (e.g. σ₂₂ percentile versus training)

Alert if rolling QLIKE exceeds the walk-forward band for several weeks.

**E6. How do you retrain?**
`python run_pipeline.py` rebuilds every model and result from `data/raw` in about a minute. CI rebuilds the
models and fails if they differ from what's committed (`git diff --exit-code models/`). So the published model
is always reproducible from the code.

**E7. How is it tested?**
- 29 unit tests: no-look-ahead checks, target definition, HAR coefficient recovery, JSON contract, input
  validation, and the false-positive rate of both permutation tests.
- 10 Streamlit AppTest tests covering every page and the error paths.
- A Playwright browser smoke test.
- GitHub Actions on Python 3.12 (research) and 3.11/3.13 (dashboard).

**E8. Dependency management?**
Exact pins, tested together. The dashboard has 4 packages, tested on Python 3.11–3.14. The research environment
is separate (`requirements-research.txt`, Python 3.12+). I deliberately avoided a root `requirements.txt`, so
Streamlit Cloud can't install the wrong file.

**E9. Secrets?**
None are needed. The old repo had a real FRED key in `.env.example`. I replaced it, and the key must be revoked
because it remains in git history. New keys go in `st.secrets` or a git-ignored `.env`.

### 3e. Critical and stress questions (9)

**C1. What's the weakest part of this project?**
The news side. Sentiment covers three months, and the GMSI result doesn't survive a correct significance test.
The forecasting model that works uses only prices, so the "sentiment" in the project's origin isn't what
delivers the result.

**C2. What would you do differently?**
- Start with the baseline (HAR) and the evaluation harness before any complex model.
- Write the significance test with autocorrelation in mind from day one.
- Never let an analysis script fall back to synthetic data silently.
- Get a multi-year text source before designing sentiment features.

**C3. What would break in production?**
- Yahoo data changes or rate limits; I'd use a licensed feed.
- Regime shifts that the 10-year history doesn't contain.
- Users pasting prices from another asset or with gaps. That's handled by validation, but not by asset checks.
- The model under-predicts exactly when it matters most (turbulent weeks), so it must not be used alone for risk
  limits.

**C4. What did you NOT do, and why?**
- No GARCH or implied-volatility benchmark, because of time.
- No intraday realised volatility, because there's no free data.
- No hyperparameter search, deliberately, to avoid overfitting a weak signal.
- No re-run of FinBERT after the fix, because the model download wasn't available in the audit environment.
- No re-run of the GMSI pipeline, because the raw GDELT event files are several GB and not in the repo.

**C5. Some of your earlier dashboard results came from simulated data. How did that happen?**
The market-dynamics script had a fallback: if its input files were missing, it simulated GARCH data "for
demo purposes". The files it expected were never produced by any code, so it always simulated. The figures got
copied into the dashboard and described as real.

I caught it during an audit:
- I re-ran the script in an empty directory and reproduced the dashboard numbers to three decimals.
- The "BTC" chart showed a cumulative return of 0.25 when real BTC did about 4.1 log points.

I fixed the root cause (fail loudly; simulate only behind an explicit `--demo` flag with a watermark) and
recomputed everything on real prices. The real data doesn't show the "t+3 secondary wave" I had described.

**C6. Isn't a "complacency effect" still interesting?**
It's an interesting hypothesis. The data shows that the lowest-stress quintile preceded the highest forward
volatility, but with a valid test the evidence is borderline for BTC (p ≈ 0.05) and absent for NIFTY. I'd
pre-register it and test it out of sample on 2026+ data before calling it a finding.

**C7. Why should we trust your numbers now?**
- Every number on the dashboard is read from files produced by one command.
- CI rebuilds the models and fails on drift.
- The tests encode the properties I care about: no look-ahead, and valid false-positive rates.
- The audit scripts in `analysis/` reproduce the original problems from the original commit, so the corrections
  are themselves checkable.

**C8. How would you extend it?**
- Add implied volatility (India VIX, Deribit DVOL) as features and as benchmarks.
- Move to intraday realised volatility.
- Use a multi-year news archive and evaluate sentiment as an *incremental* signal on top of HAR, with the same
  purged walk-forward test.
- Produce quantile forecasts for risk use.

**C9. If you had one more week?**
GARCH and India VIX as benchmarks, plus a proper "does news add to HAR?" test on whatever multi-year text I
could license. That's the experiment the project was originally meant to answer.

---

## 4. Do not bluff: honest answers to weak spots

| Topic | Honest line |
|---|---|
| Sentiment value | "Untested. I only had three months of headlines, about 13 usable rows. I'd need years of text." |
| FinBERT accuracy | "I never evaluated FinBERT on labelled data. An earlier dashboard claimed 82.7%; no code produced that, so I removed it." |
| GMSI significance | "Borderline for BTC, not significant for NIFTY once you account for autocorrelation. My original test was wrong." |
| GMSI reproducibility | "The GMSI is built from several GB of GDELT event files that aren't in the repo; the charts are from the original run." |
| Model strength | "It beats naive baselines, but R² is about 0.05–0.10. It's a baseline-quality forecaster, not a trading signal." |
| GARCH / implied vol | "I didn't benchmark against them. That's the first thing I'd add." |
| MFI | "It's a descriptive index with equal weights by assumption. I haven't validated it against VIX; the earlier claim that I had was wrong." |
| Causality | "Everything here is association. Nothing establishes that news causes volatility." |
| Synthetic figures | Own it: "An analysis script silently fell back to simulated data. I found it, proved it, fixed the root cause and recomputed." |
| AI-written reports | "Some early write-ups in `reports/` were drafted with an AI assistant. I've annotated where they were wrong, and I can defend the code and numbers myself." |

---

## 5. Glossary

**Data and returns**
- **Log return**: ln(Pₜ/Pₜ₋₁). Additive over time; close to a % change for small moves.
- **Simple return**: Pₜ/Pₜ₋₁ − 1 (`pct_change`). It is what the raw CSVs store.
- **Realised volatility (σₙ)**: sample std of the last n daily log returns.
- **Annualisation**: σ_daily × √(periods per year): √365 for BTC, √252 trading days for NIFTY.
- **Fat tails / excess kurtosis**: more extreme moves than a normal distribution (normal = 0; BTC ≈ 8, NIFTY ≈ 12).
- **Skewness**: asymmetry. NIFTY returns are negatively skewed: crashes are bigger than rallies.
- **Volatility clustering**: large moves follow large moves. This is why past volatility predicts future volatility.
- **Mean reversion**: volatility drifts back to its long-run level. HAR coefficients summing to less than 1 encode it.
- **Jarque–Bera test**: a normality test based on skewness and kurtosis.

**Model and target**
- **Target (here)**: log of the std of the next 5 log returns.
- **Conditional median**: exp(E[log σ]). What a log-scale model returns after exponentiating; below the mean.
- **HAR (Heterogeneous AutoRegressive)**: regression of future volatility on past volatility over several windows (Corsi, 2009).
- **OLS**: ordinary least squares; minimises squared residuals. Here via `numpy.linalg.lstsq`.
- **Random Forest**: a bagged ensemble of decision trees, each with random feature subsets (`max_features="sqrt"`).
- **Bagging / bootstrap**: training on resamples drawn with replacement; averaging reduces variance.
- **min_samples_leaf**: the minimum rows per leaf. A split needs at least 2× this many rows.
- **max_depth**: the maximum tree depth.
- **Impurity importance**: the total squared-error reduction from splits on a feature (computed on training data; biased).
- **Permutation importance**: the increase in held-out error when a feature is shuffled.
- **DummyRegressor**: predicts the training mean. The trivial baseline.
- **Persistence forecast**: tomorrow equals today (here: next-week σ = last-week σ).
- **GARCH(1,1)**: conditional variance σ²ₜ = ω + α·ε²ₜ₋₁ + β·σ²ₜ₋₁. The old code used it to *simulate* data.
- **CNN, Huber loss, early stopping**: the abandoned NIFTY neural model. Huber is quadratic near 0 and linear in
  the tails; early stopping halts training when validation loss stops improving.
- **KMeans**: clustering into k groups by distance. Used descriptively in the old notebooks.

**Validation and leakage**
- **Walk-forward (expanding window)**: train on everything before a block, test on the block, repeat.
- **TimeSeriesSplit(gap)**: scikit-learn's walk-forward splitter; `gap` drops rows between train and test.
- **Purging / embargo**: removing training rows whose labels overlap the test period.
- **Leakage / look-ahead bias**: using information not available at prediction time.
- **Expanding vs rolling window**: all history up to t, versus the last n days.
- **StandardScaler / z-score**: (x − mean)/std. Leaks when fitted on the full sample.
- **Expanding min-max**: (x − min_past)/(max_past − min_past), mapping to [0, 1] with past-only extremes.

**Metrics**
- **RMSE / MAE**: root mean squared / mean absolute error, in volatility units.
- **R² (log σ)**: 1 − SSE/SST on the log scale. Negative means worse than predicting the test mean.
- **QLIKE**: mean(σ²/σ̂² − ln(σ²/σ̂²) − 1). A robust volatility-forecast loss that penalises under-prediction more.

**Statistics**
- **Pearson / Spearman correlation**: linear correlation, and correlation of ranks (robust to outliers and monotone transforms).
- **Autocorrelation (ACF), lag-1 AC1**: correlation of a series with its own past.
- **AR(1)**: xₜ = φxₜ₋₁ + εₜ. The simplest persistent process; φ is its lag-1 autocorrelation.
- **Effective sample size / variance inflation**: autocorrelation makes n observations worth fewer independent
  ones, so null distributions widen (about 6.8× variance here).
- **Permutation (placebo) test**: compare the real statistic with its distribution under reshuffled data.
- **Exchangeability**: the requirement that all reorderings are equally likely under H₀. It fails for time series under i.i.d. shuffling.
- **Circular-shift test**: rotate one series against the other. It keeps both autocorrelations and is valid here.
- **p-value**: P(a statistic at least this extreme | H₀).
- **KS test / Mann–Whitney U / Kruskal–Wallis**: distribution-equality, rank-shift and multi-group rank tests. All assume independent observations.
- **Granger causality**: does the past of X improve the prediction of Y beyond Y's own past?
- **ADF test**: augmented Dickey–Fuller unit-root (stationarity) test.
- **Quantile regression**: models conditional quantiles instead of the mean.
- **VIF**: variance inflation factor, a multicollinearity measure (1/(1−R²) of a feature on the others).
- **Mutual information**: a dependence measure that captures non-linear relationships.
- **Conditional expectation by quintile**: mean target within each fifth of a signal's distribution.
- **Wasserstein distance**: "earth-mover" distance between distributions. Mentioned on the old dashboard, never computed.

**Project-specific indices**
- **GMSI (Global Market Stress Index)**: equal-weight mean of expanding z-scores of GDELT event count, negative-event share, −Goldstein, −sentiment, −sentiment surprise and attention.
- **VSI (Volatility-weighted Sentiment Index)**: the earlier index. It multiplied sentiment by volatility, so it was mechanically correlated with volatility.
- **MFI (Market Fragility Index)**: the mean of normalised 30-day AC1(|r|) (persistence), CoV of 7-day volatility (vol-of-vol) and P(|r| > 2σ₃₀) (tail frequency).
- **Vol-of-vol / CoV**: std/mean of rolling volatility.
- **Shock**: a day with |r| above the expanding 95th percentile of past |r| (250-day burn-in).
- **Regime**: low/medium/high stress by expanding 20%/80% GMSI quantiles.
- **Within-spell AC1**: autocorrelation using only consecutive days inside the same regime.
- **Sentiment surprise**: today's sentiment minus its trailing mean.

**Text and NLP**
- **GDELT**: Global Database of Events, Language and Tone. **Events**: coded actor-action records with a Goldstein score. **GKG**: themes, tone and entities per article. **DOC API**: an article search endpoint.
- **Goldstein scale**: −10 to +10 rating of an event's effect on stability.
- **FinBERT (ProsusAI)**: BERT fine-tuned for financial sentiment. Labels {0: positive, 1: negative, 2: neutral}.
- **id2label**: the model config's mapping from output index to label name.
- **Softmax**: turns logits into probabilities.
- **VADER**: rule- and lexicon-based sentiment; outputs a compound score in [−1, 1].
- **Google Trends**: search interest 0–100, scaled to the peak of the queried window.

**Engineering**
- **st.cache_data / st.cache_resource**: Streamlit caches for serialisable data, and for shared objects like models.
- **AppTest**: Streamlit's headless test harness.
- **Playwright**: browser automation, used for the end-to-end smoke test.
- **session_state / rerun**: Streamlit's per-user state; the script re-executes top to bottom on each interaction.
- **Plotly add_vrect**: shaded vertical band. Calling it per segment with row/col was the 185-second bottleneck.
- **joblib / pickle**: Python object serialisation; version-coupled to the library that wrote it.
- **JSON model**: the HAR model stored as numbers plus a feature contract; library-independent.
- **CI drift check**: CI rebuilds the models and fails if they differ from the committed ones.
