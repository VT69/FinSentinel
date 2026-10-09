import streamlit as st


def render():
    st.title("Methodology & limitations")
    st.markdown("""
### Data
| Source | Used for | Notes |
|---|---|---|
| Yahoo Finance via `yfinance` | daily BTC-USD (2015–2025) and ^NSEI (2010–2025) closes | committed in `data/raw/`; research use only |
| GDELT 2.0 DOC API | 1,500 headline titles, Oct 2024 – Jan 2025 | sentiment experiments only |
| GDELT 1.0 event files | daily event counts / tone / Goldstein for GMSI | not committed (several GB) |

### Volatility forecast (served model)
- **Target:** standard deviation of the next 5 daily log returns, modelled as its log.
- **Features:** trailing 5-, 22- and 60-day realised volatility (log). Computed by one function used for both training and serving.
- **Model:** HAR-style OLS (Corsi, 2009). Chosen over a Random Forest because it scored better out of sample on
  both assets while being far simpler; the forest over-fitted (train R² ≈ 0.5, test ≈ 0).
- **Validation:** 5-fold expanding-window walk-forward with a 5-day purge gap (so no training label overlaps the
  test period). Metrics: RMSE / MAE of σ, R² of log σ, QLIKE.

### Market dynamics
- **MFI:** equal-weight mean of rolling persistence, vol-of-vol and tail frequency, each scaled by expanding min-max.
- **Shocks:** |r| above the expanding 95th percentile of past days; post-shock |r| compared with the normal level.

### GMSI significance
- Spearman correlation with forward 7-day volatility; significance from a **circular-shift** permutation test
  (preserves autocorrelation) and a calibrated AR(1) null. The original i.i.d. shuffle is invalid for persistent series.

### Limitations
- Sentiment covers ~3 months — far too little to train or validate a sentiment model; the sentiment Random Forest
  was archived.
- The GMSI series is not in the repository, so GMSI charts cannot be regenerated from it without the GDELT pipeline.
- All analyses are associations in historical data; nothing here establishes causality.
- Two assets, daily frequency, one data vendor. Not suitable for trading or risk decisions.

### Corrections log
Earlier versions of this dashboard showed results generated from simulated data, a fabricated null distribution,
p-values no code produced and a constant "model". All were removed or recomputed — see `docs/ISSUES.md`.
""")
