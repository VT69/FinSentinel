"""
pipeline/market_dynamics.py — Market Fragility Index (MFI), shock propagation
and regime helpers, computed on REAL prices only.

Ported from scripts/08_market_dynamics_analysis.py with these corrections
(docs/ISSUES.md):
  * no synthetic fallback — callers pass real data or get an error (P0-2)
  * shocks need a 250-day burn-in before the expanding 95% threshold is trusted
    (the original threshold on day 2 was computed from 1 observation)
  * shock curves are reported against the unconditional mean |r| with 95% CIs,
    instead of mean ± 0.5 sd and a 5-point "half-life" fit (P1-16)
  * regime autocorrelation uses only consecutive days inside one regime spell
    (the original paired the last day of one spell with the first day of the
    next, weeks apart — P1-10)
  * regime thresholds use expanding (past-only) quantiles (P1-3)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

MFI_WINDOW = 30
SHOCK_QUANTILE = 0.95
SHOCK_BURN_IN = 250
HORIZONS = (1, 3, 7, 14, 21)


def _expanding_minmax(s: pd.Series) -> pd.Series:
    lo, hi = s.expanding().min(), s.expanding().max()
    return (s - lo) / (hi - lo).replace(0, np.nan)


def compute_mfi(log_return: pd.Series, window: int = MFI_WINDOW) -> pd.DataFrame:
    """MFI = mean of three expanding-min-max-normalised rolling components:

    A persistence  lag-1 autocorrelation of |r| within the last `window` days (window-1 pairs)
    B vol-of-vol   std/mean of the 7-day rolling vol over `window` days
    C tail freq.   share of the last `window` days with |r| > 2 × trailing 30-day std
    Everything uses data up to and including day t only.
    """
    r = log_return.astype("float64")
    a = r.abs()
    out = pd.DataFrame(index=r.index)
    persistence = a.rolling(window - 1).corr(a.shift(1))
    full = a.rolling(window).count() == window
    out["persistence"] = persistence.where(~full | persistence.notna(), 0.0)   # constant window → 0, as before
    vol7 = r.rolling(7).std()
    out["vol_of_vol"] = vol7.rolling(window).std() / vol7.rolling(window).mean().replace(0, np.nan)
    out["tail_freq"] = (a > 2 * r.rolling(30).std()).astype(float).where(r.rolling(30).std().notna()).rolling(window).mean()
    for c in ["persistence", "vol_of_vol", "tail_freq"]:
        out[f"{c}_norm"] = _expanding_minmax(out[c])
    out["MFI"] = out[["persistence_norm", "vol_of_vol_norm", "tail_freq_norm"]].mean(axis=1, skipna=False)
    return out


def identify_shocks(log_return: pd.Series, quantile: float = SHOCK_QUANTILE,
                    burn_in: int = SHOCK_BURN_IN) -> tuple[pd.Series, pd.Series]:
    """Shock on day t ⇔ |r_t| > expanding quantile of |r| over days < t (needs `burn_in` past days)."""
    a = log_return.abs()
    threshold = a.shift(1).expanding(min_periods=burn_in).quantile(quantile)
    return (a > threshold) & threshold.notna(), threshold


def shock_propagation(log_return: pd.Series, shocks: pd.Series, horizons=HORIZONS) -> pd.DataFrame:
    """Mean |r_{t+h}| after shock days, vs the unconditional mean |r| over the same eligible period."""
    a = log_return.abs().to_numpy()
    idx = np.flatnonzero(shocks.to_numpy())
    start = idx.min() if len(idx) else 0
    baseline = np.nanmean(a[start:])
    rows = []
    for h in horizons:
        fwd = a[idx[idx + h < len(a)] + h]
        m, sd, n = np.nanmean(fwd), np.nanstd(fwd, ddof=1), np.sum(np.isfinite(fwd))
        half = 1.96 * sd / np.sqrt(n)
        rows.append({"horizon": h, "n_shocks": int(n), "mean_abs_return": m, "ci_low": m - half,
                     "ci_high": m + half, "baseline_mean_abs_return": baseline, "ratio_to_baseline": m / baseline})
    return pd.DataFrame(rows)


def expanding_regimes(signal: pd.Series, low_q: float = 0.2, high_q: float = 0.8,
                      min_periods: int = 250) -> pd.Series:
    """'low' / 'medium' / 'high' using quantiles of the signal's PAST values only."""
    past = signal.shift(1).expanding(min_periods=min_periods)
    lo, hi = past.quantile(low_q), past.quantile(high_q)
    reg = pd.Series(np.where(signal <= lo, "low", np.where(signal >= hi, "high", "medium")), index=signal.index)
    return reg.where(lo.notna() & signal.notna())


def within_spell_ac1(x: pd.Series, regimes: pd.Series, regime: str) -> tuple[float, int]:
    """Lag-1 autocorrelation using only pairs (x_{t-1}, x_t) where both days are consecutive rows in `regime`."""
    same = (regimes == regime) & (regimes.shift(1) == regime)
    pairs = pd.DataFrame({"x": x, "lag": x.shift(1)})[same].dropna()
    if len(pairs) < 10:
        return float("nan"), len(pairs)
    return float(pairs["x"].corr(pairs["lag"])), len(pairs)
