"""
pipeline/stats.py — significance tests that remain valid for autocorrelated
time series.

Why this exists: the original placebo test (scripts/validate_gmsi.py, before
the fix) shuffled GMSI i.i.d. A permutation test is only valid if every
reordering is equally likely under H0 (exchangeability). Shuffling a persistent
series destroys its autocorrelation, so the null distribution is far too narrow
and "significant" results appear for unrelated series (57.5% false-positive
rate at AR(1) φ=0.9; docs/MODEL_REPORT.md §D).

`circular_shift_test` rotates x relative to y instead: each rotation keeps both
series' autocorrelation intact while breaking their alignment.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import signal, stats


def _standardised_ranks(a) -> np.ndarray:
    r = stats.rankdata(np.asarray(a, dtype="float64"))
    return (r - r.mean()) / r.std()


def spearman(x, y) -> float:
    return float(np.mean(_standardised_ranks(x) * _standardised_ranks(y)))


@dataclass
class PermutationResult:
    statistic: float
    p_value: float          # two-sided: P(|null| >= |statistic|)
    null_sd: float
    n: int
    n_perm: int
    method: str


def permutation_test(x, y, n_perm: int = 1000, method: str = "circular", min_shift: int = 30,
                     seed: int = 42) -> PermutationResult:
    """Spearman correlation with a permutation null.

    method="circular": rotate x by a random offset in [min_shift, n-min_shift) — valid under autocorrelation.
    method="iid":      shuffle x i.i.d. — kept only to show how the old test misbehaves.
    """
    x = np.asarray(x, dtype="float64"); y = np.asarray(y, dtype="float64")
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("x and y must be 1-D arrays of equal length")
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    n = len(x)
    if n < 2 * min_shift + 10:
        raise ValueError(f"need at least {2 * min_shift + 10} paired observations, got {n}")
    rx, ry = _standardised_ranks(x), _standardised_ranks(y)
    real = float(np.mean(rx * ry))
    rng = np.random.default_rng(seed)
    if method == "circular":
        shifts = rng.integers(min_shift, n - min_shift, n_perm)
        null = np.array([np.mean(np.roll(rx, s) * ry) for s in shifts])
    elif method == "iid":
        null = np.array([np.mean(rng.permutation(rx) * ry) for _ in range(n_perm)])
    else:
        raise ValueError("method must be 'circular' or 'iid'")
    p = float(np.mean(np.abs(null) >= abs(real)))
    return PermutationResult(real, p, float(null.std()), n, n_perm, method)


def ar1(n: int, phi: float, rng: np.random.Generator) -> np.ndarray:
    """Stationary-start-agnostic AR(1) path x_t = φ x_{t-1} + ε_t (vectorised via lfilter)."""
    return signal.lfilter([1.0], [1.0, -phi], rng.standard_normal(n))


def ar1_null(y, statistic: float, phi: float, n_sims: int = 2000, seed: int = 42) -> dict:
    """How often does an INDEPENDENT AR(1)(φ) series reach |Spearman| >= |statistic| with y?

    A Monte-Carlo p-value for a correlation between y and a persistent regressor whose
    lag-1 autocorrelation is about φ. If the real regressor's ACF decays more slowly than φ^k
    the true null is wider still, so this p-value is a lower bound.
    """
    y = np.asarray(y, dtype="float64"); y = y[np.isfinite(y)]
    ry = _standardised_ranks(y)
    rng = np.random.default_rng(seed)
    null = np.array([np.mean(_standardised_ranks(ar1(len(y), phi, rng)) * ry) for _ in range(n_sims)])
    iid_sd = 1 / np.sqrt(len(y) - 1)
    return {"n": len(y), "statistic": statistic, "phi": phi, "n_sims": n_sims,
            "null_sd": float(null.std()), "iid_null_sd": float(iid_sd),
            "variance_inflation": float((null.std() / iid_sd) ** 2),
            "p_value": float(np.mean(np.abs(null) >= abs(statistic))),
            "p_value_if_iid": float(2 * stats.norm.sf(abs(statistic) / iid_sd))}
