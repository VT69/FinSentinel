import numpy as np
import pytest

pytest.importorskip("scipy")
from pipeline import stats  # noqa: E402


def _fpr(method, phi=0.95, n=600, sims=80):
    rng = np.random.default_rng(123)
    hits = 0
    for i in range(sims):
        x, y = stats.ar1(n, phi, rng), stats.ar1(n, phi, rng)    # independent by construction
        hits += stats.permutation_test(x, y, n_perm=200, method=method, seed=i).p_value < 0.05
    return hits / sims


def test_iid_shuffle_is_anticonservative_but_circular_shift_is_not():
    iid, circ = _fpr("iid"), _fpr("circular")
    assert iid > 0.3            # the old test "finds" relationships between independent series
    assert circ < 0.15          # nominal 5%, allowing Monte-Carlo error


def test_detects_a_real_relationship():
    rng = np.random.default_rng(0)
    x = stats.ar1(800, 0.5, rng)
    y = x + rng.normal(0, 1, 800)
    res = stats.permutation_test(x, y, n_perm=300)
    assert res.statistic > 0.4 and res.p_value < 0.01


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr
    rng = np.random.default_rng(2)
    x, y = rng.normal(size=300), rng.normal(size=300)
    assert stats.spearman(x, y) == pytest.approx(spearmanr(x, y).statistic, abs=1e-12)


def test_ar1_null_inflates_variance_for_persistent_series():
    rng = np.random.default_rng(3)
    y = stats.ar1(1500, 0.95, rng)
    out = stats.ar1_null(y, statistic=0.05, phi=0.9, n_sims=200)
    assert out["variance_inflation"] > 3
    assert out["p_value"] > out["p_value_if_iid"]


def test_input_validation():
    with pytest.raises(ValueError):
        stats.permutation_test([1, 2, 3], [1, 2, 3])
    with pytest.raises(ValueError):
        stats.permutation_test(np.arange(100.0), np.arange(100.0), method="bogus")
