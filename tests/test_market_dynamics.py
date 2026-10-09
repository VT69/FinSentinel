import numpy as np
import pandas as pd

from pipeline import market_dynamics as md


def _returns(n=800, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_t(4, n) * 0.02, index=pd.date_range("2018-01-01", periods=n))


def test_mfi_bounded_and_past_only():
    r = _returns()
    m = md.compute_mfi(r)["MFI"]
    assert m.dropna().between(0, 1).all()
    t = 500
    r2 = r.copy(); r2.iloc[t + 1:] *= 5
    pd.testing.assert_series_equal(m.iloc[: t + 1], md.compute_mfi(r2)["MFI"].iloc[: t + 1])


def test_persistence_matches_original_definition():
    r = _returns(200)
    a = r.abs()
    i, w = 150, md.MFI_WINDOW
    chunk = a.iloc[i - w + 1: i + 1].to_numpy()
    expected = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]
    assert abs(md.compute_mfi(r)["persistence"].iloc[i] - expected) < 1e-10


def test_shocks_need_burn_in_and_use_past_threshold():
    r = _returns()
    shocks, thr = md.identify_shocks(r, burn_in=250)
    assert not shocks.iloc[:251].any()
    t = 400
    assert thr.iloc[t] == r.abs().iloc[:t].quantile(0.95)


def test_shock_propagation_reports_baseline():
    r = _returns()
    shocks, _ = md.identify_shocks(r)
    sp = md.shock_propagation(r, shocks)
    assert list(sp["horizon"]) == list(md.HORIZONS)
    assert (sp["ci_low"] <= sp["mean_abs_return"]).all()
    assert sp["baseline_mean_abs_return"].nunique() == 1


def test_within_spell_ac1_ignores_pairs_across_spells():
    # x alternates high/low between spells; inside spells it is i.i.d. noise
    rng = np.random.default_rng(1)
    x = pd.Series(rng.normal(size=600))
    reg = pd.Series((["low"] * 10 + ["high"] * 10) * 30)
    ac, n = md.within_spell_ac1(x, reg, "low")
    assert n == 30 * 9            # 9 consecutive pairs per 10-day spell
    assert abs(ac) < 0.2


def test_expanding_regimes_are_past_only():
    s = pd.Series(np.random.default_rng(2).normal(size=600))
    reg = md.expanding_regimes(s, min_periods=100)
    assert reg.iloc[:100].isna().all()
    s2 = s.copy(); s2.iloc[400:] += 10
    pd.testing.assert_series_equal(reg.iloc[:400], md.expanding_regimes(s2, min_periods=100).iloc[:400])
