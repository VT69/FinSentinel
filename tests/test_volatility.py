import numpy as np
import pandas as pd
import pytest

from pipeline import volatility as v


def _closes(n=400, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, n))),
                     index=pd.date_range("2020-01-01", periods=n, freq="D"))


def test_features_use_only_past_data():
    c = _closes()
    base = v.price_features(c)
    t = 200
    bumped = c.copy()
    bumped.iloc[t + 1:] *= 3.0                      # change only the future
    after = v.price_features(bumped)
    feat_cols = v.HAR_FEATURES + v.RF_FEATURES
    pd.testing.assert_frame_equal(base.iloc[: t + 1][feat_cols], after.iloc[: t + 1][feat_cols])
    # target at t-5 covers r[t-4..t] -> unchanged; target at t covers r[t+1..t+5] -> changed
    assert base["target_vol"].iloc[t - 5] == pytest.approx(after["target_vol"].iloc[t - 5])
    assert base["target_vol"].iloc[t] != pytest.approx(after["target_vol"].iloc[t])


def test_target_is_std_of_next_horizon_log_returns():
    c = _closes()
    f = v.price_features(c)
    r = np.log(c / c.shift(1))
    t = 100
    assert f["target_vol"].iloc[t] == pytest.approx(r.iloc[t + 1: t + 1 + v.HORIZON].std())


def test_har_fit_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    n = 2000
    X = pd.DataFrame(rng.normal(-4, 0.5, (n, 3)), columns=v.HAR_FEATURES)
    X[v.TARGET] = -0.5 + 0.2 * X.iloc[:, 0] + 0.5 * X.iloc[:, 1] + 0.1 * X.iloc[:, 2] + rng.normal(0, 0.01, n)
    X.index = pd.date_range("2000-01-01", periods=n)
    m = v.HARModel.fit(X, "btc")
    assert m.intercept == pytest.approx(-0.5, abs=0.02)
    assert [m.coef[c] for c in v.HAR_FEATURES] == pytest.approx([0.2, 0.5, 0.1], abs=0.01)
    assert m.asset == "BTC" and m.n_train == n


def test_model_json_roundtrip_and_contract_check(tmp_path):
    f = v.training_frame(_closes(800))
    m = v.HARModel.fit(f, "NIFTY")
    p = tmp_path / "m.json"
    m.to_json(p)
    m2 = v.HARModel.from_json(p)
    np.testing.assert_allclose(m.predict_log(f), m2.predict_log(f))
    bad = p.read_text().replace('"horizon": 5', '"horizon": 7')
    p.write_text(bad)
    with pytest.raises(ValueError, match="different feature contract"):
        v.HARModel.from_json(p)


@pytest.mark.parametrize("text,msg", [
    ("", "at least"),
    ("1 2 3", "at least"),
    ("abc " * 70, "not a number"),
    (" ".join(["100"] * 60 + ["-1"]), "strictly positive"),
    (" ".join(["100"] * 60 + ["nan"]), "finite"),
])
def test_parse_closes_rejects_bad_input(text, msg):
    with pytest.raises(v.InputError, match=msg):
        v.parse_closes(text)


def test_parse_closes_accepts_mixed_separators():
    vals = v.parse_closes("\n".join(f"{100 + i},{101 + i};" for i in range(31)))
    assert len(vals) == 62 and vals[0] == 100


def test_forecast_matches_model_on_training_features():
    c = _closes(600)
    f = v.training_frame(c)
    m = v.HARModel.fit(f, "BTC")
    out = v.forecast(m, c)
    expected = m.predict_vol(v.price_features(c, with_target=False).iloc[[-1]])[0]
    assert out["daily_vol"] == pytest.approx(expected)
    assert out["annualised_vol"] == pytest.approx(expected * np.sqrt(365))
    assert out["warning"] is None


def test_forecast_warns_on_suspicious_jump():
    c = _closes(100).to_numpy(copy=True)
    c[-1] = c[-2] * 3
    m = v.HARModel(asset="NIFTY", intercept=0.0, coef={k: 1 / 3 for k in v.HAR_FEATURES})
    assert "check for data errors" in v.forecast(m, c)["warning"]


def test_annualise_unknown_asset():
    with pytest.raises(ValueError):
        v.annualise(0.01, "ETH")
