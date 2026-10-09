"""
run_pipeline.py — single entry point that turns the committed raw data into
every model and result the dashboard shows.

    python run_pipeline.py                     # volatility models + market dynamics + GMSI calibration
    python run_pipeline.py --fast              # fewer Monte-Carlo draws (for CI / quick checks)
    python run_pipeline.py sentiment-rf        # archived sentiment RF (needs data/processed/, not committed)

Outputs
  models/har_{btc,nifty}.json            served 5-day volatility models (plain JSON)
  dashboard/data/metrics.json            walk-forward metrics for all candidate models
  dashboard/data/oos_{asset}.csv         out-of-sample forecasts vs realised vol
  dashboard/data/mfi_{asset}.csv         Market Fragility Index on real prices
  dashboard/data/shocks_{asset}.csv      shock propagation on real prices
  dashboard/data/gmsi_calibration.json   corrected significance for the published GMSI correlations
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd

from pipeline import market_dynamics as md
from pipeline import stats
from pipeline import volatility as vol
from pipeline.data import ASSETS, DASHBOARD_DATA, MODELS, ROOT, load_close

N_SPLITS = 5
SEED = 42
# Published by scripts/validate_gmsi.py (reports/exogenous_vsi_v1_results): Spearman(GMSI_t, fwd 7d vol).
# The GMSI series itself is not in the repo, so these two numbers are inputs, not recomputed.
PUBLISHED_GMSI_SPEARMAN = {"BTC": -0.0837, "NIFTY": -0.0580}
GMSI_LAG1_ACF = 0.82   # read from reports/figures/gmsi_sanity_checks.png (ACF panel); see docs/MODEL_REPORT.md §E


def vol_metrics(y_log, p_log):
    a, p = np.exp(y_log), np.exp(p_log)
    ratio = a ** 2 / p ** 2
    ss_res, ss_tot = np.sum((y_log - p_log) ** 2), np.sum((y_log - y_log.mean()) ** 2)
    return {"rmse_vol": float(np.sqrt(np.mean((a - p) ** 2))), "mae_vol": float(np.mean(np.abs(a - p))),
            "r2_log": float(1 - ss_res / ss_tot), "qlike": float(np.mean(ratio - np.log(ratio) - 1))}


def walk_forward(frame: pd.DataFrame, asset: str):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import TimeSeriesSplit

    from pipeline.train_rf_model import RF_PARAMS

    y = frame[vol.TARGET].to_numpy()
    folds, oos = [], []
    # gap = horizon: drop training rows whose 5-day target window overlaps the test fold (purging)
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS, gap=vol.HORIZON).split(frame)):
        ftr, fte = frame.iloc[tr], frame.iloc[te]
        har = vol.HARModel.fit(ftr, asset)
        rf = RandomForestRegressor(**RF_PARAMS).fit(ftr[vol.RF_FEATURES], y[tr])
        preds = {"HAR (served)": (har.predict_log(fte), har.predict_log(ftr)),
                 "Random Forest": (rf.predict(fte[vol.RF_FEATURES]), rf.predict(ftr[vol.RF_FEATURES])),
                 "Persistence (vol_5)": (fte["log_vol_5"].to_numpy(), ftr["log_vol_5"].to_numpy()),
                 "Mean (dummy)": (np.full(len(te), y[tr].mean()), np.full(len(tr), y[tr].mean()))}
        for name, (p_te, p_tr) in preds.items():
            folds.append({"fold": k, "model": name, "test_start": str(fte.index[0])[:10],
                          "test_end": str(fte.index[-1])[:10], "n_train": len(tr), "n_test": len(te),
                          **{f"test_{m}": v for m, v in vol_metrics(y[te], p_te).items()},
                          **{f"train_{m}": v for m, v in vol_metrics(y[tr], p_tr).items()}})
        oos.append(pd.DataFrame({"date": fte.index, "fold": k, "actual_vol": np.exp(y[te]) - vol.EPS,
                                 "har_vol": np.exp(preds["HAR (served)"][0]) - vol.EPS,
                                 "rf_vol": np.exp(preds["Random Forest"][0]) - vol.EPS,
                                 "mean_vol": np.exp(preds["Mean (dummy)"][0]) - vol.EPS}))
    folds = pd.DataFrame(folds)
    summary = (folds.groupby("model")[[c for c in folds if c.startswith(("test_", "train_"))
                                       and c not in ("test_start", "test_end")]].mean()
               .sort_values("test_qlike"))
    return folds, summary, pd.concat(oos, ignore_index=True)


def run_volatility(fast: bool):
    DASHBOARD_DATA.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(exist_ok=True)
    metrics = {}
    for asset in ASSETS:
        t0 = time.time()
        close = load_close(asset)
        frame = vol.training_frame(close)
        folds, summary, oos = walk_forward(frame, asset)
        model = vol.HARModel.fit(frame, asset)
        model.metrics = {k: float(v) for k, v in summary.loc["HAR (served)"].items()}
        model.to_json(MODELS / f"har_{asset.lower()}.json")
        oos.to_csv(DASHBOARD_DATA / f"oos_{asset.lower()}.csv", index=False, float_format="%.6g")
        metrics[asset] = {"summary": json.loads(summary.round(6).to_json(orient="index")),
                          "folds": json.loads(folds.round(6).to_json(orient="records")),
                          "n_rows": len(frame), "date_range": [str(frame.index[0])[:10], str(frame.index[-1])[:10]]}
        print(f"\n{asset}: {len(frame)} rows, walk-forward ({time.time() - t0:.1f}s)")
        print(summary[["test_rmse_vol", "test_mae_vol", "test_r2_log", "test_qlike", "train_r2_log"]].round(5)
              .to_string())

        # Market dynamics on REAL prices
        r = np.log(close / close.shift(1))
        mfi = md.compute_mfi(r)
        mfi[["MFI", "persistence_norm", "vol_of_vol_norm", "tail_freq_norm"]].dropna().to_csv(
            DASHBOARD_DATA / f"mfi_{asset.lower()}.csv", float_format="%.5g", index_label="date")
        shocks, _ = md.identify_shocks(r)
        sp = md.shock_propagation(r, shocks)
        sp.to_csv(DASHBOARD_DATA / f"shocks_{asset.lower()}.csv", index=False, float_format="%.6g")
        peak = mfi["MFI"].idxmax()
        metrics[asset]["market_dynamics"] = {
            "n_shocks": int(shocks.sum()), "mfi_max": float(mfi["MFI"].max()), "mfi_max_date": str(peak)[:10],
            "mfi_2020_03_max": float(mfi["MFI"].loc["2020-03"].max()),
            "mfi_mean_since_2016": float(mfi["MFI"].loc["2016":].mean())}
        print(sp.round(5).to_string(index=False))
    (DASHBOARD_DATA / "metrics.json").write_text(json.dumps(metrics, indent=2))


def run_gmsi_calibration(fast: bool):
    """Corrected significance for the published GMSI→forward-vol correlations."""
    n_sims = 300 if fast else 2000
    out = {"published_spearman": PUBLISHED_GMSI_SPEARMAN, "assumed_gmsi_lag1_acf": GMSI_LAG1_ACF, "assets": {},
           "false_positive_rate": []}
    fwd = {}
    for asset in ASSETS:
        close = load_close(asset)
        r = np.log(close / close.shift(1))
        # forward 7-day vol exactly as scripts/construct_vsi_full.py + scripts/validate_gmsi.py define it
        fwd[asset] = r.rolling(7, min_periods=1).std().shift(-7).dropna().to_numpy()
        out["assets"][asset] = stats.ar1_null(fwd[asset], PUBLISHED_GMSI_SPEARMAN[asset], GMSI_LAG1_ACF,
                                              n_sims=n_sims, seed=SEED)
        out["assets"][asset]["fwd_vol_lag1_acf"] = float(pd.Series(fwd[asset]).autocorr(1))
        print(f"{asset}: {out['assets'][asset]}")
    # How often does each placebo method flag an INDEPENDENT persistent series as significant?
    rng = np.random.default_rng(SEED)
    y = fwd["BTC"]
    sims = 40 if fast else 200
    for phi in (0.0, 0.5, 0.9):
        hits = {"iid": 0, "circular": 0}
        for i in range(sims):
            x = stats.ar1(len(y), phi, rng)
            for m in hits:
                hits[m] += stats.permutation_test(x, y, n_perm=500, method=m, seed=i).p_value < 0.05
        out["false_positive_rate"].append({"phi": phi, "n_sims": sims,
                                           "iid_shuffle": hits["iid"] / sims, "circular_shift": hits["circular"] / sims})
        print(out["false_positive_rate"][-1])
    (DASHBOARD_DATA / "gmsi_calibration.json").write_text(json.dumps(out, indent=2))


def run_sentiment_rf():
    """Archived experiment; kept runnable so its failure mode is explicit, not silent."""
    from pipeline.evaluate import feature_importance, plot_predictions
    from pipeline.feature_engineering import build_features
    from pipeline.train_rf_model import train_random_forest

    path = ROOT / "data" / "processed" / "btc_sentiment_aligned.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. It is produced by notebooks/02-04 (needs FinBERT) and is not committed.")
    X, y, features = build_features(pd.read_csv(path, parse_dates=["date"]))
    n_test = int(np.ceil(0.2 * len(X)))
    gap = 5   # purge: last training targets would otherwise overlap the test window
    X_train, y_train = X.iloc[: len(X) - n_test - gap], y.iloc[: len(X) - n_test - gap]
    X_test, y_test = X.iloc[-n_test:], y.iloc[-n_test:]
    model, *_ = train_random_forest(X_train, X_test, y_train, y_test, asset="BTC")
    feature_importance(model, features)
    plot_predictions(y_test, model.predict(X_test), asset="BTC")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", nargs="?", default="all", choices=["all", "volatility", "gmsi", "sentiment-rf"])
    ap.add_argument("--fast", action="store_true", help="fewer Monte-Carlo draws")
    a = ap.parse_args()
    if a.command == "sentiment-rf":
        return run_sentiment_rf()
    if a.command in ("all", "volatility"):
        run_volatility(a.fast)
    if a.command in ("all", "gmsi"):
        run_gmsi_calibration(a.fast)


if __name__ == "__main__":
    main()
