"""
analysis/model_eval.py — re-runs and audits every model / statistical test in
the repo that can be re-run from committed data.

    python analysis/data_profile.py   # first: builds the reconstructed training table
    python analysis/model_eval.py

Sections
  A. Inspect the committed artifact models/rf_btc.pkl
  B. Re-run run_pipeline.py's training on the reconstructed table
  C. Honest walk-forward re-evaluation on the FULL price history (price features only)
  D. Is the repo's placebo test (scripts/validate_gmsi.py:77-96) valid? Null simulation
Outputs → analysis/outputs/model_eval_*.{txt,csv,png}, model_eval_summary.json
"""
from __future__ import annotations

import io
import json
import math
import sys
from contextlib import redirect_stdout
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (OUT, PIPELINE_FEATURES, PRICE_FEATURES, ROOT, SEED,  # noqa: E402
                    load_prices, price_feature_frame)

from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import LinearRegression  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.model_selection import TimeSeriesSplit  # noqa: E402

RF_PARAMS = dict(n_estimators=600, max_depth=8, min_samples_leaf=10, max_features="sqrt",
                 random_state=42, n_jobs=-1)          # pipeline/train_rf_model.py:8-15 verbatim
SUMMARY: dict = {}


def section(t):
    print("\n" + "=" * 100 + f"\n{t}\n" + "=" * 100)


def vol_metrics(y_log_true, y_log_pred):
    """Same space as pipeline/train_rf_model.py:20-24 (exp of log-vol) + log-space R² + QLIKE."""
    a, p = np.exp(y_log_true), np.exp(y_log_pred)
    a2, p2 = a ** 2, p ** 2
    return {"rmse_vol": float(np.sqrt(mean_squared_error(a, p))), "mae_vol": float(mean_absolute_error(a, p)),
            "r2_log": float(r2_score(y_log_true, y_log_pred)),
            "qlike": float(np.mean(a2 / p2 - np.log(a2 / p2) - 1))}


# ═════════════════════════════════════════════════════════════════════════════
def section_a():
    section("A. COMMITTED ARTIFACT models/rf_btc.pkl")
    import joblib
    import sklearn
    raw = (ROOT / "models" / "rf_btc.pkl").read_bytes()
    m = joblib.load(ROOT / "models" / "rf_btc.pkl")
    depths = sorted({e.get_depth() for e in m.estimators_})
    nodes = sorted({e.tree_.node_count for e in m.estimators_})
    roots = np.array([e.tree_.value[0, 0, 0] for e in m.estimators_])
    probe = pd.DataFrame(np.random.default_rng(SEED).normal(scale=1e3, size=(1000, m.n_features_in_)),
                         columns=m.feature_names_in_)
    preds = m.predict(probe)
    info = {"file_bytes": len(raw), "pickled_with_sklearn": raw.split(b"_sklearn_version")[1][3:8].decode(),
            "loaded_with_sklearn": sklearn.__version__, "class": type(m).__name__, "params": m.get_params(),
            "feature_names_in_": list(m.feature_names_in_), "n_trees": len(m.estimators_),
            "distinct_tree_depths": depths, "distinct_node_counts": nodes,
            "feature_importances_": m.feature_importances_.tolist(),
            "distinct_bootstrap_sizes_at_root": sorted({int(e.tree_.n_node_samples[0]) for e in m.estimators_}),
            "per_tree_root_value_min_max": [float(roots.min()), float(roots.max())],
            "prediction_on_1000_random_inputs_unique": sorted(set(np.round(preds, 10))),
            "exp(prediction)=forecast_vol": float(np.exp(preds[0]))}
    for k, v in info.items():
        print(f"  {k}: {v}")
    SUMMARY["A_artifact"] = info


# ═════════════════════════════════════════════════════════════════════════════
def section_b():
    section("B. RE-RUN run_pipeline.py training on the reconstructed table")
    full = pd.read_csv(OUT / "btc_training_table_reconstructed.csv", parse_dates=["date"])
    X, y = full[PIPELINE_FEATURES], full["log_vol_target"]
    n_test = math.ceil(0.2 * len(X))
    Xtr, Xte, ytr, yte = X.iloc[:-n_test], X.iloc[-n_test:], y.iloc[:-n_test], y.iloc[-n_test:]
    rf = RandomForestRegressor(**RF_PARAMS).fit(Xtr, ytr)
    dummy = DummyRegressor().fit(Xtr, ytr)
    info = {"n_train": len(Xtr), "n_test": len(Xte),
            "min_rows_needed_for_one_split (2*min_samples_leaf)": 2 * RF_PARAMS["min_samples_leaf"],
            "tree_depths": sorted({e.get_depth() for e in rf.estimators_}),
            "importances": rf.feature_importances_.tolist(),
            "rf_test_preds": rf.predict(Xte).tolist(), "dummy_test_preds": dummy.predict(Xte).tolist(),
            "rf_metrics": vol_metrics(yte.values, rf.predict(Xte)),
            "dummy_metrics": vol_metrics(yte.values, dummy.predict(Xte)),
            "note": "FinBERT columns hold VADER as a stand-in; irrelevant here because no tree can split."}
    for k, v in info.items():
        print(f"  {k}: {v}")
    SUMMARY["B_rerun_pipeline"] = info


# ═════════════════════════════════════════════════════════════════════════════
def walk_forward(asset, n_splits=5, horizon=5):
    f = price_feature_frame(load_prices(asset), horizon=horizon)
    X, y = f[PRICE_FEATURES], f["log_vol_target"]
    lin_cols = ["vol_5", "vol_22", "vol_60"]
    # gap=horizon purges training rows whose 5-day-ahead target window overlaps the test fold
    tss = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    rows, last = [], None
    for k, (tr, te) in enumerate(tss.split(X)):
        Xtr, Xte, ytr, yte = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        models = {
            "dummy_mean": DummyRegressor().fit(Xtr, ytr),
            "persistence_log_vol5": None,
            "HAR_ols_log(vol5,22,60)": LinearRegression().fit(np.log(Xtr[lin_cols] + 1e-6), ytr),
            "rf_repo_params": RandomForestRegressor(**RF_PARAMS).fit(Xtr, ytr),
        }
        for name, mdl in models.items():
            if name == "persistence_log_vol5":
                p_te, p_tr = np.log(Xte["vol_5"] + 1e-6), np.log(Xtr["vol_5"] + 1e-6)
            elif name.startswith("HAR"):
                p_te, p_tr = mdl.predict(np.log(Xte[lin_cols] + 1e-6)), mdl.predict(np.log(Xtr[lin_cols] + 1e-6))
            else:
                p_te, p_tr = mdl.predict(Xte), mdl.predict(Xtr)
            m_te, m_tr = vol_metrics(yte.values, np.asarray(p_te)), vol_metrics(ytr.values, np.asarray(p_tr))
            rows.append({"asset": asset, "fold": k, "train_end": str(f["date"].iloc[tr[-1]].date()),
                         "test": f"{f['date'].iloc[te[0]].date()}→{f['date'].iloc[te[-1]].date()}",
                         "n_train": len(tr), "n_test": len(te), "model": name,
                         **{f"test_{a}": b for a, b in m_te.items()}, **{f"train_{a}": b for a, b in m_tr.items()}})
        last = (f.iloc[te], Xte, yte, models["rf_repo_params"], models["dummy_mean"], Xtr, ytr)
    return pd.DataFrame(rows), last, f


def section_c():
    section("C. WALK-FORWARD RE-EVALUATION (price features only, full history, TimeSeriesSplit(5, gap=5))")
    allres = []
    for asset in ["BTC", "NIFTY"]:
        res, last, f = walk_forward(asset)
        allres.append(res)
        agg = res.groupby("model")[["test_rmse_vol", "test_mae_vol", "test_r2_log", "test_qlike",
                                    "train_rmse_vol", "train_r2_log"]].mean().sort_values("test_qlike")
        print(f"\n{asset} — mean over 5 folds:\n", agg.round(5).to_string())
        print(f"{asset} — per fold test R² (log):\n",
              res.pivot(index="fold", columns="model", values="test_r2_log").round(3).to_string())
        SUMMARY.setdefault("C_walk_forward", {})[asset] = json.loads(agg.round(6).to_json())

        df_te, Xte, yte, rf, dummy, Xtr, ytr = last
        # importances on the last fold
        imp = pd.Series(rf.feature_importances_, index=PRICE_FEATURES)
        pim = permutation_importance(rf, Xte, yte, n_repeats=20, random_state=SEED, scoring="neg_mean_squared_error")
        imps = pd.DataFrame({"impurity": imp, "permutation_mean": pim.importances_mean,
                             "permutation_std": pim.importances_std}).sort_values("permutation_mean", ascending=False)
        print(f"{asset} — RF importances (last fold):\n", imps.round(5).to_string())
        SUMMARY["C_walk_forward"][asset + "_importances"] = json.loads(imps.round(6).to_json())

        # error analysis by regime of realised target (last fold)
        pred = rf.predict(Xte)
        e = pd.DataFrame({"date": df_te["date"].values, "y": yte.values, "pred": pred,
                          "abs_err_vol": np.abs(np.exp(yte.values) - np.exp(pred))})
        e["actual_quintile"] = pd.qcut(e["y"], 5, labels=["Q1 calm", "Q2", "Q3", "Q4", "Q5 turbulent"])
        by_q = e.groupby("actual_quintile", observed=True).agg(n=("y", "size"), mae_vol=("abs_err_vol", "mean"),
                                                                bias_log=("pred", "mean"))
        by_q["bias_log"] = by_q["bias_log"] - e.groupby("actual_quintile", observed=True)["y"].mean()
        print(f"{asset} — RF error by quintile of ACTUAL future vol (last fold):\n", by_q.round(5).to_string())
        worst = e.nlargest(5, "abs_err_vol")[["date", "abs_err_vol"]]
        worst["actual_vol"] = np.exp(e.loc[worst.index, "y"]); worst["pred_vol"] = np.exp(e.loc[worst.index, "pred"])
        print(f"{asset} — 5 worst misses:\n", worst.round(5).to_string(index=False))
        SUMMARY["C_walk_forward"][asset + "_error_by_quintile"] = json.loads(by_q.round(6).to_json())

        # learning curve: last-fold test set, growing training window
        sizes, lc = [250, 500, 1000, 2000, len(Xtr)], []
        for s in sizes:
            m = RandomForestRegressor(**RF_PARAMS).fit(Xtr.iloc[-s:], ytr.iloc[-s:])
            lc.append({"n_train": s, "train_r2_log": r2_score(ytr.iloc[-s:], m.predict(Xtr.iloc[-s:])),
                       "test_r2_log": r2_score(yte, m.predict(Xte))})
        lc = pd.DataFrame(lc); print(f"{asset} — learning curve:\n", lc.round(4).to_string(index=False))
        SUMMARY["C_walk_forward"][asset + "_learning_curve"] = lc.round(5).to_dict(orient="records")
        fig, ax = plt.subplots(1, 3, figsize=(16, 3.6))
        ax[2].scatter(pred, yte.values - pred, s=4, alpha=.5)
        ax[2].axhline(0, color="k", lw=.8)
        ax[2].set_xlabel("RF prediction (log vol)"); ax[2].set_ylabel("residual (actual − pred, log)")
        ax[2].set_title("residuals vs prediction (last fold)")
        ax[0].plot(df_te["date"], np.exp(yte), lw=.7, label="actual 5d vol (t+1..t+5)")
        ax[0].plot(df_te["date"], np.exp(pred), lw=.9, label="RF forecast")
        ax[0].axhline(np.exp(dummy.predict(Xte)[0]), color="k", ls="--", lw=.8, label="dummy mean")
        ax[0].legend(fontsize=7); ax[0].set_title(f"{asset} last walk-forward fold")
        ax[1].plot(lc["n_train"], lc["train_r2_log"], "o-", label="train R² (log)")
        ax[1].plot(lc["n_train"], lc["test_r2_log"], "o-", label="test R² (log)")
        ax[1].set_xscale("log"); ax[1].legend(fontsize=7); ax[1].set_title("learning curve")
        plt.tight_layout(); plt.savefig(OUT / f"model_eval_{asset.lower()}.png", dpi=90); plt.close()
    pd.concat(allres).to_csv(OUT / "model_eval_walk_forward.csv", index=False)


# ═════════════════════════════════════════════════════════════════════════════
def _perm_pvals(x, y, n_perm, rng, mode):
    """Spearman via ranks; returns two-sided empirical p as in scripts/validate_gmsi.py:95."""
    rx = stats.rankdata(x); ry = stats.rankdata(y)
    rx = (rx - rx.mean()) / rx.std(); ry = (ry - ry.mean()) / ry.std()
    real = float(np.mean(rx * ry))
    n = len(rx)
    if mode == "iid_shuffle":            # what the repo does
        idx = np.argsort(rng.random((n_perm, n)), axis=1)
    else:                                # circular shift: keeps the autocorrelation of x intact
        shifts = rng.integers(30, n - 30, n_perm)
        idx = (np.arange(n)[None, :] + shifts[:, None]) % n
    null = (rx[idx] * ry[None, :]).mean(axis=1)
    return real, float(np.mean(np.abs(null) >= abs(real)))


def section_d(n_sims=200, n_perm=1000):
    section("D. PLACEBO-TEST VALIDITY — real BTC forward vol vs an INDEPENDENT AR(1) fake 'GMSI'")
    # forward 7d vol exactly as scripts/construct_vsi_full.py:26 + scripts/validate_gmsi.py:23
    p = load_prices("BTC")
    r = np.log(p["close"] / p["close"].shift(1))
    fwd = r.rolling(7, min_periods=1).std().shift(-7).dropna().values
    n = len(fwd)
    acf1 = float(pd.Series(fwd).autocorr(1))
    print(f"  BTC fwd-7d-vol series: n={n}, lag-1 autocorrelation={acf1:.3f}")
    rng = np.random.default_rng(SEED)
    rows = []
    for phi in [0.0, 0.5, 0.9, 0.98]:
        rej = {"iid_shuffle": 0, "circular_shift": 0}
        for _ in range(n_sims):
            e = rng.normal(size=n); g = np.empty(n); g[0] = e[0]
            for t in range(1, n):
                g[t] = phi * g[t - 1] + e[t]
            for mode in rej:
                _, pv = _perm_pvals(g, fwd, n_perm, rng, mode)
                rej[mode] += pv < 0.05
        rows.append({"fake_gmsi_ar1_phi": phi, "n_sims": n_sims,
                     "false_positive_rate_iid_shuffle (repo)": rej["iid_shuffle"] / n_sims,
                     "false_positive_rate_circular_shift": rej["circular_shift"] / n_sims})
        print("  ", rows[-1], flush=True)
    res = pd.DataFrame(rows); res.to_csv(OUT / "model_eval_placebo_validity.csv", index=False)
    SUMMARY["D_placebo_validity"] = {"fwd_vol_acf1": acf1, "n": n, "results": res.to_dict(orient="records")}


def section_e(n_sims=2000, phi=0.82):
    section(f"E. CALIBRATED NULL for the published GMSI correlations (AR(1) phi={phi} fake GMSI)")
    # phi=0.82 ≈ lag-1 ACF visible in reports/figures/gmsi_sanity_checks.png (bottom-left panel).
    # That ACF decays far slower than 0.82^k (≈0.2 at lag 30), so this null is a LOWER bound on
    # the true null spread — p-values below are therefore optimistic for the repo's claim.
    published = {"BTC": -0.0837, "NIFTY": -0.0580}   # scripts/validate_gmsi.py output, dashboard/app.py:389-390
    rng = np.random.default_rng(SEED)
    out = {}
    for asset, rho in published.items():
        p = load_prices(asset)
        r = np.log(p["close"] / p["close"].shift(1))
        fwd = r.rolling(7, min_periods=1).std().shift(-7).dropna().values
        n = len(fwd); ry = stats.rankdata(fwd); ry = (ry - ry.mean()) / ry.std()
        null = np.empty(n_sims)
        for i in range(n_sims):
            e = rng.normal(size=n); g = np.empty(n); g[0] = e[0]
            for t in range(1, n):
                g[t] = phi * g[t - 1] + e[t]
            rx = stats.rankdata(g); rx = (rx - rx.mean()) / rx.std()
            null[i] = np.mean(rx * ry)
        iid_sd = 1 / np.sqrt(n - 1)
        out[asset] = {"n": n, "published_spearman": rho, "null_sd_ar1": float(null.std()),
                      "null_sd_iid_shuffle_theory": float(iid_sd),
                      "variance_inflation_vs_iid": float((null.std() / iid_sd) ** 2),
                      "two_sided_p_ar1_null": float(np.mean(np.abs(null) >= abs(rho))),
                      "two_sided_p_iid_normal_approx": float(2 * stats.norm.sf(abs(rho) / iid_sd))}
        print(f"  {asset}: {out[asset]}", flush=True)
    SUMMARY["E_calibrated_null"] = {"phi": phi, "n_sims": n_sims, **out}


def main():
    log = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s); log.write(s); return len(s)

    with redirect_stdout(Tee()):
        section_a(); section_b(); section_c(); section_d(); section_e()
    (OUT / "model_eval_stdout.txt").write_text(log.getvalue())
    (OUT / "model_eval_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))


if __name__ == "__main__":
    main()
