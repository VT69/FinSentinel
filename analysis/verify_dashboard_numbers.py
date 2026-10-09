"""
analysis/verify_dashboard_numbers.py — traces every hard-coded number in
dashboard/app.py back to the code that produced it.

Specifically: re-runs scripts/08_market_dynamics_analysis.py's *synthetic*
branch (np.random.seed(42) GARCH + AR(1) "GMSI") and compares its outputs to
dashboard/app.py's SHOCK_DECAY and NIFTY_REGIME_STATS constants and the
"MFI peak 0.76" claim. If they match, those dashboard "real data" numbers are
simulation outputs.

    python analysis/verify_dashboard_numbers.py
"""
from __future__ import annotations

import ast
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, ROOT  # noqa: E402


def dashboard_constants():
    tree = ast.parse((ROOT / "dashboard" / "app.py").read_text())
    want = {"SHOCK_DECAY", "NIFTY_REGIME_STATS", "COND_EXP_BTC", "COND_EXP_NIFTY", "REAL_CORR_BTC", "REAL_CORR_NIFTY"}
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in want:
            out[node.targets[0].id] = ast.literal_eval(node.value)
    return out


def run_script08_synthetic():
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:   # empty cwd → every data path in load_or_generate_data() misses
        os.chdir(tmp)
        try:
            spec = importlib.util.spec_from_file_location("s08", ROOT / "scripts" / "08_market_dynamics_analysis.py")
            s08 = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(s08)          # module import runs np.random.seed(42) (line 39)
            btc, nifty, gmsi = s08.generate_synthetic_data()   # same single call as the __main__ "all missing" path
            res = {}
            for name, df in [("BTC", btc), ("NIFTY", nifty)]:
                g = gmsi["gmsi"].reindex(df.index).ffill()
                mfi = s08.compute_mfi(df, window=30)
                reg = s08.define_regimes(g)
                mask, _ = s08.identify_shocks(df, 0.95)
                _, summ = s08.compute_shock_propagation(df, mask, (1, 3, 7, 14, 21))
                rs = s08.regime_stats(df, reg, mfi)
                res[name] = {"date_range": [str(df.index[0].date()), str(df.index[-1].date())], "rows": len(df),
                             "shock_mean_fwd_abs_ret": summ["mean_fwd_vol"].round(4).tolist(),
                             "regime_stats": rs[["mean_vol30d", "mean_MFI", "persistence_ac1"]].round(3).to_dict(orient="index"),
                             "mfi_max": round(float(mfi["MFI"].max()), 3),
                             "mfi_max_date": str(mfi["MFI"].idxmax().date()),
                             "mfi_max_after_2017": round(float(mfi["MFI"].loc["2017":].max()), 3),
                             "mfi_max_after_2017_date": str(mfi["MFI"].loc["2017":].idxmax().date()),
                             "corr_gmsi_vs_vol30d": round(float(g.corr(df["vol_30d"])), 4)}
        finally:
            os.chdir(cwd)
    return res


def run_script08_on_real_prices():
    """Same MFI / shock functions, fed the committed real prices (no GMSI exists → no regime stats)."""
    import pandas as pd
    from common import load_prices
    spec = importlib.util.spec_from_file_location("s08r", ROOT / "scripts" / "08_market_dynamics_analysis.py")
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            s08 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s08)
        finally:
            os.chdir(cwd)
    out = {}
    for a in ["BTC", "NIFTY"]:
        p = load_prices(a).set_index("date")
        df = pd.DataFrame({"log_return": np.log(p["close"] / p["close"].shift(1))})
        for w in (7, 14, 30):   # same annualisation as the script (×√252), main-block lines 884-887
            df[f"vol_{w}d"] = df["log_return"].rolling(w).std() * np.sqrt(252)
        df["abs_return"] = df["log_return"].abs()
        df = df.loc["2016-01-01":"2023-08-31"].dropna()   # same window the synthetic figures show
        mfi = s08.compute_mfi(df, 30)
        mask, _ = s08.identify_shocks(df, 0.95)
        _, summ = s08.compute_shock_propagation(df, mask, (1, 3, 7, 14, 21))
        out[a] = {"rows": len(df), "shock_mean_fwd_abs_ret": summ["mean_fwd_vol"].round(4).tolist(),
                  "mfi_max": round(float(mfi["MFI"].max()), 3), "mfi_max_date": str(mfi["MFI"].idxmax().date()),
                  "mfi_2020_03": round(float(mfi["MFI"].loc["2020-03"].max()), 3),
                  "vol_30d_ann_max": round(float(df["vol_30d"].max()), 3),
                  "vol_30d_ann_max_date": str(df["vol_30d"].idxmax().date()),
                  "cum_log_return_end": round(float(df["log_return"].sum()), 3)}
    return out


def main():
    dash = dashboard_constants()
    syn = run_script08_synthetic()
    real = run_script08_on_real_prices()
    print("REAL-PRICE counterparts (2016-01-01 … 2023-08-31):")
    for a, v in real.items():
        print(f"    {a}: {v}")
    cmp = {
        "SHOCK_DECAY.btc (dashboard)": dash["SHOCK_DECAY"]["btc"],
        "synthetic BTC mean fwd |r| at h=1,3,7,14,21": syn["BTC"]["shock_mean_fwd_abs_ret"],
        "SHOCK_DECAY.nifty (dashboard)": dash["SHOCK_DECAY"]["nifty"],
        "synthetic NIFTY mean fwd |r|": syn["NIFTY"]["shock_mean_fwd_abs_ret"],
        "NIFTY_REGIME_STATS (dashboard)": dash["NIFTY_REGIME_STATS"],
        "synthetic NIFTY regime stats": syn["NIFTY"]["regime_stats"],
        "dashboard claim": "MFI peak 0.76 · COVID crash · Mar 2020 (app.py:636, 996-999)",
        "synthetic BTC MFI max after 2017": [syn["BTC"]["mfi_max_after_2017"], syn["BTC"]["mfi_max_after_2017_date"]],
        "synthetic BTC MFI global max": [syn["BTC"]["mfi_max"], syn["BTC"]["mfi_max_date"]],
        "synthetic date ranges": {k: v["date_range"] for k, v in syn.items()},
        "synthetic corr(GMSI, vol_30d)": {k: v["corr_gmsi_vs_vol30d"] for k, v in syn.items()},
    }
    for k, v in cmp.items():
        print(f"{k}:\n    {v}")
    (OUT / "verify_dashboard_numbers.json").write_text(json.dumps({"dashboard": dash, "synthetic": syn, "real": real}, indent=2))


if __name__ == "__main__":
    main()
