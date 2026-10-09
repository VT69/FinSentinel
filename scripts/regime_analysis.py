"""
scripts/regime_analysis.py — volatility behaviour inside GMSI stress regimes.

    python scripts/regime_analysis.py      # needs data/processed/{gmsi_exogenous,btc_vsi_full,nifty_vsi_full}.csv

Fixes (docs/ISSUES.md):
  P1-3  regime cut-offs (20%/80%) and the shock threshold (95%) are EXPANDING,
        past-only quantiles — the old full-sample quantiles used future data.
  P1-10 persistence is a lag-1 autocorrelation over consecutive days inside the
        same regime spell — the old code concatenated non-adjacent days.
  KS / Mann-Whitney tests assume independent observations; with autocorrelated
  volatility their p-values are far too small, so they are printed as descriptive.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, kurtosis, mannwhitneyu, skew

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline import market_dynamics as md  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "reports" / "figures" / "regime"
REGIMES = ["low", "medium", "high"]


def require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"{path} not found (built from GDELT data that is not committed to the repo).")
    return path


def load_data():
    gmsi = pd.read_csv(require(PROCESSED / "gmsi_exogenous.csv"), parse_dates=["date"])
    btc = pd.read_csv(require(PROCESSED / "btc_vsi_full.csv"), parse_dates=["date"])
    nifty = pd.read_csv(require(PROCESSED / "nifty_vsi_full.csv"), parse_dates=["date"])
    return gmsi, btc, nifty


def attach_regimes(asset_df, gmsi):
    g = gmsi.set_index("date")["pure_gmsi"].sort_index()
    reg = md.expanding_regimes(g)                       # past-only thresholds on the GMSI's own calendar
    df = asset_df.merge(pd.DataFrame({"date": g.index, "pure_gmsi": g.values, "Regime": reg.values}),
                        on="date", how="inner").dropna(subset=["volatility_7d", "Regime"])
    return df.sort_values("date").reset_index(drop=True)


def regime_diagnostics(df, asset):
    print(f"\n{'*' * 50}\nREGIME DIAGNOSTICS: {asset}\n{'*' * 50}")
    rows = []
    for r in REGIMES:
        v = df.loc[df["Regime"] == r, "volatility_7d"].dropna()
        ac1, npairs = md.within_spell_ac1(df["volatility_7d"], df["Regime"], r)
        rows.append({"Regime": r, "N_days": len(v), "Mean_7d": v.mean(), "Median_7d": v.median(),
                     "Vol_of_Vol_7d": v.std(), "Skewness_7d": skew(v), "Kurtosis_7d": kurtosis(v),
                     "AC1_within_spell": ac1, "N_pairs": npairs})
    res = pd.DataFrame(rows)
    print(res.round(4).to_string(index=False))
    return res


def shock_response(df, asset):
    print(f"\n--- Shock response: {asset} ---")
    r = df["return"]
    shocks, _ = md.identify_shocks(r)                   # expanding 95% threshold, 250-day burn-in
    df = df.assign(shock=shocks.values, regime_prev=df["Regime"].shift(1))
    rows = []
    for reg in REGIMES:
        idx = df.index[df["shock"] & (df["regime_prev"] == reg)]
        row = {"Regime": reg, "Shock_Count": len(idx)}
        for h in (1, 7, 14):
            row[f"Fwd_Vol_t+{h}"] = df["volatility_7d"].reindex(idx + h).mean()
        rows.append(row)
    res = pd.DataFrame(rows)
    print(res.round(5).to_string(index=False))
    return res


def tail_risk_conditioning(df, asset):
    print(f"\n--- Tail risk (contemporaneous, descriptive): {asset} ---")
    thr = df["volatility_7d"].shift(1).expanding(min_periods=250).quantile(0.95)
    tail = (df["volatility_7d"] >= thr) & thr.notna()
    print(f"Unconditional P(tail) = {tail[thr.notna()].mean():.4f}")
    for r in REGIMES:
        m = (df["Regime"] == r) & thr.notna()
        print(f"P(tail | {r}) = {tail[m].mean():.4f}  (N={int(m.sum())})")
    lo = df.loc[df["Regime"] == "low", "volatility_7d"]; hi = df.loc[df["Regime"] == "high", "volatility_7d"]
    print(f"KS={ks_2samp(hi, lo).statistic:.4f}, MWU p={mannwhitneyu(hi, lo).pvalue:.3e} "
          "(assumes independence — descriptive only)")


def visualise(df_btc, df_nifty, shocks):
    OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (df, name) in zip(axes, [(df_btc, "BTC"), (df_nifty, "NIFTY")]):
        sns.boxplot(x="Regime", y="volatility_7d", data=df, order=REGIMES, ax=ax)
        ax.set_title(f"{name}: 7d volatility by GMSI regime (expanding thresholds)")
    plt.tight_layout(); plt.savefig(OUT / "boxplot_vol_by_regime.png"); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, (df, name) in zip(axes, [(df_btc, "BTC"), (df_nifty, "NIFTY")]):
        q = pd.qcut(df["volatility_7d"], 5, labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"])
        ct = pd.crosstab(pd.Categorical(df["Regime"], REGIMES), q, normalize="index") * 100
        sns.heatmap(ct, annot=True, fmt=".1f", ax=ax, cbar_kws={"label": "% of regime"})
        ax.set_title(f"{name}: regime vs volatility quintile")
    plt.tight_layout(); plt.savefig(OUT / "heatmap_regime_vs_vol.png"); plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, (res, name) in zip(axes, shocks):
        res.set_index("Regime")[["Fwd_Vol_t+1", "Fwd_Vol_t+7", "Fwd_Vol_t+14"]].T.plot(marker="o", ax=ax)
        ax.set_title(f"{name}: shock response by prior-day regime"); ax.set_ylabel("mean 7d vol")
    plt.tight_layout(); plt.savefig(OUT / "shock_response.png"); plt.close()
    print(f"Saved plots to {OUT}")


def main():
    gmsi, btc, nifty = load_data()
    df_btc, df_nifty = attach_regimes(btc, gmsi), attach_regimes(nifty, gmsi)
    shocks = []
    for df, name in [(df_btc, "BTC"), (df_nifty, "NIFTY")]:
        regime_diagnostics(df, name)
        shocks.append((shock_response(df, name), name))
        tail_risk_conditioning(df, name)
    visualise(df_btc, df_nifty, shocks)


if __name__ == "__main__":
    main()
