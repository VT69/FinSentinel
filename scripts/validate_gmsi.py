"""
scripts/validate_gmsi.py — does GMSI(t) relate to forward realised volatility?

    python scripts/validate_gmsi.py            # needs data/processed/{gmsi_exogenous,btc_vsi_full,nifty_vsi_full}.csv

Fix (docs/ISSUES.md P1-1): significance now comes from a CIRCULAR-SHIFT
permutation test, which preserves the autocorrelation of both series. The old
i.i.d. shuffle is still printed for comparison only — it is anti-conservative
for persistent series (57.5% false positives at AR(1) φ=0.9, MODEL_REPORT §D).
Conditional expectations are full-sample quintiles: an in-sample DESCRIPTION,
not a forecast (P1-3).
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp, mannwhitneyu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline.stats import permutation_test  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "reports" / "figures"
N_PERM = 1000


def require(path: Path) -> Path:
    if not path.exists():
        raise SystemExit(f"{path} not found. It is produced by scripts/reconstruct_gmsi.py / "
                         "scripts/construct_vsi_full.py from GDELT data that is not committed to the repo.")
    return path


def validate_asset(gmsi, asset_df, asset_name):
    print(f"\n{'=' * 50}\nValidating GMSI against {asset_name} volatility\n{'=' * 50}")
    df = pd.merge(gmsi[["date", "pure_gmsi"]],
                  asset_df[["date", "return", "volatility_7d", "volatility_14d", "volatility_30d"]],
                  on="date", how="inner").dropna()
    df["fwd_vol_1d"] = df["return"].abs().shift(-1)            # |r_{t+1}|
    df["fwd_vol_7d"] = df["volatility_7d"].shift(-7)           # std r[t+1..t+7]
    df["fwd_vol_14d"] = df["volatility_14d"].shift(-14)        # std r[t+1..t+14]
    df = df.dropna()
    targets = {"1 Day": "fwd_vol_1d", "7 Days": "fwd_vol_7d", "14 Days": "fwd_vol_14d"}

    print("\n--- Spearman(GMSI_t, forward vol) with permutation p-values ---")
    print(f"{'horizon':8s} {'rho':>8s} {'p_circular (valid)':>20s} {'p_iid (invalid, ref)':>22s}")
    circ = {}
    for label, col in targets.items():
        c = permutation_test(df["pure_gmsi"], df[col], n_perm=N_PERM, method="circular")
        i = permutation_test(df["pure_gmsi"], df[col], n_perm=N_PERM, method="iid")
        circ[label] = c
        print(f"{label:8s} {c.statistic:8.4f} {c.p_value:20.4f} {i.p_value:22.4f}")

    print("\n--- Conditional expectations (full-sample quintiles: descriptive, in-sample) ---")
    df["gmsi_quintile"] = pd.qcut(df["pure_gmsi"], 5, labels=["Q1(Low)", "Q2", "Q3", "Q4", "Q5(High)"])
    cond = df.groupby("gmsi_quintile", observed=True)[["fwd_vol_7d", "fwd_vol_14d"]].mean()
    print(cond)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    cond["fwd_vol_7d"].plot(kind="bar", ax=axes[0], color="skyblue", title=f"{asset_name}: E[7d vol | GMSI quintile]")
    cond["fwd_vol_14d"].plot(kind="bar", ax=axes[1], color="salmon", title=f"{asset_name}: E[14d vol | GMSI quintile]")
    plt.tight_layout(); plt.savefig(FIGURES / f"cond_exp_{asset_name}.png"); plt.close()

    print("\n--- Tail events (top 5% forward 7d vol) ---")
    thr = df["fwd_vol_7d"].quantile(0.95)
    tail, normal = df[df["fwd_vol_7d"] >= thr], df[df["fwd_vol_7d"] < thr]
    print(f"threshold {thr:.4f} (N={len(tail)}); KS p={ks_2samp(tail['pure_gmsi'], normal['pure_gmsi']).pvalue:.3e}; "
          f"MWU p={mannwhitneyu(tail['pure_gmsi'], normal['pure_gmsi']).pvalue:.3e}  "
          "(both assume independent observations — treat as descriptive)")

    c7 = circ["7 Days"]
    plt.figure(figsize=(8, 5))
    plt.axvline(c7.statistic, color="red", ls="--", lw=2, label=f"real ρ = {c7.statistic:.4f}")
    plt.axvspan(-1.96 * c7.null_sd, 1.96 * c7.null_sd, color="gray", alpha=.3,
                label=f"95% circular-shift null (sd {c7.null_sd:.4f})")
    plt.xlim(-max(0.15, abs(c7.statistic) * 1.5), max(0.15, abs(c7.statistic) * 1.5))
    plt.title(f"{asset_name}: GMSI vs forward 7d vol — circular-shift placebo (p = {c7.p_value:.3f})")
    plt.legend(); plt.savefig(FIGURES / f"placebo_test_{asset_name}.png"); plt.close()


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    gmsi = pd.read_csv(require(PROCESSED / "gmsi_exogenous.csv"), parse_dates=["date"])
    for asset in ["BTC", "NIFTY"]:
        df = pd.read_csv(require(PROCESSED / f"{asset.lower()}_vsi_full.csv"), parse_dates=["date"])
        validate_asset(gmsi, df, asset)


if __name__ == "__main__":
    main()
