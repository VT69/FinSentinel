"""
scripts/08_market_dynamics_analysis.py — Market Fragility Index (MFI), shock
propagation and (optionally) GMSI-regime statistics on REAL prices.

    python scripts/08_market_dynamics_analysis.py                       # real BTC + NIFTY prices
    python scripts/08_market_dynamics_analysis.py --gmsi data/processed/gmsi_exogenous.csv
    python scripts/08_market_dynamics_analysis.py --demo                # synthetic, watermarked, separate folder

History (docs/ISSUES.md P0-1/P0-2): the previous version silently fell back to
seeded synthetic GARCH data whenever its (never-produced) input files were
missing, and those synthetic figures/numbers were published as real results.
This version reads data/raw/*_prices.csv, fails loudly if anything is missing,
and only simulates under an explicit --demo flag.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import market_dynamics as md  # noqa: E402
from pipeline.data import ASSETS, FIGURES, load_close  # noqa: E402
from pipeline.volatility import annualise  # noqa: E402

COLORS = {"BTC": "#f7931a", "NIFTY": "#0070f3", "low": "#3fb950", "medium": "#d29922", "high": "#f85149"}
DISPLAY_FROM = "2016-01-01"   # earlier rows are burn-in for expanding normalisation / thresholds


def synthetic_returns(n=2000, seed=42) -> dict[str, pd.Series]:
    """GARCH(1,1)-t returns for --demo ONLY. Never mixed with real outputs."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-01", periods=n)
    out = {}
    for asset, (omega, alpha, beta, dof) in {"BTC": (1e-5, .12, .85, 4), "NIFTY": (3e-6, .07, .90, 6)}.items():
        s2, r = omega / (1 - alpha - beta), np.zeros(n)
        for t in range(1, n):
            s2 = omega + alpha * r[t - 1] ** 2 + beta * s2
            r[t] = np.sqrt(s2) * rng.standard_t(dof) / np.sqrt(dof / (dof - 2))
        out[asset] = pd.Series(r, index=dates)
    return out


def watermark(fig, demo):
    if demo:
        fig.text(0.5, 0.5, "SYNTHETIC DEMO DATA — NOT A RESULT", ha="center", va="center", fontsize=28,
                 color="red", alpha=0.25, rotation=20)


def fig_mfi(r, mfi, asset, out, demo):
    m = mfi.loc[DISPLAY_FROM:]
    fig, ax = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    ax[0].plot(r.loc[DISPLAY_FROM:].cumsum(), color=COLORS[asset], lw=.9); ax[0].set_ylabel("cum. log return")
    ax[1].plot(m["MFI"], lw=.8); ax[1].set_ylim(0, 1); ax[1].set_ylabel("MFI")
    v30 = annualise(r.rolling(30).std(), asset).loc[DISPLAY_FROM:]
    ax[2].plot(v30, color="0.3", lw=.8); ax[2].set_ylabel("30d vol (annualised)")
    fig.suptitle(f"Market Fragility Index — {asset} ({'SYNTHETIC' if demo else 'real prices'})")
    watermark(fig, demo); fig.tight_layout(); fig.savefig(out / f"fig1_mfi_{asset.lower()}.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(3, 1, figsize=(13, 7), sharex=True)
    for a, (c, lbl) in zip(ax, [("persistence_norm", "A: persistence AC1(|r|)"), ("vol_of_vol_norm", "B: vol-of-vol"),
                                ("tail_freq_norm", "C: tail frequency")]):
        a.plot(m[c], lw=.7); a.set_ylabel(lbl, fontsize=8); a.set_ylim(0, 1)
    fig.suptitle(f"MFI components — {asset}"); watermark(fig, demo); fig.tight_layout()
    fig.savefig(out / f"fig5_mfi_components_{asset.lower()}.png", dpi=110); plt.close(fig)


def fig_shocks(tables, out, demo):
    fig, ax = plt.subplots(figsize=(10, 5))
    for asset, sp in tables.items():
        ax.plot(sp["horizon"], sp["ratio_to_baseline"], "o-", color=COLORS[asset], label=asset)
        lo = sp["ci_low"] / sp["baseline_mean_abs_return"]; hi = sp["ci_high"] / sp["baseline_mean_abs_return"]
        ax.fill_between(sp["horizon"], lo, hi, color=COLORS[asset], alpha=.15)
    ax.axhline(1, color="k", ls="--", lw=.8, label="unconditional mean |r|")
    ax.set_xlabel("days after shock (top-5% |r|, expanding threshold)")
    ax.set_ylabel("mean |r| ÷ unconditional mean |r|  (95% CI)")
    ax.set_title("Shock propagation" + (" — SYNTHETIC" if demo else " — real prices")); ax.legend()
    watermark(fig, demo); fig.tight_layout(); fig.savefig(out / "fig6_shock_decay_comparison.png", dpi=110); plt.close(fig)


def regime_analysis(r, mfi, gmsi, asset, out, demo):
    """Needs a real GMSI series. Expanding (past-only) regime thresholds; AC1 within contiguous spells."""
    g = gmsi.reindex(r.index).ffill()
    reg = md.expanding_regimes(g)
    rows = []
    for name in ["low", "medium", "high"]:
        mask = reg == name
        ac1, npairs = md.within_spell_ac1(r.abs(), reg, name)
        rows.append({"regime": name, "n_days": int(mask.sum()),
                     "mean_vol30d_annualised": float(annualise(r.rolling(30).std(), asset)[mask].mean()),
                     "mean_MFI": float(mfi["MFI"][mask].mean()), "ac1_abs_return_within_spell": ac1,
                     "n_consecutive_pairs": npairs})
    tab = pd.DataFrame(rows).set_index("regime")
    print(f"\n{asset} regime statistics (expanding thresholds, within-spell AC1):\n{tab.round(4)}")
    tab.to_csv(out / f"regime_stats_{asset.lower()}.csv")
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for a, c in zip(ax, ["mean_vol30d_annualised", "mean_MFI", "ac1_abs_return_within_spell"]):
        a.bar(tab.index, tab[c], color=[COLORS[i] for i in tab.index]); a.set_title(c, fontsize=9)
    fig.suptitle(f"GMSI regime statistics — {asset}"); watermark(fig, demo); fig.tight_layout()
    fig.savefig(out / f"fig4_regime_stats_{asset.lower()}.png", dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gmsi", type=Path, help="CSV with columns date,pure_gmsi (scripts/reconstruct_gmsi.py output)")
    ap.add_argument("--demo", action="store_true", help="synthetic GARCH data, written to reports/figures/demo/")
    a = ap.parse_args()

    out = FIGURES / "demo" if a.demo else FIGURES
    out.mkdir(parents=True, exist_ok=True)
    if a.demo:
        print("DEMO MODE: synthetic data. Nothing written here is a research result.")
        returns = synthetic_returns()
    else:
        returns = {asset: np.log(c / c.shift(1)) for asset, c in ((x, load_close(x)) for x in ASSETS)}

    gmsi = None
    if a.gmsi:
        if not a.gmsi.exists():
            raise SystemExit(f"--gmsi file {a.gmsi} not found")
        gdf = pd.read_csv(a.gmsi, parse_dates=["date"])
        if "pure_gmsi" not in gdf:
            raise SystemExit(f"{a.gmsi} has no 'pure_gmsi' column")
        gmsi = gdf.set_index("date")["pure_gmsi"]
    else:
        print("No --gmsi given: skipping regime statistics (the GMSI series is not committed to the repo).")

    tables = {}
    for asset, r in returns.items():
        mfi = md.compute_mfi(r)
        shocks, _ = md.identify_shocks(r)
        tables[asset] = md.shock_propagation(r, shocks)
        print(f"\n{asset}: {int(shocks.sum())} shocks; MFI max {mfi['MFI'].max():.3f} on {mfi['MFI'].idxmax():%Y-%m-%d}")
        print(tables[asset].round(5).to_string(index=False))
        fig_mfi(r, mfi, asset, out, a.demo)
        if gmsi is not None:
            regime_analysis(r, mfi, gmsi, asset, out, a.demo)
    fig_shocks(tables, out, a.demo)
    print(f"\nFigures written to {out}")


if __name__ == "__main__":
    main()
