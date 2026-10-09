# reports/

| File | Produced by | Data | Status |
|---|---|---|---|
| `figures/fig1_mfi_{btc,nifty}.png`, `fig5_mfi_components_{btc,nifty}.png`, `fig6_shock_decay_comparison.png` | `scripts/08_market_dynamics_analysis.py` | real prices (`data/raw`) | current — regenerate any time |
| `figures/cond_exp_{BTC,NIFTY}.png`, `gmsi_sanity_checks.png` | `scripts/validate_gmsi.py`, `scripts/sanity_check_gmsi.py` | GMSI (GDELT, not committed) | original run; descriptive in-sample quintiles |
| `figures/placebo_test_{BTC,NIFTY}.png` | old `scripts/validate_gmsi.py` | GMSI | **superseded**: shows the invalid i.i.d. placebo. The fixed script overwrites them with circular-shift results when re-run with GMSI data |
| `figures/regime/*.png` | old `scripts/regime_analysis.py` | GMSI | **superseded**: full-sample (look-ahead) regime thresholds |
| `gmsi_validation_walkthrough.md` | original analysis session | — | historical record with a correction banner |

Earlier versions of this folder contained `fig3_regime_shock_nifty.png` and `fig4_regime_stats_nifty.png`, and
versions of fig1/fig5/fig6 generated from **synthetic** data by a silent fallback in the old script 08. They were
removed or regenerated; see `docs/ISSUES.md` (P0-1).
