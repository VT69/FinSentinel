# FinSentinel dashboard

```bash
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py          # run from the repo root
```

| Page | Shows | Source |
|---|---|---|
| Overview | headline results, BTC & NIFTY price history | `dashboard/data/*`, `data/raw/*_prices.csv` |
| Volatility forecast | **live model inference**: 5-day volatility forecast for any date in the data or pasted prices; walk-forward model comparison | `models/har_*.json`, `pipeline/volatility.py` |
| GMSI & volatility | conditional-expectation charts, corrected significance | `reports/figures/`, `dashboard/data/gmsi_calibration.json` |
| Market fragility (MFI) | MFI and its components on real prices | `dashboard/data/mfi_*.csv` |
| Shock propagation | volatility after top-5% moves vs normal | `dashboard/data/shocks_*.csv` |
| Methodology & limitations | definitions, validation, caveats, corrections log | — |

All files under `dashboard/data/` and `models/` are produced by `python run_pipeline.py` (repo root) and committed,
so the app needs no network access and no heavy ML libraries. Deployment: `docs/DEPLOY.md`.
