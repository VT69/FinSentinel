# FinSentinel — stress, fragility and volatility in BTC and NIFTY 50

A research project on **how volatile BTC-USD and the NIFTY 50 index will be next week**, and whether
**news-based stress** and **market fragility** measures add anything to that picture.
Daily data: BTC 2015–2025, NIFTY 2010–2025.

Dashboard: `streamlit run dashboard/app.py` · Deployment guide: [`docs/DEPLOY.md`](docs/DEPLOY.md) ·
Start reading: [`docs/README.md`](docs/README.md)

---

## Results (all reproducible with `python run_pipeline.py`)

| Question | Answer from the data |
|---|---|
| Can next-week volatility be forecast? | **A little, mostly from its own past.** A 4-parameter HAR model on trailing 5/22/60-day realised volatility gets out-of-sample R² (log σ) of **0.047 (BTC)** and **0.100 (NIFTY)** in purged walk-forward tests. It beats a Random Forest (−0.026 / 0.062), a persistence forecast and the historical mean on RMSE, MAE, R² and QLIKE. |
| Where does it fail? | Event weeks it cannot see coming (Aug 2024 global sell-off for BTC; the June 2024 Indian election-result week for NIFTY). It shrinks toward the mean: it over-predicts calm weeks and under-predicts turbulent ones. |
| How long do shocks last? | After a top-5% daily move, BTC's absolute returns stay **~1.5–1.7× normal for three weeks**; NIFTY's start at **2.0×** and fade to **1.3×** after 21 days. |
| Does the news-based stress index (GMSI) predict volatility? | **Weakly at best.** Its lowest-stress quintile precedes the highest forward volatility (Spearman ρ = −0.084 BTC, −0.058 NIFTY), but once autocorrelation is handled the significance is borderline for BTC (**p ≈ 0.05**) and absent for NIFTY (**p ≈ 0.16**). The "complacency effect" is a hypothesis, not a finding. |
| Does sentiment (FinBERT/VADER) help? | **Untested.** Headlines exist for only ~3 months (Oct 2024 – Jan 2025), about 13 usable rows. The sentiment Random Forest is archived. |

> **Corrections.** Earlier versions of this repository and dashboard reported fragility, shock-decay and
> "AC₁ paradox" results that had been generated from **simulated data** by a silent fallback. They also showed
> p-values from a placebo test that is invalid for persistent series, and served a Random Forest that predicted
> a constant. All of these were removed or recomputed. The full audit is in [`docs/ISSUES.md`](docs/ISSUES.md).

## Repository layout

```
run_pipeline.py              one command: raw prices → models/ + dashboard/data/
pipeline/                    tested library code shared by training, scripts and dashboard
  volatility.py              features, target, HAR model (JSON), input validation, forecast()
  market_dynamics.py         Market Fragility Index, shocks, regimes (past-only)
  stats.py                   circular-shift permutation test, calibrated AR(1) null
  sentiment.py               FinBERT polarity by label name, text cleaning
  data.py                    paths + validated price loaders
  feature_engineering.py, train_rf_model.py, evaluate.py   archived sentiment-RF experiment
dashboard/                   Streamlit app (app.py, data.py, views/), pinned requirements
models/har_{btc,nifty}.json  served models (plain JSON, no pickle)
data/raw/                    committed prices, headlines, Google Trends
scripts/                     GDELT download, GMSI construction/validation, market-dynamics figures
notebooks/                   archived exploration (each starts with its known issues)
analysis/                    forensic audit scripts + outputs (reproduce docs/ numbers)
tests/                       pytest unit + Streamlit AppTest + Playwright smoke test
docs/                        architecture, data, model, issues, deployment, interview prep
```

## Run it

```bash
# training / analysis (Python 3.12+)
pip install -r requirements-research.txt
python run_pipeline.py            # ~1 min: walk-forward eval, models, MFI, shocks, GMSI calibration
pytest -q

# dashboard (Python 3.11–3.14)
pip install -r dashboard/requirements.txt
streamlit run dashboard/app.py

# optional
pip install -r data_pipeline/requirements.txt && python data_pipeline/fetch_all.py --source yfinance
pip install -r requirements-nlp.txt   # FinBERT re-scoring (torch / transformers)
```

The GMSI scripts (`scripts/reconstruct_gmsi.py`, `validate_gmsi.py`, `regime_analysis.py`) need GDELT-derived
files in `data/processed/` that are not committed (several GB of raw event files). They fail with a clear message
when the files are missing.

## Data sources and terms
- **Yahoo Finance via `yfinance`**: prices. Personal / research use only.
- **GDELT** (DOC API headlines; 1.0 event files): open data, attribution requested.
- **Google Trends**: monthly interest exports.

API keys for the optional fetchers go in `.env` (template: `.env.example`). Never commit `.env`.

## Author
Vaibhav Tiwari · B.Tech AI & ML, VIT Bhopal University · [github.com/VT69](https://github.com/VT69)

*Research code. Not investment advice; not suitable for trading or risk decisions.*
