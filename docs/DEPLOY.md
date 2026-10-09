# Deploying the dashboard to Streamlit Community Cloud

## What gets deployed

| Item | Value |
|---|---|
| Entrypoint | `dashboard/app.py` |
| Dependencies | `dashboard/requirements.txt`: streamlit 1.63.0, pandas 3.0.5, numpy 2.4.6, plotly 6.9.0 |
| Config | `.streamlit/config.toml` (repo root: Cloud requires it there even when the entrypoint is in a subfolder) |
| Python | tested on 3.11, 3.12, 3.13 and 3.14; choose **3.12** in Advanced settings |
| Secrets | none needed |
| Data and model | committed: `data/raw/*_prices.csv` (~0.8 MB), `models/har_*.json` (<1 KB each), `dashboard/data/` (~0.7 MB), `reports/figures/` |
| Network at runtime | none (no live data fetches) |

Size is far below any hosting limit, so **no Git LFS or external storage is needed**. The model is plain JSON, so
there is **no pickle or scikit-learn version to match** at deploy time.

There is deliberately **no `requirements.txt` at the repo root**: Community Cloud accepts a dependency file
either at the root or next to the entrypoint. The heavier research pins live in `requirements-research.txt`, so the
dashboard file is the only candidate.

## Step by step

1. Push this branch to GitHub and merge it into the branch you want to deploy (e.g. `main`).
2. Go to **share.streamlit.io** and sign in with GitHub.
3. **Create app → Deploy a public app from GitHub.**
4. Fill in:
   - Repository: `VT69/<repo-name>`
   - Branch: `main`
   - Main file path: `dashboard/app.py`
5. Open **Advanced settings** and set the **Python version to 3.12**. Community Cloud ignores `runtime.txt`; the
   version is chosen here, and it can only be changed by deleting and redeploying the app.
6. Leave Secrets empty. Click **Deploy**.
7. When the build finishes, check the first lines of the build log show Python 3.12 and
   `streamlit==1.63.0`. Then open each page. Expected: the landing page renders in a few seconds, and no red error boxes appear.

If you are replacing the existing app at `financial-sentiment-market-analysis.streamlit.app`, change the main file
path to `dashboard/app.py` if needed, then **Reboot** it from the app menu so it rebuilds with the new requirements.

## Updating the results

```bash
pip install -r requirements-research.txt
python run_pipeline.py            # rewrites models/ and dashboard/data/ from data/raw/
pytest -q
git add models dashboard/data && git commit -m "Refresh results" && git push
```

Community Cloud redeploys on push. To refresh prices first:
`pip install -r data_pipeline/requirements.txt && python data_pipeline/fetch_all.py --source yfinance`.
Then copy the new BTC-USD / ^NSEI files into `data/raw/{btc,nifty}_prices.csv`. That step is manual on purpose:
it keeps the published numbers tied to a committed snapshot.

## Verify locally before pushing

```bash
pip install -r dashboard/requirements.txt pytest
streamlit run dashboard/app.py --server.headless true --server.port 8501   # from the repo root, like Cloud
pytest -q tests/test_dashboard.py                                         # every page + input validation
pip install playwright && playwright install chromium
python tests/e2e/browser_smoke.py http://localhost:8501                   # real browser
```

Evidence from the run before this commit (Python 3.11.17, Streamlit 1.63.0, headless Chromium):

```
first paint: 3.8s
Overview                      1.6s  h1='FinSentinel — stress, fragility and volatility'  charts=1 exceptions=0 errors=0
Volatility forecast           2.6s  h1='Volatility forecast'  charts=1 exceptions=0 errors=0
  historical: Forecast daily σ | 1.38% | std of next 5 daily log returns
  bad input -> Value #3 ('abc') is not a number.
  good input -> Forecast daily σ | 1.32% | std of next 5 daily log returns
GMSI & volatility             1.6s  h1='GMSI and forward volatility'  charts=1 exceptions=0 errors=0
Market fragility (MFI)        3.7s  h1='Market Fragility Index (MFI)'  charts=2 exceptions=0 errors=0
Shock propagation             1.6s  h1='Shock propagation'  charts=1 exceptions=0 errors=0
Methodology & limitations     1.6s  h1='Methodology & limitations'  charts=0 exceptions=0 errors=0
browser console errors: 0
PASS
```

Timings include a fixed 1.5 s wait per page. `pytest tests/test_dashboard.py` → 10 passed on Python 3.11, 3.13 and 3.14.
The old app took **185 s** to render its landing page.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Build log shows a different Python | Cloud ignores `runtime.txt`; the version is set at creation | Delete and redeploy with Python 3.12 in Advanced settings (the dependencies also work on 3.11–3.14) |
| Red box "A required data file is missing … run `python run_pipeline.py`" | `models/` or `dashboard/data/` not committed | Run the pipeline and commit both folders |
| `ModuleNotFoundError: pipeline` | app started from a copy without the repo root | Deploy from the full repo; `dashboard/data.py` adds the repo root to `sys.path` |
| Theme is not dark | `.streamlit/config.toml` not at the repo root | Keep it at the root; reboot the app |
| "This app has gone over its resource limits" | Unlikely: the app holds about 10k rows | Reboot; the caches are small |

## Secrets
None are used. If you later add an API (e.g. live prices), put keys in the Cloud app's **Secrets** box and read them
with `st.secrets["NAME"]`. Locally use `.streamlit/secrets.toml`, which is in `.gitignore`. Never put keys in
`.env.example`.
