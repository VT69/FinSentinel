# docs/ — reading guide

## If you are studying this project cold (recommended order)

| # | Read | Time | Why |
|---|---|---|---|
| 1 | [`../README.md`](../README.md) | 5 min | what the project claims now, and the results table |
| 2 | [`ARCHITECTURE.md`](ARCHITECTURE.md) **Part A** | 15 min | how the current system fits together; the training/inference boundary |
| 3 | [`MODEL_REPORT.md`](MODEL_REPORT.md) §1, §4–§10 | 25 min | why HAR, how it was validated, metrics, failure modes |
| 4 | [`DATA_REPORT.md`](DATA_REPORT.md) §0, §1, §6–§8, §10 | 20 min | the data, the target, leakage audit, limitations |
| 5 | [`INTERVIEW_DEFENSE.md`](INTERVIEW_DEFENSE.md) | 45 min | pitch, ~50 Q&As, "do not bluff" list, glossary. Rehearse out loud. |
| 6 | [`ISSUES.md`](ISSUES.md) resolution table, then P0 rows | 15 min | what was wrong in the original and how each item was fixed |
| 7 | [`ARCHITECTURE.md`](ARCHITECTURE.md) **Part B** | optional | the full forensic audit of the original code |
| 8 | [`DEPLOY.md`](DEPLOY.md) | 10 min | how to host the dashboard; verification evidence |

Then open the code in this order: `pipeline/volatility.py` → `run_pipeline.py` → `dashboard/data.py` →
`dashboard/views/forecast.py` → `pipeline/stats.py` → `tests/`.

## Index

| File | Contents |
|---|---|
| `ARCHITECTURE.md` | A: current architecture (diagrams, module table, artifact contract). B: audit of commit `e1eb270` |
| `DATA_REPORT.md` | provenance, schema, full numeric/categorical/text profile, target, correlations, missingness, leakage audit, limitations |
| `MODEL_REPORT.md` | preprocessing, features, algorithms, alternatives, validation, re-run metrics, importance, overfitting, error analysis, placebo validity |
| `ISSUES.md` | resolution table + P0–P3 issues with file:line, impact, fix, risk |
| `DEPLOY.md` | Streamlit Community Cloud steps, local verification, troubleshooting |
| `INTERVIEW_DEFENSE.md` | 2-min / 30-s pitch, 48 Q&As, honest weak spots, glossary |

## Reproducing the numbers

```bash
pip install -r requirements-research.txt
python run_pipeline.py                  # current models and results (dashboard/data, models/)
pytest -q

pip install -r analysis/requirements.txt
python analysis/data_profile.py         # DATA_REPORT tables (analysis/outputs/)
python analysis/model_eval.py           # MODEL_REPORT audit (reads the original artefacts from commit e1eb270 via git)
python analysis/verify_dashboard_numbers.py   # proves the old dashboard numbers were synthetic
```

`analysis/` needs a full git clone, because it reads the deleted original files (`models/rf_btc.pkl`, the old
`scripts/08…py`, the old `dashboard/app.py`) from the audited commit with `git show`.
