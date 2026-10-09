import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
from pipeline import volatility as vol
from views._ui import COLORS, kcard, show, style

MODES = ("A date in the committed price history", "Paste your own closing prices")
EXAMPLE = "\n".join(f"{100 * np.exp(0.01 * np.sin(i / 3)):.2f}" for i in range(70))


def _realised_after(close: pd.Series, asof: pd.Timestamp):
    """Realised σ of the next h log returns after `asof`, if the committed data contains them."""
    pos = close.index.get_loc(asof)
    nxt = close.iloc[pos: pos + vol.HORIZON + 1]
    if len(nxt) < vol.HORIZON + 1:
        return None
    return float(np.log(nxt / nxt.shift(1)).dropna().std())


def _show_result(out: dict, asset: str, realised=None):
    c = st.columns(4)
    with c[0]:
        kcard("Forecast daily σ", f"{out['daily_vol']:.2%}", f"std of next {out['horizon_days']} daily log returns")
    with c[1]:
        days = vol.ANNUALISATION_DAYS[asset]
        kcard("Annualised", f"{out['annualised_vol']:.1%}", f"× √{days} ({'calendar' if days == 365 else 'trading'} days)")
    with c[2]:
        if realised is None:
            kcard("Realised", "—", "not in the committed data (future or too recent)")
        else:
            err = realised - out["daily_vol"]
            kcard("Realised (from data)", f"{realised:.2%}", f"forecast error {err:+.2%}")
    with c[3]:
        i = out["inputs"]
        kcard("Recent realised daily σ", f"{i['log_vol_5']:.2%}",
              f"last 5 days · 22d {i['log_vol_22']:.2%} · 60d {i['log_vol_60']:.2%}")
    if out["warning"]:
        st.warning(out["warning"])


def _model_quality(asset: str):
    st.subheader("How good is this model?")
    m = data.metrics()[asset]
    rows = []
    for name, s in m["summary"].items():
        rows.append({"model": name, "test RMSE (σ)": s["test_rmse_vol"], "test MAE (σ)": s["test_mae_vol"],
                     "test R² (log σ)": s["test_r2_log"], "test QLIKE ↓": s["test_qlike"],
                     "train R² (log σ)": s["train_r2_log"]})
    st.dataframe(pd.DataFrame(rows).set_index("model").style.format("{:.4f}"), width="stretch")
    st.caption(f"Purged walk-forward evaluation: 5 expanding-window folds, 5-day gap between train and test, "
               f"{m['n_rows']:,} days ({m['date_range'][0]} → {m['date_range'][1]}). QLIKE is the standard loss for "
               "volatility forecasts and penalises under-prediction most. The Random Forest fits training data "
               "better but generalises worse than the 4-parameter HAR model, which is why HAR is served.")

    oos = data.table("oos", asset)
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=oos["date"], y=oos["actual_vol"], name="realised next-5-day σ",
                               line=dict(color="#8ba3c4", width=0.8)))
    fig.add_trace(go.Scattergl(x=oos["date"], y=oos["har_vol"], name="HAR forecast (out-of-sample)",
                               line=dict(color=COLORS[asset], width=1.3)))
    fig.add_trace(go.Scattergl(x=oos["date"], y=oos["mean_vol"], name="historical-mean baseline",
                               line=dict(color="#ffffff", width=0.8, dash="dot")))
    show(style(fig, title=f"{asset}: out-of-sample forecasts vs realised volatility", yaxis_tickformat=".1%"))


def _model_card(mdl: vol.HARModel):
    with st.expander("Model card"):
        st.markdown(f"""
**log σ̂(t+1..t+5) = {mdl.intercept:+.3f} {mdl.coef['log_vol_5']:+.3f}·log σ₅ {mdl.coef['log_vol_22']:+.3f}·log σ₂₂ {mdl.coef['log_vol_60']:+.3f}·log σ₆₀**

- σₙ = sample std of the last n daily log returns, known at the close of day t. Target = std of the next 5.
- Fitted by OLS on {mdl.n_train:,} days ({mdl.trained_from} → {mdl.trained_through}). Stored as JSON
  (`models/har_{mdl.asset.lower()}.json`) — no pickle, no scikit-learn needed to serve it.
- The coefficients sum to {sum(mdl.coef.values()):.2f} < 1: forecasts mean-revert toward the long-run level.
- `exp(log forecast)` is a conditional *median*; the model systematically under-predicts the most turbulent weeks
  and over-predicts the calmest ones (it shrinks toward the mean).
- Not for trading or risk decisions. Trained on {mdl.asset} only; pasted prices from other assets are outside its training data.
""")


def render():
    st.title("Volatility forecast")
    st.caption("Forecasts how volatile the next 5 trading days will be, from the last 61+ daily closes. "
               "Training and this page use the same feature code (`pipeline/volatility.py`).")
    asset = st.selectbox("Asset", data.ASSETS)
    mdl, close = data.model(asset), data.prices(asset)
    mode = st.radio("Input", MODES, horizontal=True)

    if mode == MODES[0]:
        first, last = close.index[vol.MIN_CLOSES - 1], close.index[-1]
        d = st.date_input("Forecast as of the close of", value=last.date(), min_value=first.date(),
                          max_value=last.date(), format="YYYY-MM-DD")
        if d is None:
            st.error("Please pick a date.")
            return
        hist = close.loc[: pd.Timestamp(d)]
        asof = hist.index[-1]
        if asof.date() != d:
            st.info(f"No {asset} close on {d} (weekend or holiday) — using the previous close, {asof:%Y-%m-%d}.")
        try:
            out = vol.forecast(mdl, hist)
        except vol.InputError as exc:
            st.error(str(exc))
            return
        _show_result(out, asset, _realised_after(close, asof))
    else:
        with st.form("paste"):
            text = st.text_area(f"Daily closing prices, oldest first (at least {vol.MIN_CLOSES}; commas, spaces or "
                                "new lines)", height=160, placeholder=EXAMPLE[:200] + " …")
            submitted = st.form_submit_button("Forecast")
        if not submitted:
            st.info(f"Paste at least {vol.MIN_CLOSES} consecutive daily closes and press **Forecast**.")
        else:
            try:
                out = vol.forecast(mdl, vol.parse_closes(text))
            except vol.InputError as exc:
                st.error(str(exc))
            else:
                _show_result(out, asset)
    _model_quality(asset)
    _model_card(mdl)
