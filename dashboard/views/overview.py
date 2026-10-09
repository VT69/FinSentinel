import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data
from views._ui import COLORS, kcard, show, style


def render():
    st.title("FinSentinel — stress, fragility and volatility")
    st.caption("Daily BTC-USD (2015–2025) and NIFTY 50 (2010–2025). Every figure on this site is computed from "
               "real prices by `run_pipeline.py`, or by the served model at request time.")

    m, cal = data.metrics(), data.gmsi_calibration()
    sh = {a: data.table("shocks", a, parse_dates=()) for a in data.ASSETS}
    c = st.columns(4)
    for col, a in zip(c[:2], data.ASSETS):
        har, base = m[a]["summary"]["HAR (served)"], m[a]["summary"]["Mean (dummy)"]
        with col:
            kcard(f"{a} 5-day vol forecast", f"R² {har['test_r2_log']:+.3f}",
                  f"out-of-sample (log scale) · mean baseline {base['test_r2_log']:+.3f}")
    with c[2]:
        b, n = sh["BTC"].iloc[-1], sh["NIFTY"].iloc[-1]
        kcard("After a top-5% shock", f"{b['ratio_to_baseline']:.2f}× / {n['ratio_to_baseline']:.2f}×",
              f"BTC / NIFTY mean |r| vs normal, {int(b['horizon'])} days later")
    with c[3]:
        bt, nf = cal["assets"]["BTC"], cal["assets"]["NIFTY"]
        kcard("GMSI → forward vol", f"p ≈ {bt['p_value']:.2f} / {nf['p_value']:.2f}",
              f"BTC / NIFTY, calibrated null (ρ = {bt['statistic']:+.3f} / {nf['statistic']:+.3f})")

    st.subheader("What the data actually shows")
    btc_s, nif_s = sh["BTC"], sh["NIFTY"]
    st.markdown(f"""
1. **Volatility is forecastable mostly from its own past.** A 4-parameter HAR model on 5/22/60-day realised
   volatility beats a Random Forest, a persistence forecast and the historical mean in purged walk-forward tests
   on both assets — but it explains only a small share of the variance
   (out-of-sample R² {m['BTC']['summary']['HAR (served)']['test_r2_log']:.3f} BTC,
   {m['NIFTY']['summary']['HAR (served)']['test_r2_log']:.3f} NIFTY, log scale). Its biggest misses are
   event weeks (e.g. Aug 2024 sell-off, Indian election week Jun 2024) that past volatility cannot see.
2. **Shocks persist differently.** After a top-5% move, BTC's absolute returns stay
   ~{btc_s['ratio_to_baseline'].min():.1f}–{btc_s['ratio_to_baseline'].max():.1f}× normal for three weeks;
   NIFTY starts at {nif_s['ratio_to_baseline'].iloc[0]:.1f}× and decays to {nif_s['ratio_to_baseline'].iloc[-1]:.1f}×.
3. **The news-based stress index (GMSI) is at best weakly related to future volatility.** Low-stress quintiles
   preceded the highest volatility, but once autocorrelation is accounted for the correlation is borderline for BTC
   (p ≈ {bt['p_value']:.2f}) and not significant for NIFTY (p ≈ {nf['p_value']:.2f}). Treat the "complacency
   effect" as a hypothesis, not a finding.
""")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("BTC-USD close (log scale)", "NIFTY 50 close (log scale)"))
    for i, a in enumerate(data.ASSETS, 1):
        s = data.prices(a).loc["2015":]
        fig.add_trace(go.Scattergl(x=s.index, y=s.values, name=a, line=dict(color=COLORS[a], width=1.3)), row=i, col=1)
        fig.update_yaxes(type="log", row=i, col=1)
    show(style(fig, height=520, showlegend=False))

    with st.expander("What changed compared with earlier versions of this dashboard"):
        st.markdown("""
- Removed the MFI, shock-decay and "AC₁ paradox" results that had been generated from **simulated** GARCH data
  (a silent fallback in the old analysis script) and labelled as real. They are now recomputed on real prices;
  the regime statistics that need the (uncommitted) GMSI series were dropped.
- Removed a fabricated "null distribution" chart and p-values that no code produced; significance is now
  computed with a test that is valid for autocorrelated series.
- The old served "model" was a Random Forest trained on ~10 rows that predicted a constant; it was replaced by
  the walk-forward-validated HAR model on the forecast page.
- Details: `docs/ISSUES.md` in the repository.
""")
