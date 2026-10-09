import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
from views._ui import COLORS, show, style


def render():
    st.title("GMSI and forward volatility")
    cal = data.gmsi_calibration()
    st.markdown("""
The **Global Market Stress Index (GMSI)** is an equal-weight average of expanding (past-only) z-scores of daily
GDELT event signals — event count, share of negative events, inverted Goldstein score — plus headline sentiment and
an attention proxy. It contains **no price data**. (Sentiment exists only for Oct 2024 – Jan 2025, so on most
days the index is driven by the three GDELT components.)

The GMSI series itself is built from GDELT files that are not committed to the repository, so the two charts
below are from the original analysis run; the significance numbers underneath are recomputed here.
""")
    c = st.columns(2)
    for col, a in zip(c, data.ASSETS):
        with col:
            st.image(str(data.figure(f"cond_exp_{a}.png")), width="stretch")
    st.caption("Mean forward 7- and 14-day volatility by GMSI quintile (full-sample quintiles: an in-sample "
               "description, not a forecast). The lowest-stress quintile Q1 has the highest forward volatility; "
               "Q2–Q5 are roughly flat, not a monotonic decline.")

    st.subheader("Is the relationship statistically significant?")
    rows = []
    for a in data.ASSETS:
        r = cal["assets"][a]
        rows.append({"asset": a, "Spearman ρ (published)": r["statistic"],
                     "p if days were independent": r["p_value_if_iid"],
                     f"p, AR(1) null φ={r['phi']}": r["p_value"],
                     "variance inflation": r["variance_inflation"],
                     "forward-vol lag-1 autocorr.": r["fwd_vol_lag1_acf"]})
    st.dataframe(pd.DataFrame(rows).set_index("asset").style.format(
        {"Spearman ρ (published)": "{:+.4f}", "p if days were independent": "{:.1e}",
         f"p, AR(1) null φ={cal['assumed_gmsi_lag1_acf']}": "{:.3f}", "variance inflation": "{:.1f}×",
         "forward-vol lag-1 autocorr.": "{:.3f}"}), width="stretch")
    st.caption(f"Both GMSI (lag-1 autocorrelation ≈ {cal['assumed_gmsi_lag1_acf']}) and forward volatility are highly "
               "persistent, so neighbouring days are not independent evidence. The AR(1) null simulates an "
               "independent index with the same persistence; its p-value is still a lower bound because the real "
               "GMSI autocorrelation decays more slowly than an AR(1).")

    fpr = pd.DataFrame(cal["false_positive_rate"])
    fig = go.Figure()
    fig.add_bar(x=[f"φ = {p}" for p in fpr["phi"]], y=fpr["iid_shuffle"], name="i.i.d. shuffle (original test)",
                marker_color=COLORS["bad"])
    fig.add_bar(x=[f"φ = {p}" for p in fpr["phi"]], y=fpr["circular_shift"], name="circular shift (corrected test)",
                marker_color=COLORS["good"])
    fig.add_hline(y=0.05, line_dash="dash", annotation_text="nominal 5%")
    show(style(fig, height=340, barmode="group", hovermode="x", yaxis_tickformat=".0%",
               title="How often each placebo test calls an UNRELATED persistent index 'significant' "
                     "(real BTC forward vol)"))
    st.info("Bottom line: the original placebo test shuffled days independently, which manufactures significance "
            "for persistent series. With a valid null the BTC correlation is borderline and the NIFTY one is not "
            "significant. The idea that calm news flow precedes volatility (\"complacency\") is a hypothesis "
            "worth testing on more data, not an established result.")
    with st.expander("GMSI sanity checks (original run)"):
        st.image(str(data.figure("gmsi_sanity_checks.png")), width="stretch")
