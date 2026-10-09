import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import data
from views._ui import COLORS, show, style


def render():
    st.title("Shock propagation")
    st.markdown("A **shock** is a day whose absolute log return exceeds the 95th percentile of all *previous* days "
                "(expanding threshold, 250-day burn-in). The chart shows the mean absolute return h days later, "
                "divided by the average absolute return over the same period, with 95% confidence intervals.")
    fig = go.Figure()
    tables = []
    for a in data.ASSETS:
        sp = data.table("shocks", a, parse_dates=())
        base = sp["baseline_mean_abs_return"]
        fig.add_trace(go.Scatter(x=sp["horizon"], y=sp["ci_high"] / base, mode="lines", line=dict(width=0), showlegend=False,
                                 hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=sp["horizon"], y=sp["ci_low"] / base, mode="lines", fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(150,150,150,0.2)", showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=sp["horizon"], y=sp["ratio_to_baseline"], name=f"{a} ({int(sp['n_shocks'][0])} shocks)",
                                 mode="lines+markers", line=dict(color=COLORS[a], width=2)))
        tables.append(sp.assign(asset=a))
    fig.add_hline(y=1, line_dash="dash", annotation_text="normal day")
    show(style(fig, height=420, hovermode="x", xaxis_title="days after the shock",
               yaxis_title="mean |r| ÷ normal mean |r|"))
    t = pd.concat(tables)[["asset", "horizon", "n_shocks", "mean_abs_return", "ci_low", "ci_high",
                           "baseline_mean_abs_return", "ratio_to_baseline"]]
    st.dataframe(t.set_index(["asset", "horizon"]).style.format("{:.4f}", subset=t.columns[3:]), width="stretch")
    st.caption("Reading: BTC's elevated volatility after a shock barely decays over three weeks (volatility "
               "clustering is long-memory); NIFTY's starts higher relative to normal and fades steadily. "
               "Overlapping post-shock windows make neighbouring horizons correlated, so the CIs are approximate.")
