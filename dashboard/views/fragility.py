import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import data
from views._ui import COLORS, kcard, show, style

COMPONENTS = {"persistence_norm": "A · persistence (lag-1 autocorr. of |r|)",
              "vol_of_vol_norm": "B · vol-of-vol (CoV of 7-day vol)",
              "tail_freq_norm": "C · tail frequency (|r| > 2σ₃₀)"}


def render():
    st.title("Market Fragility Index (MFI)")
    st.markdown("**MFI = mean of three 30-day components**, each rescaled to [0, 1] by its own expanding (past-only) "
                "min and max. Higher = shocks have recently been more persistent, volatility less stable and large "
                "moves more frequent. Computed on real daily prices.")
    asset = st.selectbox("Asset", data.ASSETS)
    mfi = data.table("mfi", asset).set_index("date").loc["2016":]
    md = data.metrics()[asset]["market_dynamics"]
    c = st.columns(3)
    with c[0]:
        kcard("Highest MFI", f"{md['mfi_max']:.2f}", md["mfi_max_date"])
    with c[1]:
        kcard("Highest in March 2020", f"{md['mfi_2020_03_max']:.2f}", "COVID crash")
    with c[2]:
        kcard("Average since 2016", f"{md['mfi_mean_since_2016']:.2f}", "")

    fig = go.Figure(go.Scattergl(x=mfi.index, y=mfi["MFI"], name="MFI", line=dict(color=COLORS[asset], width=1)))
    show(style(fig, height=320, title=f"{asset} MFI", yaxis_range=[0, 1]))
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=list(COMPONENTS.values()))
    for i, col in enumerate(COMPONENTS, 1):
        fig.add_trace(go.Scattergl(x=mfi.index, y=mfi[col], line=dict(width=0.8)), row=i, col=1)
    show(style(fig, height=560, showlegend=False))
    st.caption("Caveats: expanding min-max scaling is unstable early in the sample, so peaks in the first years are "
               "partly a short-history effect. The MFI is descriptive; it has not been validated against an external "
               "fragility measure (e.g. VIX), and its weights are equal by assumption.")
