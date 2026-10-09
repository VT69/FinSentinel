"""
FinSentinel dashboard — entry point for `streamlit run dashboard/app.py`.

Every number shown is either read from files produced by `python run_pipeline.py`
(real BTC-USD / NIFTY 50 prices in data/raw) or computed live by the served
volatility model. Nothing is simulated.
"""
import streamlit as st

st.set_page_config(page_title="FinSentinel", page_icon="📡", layout="wide",
                   initial_sidebar_state="expanded")

from data import MissingArtifact  # noqa: E402
from views import fragility, forecast, gmsi, methodology, overview, shocks  # noqa: E402

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1400px;}
.kcard {background: #131a26; border: 1px solid #1e2d45; border-left: 3px solid #00c8f8;
        border-radius: 8px; padding: 14px 16px; height: 100%;}
.kcard .l {font-size: 12px; letter-spacing: .03em; color: #8ba3c4;}
.kcard .v {font-size: 24px; font-weight: 600; margin: 4px 0;}
.kcard .s {font-size: 12px; color: #8ba3c4;}
</style>""", unsafe_allow_html=True)

PAGES = {
    "Overview": overview,
    "Volatility forecast": forecast,
    "GMSI & volatility": gmsi,
    "Market fragility (MFI)": fragility,
    "Shock propagation": shocks,
    "Methodology & limitations": methodology,
}

with st.sidebar:
    st.markdown("### 📡 FinSentinel")
    st.caption("Stress, fragility and volatility — BTC-USD & NIFTY 50, daily data 2015–2025")
    choice = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("All results are reproducible with `python run_pipeline.py`. "
               "Data: Yahoo Finance via yfinance (research use only).")

try:
    PAGES[choice].render()
except MissingArtifact as exc:
    st.error(f"A required data file is missing. {exc}")
