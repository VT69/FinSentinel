"""Small presentation helpers shared by the dashboard pages."""
import plotly.graph_objects as go
import streamlit as st

COLORS = {"BTC": "#f7931a", "NIFTY": "#4488ff", "accent": "#00c8f8", "muted": "#8ba3c4", "bad": "#ff4d6d",
          "good": "#00d68f"}


def kcard(label: str, value: str, sub: str = "") -> None:
    st.markdown(f"<div class='kcard'><div class='l'>{label}</div><div class='v'>{value}</div>"
                f"<div class='s'>{sub}</div></div>", unsafe_allow_html=True)


def style(fig: go.Figure, height: int = 380, **layout) -> go.Figure:
    base = dict(height=height, margin=dict(l=10, r=10, t=40, b=10), hovermode="x unified",
                legend=dict(orientation="h", y=1.08, x=0))
    fig.update_layout(**{**base, **layout})
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, width="stretch", config={"displaylogo": False})
