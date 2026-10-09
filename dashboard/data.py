"""
dashboard/data.py — every file the dashboard reads, behind Streamlit caches.

Nothing here computes results: models and result tables are produced offline by
`python run_pipeline.py` and committed. The only live computation is the
forecast, which calls pipeline.volatility.forecast — the same feature code the
model was trained with.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:          # Streamlit puts only dashboard/ on sys.path
    sys.path.insert(0, str(ROOT))

from pipeline import volatility as vol  # noqa: E402
from pipeline.data import DASHBOARD_DATA, FIGURES, MODELS, load_close  # noqa: E402

ASSETS = ("BTC", "NIFTY")
REBUILD_HINT = "Run `python run_pipeline.py` from the repo root to regenerate it."


class MissingArtifact(RuntimeError):
    pass


def _require(path: Path) -> Path:
    if not path.exists():
        raise MissingArtifact(f"`{path.relative_to(ROOT)}` is missing. {REBUILD_HINT}")
    return path


@st.cache_data(show_spinner=False)
def prices(asset: str) -> pd.Series:
    return load_close(asset)


@st.cache_resource(show_spinner=False)
def model(asset: str) -> vol.HARModel:
    return vol.HARModel.from_json(_require(MODELS / f"har_{asset.lower()}.json"))


@st.cache_data(show_spinner=False)
def metrics() -> dict:
    return json.loads(_require(DASHBOARD_DATA / "metrics.json").read_text())


@st.cache_data(show_spinner=False)
def gmsi_calibration() -> dict:
    return json.loads(_require(DASHBOARD_DATA / "gmsi_calibration.json").read_text())


@st.cache_data(show_spinner=False)
def table(name: str, asset: str, parse_dates=("date",)) -> pd.DataFrame:
    path = _require(DASHBOARD_DATA / f"{name}_{asset.lower()}.csv")
    return pd.read_csv(path, parse_dates=list(parse_dates) or False)


def figure(name: str) -> Path:
    return _require(FIGURES / name)
