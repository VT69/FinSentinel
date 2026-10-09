"""
pipeline/volatility.py — the single source of truth for volatility features,
the forecasting target, and the served model.

Training (run_pipeline.py) and inference (dashboard/) both call
`price_features()`, so the features a model sees in production are computed by
exactly the same code it was trained on.

Target: σ_{t+1..t+h} = sample std of the next h daily log returns
        (h = 5 → "next-week volatility"), modelled in log space.
Model:  HAR-style log-linear regression on trailing 5/22/60-day realised vol
        (Corsi 2009, with a quarterly instead of daily component). It beat the
        Random Forest out of sample — see docs/MODEL_REPORT.md.

The model is stored as plain JSON (4 numbers + metadata), so loading it needs
neither scikit-learn nor pickle, and there is no pickle-version mismatch risk.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

EPS = 1e-6                     # log floor; daily σ is ~1e-3..1e-1, so it never binds in practice
HORIZON = 5                    # forecast window in trading days
WINDOWS = (5, 22, 60)          # ~1 week, 1 month, 1 quarter of trading days
MIN_CLOSES = max(WINDOWS) + 1  # 61 closes → 60 returns → one complete feature row
MAX_CLOSES = 10_000
ANNUALISATION_DAYS = {"BTC": 365, "NIFTY": 252}   # BTC trades every calendar day

HAR_FEATURES = [f"log_vol_{w}" for w in WINDOWS]
RF_FEATURES = ["abs_return", "sq_return"] + [f"vol_{w}" for w in WINDOWS]
TARGET = "log_target_vol"


class InputError(ValueError):
    """Raised for user-supplied data that cannot produce a forecast; message is user-facing."""


def annualise(daily_vol, asset: str):
    try:
        return daily_vol * math.sqrt(ANNUALISATION_DAYS[asset.upper()])
    except KeyError:
        raise ValueError(f"Unknown asset {asset!r}; expected one of {sorted(ANNUALISATION_DAYS)}") from None


def price_features(close: pd.Series, horizon: int = HORIZON, with_target: bool = True) -> pd.DataFrame:
    """Features (all trailing, i.e. known at the close of day t) and, optionally, the target.

    Rows are NOT dropped here; callers drop NaNs on exactly the columns they use.
    """
    close = pd.Series(close, dtype="float64")
    r = np.log(close / close.shift(1))
    out = pd.DataFrame(index=close.index)
    out["close"] = close
    out["log_return"] = r
    out["abs_return"] = r.abs()
    out["sq_return"] = r ** 2
    for w in WINDOWS:
        out[f"vol_{w}"] = r.rolling(w).std()
        out[f"log_vol_{w}"] = np.log(out[f"vol_{w}"] + EPS)
    if with_target:
        out["target_vol"] = r.rolling(horizon).std().shift(-horizon)   # std of r[t+1 .. t+h]
        out[TARGET] = np.log(out["target_vol"] + EPS)
    return out


def training_frame(close: pd.Series, horizon: int = HORIZON) -> pd.DataFrame:
    f = price_features(close, horizon=horizon, with_target=True)
    return f.dropna(subset=HAR_FEATURES + RF_FEATURES + [TARGET])


@dataclass
class HARModel:
    asset: str
    intercept: float
    coef: dict
    horizon: int = HORIZON
    windows: tuple = WINDOWS
    trained_from: str = ""
    trained_through: str = ""
    n_train: int = 0
    metrics: dict = field(default_factory=dict)

    @classmethod
    def fit(cls, frame: pd.DataFrame, asset: str) -> "HARModel":
        X = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy() for c in HAR_FEATURES])
        beta, *_ = np.linalg.lstsq(X, frame[TARGET].to_numpy(), rcond=None)
        idx = frame.index
        return cls(asset=asset.upper(), intercept=float(beta[0]),
                   coef={c: float(b) for c, b in zip(HAR_FEATURES, beta[1:])},
                   trained_from=str(idx[0])[:10], trained_through=str(idx[-1])[:10], n_train=len(frame))

    def predict_log(self, frame: pd.DataFrame) -> np.ndarray:
        missing = [c for c in HAR_FEATURES if c not in frame]
        if missing:
            raise KeyError(f"missing feature columns: {missing}")
        return self.intercept + sum(frame[c].to_numpy() * self.coef[c] for c in HAR_FEATURES)

    def predict_vol(self, frame: pd.DataFrame) -> np.ndarray:
        # exp of a log forecast is the conditional *median*, not mean; we report it as such.
        return np.exp(self.predict_log(frame)) - EPS

    def to_json(self, path: Path) -> None:
        d = asdict(self); d["windows"] = list(self.windows); d["model_type"] = "HAR-log-linear"
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def from_json(cls, path: Path) -> "HARModel":
        d = json.loads(Path(path).read_text())
        if d.get("model_type") != "HAR-log-linear":
            raise ValueError(f"{path} is not a HAR model file")
        if set(d["coef"]) != set(HAR_FEATURES) or tuple(d["windows"]) != WINDOWS or d["horizon"] != HORIZON:
            raise ValueError(f"{path} was trained with a different feature contract "
                             f"(coef={sorted(d['coef'])}, windows={d['windows']}, horizon={d['horizon']}); "
                             "retrain with `python run_pipeline.py`")
        d.pop("model_type")
        d["windows"] = tuple(d["windows"])
        return cls(**d)


def parse_closes(text: str) -> np.ndarray:
    """Parse user-pasted closing prices (comma / whitespace / newline separated, oldest first)."""
    if text is None or not text.strip():
        raise InputError("Please paste at least %d closing prices (oldest first)." % MIN_CLOSES)
    tokens = [t for t in text.replace(",", " ").replace(";", " ").split() if t]
    values = []
    for i, t in enumerate(tokens, 1):
        try:
            values.append(float(t))
        except ValueError:
            raise InputError(f"Value #{i} ('{t[:20]}') is not a number.") from None
    return validate_closes(values)


def validate_closes(values) -> np.ndarray:
    a = np.asarray(values, dtype="float64")
    if a.ndim != 1:
        raise InputError("Closing prices must be a single list of numbers.")
    if len(a) < MIN_CLOSES:
        raise InputError(f"Need at least {MIN_CLOSES} closing prices (60-day volatility window + 1); got {len(a)}.")
    if len(a) > MAX_CLOSES:
        raise InputError(f"Too many values ({len(a)}); at most {MAX_CLOSES} are accepted.")
    if not np.all(np.isfinite(a)):
        raise InputError("Prices must be finite numbers (no NaN / inf).")
    if np.any(a <= 0):
        raise InputError("Prices must be strictly positive (log returns are undefined otherwise).")
    return a


def forecast(model: HARModel, close: pd.Series) -> dict:
    """Forecast σ for the h days after the last close in `close` (oldest→newest)."""
    s = close.astype("float64") if isinstance(close, pd.Series) else pd.Series(np.asarray(close, dtype="float64"))
    validate_closes(s.to_numpy())
    f = price_features(s, with_target=False)
    last = f.iloc[[-1]]
    if last[HAR_FEATURES].isna().any(axis=None):
        raise InputError("Not enough consecutive valid prices to compute the 60-day window.")
    daily = float(model.predict_vol(last)[0])
    max_abs_ret = float(f["abs_return"].max())
    return {"daily_vol": daily, "annualised_vol": float(annualise(daily, model.asset)),
            "horizon_days": model.horizon,
            "inputs": {c: float(np.exp(last[c].iloc[0]) - EPS) for c in HAR_FEATURES},
            "warning": (f"Largest daily move in your input is {max_abs_ret:.0%} — check for data errors "
                        "(splits, typos)." if max_abs_ret > 0.5 else None)}
