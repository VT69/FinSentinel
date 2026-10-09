"""
analysis/data_profile.py — exhaustive, reproducible profile of every dataset
committed to this repository.

Run from the repo root:
    pip install -r analysis/requirements.txt
    python analysis/data_profile.py

Everything printed here is also written to analysis/outputs/ (CSV / TXT / PNG,
plus profile_summary.json). docs/DATA_REPORT.md quotes these outputs.

What CAN'T be profiled (not committed / not reachable): everything under
data/processed/ (gitignored), the GDELT events files used to build GMSI, and
FinBERT scores (model download blocked in the analysis sandbox). Where a
FinBERT value is needed to reproduce row counts, VADER is used as a clearly
labelled stand-in; no FinBERT statistic is reported.
"""
from __future__ import annotations

import csv
import io
import json
import math
import sys
import unicodedata
from contextlib import redirect_stdout

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import (OUT, RAW, SEED, PRICE_FEATURES, align_btc, build_features,  # noqa: E402
                    load_prices, load_text, load_trends, preprocess_text,
                    price_feature_frame, vader_scores)

np.random.seed(SEED)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)
SUMMARY: dict = {}

SEMANTICS = {
    # prices (notebooks/01_data_collection.ipynb cell 3)
    "date": ("Trading/calendar day of the bar (yfinance index)", "high"),
    "open": ("Opening price of the day, quote currency (USD for BTC, INR index points for NIFTY)", "high"),
    "high": ("Intraday high", "high"),
    "low": ("Intraday low", "high"),
    "close": ("Close price (yfinance default auto_adjust; index has no dividends)", "high"),
    "volume": ("Traded volume (USD notional for BTC-USD; NIFTY index volume as reported by Yahoo)", "medium"),
    "return": ("Simple return close.pct_change() — NOT log; notebooks later overwrite it with log return", "high"),
    # text (notebooks/01a_text_data_collection.ipynb)
    "timestamp": ("GDELT 'seendate' of the headline, UTC", "high"),
    "text": ("Headline title only (no body)", "high"),
    "source": ("Intended: publisher name. Empty in the committed file", "high"),
    "asset": ("Which query produced the headline: BTC or NIFTY", "high"),
    "channel": ("Collection channel label", "high"),
    # trends
    "google_trend": ("Google Trends interest, 0-100, relative to the peak month of the queried window", "high"),
}


def section(title):
    print("\n" + "=" * 100 + f"\n{title}\n" + "=" * 100)


def save_df(df, name, index=True):
    df.to_csv(OUT / f"{name}.csv", index=index)


# ═════════════════════════════════════════════════════════════════════════════
# 1. File-level metadata
# ═════════════════════════════════════════════════════════════════════════════

def file_meta():
    section("1. FILE METADATA (data/raw)")
    rows = []
    for p in sorted(RAW.iterdir()):
        b = p.read_bytes()
        try:
            b.decode("ascii"); enc = "ascii (subset of utf-8)"
        except UnicodeDecodeError:
            try:
                b.decode("utf-8"); enc = "utf-8 (non-ASCII present)"
            except UnicodeDecodeError:
                enc = "not utf-8"
        delim, header = None, None
        if p.suffix == ".csv":
            sample = b[:4096].decode("utf-8", errors="replace")
            delim = repr(csv.Sniffer().sniff(sample).delimiter)
            header = sample.splitlines()[0]
        rows.append({"file": p.name, "bytes": len(b), "lines": b.count(b"\n") + (0 if b.endswith(b"\n") else 1),
                     "encoding": enc, "delimiter": delim, "header": header})
    meta = pd.DataFrame(rows)
    print(meta.to_string(index=False))
    save_df(meta, "file_metadata", index=False)
    SUMMARY["files"] = meta.to_dict(orient="records")


# ═════════════════════════════════════════════════════════════════════════════
# 2. Structure / inventory
# ═════════════════════════════════════════════════════════════════════════════

def inventory(df, name):
    n = len(df)
    rows = []
    for c in df.columns:
        nn = int(df[c].notna().sum())
        nu = int(df[c].nunique(dropna=True))
        ex = df[c].dropna().astype(str).unique()[:3]
        sem, conf = SEMANTICS.get(c, ("(derived)", "n/a"))
        rows.append({"column": c, "dtype": str(df[c].dtype), "non_null": nn, "null": n - nn,
                     "null_pct": round(100 * (n - nn) / n, 2) if n else None, "n_unique": nu,
                     "cardinality_ratio": round(nu / n, 4) if n else None,
                     "examples": " | ".join(e[:40] for e in ex), "meaning": sem, "confidence": conf})
    inv = pd.DataFrame(rows)
    save_df(inv, f"inventory_{name}", index=False)
    return inv


def structure(df, name):
    section(f"2. STRUCTURE — {name}")
    mem = df.memory_usage(deep=True).sum()
    print(f"shape: {df.shape[0]} rows x {df.shape[1]} cols | memory (deep): {mem/1024:.1f} KiB")
    buf = io.StringIO()
    df.info(buf=buf)
    print(buf.getvalue())
    (OUT / f"info_{name}.txt").write_text(buf.getvalue())
    inv = inventory(df, name)
    print(inv.drop(columns=["examples"]).to_string(index=False))
    SUMMARY.setdefault("shapes", {})[name] = {"rows": df.shape[0], "cols": df.shape[1],
                                              "memory_kib": round(mem / 1024, 1)}


# ═════════════════════════════════════════════════════════════════════════════
# 3. Numeric profile
# ═════════════════════════════════════════════════════════════════════════════

def numeric_profile(df, name):
    section(f"3. NUMERIC PROFILE — {name}")
    rows = []
    for c in df.select_dtypes(include="number").columns:
        s = df[c].dropna().astype(float)
        if s.empty:
            continue
        q1, q2, q3 = s.quantile([.25, .5, .75])
        iqr = q3 - q1
        z = (s - s.mean()) / s.std(ddof=1) if s.std(ddof=1) > 0 else s * 0
        mode = s.mode()
        rows.append({
            "column": c, "count": len(s), "mean": s.mean(), "median": q2,
            "mode": mode.iloc[0] if len(mode) else np.nan, "mode_count": int((s == mode.iloc[0]).sum()) if len(mode) else 0,
            "std": s.std(ddof=1), "var": s.var(ddof=1), "min": s.min(), "max": s.max(), "range": s.max() - s.min(),
            "q1": q1, "q2": q2, "q3": q3, "iqr": iqr,
            "p01": s.quantile(.01), "p05": s.quantile(.05), "p95": s.quantile(.95), "p99": s.quantile(.99),
            "skew": stats.skew(s, bias=False), "kurtosis_excess": stats.kurtosis(s, bias=False),
            "cv": s.std(ddof=1) / s.mean() if s.mean() != 0 else np.nan,
            "zeros": int((s == 0).sum()), "negatives": int((s < 0).sum()),
            "outliers_iqr_1.5": int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum()),
            "outliers_abs_z_gt_3": int((z.abs() > 3).sum()),
        })
    prof = pd.DataFrame(rows).set_index("column")
    with pd.option_context("display.float_format", lambda v: f"{v:.6g}"):
        print(prof.T.to_string())
    save_df(prof, f"numeric_profile_{name}")
    SUMMARY.setdefault("numeric", {})[name] = json.loads(prof.to_json())
    return prof


def price_plausibility(df, asset):
    section(f"3b. PLAUSIBILITY CHECKS — {asset} prices")
    d = df.copy()
    out = {}
    out["date_min"] = str(d["date"].min().date()); out["date_max"] = str(d["date"].max().date())
    out["future_dates_vs_2026-10-09"] = int((d["date"] > pd.Timestamp("2026-10-09")).sum())
    out["duplicate_dates"] = int(d["date"].duplicated().sum())
    out["nonpositive_prices"] = int((d[["open", "high", "low", "close"]] <= 0).any(axis=1).sum())
    out["high_lt_low"] = int((d["high"] < d["low"]).sum())
    out["close_outside_low_high"] = int(((d["close"] > d["high"] + 1e-9) | (d["close"] < d["low"] - 1e-9)).sum())
    out["open_outside_low_high"] = int(((d["open"] > d["high"] + 1e-9) | (d["open"] < d["low"] - 1e-9)).sum())
    out["volume_zero_rows"] = int((d["volume"] == 0).sum())
    if out["volume_zero_rows"]:
        z = d.loc[d["volume"] == 0, "date"]
        out["volume_zero_first_last"] = [str(z.min().date()), str(z.max().date())]
        out["volume_zero_by_year"] = {int(k): int(v) for k, v in z.dt.year.value_counts().sort_index().items()}
    # stored 'return' vs recomputed pct_change on the same file
    recomputed = d["close"].pct_change()
    diff = (d["return"] - recomputed).abs()
    out["stored_return_vs_pct_change_maxabsdiff_excl_first"] = float(diff.iloc[1:].max())
    out["stored_return_first_row"] = float(d["return"].iloc[0])
    gaps = d["date"].diff().dt.days.value_counts().sort_index()
    out["day_gap_histogram"] = {int(k): int(v) for k, v in gaps.items()}
    full = pd.date_range(d["date"].min(), d["date"].max(), freq="D")
    out["calendar_days_missing"] = int(len(full) - d["date"].nunique())
    if asset == "BTC":
        miss = full.difference(d["date"])
        out["btc_missing_dates_sample"] = [str(x.date()) for x in miss[:10]]
    lr = np.log(d["close"]).diff().abs()
    top = d.assign(abs_log_ret=lr).nlargest(5, "abs_log_ret")[["date", "close", "abs_log_ret"]]
    out["largest_abs_log_returns"] = [{"date": str(r.date.date()), "close": round(r.close, 2),
                                       "abs_log_ret": round(r.abs_log_ret, 4)} for r in top.itertuples()]
    out["rows_per_year"] = {int(k): int(v) for k, v in d["date"].dt.year.value_counts().sort_index().items()}
    for k, v in out.items():
        print(f"  {k}: {v}")
    SUMMARY.setdefault("plausibility", {})[asset] = out


# ═════════════════════════════════════════════════════════════════════════════
# 4. Categorical + text profile
# ═════════════════════════════════════════════════════════════════════════════

def entropy_bits(vc):
    p = vc / vc.sum()
    return float(-(p * np.log2(p)).sum())


def categorical_profile(df, cols, name):
    section(f"4. CATEGORICAL PROFILE — {name}")
    res = {}
    for c in cols:
        s = df[c]
        raw = s.astype("string")
        vc = s.value_counts(dropna=False)
        rel = (vc / len(s)).round(4)
        nonnull = s.dropna()
        variants = {}
        if len(nonnull):
            norm = nonnull.astype(str).str.strip().str.lower()
            for k, g in nonnull.astype(str).groupby(norm):
                u = sorted(set(g))
                if len(u) > 1:
                    variants[k] = u
        na_like = int(raw.str.strip().str.upper().isin(["", "N/A", "NA", "NAN", "NONE", "NULL"]).sum())
        info = {
            "value_counts": {str(k): int(v) for k, v in vc.items()},
            "relative_freq": {str(k): float(v) for k, v in rel.items()},
            "mode": str(nonnull.mode().iloc[0]) if len(nonnull) else None,
            "entropy_bits": entropy_bits(nonnull.value_counts()) if len(nonnull) else None,
            "n_categories": int(nonnull.nunique()),
            "rare_lt_1pct": int((nonnull.value_counts(normalize=True) < 0.01).sum()) if len(nonnull) else 0,
            "case_or_whitespace_variants": variants,
            "na_like_strings": na_like, "true_NaN": int(s.isna().sum()),
        }
        res[c] = info
        print(f"\n[{c}]")
        for k, v in info.items():
            print(f"  {k}: {v}")
    SUMMARY.setdefault("categorical", {})[name] = res
    return res


SCRIPT_NAMES = ["LATIN", "DEVANAGARI", "HEBREW", "ARABIC", "CJK", "HANGUL", "CYRILLIC", "BENGALI",
                "TAMIL", "TELUGU", "GUJARATI", "KANNADA", "MALAYALAM", "GURMUKHI", "THAI", "GREEK",
                "HIRAGANA", "KATAKANA", "ORIYA"]


def dominant_script(text):
    counts = {}
    for ch in str(text):
        if ch.isalpha():
            name = unicodedata.name(ch, "")
            sc = next((s for s in SCRIPT_NAMES if name.startswith(s) or s in name.split()), "OTHER")
            counts[sc] = counts.get(sc, 0) + 1
    return max(counts, key=counts.get) if counts else "NONE"


def text_profile(text):
    section("4b. TEXT PROFILE — data/raw/text_data.csv")
    t = text.copy()
    t["ts"] = pd.to_datetime(t["timestamp"], utc=True)
    out = {"rows": len(t), "ts_min": str(t["ts"].min()), "ts_max": str(t["ts"].max()),
           "distinct_utc_days": int(t["ts"].dt.date.nunique())}
    t["script"] = t["text"].apply(dominant_script)
    out["dominant_script_counts"] = t["script"].value_counts().to_dict()
    out["dominant_script_by_asset"] = {a: g["script"].value_counts().to_dict() for a, g in t.groupby("asset")}
    t["len_chars"] = t["text"].astype(str).str.len()
    out["headline_len_chars"] = t["len_chars"].describe().round(2).to_dict()
    out["exact_duplicate_rows"] = int(t.duplicated(subset=["timestamp", "text", "asset"]).sum())
    out["duplicate_text_any_time"] = int(t.duplicated(subset=["text"]).sum())
    pp = preprocess_text(text)
    out["rows_after_clean_len_gt_10 (nb02 reports 1377)"] = len(pp)
    out["rows_dropped_by_clean_filter"] = len(text) - len(pp)
    dropped = text.loc[~text.index.isin(pp.index)].copy()
    dropped["script"] = dropped["text"].apply(dominant_script)
    out["dropped_rows_by_script"] = dropped["script"].value_counts().to_dict()
    kept = text.loc[pp.index].copy()
    kept["script"] = kept["text"].apply(dominant_script)
    out["kept_non_latin_rows (garbled by ascii-only regex)"] = int((kept["script"] != "LATIN").sum())
    out["near_duplicate_clean_text"] = int(pp.duplicated(subset=["clean_text"]).sum())
    per_day = pd.to_datetime(pp["timestamp"], utc=True).dt.date
    pdc = pp.groupby([per_day, "asset"]).size().unstack(fill_value=0)
    out["headlines_per_day_per_asset"] = pdc.describe().round(2).to_dict()
    out["days_with_headlines"] = {a: int((pdc[a] > 0).sum()) for a in pdc.columns}
    save_df(pdc, "text_headlines_per_day")
    for k, v in out.items():
        print(f"  {k}: {v}")
    SUMMARY["text"] = out
    return pp


# ═════════════════════════════════════════════════════════════════════════════
# 5. Reconstruct the ML training table + target deep dive + baseline
# ═════════════════════════════════════════════════════════════════════════════

def reconstruct_training_table(pp):
    section("5. RECONSTRUCTED TRAINING TABLE (nb02 → nb03 → nb04 → pipeline/feature_engineering.py)")
    sent = pp.copy()
    sent["vader_score"] = vader_scores(sent["clean_text"])
    sent["finbert_score"] = sent["vader_score"]   # STAND-IN: FinBERT not reproducible here (see module docstring)
    btc = load_prices("BTC")
    aligned = align_btc(btc, sent)
    X, y, feats, full = build_features(aligned.copy(), target_horizon=5)
    n_test = math.ceil(0.2 * len(X))   # sklearn train_test_split(test_size=0.2) uses ceil for the test fold
    info = {
        "vader_on_clean_text": sent["vader_score"].describe().round(4).to_dict(),
        "aligned_rows (nb05 printed BTC shape (75, 15))": len(aligned),
        "aligned_date_range": [str(aligned["date"].min().date()), str(aligned["date"].max().date())],
        "rows_after_build_features_dropna": len(X),
        "train_rows": len(X) - n_test, "test_rows": n_test,
        "train_dates": [str(full["date"].iloc[0].date()), str(full["date"].iloc[len(X) - n_test - 1].date())] if len(X) else None,
        "test_dates": [str(full["date"].iloc[len(X) - n_test].date()), str(full["date"].iloc[-1].date())] if len(X) else None,
        "why_so_few": "rolling(60) in pipeline/feature_engineering.py:17 consumes 59 of the aligned rows; "
                      "shift(-5) target consumes the last 5",
    }
    for k, v in info.items():
        print(f"  {k}: {v}")
    full.to_csv(OUT / "btc_training_table_reconstructed.csv", index=False)
    aligned.to_csv(OUT / "btc_sentiment_aligned_reconstructed.csv", index=False)
    SUMMARY["training_table"] = info

    # Target deep dive on the actual training target
    section("5b. TARGET — log_vol_target in the reconstructed training table")
    print(y.describe().to_string())
    yv = np.exp(y.values) - 1e-6
    ytr, yte = y.iloc[:-n_test], y.iloc[-n_test:]
    pred = np.full(len(yte), ytr.mean())
    base = {"dummy_mean_rmse_vol_space": float(np.sqrt(np.mean((np.exp(pred) - np.exp(yte)) ** 2))),
            "dummy_mean_mae_vol_space": float(np.mean(np.abs(np.exp(pred) - np.exp(yte)))),
            "y_train_values": [round(v, 5) for v in ytr.tolist()], "y_test_values": [round(v, 5) for v in yte.tolist()],
            "vol_space_min_max": [float(yv.min()), float(yv.max())]}
    for k, v in base.items():
        print(f"  {k}: {v}")
    SUMMARY["training_target_baseline"] = base
    return aligned


def research_target(asset):
    section(f"5c. FULL-HISTORY TARGET — {asset} 5-day realised vol, 5 days ahead (same definition as pipeline)")
    f = price_feature_frame(load_prices(asset))
    out = {}
    for col in ["vol_target", "log_vol_target"]:
        s = f[col]
        jb = stats.jarque_bera(s)
        out[col] = {"n": len(s), "mean": s.mean(), "median": s.median(), "std": s.std(),
                    "skew": stats.skew(s), "excess_kurtosis": stats.kurtosis(s),
                    "jarque_bera_stat": float(jb.statistic), "jarque_bera_p": float(jb.pvalue)}
    for k, v in out.items():
        print(f"  {k}: { {a: (round(b, 6) if isinstance(b, float) else b) for a, b in v.items()} }")
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6))
    sns.histplot(f["vol_target"], kde=True, ax=ax[0], bins=60); ax[0].set_title(f"{asset}: vol_target (raw)")
    sns.histplot(f["log_vol_target"], kde=True, ax=ax[1], bins=60); ax[1].set_title(f"{asset}: log(vol_target)")
    plt.tight_layout(); plt.savefig(OUT / f"target_distribution_{asset.lower()}.png", dpi=90); plt.close()
    SUMMARY.setdefault("research_target", {})[asset] = out
    return f


# ═════════════════════════════════════════════════════════════════════════════
# 6. Relationships
# ═════════════════════════════════════════════════════════════════════════════

def relationships(f, asset):
    section(f"6. RELATIONSHIPS — {asset} price features vs target (full history)")
    from sklearn.feature_selection import mutual_info_regression
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    cols = PRICE_FEATURES + ["log_vol_target"]
    pear = f[cols].corr("pearson"); spear = f[cols].corr("spearman")
    print("Pearson:\n", pear.round(3).to_string()); print("Spearman:\n", spear.round(3).to_string())
    save_df(pear, f"corr_pearson_{asset.lower()}"); save_df(spear, f"corr_spearman_{asset.lower()}")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    sns.heatmap(pear, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[0]); ax[0].set_title(f"{asset} Pearson")
    sns.heatmap(spear, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax[1]); ax[1].set_title(f"{asset} Spearman")
    plt.tight_layout(); plt.savefig(OUT / f"corr_heatmap_{asset.lower()}.png", dpi=90); plt.close()
    pairs = (pear.loc[PRICE_FEATURES, PRICE_FEATURES].where(np.triu(np.ones((5, 5)), 1).astype(bool))
             .stack().abs().sort_values(ascending=False))
    print("Top |Pearson| feature pairs:\n", pairs.round(3).head(6).to_string())
    X = f[PRICE_FEATURES].assign(const=1.0)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(len(PRICE_FEATURES))], index=PRICE_FEATURES)
    print("VIF:\n", vif.round(2).to_string())
    mi = pd.Series(mutual_info_regression(f[PRICE_FEATURES], f["log_vol_target"], random_state=SEED), index=PRICE_FEATURES)
    rank = pd.DataFrame({"pearson_with_target": pear.loc[PRICE_FEATURES, "log_vol_target"],
                         "spearman_with_target": spear.loc[PRICE_FEATURES, "log_vol_target"],
                         "mutual_info": mi}).sort_values("mutual_info", ascending=False)
    print("Feature ↔ target ranking:\n", rank.round(4).to_string())
    save_df(rank, f"feature_target_ranking_{asset.lower()}")
    SUMMARY.setdefault("relationships", {})[asset] = {
        "top_pairs": {f"{a}~{b}": round(v, 4) for (a, b), v in pairs.head(6).items()},
        "vif": vif.round(3).to_dict(), "ranking": json.loads(rank.round(5).to_json())}


# ═════════════════════════════════════════════════════════════════════════════
# 7. Missingness across the daily calendar the project actually joins on
# ═════════════════════════════════════════════════════════════════════════════

def missingness(text_pp):
    section("7. MISSINGNESS — joined daily calendar 2015-01-01 … 2025-12-31")
    cal = pd.DataFrame(index=pd.date_range("2015-01-01", "2025-12-31", freq="D"))
    btc = load_prices("BTC").set_index("date"); nif = load_prices("NIFTY").set_index("date")
    cal["btc_close"] = btc["close"]; cal["nifty_close"] = nif["close"]
    ts = pd.to_datetime(text_pp["timestamp"], utc=True).dt.tz_localize(None).dt.normalize()
    for a in ["BTC", "NIFTY"]:
        cnt = ts[text_pp["asset"] == a].value_counts()
        cal[f"{a.lower()}_headlines"] = cnt.reindex(cal.index)
    for a in ["BTC", "NIFTY"]:
        g = load_trends(a).drop_duplicates()   # nifty file repeats 2025-11/2025-12 (see DUPLICATES)
        g["date"] = pd.to_datetime(g["date"] + "-01")
        cal[f"{a.lower()}_trend_monthly"] = g.set_index("date")["google_trend"].reindex(cal.index)
    miss = cal.isna()
    pct = (100 * miss.mean()).round(2)
    print("% missing per column:\n", pct.to_string())
    co = miss.astype(int).T.dot(miss.astype(int))
    print("Co-missing day counts:\n", co.to_string())
    print("Missing-indicator correlation:\n", miss.astype(int).corr().round(3).to_string())
    nif_missing_dow = cal.index[miss["nifty_close"]].dayofweek.value_counts().sort_index()
    print("NIFTY-missing days by weekday (0=Mon):", nif_missing_dow.to_dict())
    wk = miss.resample("W").mean()   # weekly share of missing days: avoids pixel aliasing of 4k daily columns
    fig, ax = plt.subplots(figsize=(11, 3.4))
    im = ax.imshow(wk.T.values, aspect="auto", interpolation="nearest", cmap="Greys", vmin=0, vmax=1)
    ax.set_yticks(range(len(cal.columns))); ax.set_yticklabels(cal.columns, fontsize=8)
    yrs = [i for i, d in enumerate(wk.index) if i == 0 or d.year != wk.index[i - 1].year]
    ax.set_xticks(yrs); ax.set_xticklabels([wk.index[i].year for i in yrs], fontsize=8)
    plt.colorbar(im, ax=ax, label="share of days missing in week")
    ax.set_title("Missingness matrix — weekly share of missing days (black = all missing)")
    plt.tight_layout(); plt.savefig(OUT / "missingness_matrix.png", dpi=90); plt.close()
    SUMMARY["missingness"] = {"pct_missing": pct.to_dict(), "nifty_missing_by_weekday": {int(k): int(v) for k, v in nif_missing_dow.items()}}


# ═════════════════════════════════════════════════════════════════════════════
# 8. Visuals
# ═════════════════════════════════════════════════════════════════════════════

def numeric_plots(df, name):
    cols = [c for c in df.select_dtypes(include="number").columns]
    fig, axes = plt.subplots(len(cols), 2, figsize=(11, 2.3 * len(cols)))
    for i, c in enumerate(cols):
        s = df[c].dropna()
        sns.histplot(s, kde=True, bins=60, ax=axes[i, 0]); axes[i, 0].set_title(f"{name}.{c} hist+KDE", fontsize=9)
        sns.boxplot(x=s, ax=axes[i, 1]); axes[i, 1].set_title(f"{name}.{c} boxplot", fontsize=9)
    plt.tight_layout(); plt.savefig(OUT / f"numeric_dist_{name}.png", dpi=80); plt.close()


def categorical_plots(text):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    text["asset"].value_counts().plot.bar(ax=axes[0], title="asset")
    text["channel"].value_counts().plot.bar(ax=axes[1], title="channel")
    text["text"].apply(dominant_script).value_counts().plot.bar(ax=axes[2], title="dominant script of headline")
    plt.tight_layout(); plt.savefig(OUT / "categorical_bars_text.png", dpi=90); plt.close()


# ═════════════════════════════════════════════════════════════════════════════

def main():
    log = io.StringIO()

    class Tee(io.TextIOBase):
        def write(self, s):
            sys.__stdout__.write(s); log.write(s); return len(s)

    with redirect_stdout(Tee()):
        file_meta()
        btc, nif, text = load_prices("BTC"), load_prices("NIFTY"), load_text()
        gt_b, gt_n = load_trends("BTC"), load_trends("NIFTY")
        for df, nm in [(btc, "btc_prices"), (nif, "nifty_prices"), (text, "text_data"),
                       (gt_b, "btc_google_trends"), (gt_n, "nifty_google_trends")]:
            structure(df, nm)
        for df, nm in [(btc, "btc_prices"), (nif, "nifty_prices"), (gt_b, "btc_google_trends"), (gt_n, "nifty_google_trends")]:
            numeric_profile(df, nm)
        price_plausibility(btc, "BTC"); price_plausibility(nif, "NIFTY")
        categorical_profile(text, ["source", "asset", "channel"], "text_data")
        pp = text_profile(text)
        reconstruct_training_table(pp)
        for a in ["BTC", "NIFTY"]:
            relationships(research_target(a), a)
        missingness(pp)
        section("8. DUPLICATES")
        dup = {nm: {"exact_duplicate_rows": int(df.duplicated().sum()),
                    "duplicate_dates": int(df["date"].duplicated().sum()) if "date" in df else None}
               for df, nm in [(btc, "btc_prices"), (nif, "nifty_prices"), (gt_b, "btc_trends"), (gt_n, "nifty_trends")]}
        dup["text_data"] = {"exact_duplicate_rows": int(text.duplicated().sum()),
                            "duplicate_text_values": int(text["text"].duplicated().sum())}
        print(dup); SUMMARY["duplicates"] = dup
        numeric_plots(btc, "btc_prices"); numeric_plots(nif, "nifty_prices")
        categorical_plots(text)
    (OUT / "data_profile_stdout.txt").write_text(log.getvalue())
    (OUT / "profile_summary.json").write_text(json.dumps(SUMMARY, indent=2, default=str))
    print(f"\nWrote outputs to {OUT}")


if __name__ == "__main__":
    main()
