import pandas as pd
from pathlib import Path

"""Keep a column subset of each GDELT event file:  python scripts/filter_gdelt_events.py"""
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "gdelt" / "events"
OUT_DIR = ROOT / "data" / "processed" / "gdelt_filtered"

COL_MAP = {
    "date": 1,
    "event_code": 26,
    "event_root": 28,
    "goldstein": 30,
    "actor1_country": 37,
    "actor2_country": 44,
    "avg_tone": 55,
}

def filter_daily_file(path):
    date = path.stem.replace("events_", "")
    print(f"🔄 Processing {date}")

    try:
        chunks = pd.read_csv(
            path,
            sep="\t",
            header=None,
            engine="python",
            on_bad_lines="skip",
            chunksize=200_000,
            low_memory=False
        )

        first = True
        out_file = OUT_DIR / f"{date}.csv"

        for chunk in chunks:
            max_col = chunk.shape[1] - 1
            valid = {k: v for k, v in COL_MAP.items() if v <= max_col}

            df = chunk[list(valid.values())].copy()
            df.columns = list(valid.keys())

            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            df.dropna(subset=["date"], inplace=True)

            df.to_csv(out_file, mode="w" if first else "a",
                      index=False, header=first)
            first = False

        print(f"  ✅ Saved {out_file.name}")

    except Exception as e:
        print(f"  ❌ Error {date}: {e}")


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(RAW_DIR.glob("*.CSV")) + sorted(RAW_DIR.glob("*.csv"))
    if not files:
        raise SystemExit(f"No event files in {RAW_DIR}; run scripts/download_gdelt.py first.")
    for f in files:
        filter_daily_file(f)
