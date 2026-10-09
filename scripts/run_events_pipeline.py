import os
import requests
import zipfile
import io
import pandas as pd
from datetime import datetime, timedelta

# ==========================
# CONFIG
# ==========================
# Usage: python scripts/run_events_pipeline.py [START YYYY-MM-DD] [END YYYY-MM-DD]
START_DATE = "2025-05-30"
END_DATE   = "2025-12-31"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(_ROOT, "data", "raw", "gdelt", "events_filtered")

# ==========================
# GDELT EVENTS SCHEMA (2.1)
# ==========================
COLS = [
    "GlobalEventID", "Day", "MonthYear", "Year",
    "FractionDate",
    "Actor1Code", "Actor1Name", "Actor1CountryCode",
    "Actor1KnownGroupCode", "Actor1EthnicCode",
    "Actor1Religion1Code", "Actor1Religion2Code",
    "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
    "Actor2Code", "Actor2Name", "Actor2CountryCode",
    "Actor2KnownGroupCode", "Actor2EthnicCode",
    "Actor2Religion1Code", "Actor2Religion2Code",
    "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
    "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
    "QuadClass", "GoldsteinScale", "NumMentions",
    "NumSources", "NumArticles", "AvgTone",
    "Actor1Geo_Type", "Actor1Geo_FullName",
    "Actor1Geo_CountryCode", "Actor1Geo_ADM1Code",
    "Actor1Geo_Lat", "Actor1Geo_Long",
    "Actor1Geo_FeatureID",
    "Actor2Geo_Type", "Actor2Geo_FullName",
    "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code",
    "Actor2Geo_Lat", "Actor2Geo_Long",
    "Actor2Geo_FeatureID",
    "ActionGeo_Type", "ActionGeo_FullName",
    "ActionGeo_CountryCode", "ActionGeo_ADM1Code",
    "ActionGeo_Lat", "ActionGeo_Long",
    "ActionGeo_FeatureID",
    "DATEADDED", "SOURCEURL"
]

# ==========================
# EVENT FILTER LOGIC
# ==========================
KEEP_EVENT_ROOTS = {
    "01",  # Make Public Statement
    "02",  # Appeal
    "03",  # Express Intent
    "04",  # Consult
    "05",  # Engage in Diplomatic Cooperation
    "06",  # Engage in Material Cooperation
    "07",  # Provide Aid
    "08",  # Yield
    "09",  # Investigate
    "10",  # Demand
    "11",  # Disapprove
    "12",  # Reject
    "13",  # Threaten
    "14",  # Protest
    "15",  # Exhibit Force
    "16",  # Reduce Relations
    "17",  # Coerce
    "18",  # Assault
    "19",  # Fight
    "20"   # Use Unconventional Mass Violence
}

IMPORTANT_COUNTRIES = {
    "USA", "RUS", "CHN", "IND", "UKR", "IRN", "ISR",
    "SAU", "DEU", "FRA", "GBR", "JPN"
}

# ==========================
# HELPERS
# ==========================
def daterange(start, end):
    start = datetime.strptime(start, "%Y-%m-%d")
    end   = datetime.strptime(end, "%Y-%m-%d")
    while start <= end:
        yield start
        start += timedelta(days=1)

# ==========================
# MAIN PIPELINE
# ==========================
def main(start=START_DATE, end=END_DATE):
    os.makedirs(OUT_DIR, exist_ok=True)
    for d in daterange(start, end):

        day_str = d.strftime("%Y%m%d")
        out_csv = os.path.join(OUT_DIR, f"{d.strftime('%Y-%m-%d')}.csv")

        if os.path.exists(out_csv):
            continue

        print(f"\n📅 Processing {d.strftime('%Y-%m-%d')}")

        url = f"http://data.gdeltproject.org/events/{day_str}.export.CSV.zip"

        try:
            r = requests.get(url, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"  ❌ Download failed: {e}")
            continue

        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = z.namelist()[0]
        except Exception as e:
            print(f"  ❌ Zip error: {e}")
            continue

        rows = []

        with z.open(csv_name) as f:
            for chunk in pd.read_csv(
                f,
                sep="\t",
                names=COLS,
                dtype=str,
                chunksize=100_000,
                on_bad_lines="skip",
                low_memory=False
            ):

                # Core filters
                chunk = chunk[
                    chunk["EventRootCode"].isin(KEEP_EVENT_ROOTS) &
                    (
                        chunk["Actor1CountryCode"].isin(IMPORTANT_COUNTRIES) |
                        chunk["Actor2CountryCode"].isin(IMPORTANT_COUNTRIES) |
                        chunk["ActionGeo_CountryCode"].isin(IMPORTANT_COUNTRIES)
                    )
                ]

                if not chunk.empty:
                    rows.append(chunk)

        if not rows:
            print("  ⚠️ No relevant events")
            continue

        out = pd.concat(rows, ignore_index=True)

        # Reduce size early
        out = out[
            [
                "Day",
                "EventRootCode",
                "EventCode",
                "GoldsteinScale",
                "AvgTone",
                "Actor1Name",
                "Actor2Name",
                "Actor1CountryCode",
                "Actor2CountryCode",
                "ActionGeo_CountryCode",
                "SOURCEURL"
            ]
        ]

        out.to_csv(out_csv, index=False)
        print(f"  ✅ Saved {len(out)} rows")

    print("\n🎯 EVENTS PIPELINE COMPLETE")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:3])
