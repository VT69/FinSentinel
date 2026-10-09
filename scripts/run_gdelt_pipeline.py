import io
import os
import zipfile

import requests
import pandas as pd
from datetime import datetime, timedelta

# ==============================
# CONFIG
# ==============================
START_DATE = "2020-03-17"
END_DATE   = "2025-12-31"   # change later (overnight run)
BASE_URL   = "http://data.gdeltproject.org/gkg/"

_ROOT      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR    = os.path.join(_ROOT, "data", "raw", "gdelt", "gkg_filtered")

# ==============================
# KEYWORDS
# ==============================
MACRO_KEYWORDS = [
    "economy", "economic", "macroeconomic",
    "recession", "depression", "slowdown",
    "inflation", "deflation", "stagflation",
    "interest rate", "rate hike", "rate cut",
    "monetary policy", "fiscal policy",
    "central bank", "federal reserve", "fed",
    "reserve bank", "rbi", "ecb", "boe",
    "liquidity", "credit", "money supply",
    "quantitative easing", "quantitative tightening",
    "bond yield", "treasury yield",
    "financial stability", "systemic risk"
]
GEO_KEYWORDS = [
    "war", "conflict", "military", "invasion",
    "terrorist attack", "terrorism",
    "geopolitical tension", "border dispute",
    "sanctions", "trade sanctions",
    "diplomatic crisis", "embargo",
    "nuclear threat", "missile test",
    "proxy war", "ceasefire violation",
    "civil unrest", "riot", "uprising",
    "regime change", "coup"
]
COMMODITY_KEYWORDS = [
    "oil price", "crude oil", "brent oil", "wti",
    "natural gas", "energy crisis",
    "supply shock", "supply chain disruption",
    "commodity prices", "raw materials",
    "gold price", "silver price",
    "food prices", "grain shortage",
    "fertilizer shortage"
]
POLICY_KEYWORDS = [
    "government policy", "economic reform",
    "budget announcement", "tax reform",
    "tariff", "import duty", "export ban",
    "trade policy", "free trade agreement",
    "regulation", "financial regulation",
    "capital controls",
    "privatization", "nationalization",
    "stimulus package", "bailout",
    "debt crisis", "sovereign debt"
]
MARKET_STRESS_KEYWORDS = [
    "market crash", "stock market crash",
    "sell-off", "panic selling",
    "market volatility", "volatility spike",
    "risk-off", "flight to safety",
    "bank failure", "bank collapse",
    "credit crunch", "liquidity crisis",
    "default", "bankruptcy",
    "financial crisis", "systemic collapse"
]

RELEVANT_KEYWORDS = (
    MACRO_KEYWORDS
    + GEO_KEYWORDS
    + COMMODITY_KEYWORDS
    + POLICY_KEYWORDS
    + MARKET_STRESS_KEYWORDS
)

def is_relevant(text):
    text = str(text).lower()
    return any(k in text for k in RELEVANT_KEYWORDS)

# ==============================
# MAIN LOOP
# ==============================
def main(start_date=START_DATE, end_date=END_DATE):
    os.makedirs(OUT_DIR, exist_ok=True)
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end   = datetime.strptime(end_date, "%Y-%m-%d")

    current = start

    while current <= end:
        date_str = current.strftime("%Y%m%d")
        url = f"{BASE_URL}{date_str}.gkg.csv.zip"

        print(f"\n📅 Processing {current.date()}")

        try:
            r = requests.get(url, timeout=60)
            if r.status_code != 200:
                print("  ⚠️ File not found")
                current += timedelta(days=1)
                continue

            z = zipfile.ZipFile(io.BytesIO(r.content))
            csv_name = z.namelist()[0]

            df = pd.read_csv(
                z.open(csv_name),
                sep="\t",
                header=None,
                encoding="latin1",
                low_memory=False
            )

            # GKG column positions (stable)
            df = df[[0, 1, 3, 4, 8, 9]]
            df.columns = [
                "date",
                "source",
                "themes",
                "locations",
                "persons",
                "organizations"
            ]

            # Filter
            mask = (
                df["themes"].apply(is_relevant) |
                df["organizations"].apply(is_relevant) |
                df["persons"].apply(is_relevant)
            )

            filtered = df[mask]

            if filtered.empty:
                print("  ➜ No relevant news")
            else:
                out_path = f"{OUT_DIR}/gkg_{date_str}.csv"
                filtered.to_csv(out_path, index=False)
                print(f"  ✅ Saved {len(filtered)} rows")

        except Exception as e:
            print(f"  ❌ Error: {e}")

        current += timedelta(days=1)

    print("\n🎯 GKG PIPELINE COMPLETE")


if __name__ == "__main__":
    import sys
    main(*sys.argv[1:3])
