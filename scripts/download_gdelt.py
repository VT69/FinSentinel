import requests
import zipfile
import io
from pathlib import Path

"""Download quarterly GDELT 1.0 event archives:  python scripts/download_gdelt.py 2015 1 [2015 2 ...]"""
import sys

BASE_URL = "http://data.gdeltproject.org/events"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw" / "gdelt" / "events"

def download_quarter(year, quarter):
    qmap = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
    fname = f"{year}{qmap[quarter]}.zip"
    url = f"{BASE_URL}/{fname}"

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n📦 Downloading {fname}")

    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
    except Exception as e:
        print(f"✗ Failed {fname}: {e}")
        return

    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        for file in z.namelist():
            out = RAW_DIR / file
            z.extract(file, RAW_DIR)
            size_mb = out.stat().st_size / 1e6
            print(f"  → Extracted {file} ({size_mb:.1f} MB)")

    print(f"✅ Completed {year} {qmap[quarter]}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args or len(args) % 2:
        sys.exit(__doc__)
    for y, q in zip(args[::2], args[1::2]):
        download_quarter(int(y), int(q))
