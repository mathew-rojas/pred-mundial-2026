"""Download raw CSVs from martj42/international_results."""
import requests
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv"]
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"


def download_all(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for fname in FILES:
        dest = RAW_DIR / fname
        if dest.exists() and not force:
            print(f"  {fname} already exists, skipping (use force=True to re-download)")
            continue
        url = f"{BASE_URL}/{fname}"
        print(f"  Downloading {fname} ...", end=" ", flush=True)
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"done ({len(resp.content) / 1024:.1f} KB)")


if __name__ == "__main__":
    download_all()
