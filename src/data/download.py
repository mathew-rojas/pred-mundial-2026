"""Download raw CSVs from martj42/international_results."""
import requests
from pathlib import Path

BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
FILES = ["results.csv", "goalscorers.csv", "shootouts.csv", "former_names.csv"]
RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

# Disable compression so HEAD Content-Length matches the bytes saved to disk.
_HEADERS = {"Accept-Encoding": "identity"}


def _remote_size(url: str) -> int | None:
    """Return Content-Length from a HEAD request (uncompressed), or None."""
    try:
        resp = requests.head(url, timeout=10, allow_redirects=True, headers=_HEADERS)
        resp.raise_for_status()
        cl = resp.headers.get("Content-Length")
        return int(cl) if cl else None
    except Exception:
        return None


def download_all(force: bool = False) -> bool:
    """Download files that are missing or whose remote size differs from local.

    Returns True if any file was (re)downloaded, False if everything was already
    up to date. Callers can use this to decide whether to retrain.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    updated = False
    for fname in FILES:
        dest = RAW_DIR / fname
        url = f"{BASE_URL}/{fname}"

        if dest.exists() and not force:
            remote_size = _remote_size(url)
            local_size = dest.stat().st_size
            if remote_size is not None and remote_size == local_size:
                print(f"  {fname} up to date ({local_size / 1024:.1f} KB)")
                continue
            if remote_size is None:
                print(f"  {fname} already exists, skipping (could not check remote size)")
                continue

        print(f"  Downloading {fname} ...", end=" ", flush=True)
        resp = requests.get(url, timeout=30, headers=_HEADERS)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"done ({len(resp.content) / 1024:.1f} KB)")
        updated = True

    return updated


if __name__ == "__main__":
    download_all()
