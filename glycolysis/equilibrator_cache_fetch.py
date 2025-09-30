from __future__ import annotations

import sys
import time
from pathlib import Path

import requests


URL = "https://zenodo.org/record/4128543/files/compounds.sqlite?download=1"


def fetch_cache(dest: Path, retries: int = 5, backoff: float = 1.5) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(URL, stream=True, timeout=60, verify=False) as r:  # nosec - user requested relaxed SSL
                r.raise_for_status()
                with dest.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                if dest.stat().st_size == 0:
                    raise RuntimeError("Empty file downloaded")
                return
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < retries:
                time.sleep(backoff ** attempt)
    assert last_err is not None
    raise RuntimeError(f"Failed to fetch equilibrator cache: {last_err}")


def main(argv: list[str] | None = None) -> int:
    out = Path.home() / ".cache" / "equilibrator" / "compounds.sqlite"
    try:
        fetch_cache(out)
        print(f"✅ Downloaded cache to {out}")
        return 0
    except Exception as e:  # noqa: BLE001
        print(f"❌ Cache download failed: {e}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())


