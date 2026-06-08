"""Download public trace samples for DeepSched C-01.

The full Google and Alibaba traces are very large. This script downloads:

1. a compact, CC-BY Alibaba 2018 machine-usage sample from Zenodo for the
   first runnable C-01 pipeline;
2. optionally, the first Google task_usage shard for later full-trace scaling.
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path


GOOGLE_TASK_USAGE_URL = (
    "https://storage.googleapis.com/clusterdata-2011-2/task_usage/"
    "part-00000-of-00500.csv.gz"
)
ALIBABA_ZENODO_URL = (
    "https://zenodo.org/records/14564935/files/"
    "machine_usage_days_1_to_8_grouped_300_seconds.csv?download=1"
)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already exists: {destination}")
        return

    print(f"Downloading {url}")
    print(f"Saving to {destination}")
    with urllib.request.urlopen(url, timeout=60) as response:
        total = response.headers.get("Content-Length")
        total_bytes = int(total) if total else None
        downloaded = 0
        chunk_size = 1024 * 1024
        with destination.open("wb") as file:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                file.write(chunk)
                downloaded += len(chunk)
                if total_bytes:
                    pct = 100 * downloaded / total_bytes
                    print(f"\r{downloaded / 1_000_000:.1f} MB ({pct:.1f}%)", end="")
                else:
                    print(f"\r{downloaded / 1_000_000:.1f} MB", end="")
    print("\nDownload complete.")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        choices=["alibaba", "google"],
        default="alibaba",
        help="Dataset sample to download. Alibaba is the default C-01 input.",
    )
    parser.add_argument("--url", default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.url is None:
        args.url = ALIBABA_ZENODO_URL if args.source == "alibaba" else GOOGLE_TASK_USAGE_URL
    if args.output is None:
        args.output = (
            Path("data/raw/alibaba/machine_usage_days_1_to_8_grouped_300_seconds.csv")
            if args.source == "alibaba"
            else Path("data/raw/google/task_usage/part-00000-of-00500.csv.gz")
        )

    try:
        download(args.url, args.output)
    except Exception as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        print(
            "You can manually place Google task_usage CSV shards under "
            "data/raw/google/task_usage/ and rerun data/preprocess.py.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
