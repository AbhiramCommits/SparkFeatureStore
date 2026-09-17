"""Download NYC TLC Yellow Taxi trip records (Parquet) into data/raw/.

Default range: 2023-01 through 2023-12 (~40M rows / several GB).
Use ``--months 1`` for a quick local smoke test (only 2023-01, ~3.3M rows,
~150 MB).

Examples:
    python scripts/fetch_data.py                      # full 2023
    python scripts/fetch_data.py --months 1           # only 2023-01
    python scripts/fetch_data.py --start 2023-03 --end 2023-06
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # noqa: E402

import requests  # noqa: E402

from common.config import REPO_ROOT  # noqa: E402
from common.logging import get_logger, setup_logging  # noqa: E402

log = get_logger(__name__)

BASE_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_{year:04d}-{month:02d}.parquet"
)
CHUNK_SIZE = 1024 * 1024
PROGRESS_EVERY_MB = 25


def month_range(start: str, end: str, limit: int | None = None) -> list[tuple[int, int]]:
    """Return (year, month) tuples from ``start`` to ``end`` (both YYYY-MM, inclusive).

    If ``limit`` is given, stop after that many months.
    """
    start_year, start_month = (int(part) for part in start.split("-"))
    end_year, end_month = (int(part) for part in end.split("-"))
    months: list[tuple[int, int]] = []
    year, month = start_year, start_month
    while (year, month) <= (end_year, end_month):
        months.append((year, month))
        if limit is not None and len(months) >= limit:
            break
        month += 1
        if month > 12:
            month, year = 1, year + 1
    return months


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a single file to ``dest``. Returns True if it was downloaded."""
    if dest.exists() and dest.stat().st_size > 0 and not force:
        log.info("Skipping %s (already present; use --force to re-download)", dest.name)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    log.info("Downloading %s -> %s", url, dest)
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        downloaded = 0
        next_report = PROGRESS_EVERY_MB * CHUNK_SIZE
        with tmp.open("wb") as fh:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                fh.write(chunk)
                downloaded += len(chunk)
                if downloaded >= next_report:
                    log.info("  %s: %.0f MB", dest.name, downloaded / 1e6)
                    next_report += PROGRESS_EVERY_MB * CHUNK_SIZE
    tmp.replace(dest)
    log.info("Saved %s (%.1f MB)", dest, dest.stat().st_size / 1e6)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--start", default="2023-01", help="First month, YYYY-MM (default: 2023-01)"
    )
    parser.add_argument(
        "--end", default="2023-12", help="Last month, inclusive, YYYY-MM (default: 2023-12)"
    )
    parser.add_argument(
        "--months",
        type=int,
        default=None,
        help="Only download the first N months of the range (e.g. 1 for a quick run)",
    )
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "data" / "raw"),
        help="Destination directory (default: data/raw)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download files that already exist")
    parser.add_argument("--verbose", action="store_true", help="Debug logging")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    months = month_range(args.start, args.end, limit=args.months)
    log.info(
        "Fetching %d month(s) of Yellow Taxi data: %s .. %s",
        len(months),
        months[0],
        months[-1],
    )
    downloaded = 0
    for year, month in months:
        url = BASE_URL.format(year=year, month=month)
        dest = Path(args.out) / f"yellow_tripdata_{year:04d}-{month:02d}.parquet"
        if download_file(url, dest, force=args.force):
            downloaded += 1
    log.info("Done: %d file(s) downloaded to %s", downloaded, args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
