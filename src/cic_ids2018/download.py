#!/usr/bin/env python3
"""
Download the CSE-CIC-IDS2018 dataset from the AWS Open Data Registry.

Downloads both raw PCAPs (zipped) and pre-extracted CICFlowMeter CSVs
into data/CIC-IDS2018/original/.

Registry page : https://registry.opendata.aws/cse-cic-ids2018/
S3 bucket     : s3://cse-cic-ids2018/
Licence       : Creative Commons Attribution 4.0 International (CC BY 4.0)

Requirements
------------
  pip install boto3

Usage
-----
  # From the thesis/ root:
  python src/download_cic_ids2018.py                   # download everything
  python src/download_cic_ids2018.py --skip-raw         # CSVs only (~6.5 GB)
  python src/download_cic_ids2018.py --skip-csv         # PCAPs only (~447 GB)
  python src/download_cic_ids2018.py --dry-run           # list files without downloading
"""

import argparse
import hashlib
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config


# ── S3 constants ──────────────────────────────────────────────────────────────
BUCKET = "cse-cic-ids2018"
RAW_PREFIX = "Original Network Traffic and Log data/"
CSV_PREFIX = "Processed Traffic Data for ML Algorithms/"

# ── Local target directories ─────────────────────────────────────────────────
BASE_DIR = Path("data/CIC-IDS2018/original")
RAW_DIR = BASE_DIR / "raw"
CSV_DIR = BASE_DIR / "full_flow"


def list_s3_objects(s3, prefix: str) -> list[dict]:
    """List all objects under a given S3 prefix."""
    paginator = s3.get_paginator("list_objects_v2")
    objects = []
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append(obj)
    return objects


def format_size(n_bytes: int) -> str:
    """Human-readable file size."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} PB"


def download_file(s3, key: str, local_path: Path, size: int) -> None:
    """Download a single file from S3 with progress reporting."""
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Skip if already downloaded and same size
    if local_path.exists() and local_path.stat().st_size == size:
        print(f"  SKIP (exists, same size): {local_path}")
        return

    print(f"  Downloading: {key}")
    print(f"    → {local_path} ({format_size(size)})")

    downloaded = 0

    def progress_callback(chunk_bytes):
        nonlocal downloaded
        downloaded += chunk_bytes
        pct = downloaded / size * 100 if size > 0 else 100
        print(f"\r    Progress: {format_size(downloaded)} / {format_size(size)} ({pct:.1f}%)", end="", flush=True)

    s3.download_file(
        Bucket=BUCKET,
        Key=key,
        Filename=str(local_path),
        Callback=progress_callback,
    )
    print()  # newline after progress


def download_prefix(s3, prefix: str, local_dir: Path, dry_run: bool) -> None:
    """Download all files under an S3 prefix, preserving directory structure."""
    objects = list_s3_objects(s3, prefix)

    if not objects:
        print(f"  No objects found under s3://{BUCKET}/{prefix}")
        return

    total_size = sum(obj["Size"] for obj in objects)
    file_count = len(objects)
    print(f"  Found {file_count} files ({format_size(total_size)})")

    if dry_run:
        for obj in objects:
            rel = obj["Key"][len(prefix):]
            print(f"    {rel}  ({format_size(obj['Size'])})")
        return

    for i, obj in enumerate(objects, 1):
        key = obj["Key"]
        # Skip "directory" markers (zero-byte keys ending with /)
        if key.endswith("/") and obj["Size"] == 0:
            continue

        rel_path = key[len(prefix):]  # strip the S3 prefix
        local_path = local_dir / rel_path

        print(f"\n  [{i}/{file_count}]")
        download_file(s3, key, local_path, obj["Size"])


def main():
    parser = argparse.ArgumentParser(
        description="Download CSE-CIC-IDS2018 from AWS Open Data Registry."
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip raw PCAPs/logs (~447 GB). Download CSVs only.",
    )
    parser.add_argument(
        "--skip-csv",
        action="store_true",
        help="Skip CICFlowMeter CSVs (~6.5 GB). Download raw PCAPs only.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files and sizes without downloading.",
    )
    args = parser.parse_args()

    # Anonymous S3 access (public bucket, no credentials needed)
    s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))

    print("=" * 60)
    print("CSE-CIC-IDS2018 Dataset Downloader")
    print("=" * 60)
    print(f"S3 bucket:  s3://{BUCKET}/")
    print(f"Local base: {BASE_DIR.resolve()}")
    if args.dry_run:
        print("MODE: dry-run (no files will be downloaded)")
    print()

    if not args.skip_csv:
        print("── CICFlowMeter CSVs ──────────────────────────────────────")
        download_prefix(s3, CSV_PREFIX, CSV_DIR, args.dry_run)
        print()

    if not args.skip_raw:
        print("── Raw PCAPs and logs ─────────────────────────────────────")
        download_prefix(s3, RAW_PREFIX, RAW_DIR, args.dry_run)
        print()

    print("=" * 60)
    if args.dry_run:
        print("Dry run complete. Re-run without --dry-run to download.")
    else:
        print("Download complete.")
        print(f"  CSVs: {CSV_DIR.resolve()}")
        print(f"  Raw:  {RAW_DIR.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
