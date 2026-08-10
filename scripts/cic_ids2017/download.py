#!/usr/bin/env python3
"""Download CIC-IDS2017 PCAPs (+ optional CSVs) from the CIC portal.

Unlike CIC-IDS2018 (public AWS S3 bucket, zero auth), the 2017 dataset
is gated behind a PHP session cookie issued by cicresearch.ca after
the user fills the UNB form.

How to obtain a session cookie
------------------------------

1. Open the form at https://www.unb.ca/cic/datasets/ids-2017.html and
   submit your name/email/organisation. You get redirected to
   https://cicresearch.ca/CICDataset/CIC-IDS-2017/browse.php?p=...
2. Open DevTools -> Application -> Cookies -> cicresearch.ca.
   Copy the value of the `Token` cookie (looks like a 24-char hex string).
3. Pass it via --cookie or set CIC_SESSION_COOKIE before running.

IP BINDING
----------

The PHP session is tied to the client IP used at login. If you fill the
form from your laptop and try to run this script from a remote server,
requests hang with no response. Run this script from the same host
that logged in, or log in again from the host you intend to download
from.

Usage
-----

    export CIC_SESSION_COOKIE=<your_token>
    python scripts/cic_ids2017/download.py                # PCAPs + MD5s
    python scripts/cic_ids2017/download.py --include-csv  # also CSV zips
    python scripts/cic_ids2017/download.py --dry-run
    python scripts/cic_ids2017/download.py --verify       # MD5 check after

Reference cURL (what the browser sends; kept here as the ground truth
for future HEADERS / URL updates):

    curl 'https://cicresearch.ca/CICDataset/CIC-IDS-2017/download.php?file=CIC-IDS-2017%2FPCAPs%2FMonday-WorkingHours.pcap' \\
      -b 'Token=<TOKEN>' \\
      -H 'Referer: https://cicresearch.ca/CICDataset/CIC-IDS-2017/browse.php?p=CIC-IDS-2017%2FPCAPs' \\
      -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36' \\
      -o Monday-WorkingHours.pcap
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "CIC-IDS2017" / "original"

BASE = "https://cicresearch.ca/CICDataset/CIC-IDS-2017/download.php?file="
BROWSE = "https://cicresearch.ca/CICDataset/CIC-IDS-2017/browse.php?p="

# Day-level PCAPs + their MD5 siblings, as exposed by browse.php?p=CIC-IDS-2017%2FPCAPs.
# NOTE: "Wednesday-workingHours" has a lowercase 'w' in 'working' on the server;
# all others are "WorkingHours". Matters: the path is case-sensitive.
PCAP_FILES = [
    "CIC-IDS-2017/PCAPs/Monday-WorkingHours.pcap",
    "CIC-IDS-2017/PCAPs/Monday-WorkingHours.md5",
    "CIC-IDS-2017/PCAPs/Tuesday-WorkingHours.pcap",
    "CIC-IDS-2017/PCAPs/Tuesday-WorkingHours.md5",
    "CIC-IDS-2017/PCAPs/Wednesday-workingHours.pcap",
    "CIC-IDS-2017/PCAPs/Wednesday-workingHours.md5",
    "CIC-IDS-2017/PCAPs/Thursday-WorkingHours.pcap",
    "CIC-IDS-2017/PCAPs/Thursday-WorkingHours.md5",
    "CIC-IDS-2017/PCAPs/Friday-WorkingHours.pcap",
    "CIC-IDS-2017/PCAPs/Friday-WorkingHours.md5",
]

# Optional CSV / labelled-flow bundles (toggle via --include-csv).
CSV_FILES = [
    "CIC-IDS-2017/MachineLearningCSV.zip",
    "CIC-IDS-2017/GeneratedLabelledFlows.zip",
]

HEADERS = [
    ("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7"),
    ("Accept-Language", "it-IT,it;q=0.9,en-US;q=0.8,en;q=0.7"),
    ("Connection", "keep-alive"),
    ("Sec-Fetch-Dest", "document"),
    ("Sec-Fetch-Mode", "navigate"),
    ("Sec-Fetch-Site", "same-origin"),
    ("Sec-Fetch-User", "?1"),
    ("Upgrade-Insecure-Requests", "1"),
    ("User-Agent", "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/147.0.0.0 Safari/537.36"),
    ("sec-ch-ua", '"Google Chrome";v="147", "Not.A/Brand";v="8", "Chromium";v="147"'),
    ("sec-ch-ua-mobile", "?0"),
    ("sec-ch-ua-platform", '"macOS"'),
]


def _check_curl() -> bool:
    if shutil.which("curl") is None:
        print("ERROR: curl not found on PATH.", file=sys.stderr)
        return False
    return True


def _build_curl_cmd(url: str, cookie: str, out_path: Path, referer: str) -> list[str]:
    cmd = [
        "curl", "--fail", "--location",
        "--connect-timeout", "30",
        "--retry", "3", "--retry-delay", "10",
        "-C", "-",  # resume partial downloads
        "-o", str(out_path),
        "-b", f"Token={cookie}",
        "-H", f"Referer: {referer}",
    ]
    for k, v in HEADERS:
        cmd += ["-H", f"{k}: {v}"]
    cmd.append(url)
    return cmd


def download_one(file_path: str, cookie: str, out_dir: Path, dry_run: bool) -> int:
    encoded = quote(file_path, safe="")
    url = BASE + encoded

    parent = Path(file_path).parent  # e.g. PosixPath("CIC-IDS-2017/PCAPs") or "CIC-IDS-2017"
    local_subdir = Path(*parent.parts[1:]) if len(parent.parts) > 1 else Path(".")
    local_dir = out_dir / local_subdir
    out_path = local_dir / Path(file_path).name

    referer = BROWSE + quote(str(parent), safe="")

    if dry_run:
        print(f"  [DRY] {url}")
        print(f"        -> {out_path}")
        return 0

    local_dir.mkdir(parents=True, exist_ok=True)
    print(f"  {file_path}")
    print(f"    -> {out_path}")
    return subprocess.call(_build_curl_cmd(url, cookie, out_path, referer))


def verify_md5(pcap_path: Path, md5_path: Path) -> bool | None:
    if not md5_path.exists() or not pcap_path.exists():
        return None
    expected = md5_path.read_text().strip().split()[0].lower()
    h = hashlib.md5()
    with pcap_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().lower() == expected


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--cookie", default=os.environ.get("CIC_SESSION_COOKIE"),
                   help="PHP Token cookie value. Falls back to $CIC_SESSION_COOKIE.")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR,
                   help=f"Local destination (default: {DEFAULT_OUT_DIR}).")
    p.add_argument("--include-csv", action="store_true",
                   help="Also fetch MachineLearningCSV.zip and GeneratedLabelledFlows.zip.")
    p.add_argument("--only-csv", action="store_true",
                   help="Skip PCAPs, fetch only CSV zips.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the plan without downloading.")
    p.add_argument("--verify", action="store_true",
                   help="After download, verify each PCAP against its .md5 sibling.")
    args = p.parse_args()

    if not _check_curl():
        return 2

    if not args.cookie and not args.dry_run:
        print("ERROR: no session cookie. Pass --cookie or set CIC_SESSION_COOKIE.", file=sys.stderr)
        print("       See module docstring for how to obtain one.", file=sys.stderr)
        return 3

    files: list[str] = []
    if not args.only_csv:
        files += PCAP_FILES
    if args.include_csv or args.only_csv:
        files += CSV_FILES

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Target: {args.out_dir}")
    print(f"Files : {len(files)}")

    failures: list[str] = []
    for i, f in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}]")
        rc = download_one(f, args.cookie or "", args.out_dir, args.dry_run)
        if rc != 0:
            failures.append(f)

    if args.verify and not args.dry_run and not args.only_csv:
        print("\n=== MD5 verification ===")
        pcap_dir = args.out_dir / "PCAPs"
        for pcap in sorted(pcap_dir.glob("*.pcap")):
            md5 = pcap.with_suffix(".md5")
            ok = verify_md5(pcap, md5)
            if ok is None:
                print(f"  {pcap.name}: skipped (missing .pcap or .md5)")
            elif ok:
                print(f"  {pcap.name}: OK")
            else:
                print(f"  {pcap.name}: MISMATCH")
                failures.append(str(pcap))

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nAll downloads complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
