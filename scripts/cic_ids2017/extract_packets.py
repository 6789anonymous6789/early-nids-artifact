#!/usr/bin/env python3
"""Run per-packet extraction (tshark) on all CIC-IDS2017 PCAPs.

Reuses the dataset-agnostic `extract_packets` from
`src.cic_ids2018.pcap_packet_extraction` (same field list, same
behaviour). Output goes to
`data/CIC-IDS2017/partial_flow/packet_base/<Day>.parquet`,
one file per PCAP (one per day for 2017, since the capture is already
a single daily PCAP).

Usage:
    python scripts/cic_ids2017/extract_packets.py
    python scripts/cic_ids2017/extract_packets.py --days Monday-WorkingHours
    python scripts/cic_ids2017/extract_packets.py --force

This is the first CIC-IDS2017 build stage. Output parquets are the
input to the flow-aggregation pipeline (TBD for 2017).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2018.pcap_packet_extraction import extract_packets  # noqa: E402

PCAP_DIR = REPO_ROOT / "data" / "CIC-IDS2017" / "original" / "PCAPs"
OUT_DIR = REPO_ROOT / "data" / "CIC-IDS2017" / "partial_flow" / "packet_base"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--days", nargs="+", default=None,
                   help="Optional list of PCAP stems (e.g. Monday-WorkingHours). Default: all.")
    p.add_argument("--force", action="store_true",
                   help="Re-extract even if the parquet already exists.")
    args = p.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    pcaps = sorted(PCAP_DIR.glob("*.pcap"))
    if args.days:
        wanted = set(args.days)
        pcaps = [p for p in pcaps if p.stem in wanted]
        if not pcaps:
            print(f"ERROR: no matching PCAPs for --days {args.days}", file=sys.stderr)
            return 2

    print(f"PCAPs to process: {len(pcaps)}")
    for pcap in pcaps:
        size_gb = pcap.stat().st_size / (1 << 30)
        print(f"  {pcap.name}  ({size_gb:.1f} GB)")

    total_t0 = time.time()
    failures = []
    for i, pcap in enumerate(pcaps, 1):
        out_path = OUT_DIR / f"{pcap.stem}.parquet"
        if out_path.exists() and not args.force:
            print(f"\n[{i}/{len(pcaps)}] SKIP (exists): {out_path}")
            continue

        print(f"\n[{i}/{len(pcaps)}] {pcap.name}")
        print(f"  -> {out_path}")
        t0 = time.time()
        try:
            df = extract_packets(pcap, source_id=pcap.stem)
        except Exception as e:
            print(f"  ERROR: {e}")
            failures.append(pcap.name)
            continue
        elapsed = time.time() - t0
        n_rows = len(df)
        print(f"  tshark+parse: {elapsed:.0f}s  ({n_rows:,} packets)")

        t0 = time.time()
        df.to_parquet(out_path, index=False)
        print(f"  parquet write: {time.time()-t0:.0f}s  ({out_path.stat().st_size/(1<<30):.2f} GB)")

    total = time.time() - total_t0
    print(f"\nTotal time: {total:.0f}s = {total/60:.1f} min")
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
