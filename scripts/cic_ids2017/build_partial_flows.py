#!/usr/bin/env python3
"""Build CIC-IDS2017 raw_flow parquets for one or more cuts.

Example:
    python scripts/cic_ids2017/build_partial_flows.py --cut-mode time_abs --cut-value 1s
    python scripts/cic_ids2017/build_partial_flows.py --cut-mode packet_pct --cut-value 1.0
    python scripts/cic_ids2017/build_partial_flows.py --days Tuesday-WorkingHours
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2017.flow_pipeline import DAYS, build_day_partial_flow  # noqa: E402


def _parse_cut_value(mode: str, raw: str) -> float:
    raw = raw.strip().lower()
    if mode == "time_abs":
        if raw.endswith("ms"):
            return float(raw[:-2]) / 1000.0
        if raw.endswith("s"):
            return float(raw[:-1])
        return float(raw)
    return float(raw)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cut-mode", required=True, choices=["packet_pct", "time_abs"])
    p.add_argument("--cut-value", required=True, help="Numeric; 500ms/1s/3s for time_abs; ratio otherwise.")
    p.add_argument("--days", nargs="+", default=None, help="Day stems; default = all 5.")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    cut_value = _parse_cut_value(args.cut_mode, args.cut_value)
    days = args.days or DAYS

    print(f"cut_mode={args.cut_mode}  cut_value={cut_value}")
    print(f"days: {days}")
    t0 = time.perf_counter()
    results = []
    for d in days:
        r = build_day_partial_flow(d, cut_mode=args.cut_mode, cut_value=cut_value, force=args.force)
        results.append(r)
    total = time.perf_counter() - t0
    print(f"\nTotal time: {total:.0f}s = {total/60:.1f} min")
    print(json.dumps([r for r in results if not r.get("skipped")], indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
