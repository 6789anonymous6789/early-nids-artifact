#!/usr/bin/env python3
"""Compute preprocessing_meta.json for CIC-IDS2017 (feature list + drop lists).

Mirrors the 2018 meta format: selects numeric columns from raw_flow parquets,
drops zero-variance and highly-correlated features, writes the result to
data/CIC-IDS2017/preprocessing_meta.json. Consumed by train_anomaly.py (2017 copy).

Usage:
    python scripts/cic_ids2017/build_preprocessing_meta.py --cut-dir time_abs_1s
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2017.pcap_extracted_flow_loading import PARTIAL_FLOW_ROOT  # noqa: E402
from src.cic_ids2017.flow_pipeline import DAYS  # noqa: E402

META_OUT = REPO_ROOT / "data" / "CIC-IDS2017" / "preprocessing_meta.json"

# Non-feature columns — identical rule to 2018.
METADATA_COLS = {
    "day", "capture_file", "flow_id",
    "flow_src_ip", "flow_dst_ip", "flow_src_port", "flow_dst_port",
    "flow_start_ts", "flow_end_ts",
    "Label", "label_encoded",
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--cut-dir", default="time_abs_1s",
                   help="Which cut to scan for feature stats. Default: time_abs_1s.")
    p.add_argument("--corr-threshold", type=float, default=0.95)
    args = p.parse_args()

    cut_dir = PARTIAL_FLOW_ROOT / args.cut_dir / "raw_flow"
    meta_out = PARTIAL_FLOW_ROOT / args.cut_dir / "preprocessing_meta.json"
    if not cut_dir.exists():
        print(f"ERROR: {cut_dir} not found. Run build_partial_flows.py first.", file=sys.stderr)
        return 2

    print(f"Scanning {cut_dir}")
    frames = []
    for day in DAYS:
        p = cut_dir / f"{day}.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p, engine="pyarrow"))
            print(f"  loaded {p.name}: {len(frames[-1]):,} rows")
    if not frames:
        print("ERROR: no parquets found", file=sys.stderr)
        return 3
    df = pd.concat(frames, ignore_index=True)
    print(f"\nTotal flows: {len(df):,}")

    # Candidate features: numeric columns, excluding metadata
    numeric = df.select_dtypes(include="number").columns.tolist()
    feature_candidates = [c for c in numeric if c not in METADATA_COLS]
    print(f"Numeric feature candidates: {len(feature_candidates)}")

    X = df[feature_candidates].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Zero-variance drop
    stds = X.std(numeric_only=True)
    zero_var = [c for c in feature_candidates if stds.get(c, 0) == 0 or np.isnan(stds.get(c, 0))]
    print(f"Zero-variance drop: {len(zero_var)} -> {zero_var}")

    keep = [c for c in feature_candidates if c not in zero_var]

    # Correlated drop (greedy, symmetrical)
    print(f"Computing correlations on {len(keep)} features ({len(X):,} rows)")
    corr = X[keep].corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    dropped_corr = []
    for col in upper.columns:
        if col in dropped_corr:
            continue
        above = upper.index[(upper[col] > args.corr_threshold)].tolist()
        for partner in above:
            if partner not in dropped_corr and partner != col:
                dropped_corr.append(partner)
    print(f"Correlated drop (thr={args.corr_threshold}): {len(dropped_corr)}")
    for c in dropped_corr:
        print(f"  - {c}")

    feature_columns = [c for c in keep if c not in dropped_corr]
    print(f"\nFinal feature count: {len(feature_columns)}")
    print(feature_columns)

    meta = {
        "source": "CIC-IDS2017 preprocessing meta, computed from time-bucketed raw_flow parquets.",
        "cut_dir_scanned": args.cut_dir,
        "n_flows_scanned": int(len(df)),
        "corr_threshold": args.corr_threshold,
        "feature_columns": feature_columns,
        "dropped_zero_var": zero_var,
        "dropped_correlated": dropped_corr,
    }
    meta_out = PARTIAL_FLOW_ROOT / args.cut_dir / "preprocessing_meta.json"
    meta_out.parent.mkdir(parents=True, exist_ok=True)
    meta_out.write_text(json.dumps(meta, indent=2))
    print(f"\nWrote {meta_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
