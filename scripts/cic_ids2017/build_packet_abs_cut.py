"""Build a raw_flow parquet by keeping only the first N packets per flow.

Mirrors the time-based cuts but with an absolute packet cap (no native pipeline
support for `packet_abs` mode). Uses the existing aggregate_flow_features and
Distrinet registry.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2017.distrinet_labeling import load_registry  # noqa: E402
from src.cic_ids2017.flow_pipeline import (  # noqa: E402
    DAYS,
    DISTRINET_FEATURE_COLUMNS,
    PARTIAL_FLOW_ROOT,
    _LEGACY_LABEL_COLS,
    load_packet_base,
)
from src.cic_ids2018.pcap_extracted_flow_pipeline import (  # noqa: E402
    aggregate_flow_features,
)


def truncate_packet_base_by_packet_abs(packet_df: pd.DataFrame, max_packets: int) -> pd.DataFrame:
    """Keep the first `max_packets` packets per flow (ordered by timestamp)."""
    if max_packets <= 0:
        raise ValueError(f"max_packets must be > 0, got {max_packets}")
    if packet_df.empty:
        return packet_df.copy()
    df = packet_df.sort_values(["flow_id", "timestamp"], kind="mergesort")
    rank = df.groupby("flow_id", sort=False).cumcount()
    return df.loc[rank < max_packets].copy()


def build_day(day: str, max_packets: int, force: bool) -> dict:
    tag = f"packet_abs_{max_packets}"
    out_path = PARTIAL_FLOW_ROOT / tag / "raw_flow" / f"{day}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not force:
        print(f"[{day} / {tag}] exists, skipping")
        return {"day": day, "skipped": True, "path": str(out_path)}

    t0 = time.perf_counter()
    print(f"[{day} / {tag}] loading packet_base + grouping into flows", flush=True)
    flows_with_packets = load_packet_base(day)
    print(f"  packets: {len(flows_with_packets):,}", flush=True)

    print(f"[{day} / {tag}] truncating to first {max_packets} packets per flow", flush=True)
    truncated = truncate_packet_base_by_packet_abs(flows_with_packets, max_packets)
    print(f"  truncated: {len(truncated):,}", flush=True)

    print(f"[{day} / {tag}] aggregating flow features", flush=True)
    flow_df = aggregate_flow_features(truncated, day=day)
    print(f"  flows: {len(flow_df):,}", flush=True)

    flow_df = flow_df.drop(columns=list(_LEGACY_LABEL_COLS & set(flow_df.columns)))

    print(f"[{day} / {tag}] joining Distrinet registry by flow_id", flush=True)
    registry = load_registry(day)[["flow_id", "Label_Distrinet", "Attempted_Category", "matched"]]
    flow_df = flow_df.merge(registry, on="flow_id", how="left")
    flow_df["matched"] = flow_df["matched"].fillna(False).astype("bool")
    flow_df["Attempted_Category"] = flow_df["Attempted_Category"].fillna(-1).astype("Int8")

    flow_df = flow_df[DISTRINET_FEATURE_COLUMNS]
    flow_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    elapsed = time.perf_counter() - t0
    n_matched = int(flow_df["matched"].sum())
    print(f"[{day} / {tag}] done in {elapsed:.0f}s -> {out_path} ({len(flow_df):,} flows, {n_matched:,} matched)", flush=True)
    return {"day": day, "path": str(out_path), "n_flows": int(len(flow_df)), "elapsed_s": float(elapsed)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-packets", type=int, required=True)
    p.add_argument("--days", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    days = args.days or DAYS
    t0 = time.perf_counter()
    for d in days:
        build_day(d, args.max_packets, args.force)
    print(f"\nTotal time: {time.perf_counter() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
