"""Build CIC-IDS2018 raw_flow parquet files keeping only the first N packets per flow."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2018.pcap_extracted_flow_pipeline import aggregate_flow_features  # noqa: E402

PARTIAL_FLOW_ROOT = REPO_ROOT / "data" / "CIC-IDS2018" / "partial_flow"
PACKET_BASE_DIR = PARTIAL_FLOW_ROOT / "packet_base"
PCT100_RAW_DIR = PARTIAL_FLOW_ROOT / "pct_100" / "raw_flow"
REGISTRY_DIR = PARTIAL_FLOW_ROOT / "distrinet_labels"


def _schema_columns() -> list[str]:
    sample = sorted(PCT100_RAW_DIR.glob("*.parquet"))[0]
    return pq.ParquetFile(sample).schema_arrow.names


def build_day(day: str, max_packets: int, force: bool) -> dict:
    tag = f"packet_abs_{max_packets}"
    packet_path = PACKET_BASE_DIR / f"{day}.parquet"
    registry_path = REGISTRY_DIR / f"{day}.parquet"
    out_path = PARTIAL_FLOW_ROOT / tag / "raw_flow" / f"{day}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f"[{day} / {tag}] exists, skipping", flush=True)
        return {"day": day, "skipped": True, "path": str(out_path)}

    t0 = time.perf_counter()
    print(f"[{day} / {tag}] loading first {max_packets} packets per flow", flush=True)
    packets = pd.read_parquet(
        packet_path,
        engine="pyarrow",
        filters=[("packet_index", "<", max_packets)],
    )
    print(f"  packets: {len(packets):,}", flush=True)

    print(f"[{day} / {tag}] aggregating flow features", flush=True)
    flow_df = aggregate_flow_features(packets, day=day)
    print(f"  flows: {len(flow_df):,}", flush=True)
    del packets

    print(f"[{day} / {tag}] joining Distrinet registry by flow_id", flush=True)
    registry = pd.read_parquet(
        registry_path,
        columns=["flow_id", "Label_Distrinet", "Attempted_Category", "matched"],
        engine="pyarrow",
    )
    flow_df = flow_df.drop(
        columns=["Label_Distrinet", "Attempted_Category", "matched"],
        errors="ignore",
    ).merge(registry, on="flow_id", how="left")
    flow_df["Label_Distrinet"] = flow_df["Label_Distrinet"].astype("string")
    flow_df["matched"] = flow_df["matched"].fillna(False).astype("bool")
    flow_df["Attempted_Category"] = (
        flow_df["Attempted_Category"].fillna(-1).astype("int16").astype("Int16")
    )

    schema_cols = _schema_columns()
    for col in schema_cols:
        if col not in flow_df.columns:
            flow_df[col] = pd.NA
    flow_df = flow_df[schema_cols]
    flow_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

    elapsed = time.perf_counter() - t0
    n_matched = int(flow_df["matched"].sum())
    print(
        f"[{day} / {tag}] done in {elapsed:.0f}s -> {out_path} "
        f"({len(flow_df):,} flows, {n_matched:,} matched)",
        flush=True,
    )
    return {
        "day": day,
        "path": str(out_path),
        "n_flows": int(len(flow_df)),
        "n_matched": n_matched,
        "elapsed_s": float(elapsed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-packets", type=int, required=True)
    parser.add_argument("--days", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    if args.max_packets <= 0:
        raise ValueError("--max-packets must be > 0")

    days = args.days or [path.stem for path in sorted(PACKET_BASE_DIR.glob("*.parquet"))]
    t0 = time.perf_counter()
    results = []
    for day in days:
        results.append(build_day(day, args.max_packets, args.force))

    tag = f"packet_abs_{args.max_packets}"
    summary_path = PARTIAL_FLOW_ROOT / tag / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2, sort_keys=True))
    print(f"\nTotal time: {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
