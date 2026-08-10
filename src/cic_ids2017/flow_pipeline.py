"""Build partial-flow artifacts for CIC-IDS2017 with Distrinet CNS 2022 labels.

Two read-only sources:
  - data/CIC-IDS2017/original/PCAPs/<day>.pcap (raw)
  - data/CIC-IDS2017/distrinet_corrected/<day>.csv (Liu et al. 2022 gold labels)

Derived (rebuildable) outputs in data/CIC-IDS2017/partial_flow/:
  - packet_base/<day>.parquet           (tshark + group_into_flows; deterministic cache)
  - distrinet_labels/<day>.parquet      (registry: flow_id → Distrinet gold label)
  - <cut_tag>/raw_flow/<day>.parquet    (truncated flow features + Distrinet label via flow_id)

Labeling: every flow_id is matched to a Distrinet CSV row by (5-tuple + flow_start_ts at
microsecond precision). Unmatched flows (~28% Wed, 17% Fri) keep `matched=False` and
`Label_Distrinet=NaN` — training pipelines should filter on `matched=True`.

The legacy IP+time labeler (pcap_flow_labeling.py) is deprecated and no longer called.
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from src.cic_ids2017.distrinet_labeling import (
    build_distrinet_labels_registry,
    coverage_summary,
    distrinet_csv_available,
    load_registry,
    save_registry,
)
from src.cic_ids2018.pcap_extracted_flow_pipeline import (
    RAW_FEATURE_COLUMNS,
    aggregate_flow_features,
    cut_tag,
    truncate_packet_base_by_cut,
)
from src.cic_ids2018.pcap_packet_extraction import group_into_flows

# Final raw_flow schema = aggregation features (without legacy Label/label_encoded)
# + Distrinet gold labels
_LEGACY_LABEL_COLS = {"Label", "label_encoded"}
_FEATURE_ONLY_COLUMNS = [c for c in RAW_FEATURE_COLUMNS if c not in _LEGACY_LABEL_COLS]
DISTRINET_FEATURE_COLUMNS = _FEATURE_ONLY_COLUMNS + [
    "Label_Distrinet",
    "Attempted_Category",
    "matched",
]

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "CIC-IDS2017"
PACKET_BASE_DIR = DATA_ROOT / "partial_flow" / "packet_base"
PARTIAL_FLOW_ROOT = DATA_ROOT / "partial_flow"

DAYS = [
    "Monday-WorkingHours",
    "Tuesday-WorkingHours",
    "Wednesday-workingHours",
    "Thursday-WorkingHours",
    "Friday-WorkingHours",
]


def _raw_flow_parquet_path(day: str, cut_mode: str, cut_value: float) -> Path:
    return PARTIAL_FLOW_ROOT / cut_tag(cut_mode, cut_value) / "raw_flow" / f"{day}.parquet"


def load_packet_base(day: str) -> pd.DataFrame:
    """Load packet_base for a day and (re)assign flow_id + direction with current code."""
    p = PACKET_BASE_DIR / f"{day}.parquet"
    if not p.exists():
        raise FileNotFoundError(f"packet_base not found: {p}")
    pkt_df = pd.read_parquet(p, engine="pyarrow")
    return group_into_flows(pkt_df, active_timeout=120.0)


def build_distrinet_registry_for_day(day: str, *, force: bool = False) -> dict:
    """Build (or load cached) per-day flow_id → Distrinet label registry.

    Aggregates the day's packet_base at pct_100 to get the 5-tuple + start_ts of
    every flow_id, then joins to Distrinet CSV. Saves to
    data/CIC-IDS2017/partial_flow/distrinet_labels/<day>.parquet.
    """
    from src.cic_ids2017.distrinet_labeling import registry_path
    out_path = registry_path(day)
    if out_path.exists() and not force:
        print(f"[{day}] registry exists, loading: {out_path}")
        df = load_registry(day)
        return {"day": day, "skipped": True, "path": str(out_path), "n_flows": len(df)}

    t0 = time.perf_counter()
    print(f"[{day}] building Distrinet registry from packet_base")
    flows_with_packets = load_packet_base(day)
    flow_df = aggregate_flow_features(flows_with_packets, day=day)
    print(f"  aggregated flows: {len(flow_df):,}")

    registry = build_distrinet_labels_registry(flow_df, day=day)
    cov = coverage_summary(registry)
    saved = save_registry(registry, day)
    elapsed = time.perf_counter() - t0
    print(
        f"[{day}] registry done in {elapsed:.0f}s -> {saved} "
        f"({cov['n_matched']:,}/{cov['n_total']:,} matched, {cov['pct_matched']:.1f}%)"
    )
    return {
        "day": day,
        "skipped": False,
        "path": str(saved),
        "n_flows": int(cov["n_total"]),
        "n_matched": int(cov["n_matched"]),
        "pct_matched": float(cov["pct_matched"]),
        "elapsed_s": float(elapsed),
    }


def build_day_partial_flow(
    day: str,
    *,
    cut_mode: str = "packet_pct",
    cut_value: float = 1.0,
    force: bool = False,
) -> dict:
    """Build a cut-specific raw_flow parquet, joining Distrinet labels by flow_id."""
    if day not in DAYS:
        raise ValueError(f"Unknown day {day!r}. Known: {DAYS}")

    tag = cut_tag(cut_mode, cut_value)
    out_path = _raw_flow_parquet_path(day, cut_mode, cut_value)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not force:
        print(f"[{day} / {tag}] exists, skipping (force=False)")
        return {"day": day, "cut_tag": tag, "skipped": True, "path": str(out_path)}

    t0 = time.perf_counter()
    print(f"[{day} / {tag}] loading packet_base + grouping into flows")
    flows_with_packets = load_packet_base(day)
    print(f"  flows_with_packets rows: {len(flows_with_packets):,}")

    print(f"[{day} / {tag}] truncating packets by {cut_mode}={cut_value}")
    truncated = truncate_packet_base_by_cut(flows_with_packets, cut_mode, cut_value)
    print(f"  truncated rows: {len(truncated):,}")

    print(f"[{day} / {tag}] aggregating flow features")
    flow_df = aggregate_flow_features(truncated, day=day)
    print(f"  flows: {len(flow_df):,}")

    flow_df = flow_df.drop(columns=list(_LEGACY_LABEL_COLS & set(flow_df.columns)))

    print(f"[{day} / {tag}] joining Distrinet registry by flow_id")
    registry = load_registry(day)[["flow_id", "Label_Distrinet", "Attempted_Category", "matched"]]
    flow_df = flow_df.merge(registry, on="flow_id", how="left")
    # Flows in the cut should always exist in the registry (same packet_base, same flow_id assignment)
    flow_df["matched"] = flow_df["matched"].fillna(False).astype("bool")
    flow_df["Attempted_Category"] = flow_df["Attempted_Category"].fillna(-1).astype("Int8")

    flow_df = flow_df[DISTRINET_FEATURE_COLUMNS]
    flow_df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    elapsed = time.perf_counter() - t0

    distri_counts = flow_df["Label_Distrinet"].value_counts(dropna=False).to_dict()
    n_matched = int(flow_df["matched"].sum())
    print(f"[{day} / {tag}] done in {elapsed:.0f}s -> {out_path} ({len(flow_df):,} flows, {n_matched:,} matched)")
    print(f"  Distrinet label distribution: {distri_counts}")
    return {
        "day": day,
        "cut_tag": tag,
        "skipped": False,
        "path": str(out_path),
        "n_flows": int(len(flow_df)),
        "n_matched": n_matched,
        "distrinet_label_counts": {str(k): int(v) for k, v in distri_counts.items()},
        "elapsed_s": float(elapsed),
    }
