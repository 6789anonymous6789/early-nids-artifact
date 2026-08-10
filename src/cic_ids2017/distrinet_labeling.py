"""Apply Distrinet CNS 2022 corrected labels to PCAP-extracted flows (2017).

The Distrinet team (Liu et al., IEEE CNS 2022 Best Paper) reverse-engineered
the CIC-IDS2017 labeling from raw PCAPs and released per-day corrected CSVs at
https://intrusion-detection.distrinet-research.be/CNS2022/Datasets/. Their
labeling uses packet-content inspection to split each attack class into the
real attack and an `X - Attempted` sibling (failed connections, empty payloads,
target-unresponsive retransmits, etc.).

We match Distrinet rows to our pct_100 raw_flow rows on (5-tuple + flow start
timestamp at microsecond precision). Flows that don't match (~38% on Wed,
mostly TCP appendices and artefacts Distrinet pruned) get `matched=False` and
NaN labels — they're kept in the parquet but training pipelines should filter
on `matched=True` to use Distrinet as ground truth.

The legacy `Label` / `label_encoded` columns from our IP+time labeler are kept
for now (deprecated, will be removed once Distrinet labels are validated).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DISTRINET_CSV_DIR = _REPO_ROOT / "data" / "CIC-IDS2017" / "distrinet_corrected"

DAY_TO_CSV = {
    "Monday-WorkingHours":    "monday.csv",
    "Tuesday-WorkingHours":   "tuesday.csv",
    "Wednesday-workingHours": "wednesday.csv",
    "Thursday-WorkingHours":  "thursday.csv",
    "Friday-WorkingHours":    "friday.csv",
}

_USECOLS = [
    "Src IP", "Src Port", "Dst IP", "Dst Port", "Protocol",
    "Timestamp", "Label", "Attempted Category",
]


def distrinet_csv_available(day: str) -> bool:
    csv = DISTRINET_CSV_DIR / DAY_TO_CSV.get(day, "")
    return csv.exists()


def _build_key(src_ip, dst_ip, src_port, dst_port, proto, ts_us) -> pd.Series:
    return (
        src_ip.astype(str) + "|" + dst_ip.astype(str) + "|"
        + src_port.astype(str) + "|" + dst_port.astype(str) + "|"
        + proto.astype(str) + "|" + ts_us.astype(str)
    )


def load_distrinet_lookup(day: str) -> dict[str, tuple[str, int]]:
    """Build a (5-tuple+ts_us) → (Label, Attempted Category) dict for one day."""
    csv = DISTRINET_CSV_DIR / DAY_TO_CSV[day]
    if not csv.exists():
        raise FileNotFoundError(f"Distrinet CSV not found: {csv}")

    df = pd.read_csv(csv, usecols=_USECOLS)
    # Distrinet Timestamp parses as datetime64[us]; .astype('int64') = epoch microseconds (UTC).
    ts_us = pd.to_datetime(df["Timestamp"]).astype("int64")
    keys = _build_key(
        df["Src IP"], df["Dst IP"], df["Src Port"], df["Dst Port"], df["Protocol"], ts_us,
    )
    cat = df["Attempted Category"].fillna(-1).astype("int64")
    return dict(zip(keys, zip(df["Label"].astype(str), cat)))


def apply_distrinet_labels(flows_df: pd.DataFrame, day: str) -> pd.DataFrame:
    """Add Label_Distrinet, Attempted_Category, matched columns by joining on
    (flow_src_ip, flow_dst_ip, flow_src_port, flow_dst_port, Protocol, flow_start_ts_us).
    """
    lookup = load_distrinet_lookup(day)

    ts_us = (flows_df["flow_start_ts"].astype("float64") * 1_000_000).round().astype("int64")
    keys = _build_key(
        flows_df["flow_src_ip"], flows_df["flow_dst_ip"],
        flows_df["flow_src_port"], flows_df["flow_dst_port"],
        flows_df["Protocol"], ts_us,
    )

    n = len(flows_df)
    label_distri = np.empty(n, dtype=object)
    attempted_cat = np.full(n, -1, dtype=np.int64)
    matched = np.zeros(n, dtype=bool)

    for i, k in enumerate(keys.to_numpy()):
        v = lookup.get(k)
        if v is not None:
            label_distri[i] = v[0]
            attempted_cat[i] = v[1]
            matched[i] = True

    out = flows_df.copy()
    out["Label_Distrinet"] = pd.array(label_distri, dtype="string")
    out["Attempted_Category"] = pd.array(attempted_cat, dtype="Int8")
    out["matched"] = pd.array(matched, dtype="bool")
    return out


def coverage_summary(flows_df: pd.DataFrame) -> dict:
    """Per-day coverage statistics for diagnostics."""
    n = len(flows_df)
    n_matched = int(flows_df["matched"].sum())
    return {
        "n_total": n,
        "n_matched": n_matched,
        "n_unmatched": n - n_matched,
        "pct_matched": (100.0 * n_matched / n) if n else 0.0,
    }


# ── Registry: one row per flow_id, gold Distrinet label per flow ──

_REGISTRY_DIR = _REPO_ROOT / "data" / "CIC-IDS2017" / "partial_flow" / "distrinet_labels"
_REGISTRY_COLUMNS = [
    "flow_id",
    "flow_src_ip", "flow_dst_ip", "flow_src_port", "flow_dst_port",
    "Protocol", "flow_start_ts",
    "Label_Distrinet", "Attempted_Category", "matched",
]


def registry_path(day: str) -> Path:
    return _REGISTRY_DIR / f"{day}.parquet"


def build_distrinet_labels_registry(flows_df: pd.DataFrame, day: str) -> pd.DataFrame:
    """Given an aggregated flows_df (one row per flow_id, with 5-tuple +
    flow_start_ts), produce the registry parquet content: flow_id → Distrinet
    label. The aggregation is done by the caller (flow_pipeline) so we don't
    duplicate that logic here.
    """
    if not distrinet_csv_available(day):
        # Monday and other no-attack days: still build a registry of all-Benign
        # but with matched=False so downstream knows there's no Distrinet entry.
        n = len(flows_df)
        out = flows_df[[
            "flow_id", "flow_src_ip", "flow_dst_ip", "flow_src_port", "flow_dst_port",
            "Protocol", "flow_start_ts",
        ]].copy()
        out["Label_Distrinet"] = pd.array([None] * n, dtype="string")
        out["Attempted_Category"] = pd.array([-1] * n, dtype="Int8")
        out["matched"] = pd.array([False] * n, dtype="bool")
        return out

    enriched = apply_distrinet_labels(flows_df, day=day)
    return enriched[_REGISTRY_COLUMNS].copy()


def save_registry(registry_df: pd.DataFrame, day: str) -> Path:
    out_path = registry_path(day)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    registry_df[_REGISTRY_COLUMNS].to_parquet(
        out_path, engine="pyarrow", compression="snappy", index=False,
    )
    return out_path


def load_registry(day: str) -> pd.DataFrame:
    p = registry_path(day)
    if not p.exists():
        raise FileNotFoundError(f"Distrinet registry missing for {day}: {p}")
    return pd.read_parquet(p, engine="pyarrow")
