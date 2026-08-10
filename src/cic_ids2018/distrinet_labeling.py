"""Apply Distrinet CNS 2022 corrected labels to CIC-IDS2018 raw-flow rows."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DISTRINET_CSV_DIR = _REPO_ROOT / "data" / "CIC-IDS2018" / "distrinet_corrected"
_REGISTRY_DIR = _REPO_ROOT / "data" / "CIC-IDS2018" / "partial_flow" / "distrinet_labels"

DAY_TO_CSV = {
    "Wednesday-14-02-2018": "Wednesday-14-02-2018.csv",
    "Thursday-15-02-2018": "Thursday-15-02-2018.csv",
    "Friday-16-02-2018": "Friday-16-02-2018.csv",
    "Tuesday-20-02-2018": "Tuesday-20-02-2018.csv",
    "Thuesday-20-02-2018": "Tuesday-20-02-2018.csv",
    "Wednesday-21-02-2018": "Wednesday-21-02-2018.csv",
    "Thursday-22-02-2018": "Thursday-22-02-2018.csv",
    "Friday-23-02-2018": "Friday-23-02-2018.csv",
    "Wednesday-28-02-2018": "Wednesday-28-02-2018.csv",
    "Thursday-01-03-2018": "Thursday-01-03-2018.csv",
    "Friday-02-03-2018": "Friday-02-03-2018.csv",
}

_USECOLS = [
    "Src IP",
    "Src Port",
    "Dst IP",
    "Dst Port",
    "Protocol",
    "Timestamp",
    "Label",
    "Attempted Category",
]

_KEY_COLUMNS = [
    "flow_src_ip",
    "flow_src_port",
    "flow_dst_ip",
    "flow_dst_port",
    "Protocol",
    "flow_start_ts_us",
]

_REGISTRY_COLUMNS = [
    "flow_id",
    "flow_src_ip",
    "flow_dst_ip",
    "flow_src_port",
    "flow_dst_port",
    "Protocol",
    "flow_start_ts",
    "Label_Distrinet",
    "Attempted_Category",
    "matched",
]


def distrinet_csv_path(day: str) -> Path:
    try:
        return DISTRINET_CSV_DIR / DAY_TO_CSV[day]
    except KeyError as exc:
        raise KeyError(f"No Distrinet CSV mapping configured for day: {day}") from exc


def distrinet_csv_available(day: str) -> bool:
    return distrinet_csv_path(day).exists()


def registry_path(day: str) -> Path:
    return _REGISTRY_DIR / f"{DAY_TO_CSV.get(day, day + '.csv').removesuffix('.csv')}.parquet"


def _timestamp_to_microseconds(series: pd.Series) -> pd.Series:
    # Pandas may infer datetime64[us] for these CSVs; force microseconds explicitly.
    values = pd.to_datetime(series).to_numpy(dtype="datetime64[us]").astype("int64")
    return pd.Series(values, index=series.index, dtype="int64")


def load_distrinet_lookup_frame(day: str) -> pd.DataFrame:
    """Load one Distrinet CSV as a merge-ready lookup frame."""
    csv_path = distrinet_csv_path(day)
    if not csv_path.exists():
        raise FileNotFoundError(f"Distrinet CSV not found: {csv_path}")

    df = pd.read_csv(csv_path, usecols=_USECOLS)
    df = df.rename(
        columns={
            "Src IP": "flow_src_ip",
            "Src Port": "flow_src_port",
            "Dst IP": "flow_dst_ip",
            "Dst Port": "flow_dst_port",
            "Timestamp": "flow_start_ts_us",
            "Label": "Label_Distrinet",
            "Attempted Category": "Attempted_Category",
        }
    )
    df["flow_start_ts_us"] = _timestamp_to_microseconds(df["flow_start_ts_us"])
    for col in ("flow_src_port", "flow_dst_port", "Protocol"):
        df[col] = df[col].astype("int64")
    df["Label_Distrinet"] = df["Label_Distrinet"].astype("string")
    df["Attempted_Category"] = df["Attempted_Category"].fillna(-1).astype("int16")

    # Duplicate keys would duplicate raw-flow rows during merge. Keep the first row,
    # matching the first-match policy used elsewhere in the pipeline.
    return df[_KEY_COLUMNS + ["Label_Distrinet", "Attempted_Category"]].drop_duplicates(
        subset=_KEY_COLUMNS,
        keep="first",
    )


def apply_distrinet_labels(flows_df: pd.DataFrame, day: str) -> pd.DataFrame:
    """Add Label_Distrinet, Attempted_Category, and matched columns."""
    lookup = load_distrinet_lookup_frame(day)

    out = flows_df.copy()
    out = out.drop(
        columns=["Label_Distrinet", "Attempted_Category", "matched", "flow_start_ts_us"],
        errors="ignore",
    )
    out["flow_start_ts_us"] = (
        out["flow_start_ts"].astype("float64").mul(1_000_000).round().astype("int64")
    )
    for col in ("flow_src_port", "flow_dst_port", "Protocol"):
        out[col] = out[col].astype("int64")

    merged = out.merge(lookup, on=_KEY_COLUMNS, how="left", sort=False, validate="many_to_one")
    merged = merged.drop(columns=["flow_start_ts_us"])
    merged["Label_Distrinet"] = merged["Label_Distrinet"].astype("string")
    merged["matched"] = merged["Label_Distrinet"].notna().astype("bool")
    merged["Attempted_Category"] = (
        merged["Attempted_Category"].fillna(-1).astype("int16").astype("Int16")
    )
    return merged


def coverage_summary(flows_df: pd.DataFrame) -> dict:
    n_total = int(len(flows_df))
    n_matched = int(flows_df["matched"].sum())
    return {
        "n_total": n_total,
        "n_matched": n_matched,
        "n_unmatched": n_total - n_matched,
        "pct_matched": (100.0 * n_matched / n_total) if n_total else 0.0,
        "distrinet_label_counts": {
            str(k): int(v)
            for k, v in flows_df["Label_Distrinet"].value_counts(dropna=False).items()
        },
    }


def build_distrinet_labels_registry(flows_df: pd.DataFrame, day: str) -> pd.DataFrame:
    labeled = apply_distrinet_labels(flows_df, day)
    return labeled[_REGISTRY_COLUMNS].copy()


def save_registry(registry_df: pd.DataFrame, day: str) -> Path:
    out_path = registry_path(day)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    registry_df[_REGISTRY_COLUMNS].to_parquet(
        out_path,
        engine="pyarrow",
        compression="snappy",
        index=False,
    )
    return out_path
