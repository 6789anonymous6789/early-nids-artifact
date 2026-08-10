"""Second-pass fuzzy Distrinet matching for CIC-IDS2018 flow labels.

The strict pass matches on 5-tuple plus exact flow-start timestamp. This script
recovers boundary-shifted flows by matching still-unmatched raw flows to still-
uncovered Distrinet rows with the same 5-tuple and a small timestamp tolerance.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2018.distrinet_labeling import load_distrinet_lookup_frame

RAW_FLOW_DIR = REPO_ROOT / "data" / "CIC-IDS2018" / "partial_flow" / "pct_100" / "raw_flow"
REGISTRY_DIR = REPO_ROOT / "data" / "CIC-IDS2018" / "partial_flow" / "distrinet_labels"
SUMMARY_PATH = REGISTRY_DIR / "summary.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "cic_ids2018" / "distrinet_fuzzy_matching"

KEY5 = ["flow_src_ip", "flow_src_port", "flow_dst_ip", "flow_dst_port", "Protocol"]
KEY6 = KEY5 + ["flow_start_ts_us"]
REGISTRY_COLUMNS = [
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


def _flow_start_us(series: pd.Series) -> pd.Series:
    return series.astype("float64").mul(1_000_000).round().astype("int64")


def _hash_frame(df: pd.DataFrame, cols: list[str]) -> np.ndarray:
    return pd.util.hash_pandas_object(df[cols], index=False).to_numpy(dtype=np.uint64)


def _match_sorted_one_to_one(
    distri_times: np.ndarray,
    distri_rows: np.ndarray,
    raw_times: np.ndarray,
    raw_positions: np.ndarray,
    tolerance_us: int,
) -> list[tuple[int, int, int]]:
    """Order-preserving one-to-one matching on sorted timestamps."""
    matches: list[tuple[int, int, int]] = []
    i = 0
    j = 0
    while i < len(distri_times) and j < len(raw_times):
        dt = int(distri_times[i])
        rt = int(raw_times[j])
        delta = rt - dt
        if delta < -tolerance_us:
            j += 1
        elif delta > tolerance_us:
            i += 1
        else:
            matches.append((int(distri_rows[i]), int(raw_positions[j]), int(delta)))
            i += 1
            j += 1
    return matches


def fuzzy_match_day(
    day: str,
    *,
    threshold_seconds: float,
    labels: set[str] | None,
) -> tuple[pd.DataFrame, dict]:
    raw_path = RAW_FLOW_DIR / f"{day}.parquet"
    if not raw_path.exists():
        raise FileNotFoundError(raw_path)

    raw_cols = [
        "flow_id",
        "flow_src_ip",
        "flow_src_port",
        "flow_dst_ip",
        "flow_dst_port",
        "Protocol",
        "flow_start_ts",
        "Label_Distrinet",
        "Attempted_Category",
        "matched",
    ]
    raw = pd.read_parquet(raw_path, columns=raw_cols)
    raw["flow_start_ts_us"] = _flow_start_us(raw["flow_start_ts"])
    for col in ("flow_src_port", "flow_dst_port", "Protocol"):
        raw[col] = raw[col].astype("int64")
    raw["_raw_pos"] = np.arange(len(raw), dtype=np.int64)

    distri = load_distrinet_lookup_frame(day).copy()
    distri["_distri_row"] = np.arange(len(distri), dtype=np.int64)

    raw_exact_hash = np.sort(_hash_frame(raw, KEY6))
    distri_exact_hash = _hash_frame(distri, KEY6)
    missing_distri = distri.loc[~np.isin(distri_exact_hash, raw_exact_hash)].copy()
    if labels is not None:
        missing_distri = missing_distri.loc[missing_distri["Label_Distrinet"].isin(labels)].copy()

    raw_unmatched = raw.loc[~raw["matched"].astype("bool")].copy()
    if missing_distri.empty or raw_unmatched.empty:
        return pd.DataFrame(), {
            "day": day,
            "threshold_seconds": threshold_seconds,
            "candidate_distrinet_rows": int(len(missing_distri)),
            "candidate_raw_rows": int(len(raw_unmatched)),
            "fuzzy_matches": 0,
        }

    missing_tuple_hash = np.unique(_hash_frame(missing_distri, KEY5))
    raw_tuple_hash = _hash_frame(raw_unmatched, KEY5)
    raw_candidates = raw_unmatched.loc[np.isin(raw_tuple_hash, missing_tuple_hash)].copy()

    tolerance_us = int(round(threshold_seconds * 1_000_000))
    raw_groups = {}
    for key, group in raw_candidates.sort_values("flow_start_ts_us").groupby(KEY5, sort=False):
        raw_groups[key] = (
            group["flow_start_ts_us"].to_numpy(dtype=np.int64),
            group["_raw_pos"].to_numpy(dtype=np.int64),
        )

    all_matches: list[tuple[int, int, int]] = []
    for key, group in missing_distri.sort_values("flow_start_ts_us").groupby(KEY5, sort=False):
        raw_group = raw_groups.get(key)
        if raw_group is None:
            continue
        distri_times = group["flow_start_ts_us"].to_numpy(dtype=np.int64)
        distri_rows = group["_distri_row"].to_numpy(dtype=np.int64)
        raw_times, raw_positions = raw_group
        all_matches.extend(
            _match_sorted_one_to_one(
                distri_times=distri_times,
                distri_rows=distri_rows,
                raw_times=raw_times,
                raw_positions=raw_positions,
                tolerance_us=tolerance_us,
            )
        )

    if not all_matches:
        return pd.DataFrame(), {
            "day": day,
            "threshold_seconds": threshold_seconds,
            "candidate_distrinet_rows": int(len(missing_distri)),
            "candidate_raw_rows": int(len(raw_candidates)),
            "fuzzy_matches": 0,
        }

    matches = pd.DataFrame(all_matches, columns=["_distri_row", "_raw_pos", "delta_us"])
    distri_match_cols = ["_distri_row", "Label_Distrinet", "Attempted_Category", *KEY6]
    raw_match_cols = ["_raw_pos", "flow_id", "flow_start_ts", *KEY5]
    matches = matches.merge(distri[distri_match_cols], on="_distri_row", how="left")
    matches = matches.merge(raw[raw_match_cols], on="_raw_pos", how="left", suffixes=("_distrinet", ""))
    matches["abs_delta_us"] = matches["delta_us"].abs().astype("int64")

    # Defensive one-to-one enforcement if duplicate positions appear after grouping.
    matches = (
        matches.sort_values(["abs_delta_us", "_distri_row", "_raw_pos"])
        .drop_duplicates(subset=["_raw_pos"], keep="first")
        .drop_duplicates(subset=["_distri_row"], keep="first")
        .sort_values("_raw_pos")
        .reset_index(drop=True)
    )

    summary = {
        "day": day,
        "threshold_seconds": threshold_seconds,
        "candidate_distrinet_rows": int(len(missing_distri)),
        "candidate_raw_rows": int(len(raw_candidates)),
        "fuzzy_matches": int(len(matches)),
        "labels": sorted(labels) if labels is not None else None,
        "delta_seconds_quantiles": {
            str(q): float(matches["abs_delta_us"].quantile(q) / 1_000_000.0)
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        },
        "matches_by_label": {
            str(k): int(v) for k, v in matches["Label_Distrinet"].value_counts().items()
        },
    }
    return matches, summary


def apply_matches(day: str, matches: pd.DataFrame) -> dict:
    raw_path = RAW_FLOW_DIR / f"{day}.parquet"
    registry_path = REGISTRY_DIR / f"{day}.parquet"
    raw = pd.read_parquet(raw_path)

    label_by_pos = matches.set_index("_raw_pos")["Label_Distrinet"]
    attempted_by_pos = matches.set_index("_raw_pos")["Attempted_Category"]
    positions = label_by_pos.index.to_numpy(dtype=np.int64)

    raw.loc[positions, "Label_Distrinet"] = raw.loc[positions].index.map(label_by_pos)
    raw.loc[positions, "Attempted_Category"] = raw.loc[positions].index.map(attempted_by_pos)
    raw.loc[positions, "matched"] = True
    raw["Label_Distrinet"] = raw["Label_Distrinet"].astype("string")
    raw["Attempted_Category"] = raw["Attempted_Category"].fillna(-1).astype("int16").astype("Int16")
    raw["matched"] = raw["matched"].astype("bool")

    tmp_raw = raw_path.with_suffix(".tmp.parquet")
    raw.to_parquet(tmp_raw, engine="pyarrow", compression="snappy", index=False)
    tmp_raw.replace(raw_path)

    registry = raw[REGISTRY_COLUMNS].copy()
    tmp_registry = registry_path.with_suffix(".tmp.parquet")
    registry.to_parquet(tmp_registry, engine="pyarrow", compression="snappy", index=False)
    tmp_registry.replace(registry_path)

    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
    else:
        summary = {}
    n_total = int(len(raw))
    n_matched = int(raw["matched"].sum())
    summary[day] = {
        "n_total": n_total,
        "n_matched": n_matched,
        "n_unmatched": n_total - n_matched,
        "pct_matched": (100.0 * n_matched / n_total) if n_total else 0.0,
        "distrinet_label_counts": {
            str(k): int(v) for k, v in raw["Label_Distrinet"].value_counts(dropna=False).items()
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, sort_keys=True))

    return {
        "n_total": n_total,
        "n_matched": n_matched,
        "n_unmatched": n_total - n_matched,
    }


def write_reports(day: str, matches: pd.DataFrame, summary: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{day}_tol_{summary['threshold_seconds']:g}s"
    (output_dir / f"{tag}_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    if matches.empty:
        return
    matches["Label_Distrinet"].value_counts().rename_axis("Label_Distrinet").reset_index(
        name="fuzzy_matches"
    ).to_csv(output_dir / f"{tag}_by_label.csv", index=False)
    delta_quantiles = pd.DataFrame(
        [
            {
                "quantile": q,
                "abs_delta_seconds": float(matches["abs_delta_us"].quantile(q) / 1_000_000.0),
            }
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        ]
    )
    delta_quantiles.to_csv(output_dir / f"{tag}_delta_quantiles.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("day", help="Day name, e.g. Friday-16-02-2018")
    parser.add_argument("--threshold-seconds", type=float, default=1.0)
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Restrict Distrinet labels to fuzzy-match. Can be repeated.",
    )
    parser.add_argument("--apply", action="store_true", help="Update raw_flow and registry parquet files.")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels = set(args.label) if args.label else None
    matches, summary = fuzzy_match_day(
        args.day,
        threshold_seconds=args.threshold_seconds,
        labels=labels,
    )
    write_reports(args.day, matches, summary, args.output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.apply and not matches.empty:
        applied = apply_matches(args.day, matches)
        print(json.dumps({"applied": applied}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
