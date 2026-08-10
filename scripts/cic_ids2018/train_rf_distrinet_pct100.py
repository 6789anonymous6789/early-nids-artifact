"""Random Forest baseline on CIC-IDS2018 pct_100 with Distrinet labels."""

from __future__ import annotations

import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split


DEFAULT_DATA_DIR = Path("data/CIC-IDS2018/partial_flow/pct_100/raw_flow")
SUMMARY_PATH = Path("data/CIC-IDS2018/partial_flow/distrinet_labels/summary.json")
DEFAULT_OUT_DIR = Path("outputs/cic_ids2018/rf_distrinet_pct100_all_classes")

DROP_COLUMNS = {
    "day",
    "capture_file",
    "flow_id",
    "flow_src_ip",
    "flow_dst_ip",
    "flow_src_port",
    "flow_dst_port",
    "flow_start_ts",
    "flow_end_ts",
    "Dst Port",
    "Label_Distrinet",
    "Attempted_Category",
    "matched",
    "Label",
    "label_encoded",
}
N_ESTIMATORS = 100
TEST_SIZE = 0.25
RANDOM_STATE = 42


def collapse_attempted(label: pd.Series) -> pd.Series:
    return label.astype("string").str.replace(r" - Attempted$", "", regex=True)


def parquet_feature_columns(path: Path) -> list[str]:
    schema_cols = pq.ParquetFile(path).schema_arrow.names
    return [col for col in schema_cols if col not in DROP_COLUMNS]


def expected_matched_rows(paths: list[Path]) -> int:
    if SUMMARY_PATH.exists():
        summary = json.loads(SUMMARY_PATH.read_text())
        return int(sum(summary[path.stem]["n_matched"] for path in paths))
    return 0


def class_real_attempted_counts(paths: list[Path]) -> pd.DataFrame:
    counts: dict[str, dict[str, int]] = {}
    for path in paths:
        df = pd.read_parquet(path, columns=["Label_Distrinet", "matched"], engine="pyarrow")
        df = df.loc[df["matched"].astype("bool")].copy()
        labels = df["Label_Distrinet"].astype("string")
        base = collapse_attempted(labels)
        attempted = labels.str.endswith(" - Attempted", na=False)
        grouped = pd.DataFrame({"class": base, "attempted": attempted}).groupby("class")[
            "attempted"
        ].agg(total="size", attempted="sum")
        grouped["real"] = grouped["total"] - grouped["attempted"]
        for cls, row in grouped.iterrows():
            item = counts.setdefault(str(cls), {"total": 0, "attempted": 0, "real": 0})
            item["total"] += int(row["total"])
            item["attempted"] += int(row["attempted"])
            item["real"] += int(row["real"])
    return pd.DataFrame([{"class": cls, **values} for cls, values in counts.items()]).sort_values(
        ["real", "total"],
        ascending=False,
    )


def load_dataset(
    paths: list[Path],
    feature_cols: list[str],
    keep_classes: set[str] | None,
) -> tuple[np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    n_expected = expected_matched_rows(paths)
    if n_expected <= 0:
        raise RuntimeError("Cannot determine matched row count from summary.json")

    x = np.empty((n_expected, len(feature_cols)), dtype=np.float32)
    y = np.empty(n_expected, dtype=np.int32)
    label_to_id: dict[str, int] = {}
    offset = 0
    counts_rows = []

    columns = feature_cols + ["Label_Distrinet", "matched"]
    for path in paths:
        start = time.perf_counter()
        df = pd.read_parquet(path, columns=columns, engine="pyarrow")
        df = df.loc[df["matched"].astype("bool")].copy()
        labels = collapse_attempted(df["Label_Distrinet"])
        if keep_classes is not None:
            keep_mask = labels.isin(keep_classes)
            df = df.loc[keep_mask].copy()
            labels = labels.loc[keep_mask]
        for label in labels.unique():
            label_str = str(label)
            if label_str not in label_to_id:
                label_to_id[label_str] = len(label_to_id)

        n = len(df)
        x_part = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        np.nan_to_num(x_part, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        y_part = labels.map(label_to_id).to_numpy(dtype=np.int32, copy=True)

        if offset + n > len(x):
            raise RuntimeError(f"Preallocated rows too small: need {offset + n}, have {len(x)}")
        x[offset : offset + n] = x_part
        y[offset : offset + n] = y_part
        offset += n

        counts = labels.value_counts().rename_axis("class").reset_index(name="rows")
        counts.insert(0, "day", path.stem)
        counts_rows.append(counts)
        print(
            f"loaded {path.name}: matched={n:,}, total_loaded={offset:,}, "
            f"elapsed={time.perf_counter() - start:.1f}s",
            flush=True,
        )

        del df, labels, x_part, y_part

    if offset != len(x):
        x = x[:offset]
        y = y[:offset]

    classes = [None] * len(label_to_id)
    for label, idx in label_to_id.items():
        classes[idx] = label
    class_by_day = pd.concat(counts_rows, ignore_index=True)
    return x, y, classes, class_by_day


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(data_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    feature_cols = parquet_feature_columns(paths[0])
    pd.DataFrame({"feature": feature_cols}).to_csv(out_dir / "feature_columns.csv", index=False)

    real_counts = class_real_attempted_counts(paths)
    real_counts["keep"] = real_counts["real"] >= args.min_real
    real_counts.to_csv(out_dir / "class_real_attempted_counts.csv", index=False)
    keep_classes = set(real_counts.loc[real_counts["keep"], "class"].astype(str))
    if args.min_real <= 0:
        keep_classes = None

    print(f"files={len(paths)} features={len(feature_cols)}", flush=True)
    if keep_classes is not None:
        print(
            f"min_real={args.min_real} kept_classes={len(keep_classes)} "
            f"dropped_classes={int((~real_counts['keep']).sum())}",
            flush=True,
        )
        print(real_counts.to_string(index=False), flush=True)
    prep_start = time.perf_counter()
    x, y, classes, class_by_day = load_dataset(paths, feature_cols, keep_classes)
    prep_time = time.perf_counter() - prep_start

    class_counts = pd.Series(y).value_counts().sort_index()
    class_counts_df = pd.DataFrame(
        {
            "class_id": class_counts.index,
            "class": [classes[i] for i in class_counts.index],
            "rows": class_counts.to_numpy(),
        }
    ).sort_values("rows", ascending=False)
    class_counts_df.to_csv(out_dir / "class_counts.csv", index=False)
    class_by_day.to_csv(out_dir / "class_counts_by_day.csv", index=False)

    print(
        f"dataset rows={len(y):,}, classes={len(classes)}, prep_time={prep_time:.1f}s",
        flush=True,
    )
    print(class_counts_df.to_string(index=False), flush=True)

    split_start = time.perf_counter()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    del x, y
    print(
        f"split train={len(y_train):,}, test={len(y_test):,}, "
        f"elapsed={time.perf_counter() - split_start:.1f}s",
        flush=True,
    )

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        verbose=1,
    )
    train_start = time.perf_counter()
    clf.fit(x_train, y_train)
    train_time = time.perf_counter() - train_start
    print(f"training_time={train_time:.1f}s", flush=True)

    pred_start = time.perf_counter()
    y_pred = clf.predict(x_test)
    pred_time = time.perf_counter() - pred_start
    print(f"prediction_time={pred_time:.1f}s", flush=True)

    labels = np.arange(len(classes), dtype=np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0,
    )
    per_class = pd.DataFrame(
        {
            "class_id": labels,
            "class": classes,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    ).sort_values("support", ascending=False)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / "confusion_matrix.csv")

    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(out_dir / "feature_importances.csv", index=False)

    summary = {
        "data_dir": str(data_dir),
        "rows": int(len(y_train) + len(y_test)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "features": int(len(feature_cols)),
        "classes": classes,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "class_weight": "balanced",
        "collapse_attempted": True,
        "matched_only": True,
        "min_real": args.min_real,
        "kept_classes": sorted(classes),
        "drop_columns": sorted(DROP_COLUMNS),
        "prep_time_s": prep_time,
        "train_time_s": train_time,
        "prediction_time_s": pred_time,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    (out_dir / "run_config.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).drop(columns=["classes", "drop_columns"]).to_csv(
        out_dir / "summary.csv",
        index=False,
    )

    print(json.dumps(summary, indent=2), flush=True)
    print(per_class.to_string(index=False), flush=True)
    print(importances.head(15).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--min-real", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    main()
