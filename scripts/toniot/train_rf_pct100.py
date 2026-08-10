"""Random Forest baseline on ToN-IoT partial-flow datasets.

Mirrors ``scripts/cic_ids2018/train_rf_distrinet_pct100.py`` (RF, 100 trees,
75/25 stratified split, macro/weighted F1 + per-class metrics + confusion
matrix + feature importances), adapted to ToN-IoT:

  * The dataset is a single pre-assembled ``<cut>/dataset.parquet`` (not one
    parquet per day), already globally capped.
  * Label column is ``Label_ToNIoT``; no "Attempted" sublabels.
  * ``normal`` is kept as a class (benign-vs-attack detection + attack typing),
    per the chosen design. ``--min-real`` can still drop rare classes
    (default 0 = keep all).

Usage:
    python -m scripts.toniot.train_rf_pct100 --cut pct_100
    python -m scripts.toniot.train_rf_pct100 --cut packet_abs_3 --min-real 10000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

PARTIAL_FLOW_ROOT = Path("data/ToN-IoT/partial_flow")

# Non-feature columns: IDs, endpoint identifiers, timestamps, label bookkeeping.
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
    "Label_ToNIoT",
    "label_encoded",
    "matched",
    "match_level",
}
LABEL_COL = "Label_ToNIoT"
N_ESTIMATORS = 100
TEST_SIZE = 0.25
RANDOM_STATE = 42


def parquet_feature_columns(path: Path) -> list[str]:
    schema_cols = pq.ParquetFile(path).schema_arrow.names
    return [c for c in schema_cols if c not in DROP_COLUMNS]


def main() -> None:
    args = parse_args()
    global RANDOM_STATE
    RANDOM_STATE = args.random_state
    data_path = args.dataset or (PARTIAL_FLOW_ROOT / args.cut / "dataset.parquet")
    if not data_path.exists():
        raise FileNotFoundError(f"dataset not found: {data_path}")
    out_dir = args.output_dir or Path(f"outputs/toniot/rf_{args.cut}")
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = parquet_feature_columns(data_path)
    if args.drop_features:
        drop = set(args.drop_features)
        missing = drop - set(feature_cols)
        if missing:
            raise ValueError(f"--drop-features not in dataset: {sorted(missing)}")
        feature_cols = [c for c in feature_cols if c not in drop]
        print(f"ablation: dropped {sorted(drop)}", flush=True)
    pd.DataFrame({"feature": feature_cols}).to_csv(out_dir / "feature_columns.csv", index=False)
    print(f"dataset={data_path} features={len(feature_cols)}", flush=True)

    prep_start = time.perf_counter()
    df = pd.read_parquet(data_path, columns=feature_cols + [LABEL_COL], engine="pyarrow")
    labels = df[LABEL_COL].astype("string")

    # Optional rare-class drop (default keep all).
    class_counts_all = labels.value_counts()
    if args.min_real > 0:
        keep_classes = set(class_counts_all[class_counts_all >= args.min_real].index.astype(str))
        mask = labels.isin(keep_classes)
        df = df.loc[mask].copy()
        labels = labels.loc[mask]
        dropped = sorted(set(class_counts_all.index.astype(str)) - keep_classes)
        print(f"min_real={args.min_real} kept={len(keep_classes)} dropped={dropped}", flush=True)

    classes = sorted(labels.unique().tolist())
    label_to_id = {c: i for i, c in enumerate(classes)}
    x = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    y = labels.map(label_to_id).to_numpy(dtype=np.int32)
    prep_time = time.perf_counter() - prep_start

    class_counts_df = (
        pd.Series(y).value_counts().rename_axis("class_id").reset_index(name="rows")
    )
    class_counts_df["class"] = class_counts_df["class_id"].map({v: k for k, v in label_to_id.items()})
    class_counts_df = class_counts_df[["class_id", "class", "rows"]].sort_values("rows", ascending=False)
    class_counts_df.to_csv(out_dir / "class_counts.csv", index=False)
    print(f"dataset rows={len(y):,}, classes={len(classes)}, prep_time={prep_time:.1f}s", flush=True)
    print(class_counts_df.to_string(index=False), flush=True)

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
    )
    del x, y
    print(f"split train={len(y_train):,} test={len(y_test):,}", flush=True)

    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    train_start = time.perf_counter()
    clf.fit(x_train, y_train)
    train_time = time.perf_counter() - train_start
    print(f"training_time={train_time:.1f}s", flush=True)

    pred_start = time.perf_counter()
    y_pred = clf.predict(x_test)
    pred_time = time.perf_counter() - pred_start

    label_ids = np.arange(len(classes), dtype=np.int32)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, labels=label_ids, zero_division=0,
    )
    per_class = pd.DataFrame({
        "class_id": label_ids,
        "class": classes,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }).sort_values("support", ascending=False)
    per_class.to_csv(out_dir / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_test, y_pred, labels=label_ids)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out_dir / "confusion_matrix.csv")

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(out_dir / "feature_importances.csv", index=False)

    summary = {
        "dataset": str(data_path),
        "cut": args.cut,
        "rows": int(len(y_train) + len(y_test)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "features": int(len(feature_cols)),
        "classes": classes,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_estimators": N_ESTIMATORS,
        "min_real": args.min_real,
        "dropped_features": list(args.drop_features),
        "normal_included": "normal" in classes,
        "prep_time_s": prep_time,
        "train_time_s": train_time,
        "prediction_time_s": pred_time,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    (out_dir / "run_config.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).drop(columns=["classes"]).to_csv(out_dir / "summary.csv", index=False)

    print(json.dumps({k: v for k, v in summary.items() if k != "classes"}, indent=2), flush=True)
    print(per_class.to_string(index=False), flush=True)
    print(importances.head(15).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cut", default="pct_100",
                   choices=["pct_100", "packet_abs_3", "packet_abs_4", "packet_abs_5"],
                   help="Which partial-flow cut's dataset.parquet to train on.")
    p.add_argument("--dataset", type=Path, default=None,
                   help="Explicit dataset.parquet path (overrides --cut).")
    p.add_argument("--min-real", type=int, default=0,
                   help="Drop classes with fewer than this many flows (0 = keep all).")
    p.add_argument("--drop-features", nargs="*", default=[],
                   help="Feature columns to exclude from training (leakage ablation).")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--random-state", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    main()
