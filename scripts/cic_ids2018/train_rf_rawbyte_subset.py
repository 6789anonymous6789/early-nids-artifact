"""Random Forest on the same CIC-IDS2018 flow subset used by raw-byte models."""

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


def collapse_attempted(labels: pd.Series) -> pd.Series:
    return labels.astype("string").str.replace(r" - Attempted$", "", regex=True)


def parquet_feature_columns(path: Path) -> list[str]:
    schema_cols = pq.ParquetFile(path).schema_arrow.names
    return [col for col in schema_cols if col not in DROP_COLUMNS]


def load_rawbyte_subset(
    raw_bytes_dir: Path,
    min_real: int,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    frames = []
    for path in sorted(raw_bytes_dir.glob("*.parquet")):
        df = pd.read_parquet(
            path,
            columns=["flow_id", "found", "Label_Distrinet"],
            engine="pyarrow",
        )
        df = df[df["found"].astype("bool")].copy()
        df["day"] = path.stem
        frames.append(df)
        print(f"raw-byte subset source {path.name}: {len(df):,} packet rows", flush=True)

    if not frames:
        raise FileNotFoundError(f"No raw-byte parquet files found in {raw_bytes_dir}")

    df = pd.concat(frames, ignore_index=True)
    df["flow_id"] = df["flow_id"].astype("int64")
    df["flow_key"] = df["day"].astype(str) + "_" + df["flow_id"].astype(str)
    df["y"] = collapse_attempted(df["Label_Distrinet"])
    df["is_attempted"] = df["Label_Distrinet"].astype("string").str.endswith(
        " - Attempted", na=False
    )

    per_flow = df.drop_duplicates("flow_key")[
        ["flow_key", "day", "flow_id", "y", "is_attempted"]
    ].copy()
    class_counts = (
        per_flow.groupby("y", dropna=False)["is_attempted"]
        .agg(total="size", real=lambda s: int((~s).sum()), attempted=lambda s: int(s.sum()))
        .reset_index()
        .rename(columns={"y": "class"})
        .sort_values(["real", "total"], ascending=False)
    )
    class_counts["keep"] = class_counts["real"] >= min_real
    keep_classes = sorted(class_counts.loc[class_counts["keep"], "class"].astype(str).tolist())

    per_flow = per_flow[per_flow["y"].astype(str).isin(keep_classes)].copy()
    per_flow = per_flow.sort_values("flow_key", kind="mergesort").reset_index(drop=True)

    print(
        f"raw-byte flow subset: flows={len(per_flow):,}, classes={len(keep_classes)}",
        flush=True,
    )
    return per_flow, keep_classes, class_counts


def load_feature_matrix(
    data_dir: Path,
    feature_cols: list[str],
    per_flow: pd.DataFrame,
) -> np.ndarray:
    keep_by_day = {
        day: set(sub["flow_id"].astype("int64").tolist())
        for day, sub in per_flow.groupby("day", sort=False)
    }
    parts = []
    columns = feature_cols + ["flow_id", "matched"]
    for path in sorted(data_dir.glob("*.parquet")):
        day = path.stem
        keep_ids = keep_by_day.get(day)
        if not keep_ids:
            continue
        start = time.perf_counter()
        df = pd.read_parquet(path, columns=columns, engine="pyarrow")
        df = df[df["matched"].astype("bool")].copy()
        df["flow_id"] = df["flow_id"].astype("int64")
        df = df[df["flow_id"].isin(keep_ids)].copy()
        df["flow_key"] = day + "_" + df["flow_id"].astype(str)
        parts.append(df[["flow_key", *feature_cols]])
        print(
            f"loaded {path.name}: kept={len(df):,}, elapsed={time.perf_counter() - start:.1f}s",
            flush=True,
        )

    if not parts:
        raise FileNotFoundError(f"No feature parquet files found in {data_dir}")

    features = pd.concat(parts, ignore_index=True).drop_duplicates("flow_key")
    features = features.set_index("flow_key")
    missing = per_flow.loc[~per_flow["flow_key"].isin(features.index), "flow_key"]
    if len(missing):
        raise RuntimeError(f"Missing {len(missing):,} raw-byte flow keys in {data_dir}")

    features = features.loc[per_flow["flow_key"].to_numpy()]
    x = features[feature_cols].to_numpy(dtype=np.float32, copy=True)
    np.nan_to_num(x, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    return x


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(args.data_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files found in {args.data_dir}")

    start = time.perf_counter()
    per_flow, classes, class_counts = load_rawbyte_subset(args.raw_bytes_dir, args.min_real)
    class_counts.to_csv(args.output_dir / "rawbyte_subset_class_counts.csv", index=False)
    feature_cols = parquet_feature_columns(paths[0])
    dropped_extra: list[str] = []
    if args.no_init_win:
        win_feats = ["Init Fwd Win Byts", "Init Bwd Win Byts"]
        dropped_extra = [c for c in win_feats if c in feature_cols]
        feature_cols = [c for c in feature_cols if c not in win_feats]
        print(f"noInitWin ablation: dropped {dropped_extra}", flush=True)
    pd.DataFrame({"feature": feature_cols}).to_csv(
        args.output_dir / "feature_columns.csv",
        index=False,
    )

    label_to_id = {label: idx for idx, label in enumerate(classes)}
    y = per_flow["y"].astype(str).map(label_to_id).to_numpy(dtype=np.int32)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    split_df = pd.DataFrame(
        {
            "flow_key": per_flow["flow_key"],
            "class": per_flow["y"].astype(str),
            "split": "train",
        }
    )
    split_df.loc[test_idx, "split"] = "test"
    split_df.to_csv(args.output_dir / "flow_split.csv", index=False)

    x = load_feature_matrix(args.data_dir, feature_cols, per_flow)
    prep_time = time.perf_counter() - start

    x_train, x_test = x[train_idx], x[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    del x

    print(
        f"dataset rows={len(y):,}, train={len(y_train):,}, test={len(y_test):,}, "
        f"features={len(feature_cols)}, prep_time={prep_time:.1f}s",
        flush=True,
    )

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.random_state,
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
    per_class.to_csv(args.output_dir / "per_class_metrics.csv", index=False)

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        args.output_dir / "confusion_matrix.csv"
    )

    importances = pd.DataFrame(
        {"feature": feature_cols, "importance": clf.feature_importances_}
    ).sort_values("importance", ascending=False)
    importances.to_csv(args.output_dir / "feature_importances.csv", index=False)

    class_counts_out = (
        pd.Series(y)
        .value_counts()
        .sort_index()
        .rename_axis("class_id")
        .reset_index(name="rows")
    )
    class_counts_out["class"] = class_counts_out["class_id"].map(lambda i: classes[int(i)])
    class_counts_out[["class_id", "class", "rows"]].to_csv(
        args.output_dir / "class_counts.csv",
        index=False,
    )

    summary = {
        "data_dir": str(args.data_dir),
        "raw_bytes_dir": str(args.raw_bytes_dir),
        "rows": int(len(y)),
        "train_rows": int(len(y_train)),
        "test_rows": int(len(y_test)),
        "features": int(len(feature_cols)),
        "classes": classes,
        "test_size": args.test_size,
        "random_state": args.random_state,
        "n_estimators": args.n_estimators,
        "class_weight": "balanced",
        "same_flow_subset_as_raw_bytes": True,
        "same_test_split_as_raw_bytes": True,
        "collapse_attempted": True,
        "matched_only": True,
        "min_real": args.min_real,
        "drop_columns": sorted(DROP_COLUMNS),
        "dropped_features": dropped_extra,
        "no_init_win": bool(args.no_init_win),
        "prep_time_s": prep_time,
        "train_time_s": train_time,
        "prediction_time_s": pred_time,
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
    }
    (args.output_dir / "run_config.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame([summary]).drop(columns=["classes", "drop_columns", "dropped_features"]).to_csv(
        args.output_dir / "summary.csv",
        index=False,
    )

    print(json.dumps(summary, indent=2), flush=True)
    print(per_class.to_string(index=False), flush=True)
    print(importances.head(15).to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument(
        "--raw-bytes-dir",
        type=Path,
        default=Path("data/CIC-IDS2018/partial_flow/raw_bytes_pkt5_b256_strict/raw_bytes"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-real", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--no-init-win", action="store_true",
                        help="Drop Init Fwd/Bwd Win Byts (window fingerprint ablation).")
    return parser.parse_args()


if __name__ == "__main__":
    main()
