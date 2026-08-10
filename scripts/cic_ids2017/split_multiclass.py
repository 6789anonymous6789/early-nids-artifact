#!/usr/bin/env python3
"""Split CIC-IDS2017 partial-flow data into train/val/test sets.

Usage:
    python scripts/cic_ids2017/split_multiclass.py --cut-dir time_abs_1s
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

REPO_ROOT = Path(__file__).resolve().parent.parent.parent

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
BENIGN_RATIO = 3

DAYS = [
    "Monday-WorkingHours",
    "Tuesday-WorkingHours",
    "Wednesday-workingHours",
    "Thursday-WorkingHours",
    "Friday-WorkingHours",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cut-dir", type=str, default="time_abs_1s")
    parser.add_argument("--task", type=str, default="multiclass",
                        choices=["multiclass", "10_classes_no_web"],
                        help="multiclass = all classes; 10_classes_no_web = drop "
                             "Web Attack - Brute Force, Web Attack - XSS, "
                             "Web Attack - SQL Injection (analog to 2018's 8_classes_no_web).")
    args = parser.parse_args()

    cut_dir = args.cut_dir
    task = args.task
    excluded_labels = set()
    if task == "10_classes_no_web":
        excluded_labels = {
            "Web Attack - Brute Force",
            "Web Attack - XSS",
            "Web Attack - SQL Injection",
        }

    data_root = REPO_ROOT / "data" / "CIC-IDS2017"
    raw_dir = data_root / "partial_flow" / cut_dir / "raw_flow"
    split_dir_name = "splits_multiclass" if task == "multiclass" else f"splits_{task}"
    split_dir = data_root / "partial_flow" / cut_dir / split_dir_name
    split_dir.mkdir(parents=True, exist_ok=True)
    
    meta_path = data_root / "partial_flow" / cut_dir / "preprocessing_meta.json"
    if not meta_path.exists():
        # Fallback to global meta if cut-specific is not found (for backwards compatibility)
        global_meta = data_root / "preprocessing_meta.json"
        if global_meta.exists():
            meta_path = global_meta
        else:
            print(f"ERROR: {meta_path} not found.")
            sys.exit(1)
        
    with open(meta_path) as f:
        meta = json.load(f)
    feature_columns = meta["feature_columns"]
    
    print(f"=== Splitting {cut_dir} task={task} ===")
    print(f"Split ratios: {TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}")
    print(f"Benign undersampling: {BENIGN_RATIO}x total attacks (train only)")
    if excluded_labels:
        print(f"Excluded labels: {sorted(excluded_labels)}")
    
    # Load all days to find common valid classes and assign int labels
    frames = []
    for day in DAYS:
        p = raw_dir / f"{day}.parquet"
        if p.exists():
            df = pd.read_parquet(p, columns=feature_columns + ["Label"], engine="pyarrow")
            # Drop inf/nan
            df[feature_columns] = df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0)
            frames.append(df)
            print(f"  {day}: {len(df):>10,} rows")
            
    full_df = pd.concat(frames, ignore_index=True)
    del frames

    # Task-specific exclusion first (web attacks for 10_classes_no_web)
    if excluded_labels:
        before = len(full_df)
        full_df = full_df[~full_df["Label"].isin(excluded_labels)].copy()
        print(f"\nExcluded {before - len(full_df):,} rows matching {sorted(excluded_labels)}")

    label_counts = full_df["Label"].value_counts()
    valid_labels = label_counts[label_counts >= 5].index
    print(f"\nFiltering classes with <5 instances:")
    for lbl, count in label_counts.items():
        if count < 5:
            print(f"  - Dropping {lbl} ({count} instances)")

    full_df = full_df[full_df["Label"].isin(valid_labels)].copy()
    
    unique_labels = sorted(full_df["Label"].unique())
    if "Benign" in unique_labels:
        unique_labels.remove("Benign")
        unique_labels = ["Benign"] + unique_labels
        
    label_to_idx = {label: i for i, label in enumerate(unique_labels)}
    idx_to_label = {i: label for label, i in label_to_idx.items()}
    full_df["label_encoded"] = full_df["Label"].map(label_to_idx)
    
    print(f"\nLabel mapping:")
    for k, v in label_to_idx.items():
        print(f"  {k}: {v}")
        
    # Stratified Split (Global)
    # Note: 2018 script does within-day splits, but global is safer here 
    # to avoid dropping classes that only appear on one day with very few samples.
    print("\nSplitting globally...")
    y = full_df["label_encoded"].values
    
    val_test_ratio = VAL_RATIO + TEST_RATIO
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=val_test_ratio, random_state=RANDOM_SEED)
    train_idx, val_test_idx = next(sss1.split(np.zeros(len(full_df)), y))
    
    df_train = full_df.iloc[train_idx].copy()
    df_val_test = full_df.iloc[val_test_idx].copy()
    
    y_vt = df_val_test["label_encoded"].values
    test_frac = TEST_RATIO / val_test_ratio
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=RANDOM_SEED)
    val_idx, test_idx = next(sss2.split(np.zeros(len(df_val_test)), y_vt))
    
    df_val = df_val_test.iloc[val_idx].copy()
    df_test = df_val_test.iloc[test_idx].copy()
    
    del full_df, df_val_test
    
    # Undersample Benign in Train only
    benign_mask = df_train["label_encoded"] == 0
    attack_mask = df_train["label_encoded"] != 0
    
    n_attack_total = int(attack_mask.sum())
    n_benign_available = int(benign_mask.sum())
    cap = int(min(n_attack_total * BENIGN_RATIO, n_benign_available))
    
    df_benign_sampled = df_train[benign_mask].sample(n=cap, random_state=RANDOM_SEED)
    df_attacks = df_train[attack_mask]
    
    df_train = pd.concat([df_benign_sampled, df_attacks], ignore_index=True)
    df_train = df_train.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
    
    print(f"\nBenign undersampling: {n_benign_available:,} -> {cap:,} (cap = {BENIGN_RATIO}x attacks)")
    print(f"Train after undersampling: {len(df_train):,}")
    
    # Compute Class Weights from the undersampled train set
    from sklearn.utils.class_weight import compute_class_weight
    y_train = df_train["label_encoded"].values
    present_classes = np.array(sorted(set(y_train.tolist())))
    weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_train)
    class_weights = dict(zip(present_classes.tolist(), weights.tolist()))
    
    # Save parquet files
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        out_path = split_dir / f"{name}.parquet"
        df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
        print(f"  Saved {name}: {len(df):>10,} rows")
        
    def split_distribution(df):
        dist = df["label_encoded"].value_counts().sort_index()
        return {
            "rows": len(df),
            "class_distribution": {idx_to_label[int(k)]: int(v) for k, v in dist.items()},
        }
        
    split_meta = {
        "task": task,
        "excluded_labels": sorted(excluded_labels) if excluded_labels else [],
        "cut_dir": cut_dir,
        "split_strategy": "global_stratified",
        "train_ratio": TRAIN_RATIO,
        "val_ratio": VAL_RATIO,
        "test_ratio": TEST_RATIO,
        "benign_undersampling_ratio": BENIGN_RATIO,
        "random_seed": RANDOM_SEED,
        "feature_columns": feature_columns,
        "label_column": "label_encoded",
        "class_names": {str(k): v for k, v in idx_to_label.items()},
        "present_classes": sorted(present_classes.tolist()),
        "n_classes": len(idx_to_label),
        "class_weights": {str(k): v for k, v in class_weights.items()},
        "splits": {
            "train": split_distribution(df_train),
            "val": split_distribution(df_val),
            "test": split_distribution(df_test),
        },
    }
    
    with open(split_dir / "split_meta.json", "w") as f:
        json.dump(split_meta, f, indent=2)
        
    print("\nDONE.")

if __name__ == "__main__":
    main()
