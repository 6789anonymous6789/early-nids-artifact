"""Random Forest on CIC-IoT2023 partial-flow features (packet-cap cuts).

Mirrors ``scripts/unsw_nb15/train_rf_pct100.py`` but:
  - reads per-class parquet files under packet_abs_{K}/raw_flow (or pct_100);
  - ``--label-mode`` selects 34-class (raw) or 8-class (group via CLASS_TO_GROUP);
  - labels are folder-derived (the ``Label`` column), no Distrinet matching.

This is the fair-masking tabular baseline the deck shows dominates raw-byte
models under strict masking.

Example
-------
python scripts/cic_iot2023/train_rf_packetcap.py \
    --data-dir data/CIC-IoT2023/partial_flow/packet_abs_5/raw_flow --label-mode group
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support,
)
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.cic_iot2023.labeling import to_group  # noqa: E402

DROP_COLUMNS = {
    "day", "capture_file", "flow_id", "flow_src_ip", "flow_dst_ip",
    "flow_src_port", "flow_dst_port", "flow_start_ts", "flow_end_ts",
    "Dst Port", "Label", "label_encoded",
}

N_ESTIMATORS = 100
TEST_SIZE = 0.25
RANDOM_STATE = 42


def feature_columns(path: Path) -> list[str]:
    return [c for c in pq.ParquetFile(path).schema_arrow.names if c not in DROP_COLUMNS]


def load_dataset(paths, feature_cols, label_mode, min_rows):
    # First pass: per-class flow counts (one file per class).
    counts: dict[str, int] = {}
    for path in paths:
        lab = pd.read_parquet(path, columns=["Label"], engine="pyarrow")["Label"].astype("string")
        y = lab.map(to_group) if label_mode == "group" else lab
        for cls, n in y.value_counts().items():
            counts[str(cls)] = counts.get(str(cls), 0) + int(n)
    keep = {c for c, n in counts.items() if n >= min_rows} if min_rows > 0 else set(counts)

    label_to_id: dict[str, int] = {}
    parts_x, parts_y = [], []
    for path in paths:
        cols = feature_cols + ["Label"]
        df = pd.read_parquet(path, columns=cols, engine="pyarrow")
        lab = df["Label"].astype("string")
        y_lab = lab.map(to_group) if label_mode == "group" else lab
        mask = y_lab.isin(keep)
        df, y_lab = df.loc[mask], y_lab.loc[mask]
        if df.empty:
            continue
        for label in y_lab.unique():
            label_to_id.setdefault(str(label), len(label_to_id))
        xp = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
        np.nan_to_num(xp, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        parts_x.append(xp)
        parts_y.append(y_lab.map(label_to_id).to_numpy(dtype=np.int32))
        print(f"loaded {path.name}: rows={len(df):,}", flush=True)

    x = np.concatenate(parts_x) if parts_x else np.empty((0, len(feature_cols)), np.float32)
    y = np.concatenate(parts_y) if parts_y else np.empty((0,), np.int32)
    classes = [None] * len(label_to_id)
    for label, idx in label_to_id.items():
        classes[idx] = label
    return x, y, classes, counts


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path,
                   default=Path("data/CIC-IoT2023/partial_flow/packet_abs_5/raw_flow"))
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--label-mode", choices=["raw", "group"], default="group")
    p.add_argument("--min-rows", type=int, default=100)
    p.add_argument("--n-estimators", type=int, default=N_ESTIMATORS)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--no-init-win", action="store_true",
                   help="Drop Init Fwd/Bwd Win Byts (window fingerprint ablation).")
    args = p.parse_args()
    global RANDOM_STATE
    RANDOM_STATE = args.random_state

    out_dir = args.out_dir or Path(
        f"outputs/cic_iot2023/rf_{args.data_dir.parent.name}_{args.label_mode}"
        + ("_noInitWin" if args.no_init_win else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.data_dir.glob("*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet files in {args.data_dir}")

    feat = feature_columns(paths[0])
    dropped_extra: list[str] = []
    if args.no_init_win:
        win_feats = ["Init Fwd Win Byts", "Init Bwd Win Byts"]
        dropped_extra = [c for c in win_feats if c in feat]
        feat = [c for c in feat if c not in win_feats]
        print(f"noInitWin ablation: dropped {dropped_extra}", flush=True)
    pd.DataFrame({"feature": feat}).to_csv(out_dir / "feature_columns.csv", index=False)

    t0 = time.perf_counter()
    x, y, classes, counts = load_dataset(paths, feat, args.label_mode, args.min_rows)
    pd.DataFrame([{"class": c, "rows": n} for c, n in sorted(counts.items(),
                 key=lambda kv: -kv[1])]).to_csv(out_dir / "class_counts.csv", index=False)
    print(f"dataset rows={len(y):,} classes={len(classes)} feats={len(feat)} "
          f"prep={time.perf_counter()-t0:.1f}s", flush=True)

    x_tr, x_te, y_tr, y_te = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)
    del x, y
    clf = RandomForestClassifier(n_estimators=args.n_estimators, random_state=RANDOM_STATE,
                                 n_jobs=-1, class_weight="balanced", verbose=1)
    ts = time.perf_counter()
    clf.fit(x_tr, y_tr)
    train_time = time.perf_counter() - ts
    y_pred = clf.predict(x_te)

    labels = np.arange(len(classes), dtype=np.int32)
    pr, rc, f1, sup = precision_recall_fscore_support(y_te, y_pred, labels=labels, zero_division=0)
    pd.DataFrame({"class_id": labels, "class": classes, "precision": pr, "recall": rc,
                  "f1": f1, "support": sup}).sort_values("support", ascending=False).to_csv(
        out_dir / "per_class_metrics.csv", index=False)
    pd.DataFrame(confusion_matrix(y_te, y_pred, labels=labels),
                 index=classes, columns=classes).to_csv(out_dir / "confusion_matrix.csv")
    pd.DataFrame({"feature": feat, "importance": clf.feature_importances_}).sort_values(
        "importance", ascending=False).to_csv(out_dir / "feature_importances.csv", index=False)

    summary = {
        "data_dir": str(args.data_dir), "label_mode": args.label_mode,
        "rows": int(len(y_tr) + len(y_te)), "train_rows": int(len(y_tr)),
        "test_rows": int(len(y_te)), "features": int(len(feat)), "classes": classes,
        "n_estimators": args.n_estimators, "min_rows": args.min_rows,
        "no_init_win": bool(args.no_init_win), "dropped_features": dropped_extra,
        "train_time_s": train_time, "accuracy": float(accuracy_score(y_te, y_pred)),
        "f1_macro": float(f1_score(y_te, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_te, y_pred, average="weighted", zero_division=0)),
    }
    (out_dir / "run_config.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("accuracy", "f1_macro", "f1_weighted", "rows", "classes")}, indent=2),
          flush=True)
    print(f"written={out_dir}", flush=True)


if __name__ == "__main__":
    main()
