"""Reproduce CIC-IDS2017 11-class Distrinet baselines.

The dataset is built from pct_100 raw flows with Distrinet corrected labels.
Classes are kept only when they have at least 100 real, non-attempted flows.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight


DEFAULT_DATA_DIR = Path("data/CIC-IDS2017/partial_flow/pct_100/raw_flow")
DEFAULT_OUT_DIR = Path("outputs/cic_ids2017/11class_distrinet_baselines")

DROP_COLUMNS = [
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
    "y",
    "is_attempted",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--models",
        default="rf,xgb,gnb,logreg,svm,mlp",
        help="Comma-separated subset: rf,xgb,gnb,logreg,svm,mlp",
    )
    parser.add_argument("--min-real", type=int, default=100)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Optional stratified sample after filtering, for quick debugging only.",
    )
    parser.add_argument(
        "--extra-drop",
        default="",
        help="Comma-separated additional CICFlowMeter feature names to drop "
             "(e.g. 'Init Fwd Win Byts,Init Bwd Win Byts'). Used for fingerprint-leakage ablation.",
    )
    return parser.parse_args()


def read_dataset(data_dir: Path) -> pd.DataFrame:
    files = sorted(data_dir.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")

    frames = []
    for path in files:
        frame = pd.read_parquet(path)
        if "matched" not in frame.columns:
            raise ValueError(f"{path} does not contain the 'matched' column")
        frames.append(frame[frame["matched"]].copy())
    return pd.concat(frames, ignore_index=True)


def prepare_dataset(
    data_dir: Path,
    min_real: int,
    sample_size: int,
    random_state: int,
    extra_drop: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    df = read_dataset(data_dir)
    if "Label_Distrinet" not in df.columns:
        raise ValueError("Label_Distrinet column is missing")

    df["y"] = df["Label_Distrinet"].astype(str).str.replace(
        " - Attempted", "", regex=False
    )
    df["is_attempted"] = df["Label_Distrinet"].astype(str).str.endswith(
        " - Attempted"
    )

    class_stats = (
        df.groupby("y", dropna=False)
        .agg(
            total=("y", "size"),
            real=("is_attempted", lambda s: int((~s).sum())),
            attempted=("is_attempted", lambda s: int(s.sum())),
        )
        .sort_values(["real", "total"], ascending=False)
    )
    class_stats["kept"] = class_stats["real"] >= min_real
    kept_classes = class_stats.index[class_stats["kept"]].tolist()

    df = df[df["y"].isin(kept_classes)].copy()
    drop_cols = list(DROP_COLUMNS) + list(extra_drop or [])
    X = df.drop(columns=drop_cols, errors="ignore")
    X = X.select_dtypes(include=[np.number])
    X = X.replace([np.inf, -np.inf], np.nan)

    finite_mask = X.notna().all(axis=1)
    X = X.loc[finite_mask].copy()
    y = df.loc[finite_mask, "y"].copy()

    if sample_size and sample_size < len(X):
        sampled_idx = (
            pd.DataFrame({"y": y})
            .groupby("y", group_keys=False)
            .sample(frac=sample_size / len(X), random_state=random_state)
            .index
        )
        X = X.loc[sampled_idx].sort_index()
        y = y.loc[sampled_idx].sort_index()

    filtered_stats = y.value_counts().rename_axis("class").reset_index(name="count")
    feature_table = pd.DataFrame({"feature": X.columns})
    return X, y, class_stats.reset_index(names="class"), filtered_stats, feature_table


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def write_report(
    model_name: str,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    classes: list[str],
    elapsed_s: float,
    out_dir: Path,
    extra: dict | None = None,
) -> dict:
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    report = classification_report(
        y_true,
        y_pred,
        labels=classes,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).T
    report_df.to_csv(model_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, y_pred, labels=classes)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(
        model_dir / "confusion_matrix.csv"
    )

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "training_time_s": float(elapsed_s),
    }
    if extra:
        metrics.update(extra)
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def train_rf(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, object, dict]:
    clf = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=args.random_state,
        max_depth=None,
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test), clf, {}


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, object, dict]:
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise RuntimeError("xgboost is not installed in this environment") from exc

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_encoded,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_encoded,
    )
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_tr)

    clf = XGBClassifier(
        objective="multi:softprob",
        num_class=len(label_encoder.classes_),
        n_estimators=600,
        learning_rate=0.08,
        max_depth=6,
        min_child_weight=1,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.0,
        reg_alpha=0.0,
        tree_method="hist",
        eval_metric="mlogloss",
        early_stopping_rounds=30,
        n_jobs=-1,
        random_state=args.random_state,
        verbosity=1,
    )
    clf.fit(
        X_tr,
        y_tr,
        sample_weight=sample_weight,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    pred_encoded = clf.predict(X_test)
    pred = label_encoder.inverse_transform(pred_encoded.astype(int))
    extra = {
        "best_iteration": int(getattr(clf, "best_iteration", -1) or -1),
    }
    return pred, {"model": clf, "label_encoder": label_encoder}, extra


def train_gnb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, object, dict]:
    clf = make_pipeline(StandardScaler(), GaussianNB())
    clf.fit(X_train, y_train)
    return clf.predict(X_test), clf, {}


def train_logreg(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, object, dict]:
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            solver="saga",
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=200,
            tol=1e-3,
            n_jobs=-1,
            random_state=args.random_state,
        ),
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test), clf, {}


def train_svm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, object, dict]:
    clf = make_pipeline(
        StandardScaler(),
        SGDClassifier(
            loss="hinge",
            penalty="l2",
            alpha=1e-4,
            class_weight="balanced",
            max_iter=100,
            tol=1e-3,
            n_jobs=-1,
            random_state=args.random_state,
            average=True,
        ),
    )
    clf.fit(X_train, y_train)
    return clf.predict(X_test), clf, {"note": "LinearSVC was too slow; this is SGD hinge SVM."}


def train_mlp(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    args: argparse.Namespace,
) -> tuple[np.ndarray, object, dict]:
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError("torch is not installed in this environment") from exc

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train,
        y_encoded,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y_encoded,
    )

    scaler = StandardScaler()
    X_tr_np = scaler.fit_transform(X_tr).astype(np.float32)
    X_val_np = scaler.transform(X_val).astype(np.float32)
    X_test_np = scaler.transform(X_test).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_features = X_train.shape[1]
    n_classes = len(label_encoder.classes_)

    model = nn.Sequential(
        nn.Linear(n_features, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(64, n_classes),
    ).to(device)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y_tr,
    ).astype(np.float32)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(X_tr_np),
            torch.from_numpy(y_tr.astype(np.int64)),
        ),
        batch_size=8192,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    X_val_t = torch.from_numpy(X_val_np).to(device)
    y_val_np = y_val.astype(np.int64)

    best_state = None
    best_f1 = -1.0
    best_epoch = 0
    patience = 4
    stale = 0
    max_epochs = 20

    for epoch in range(1, max_epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t).argmax(dim=1).cpu().numpy()
        val_f1 = f1_score(y_val_np, val_pred, average="macro", zero_division=0)
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_loader = DataLoader(
        torch.from_numpy(X_test_np),
        batch_size=32768,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    preds = []
    model.eval()
    with torch.no_grad():
        for xb in test_loader:
            xb = xb.to(device, non_blocking=True)
            preds.append(model(xb).argmax(dim=1).cpu().numpy())
    pred_encoded = np.concatenate(preds)
    pred = label_encoder.inverse_transform(pred_encoded)
    artifact = {"model": model, "scaler": scaler, "label_encoder": label_encoder}
    extra = {
        "device": str(device),
        "best_epoch": int(best_epoch),
        "best_val_macro_f1": float(best_f1),
    }
    return pred, artifact, extra


TRAINERS: dict[str, Callable] = {
    "rf": train_rf,
    "xgb": train_xgb,
    "gnb": train_gnb,
    "logreg": train_logreg,
    "svm": train_svm,
    "mlp": train_mlp,
}


def save_model(model_name: str, artifact: object, out_dir: Path) -> None:
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_name == "mlp":
        import torch

        torch.save(artifact, model_dir / "model.pt")
    else:
        joblib.dump(artifact, model_dir / "model.joblib")


def build_wide_f1_table(out_dir: Path, classes: list[str]) -> None:
    rows = pd.DataFrame({"class": classes}).set_index("class")
    support = None
    for model_dir in sorted(path for path in out_dir.iterdir() if path.is_dir()):
        report_path = model_dir / "classification_report.csv"
        if not report_path.exists():
            continue
        report = pd.read_csv(report_path, index_col=0)
        per_class = report.loc[report.index.intersection(classes)]
        rows[model_dir.name] = per_class["f1-score"]
        if support is None:
            support = per_class["support"].astype(int)
    if support is not None:
        rows["support"] = support
    rows.reset_index().to_csv(out_dir / "per_class_f1_support.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    requested_models = [name.strip() for name in args.models.split(",") if name.strip()]
    unknown = sorted(set(requested_models) - set(TRAINERS))
    if unknown:
        raise ValueError(f"Unknown models: {unknown}")

    extra_drop = [s.strip() for s in args.extra_drop.split(",") if s.strip()]

    start = time.perf_counter()
    X, y, class_stats, filtered_stats, feature_table = prepare_dataset(
        args.data_dir,
        min_real=args.min_real,
        sample_size=args.sample_size,
        random_state=args.random_state,
        extra_drop=extra_drop,
    )
    classes = sorted(y.unique().tolist())

    class_stats.to_csv(out_dir / "class_real_attempted_counts.csv", index=False)
    filtered_stats.to_csv(out_dir / "filtered_class_counts.csv", index=False)
    feature_table.to_csv(out_dir / "feature_columns.csv", index=False)

    config = {
        "data_dir": str(args.data_dir),
        "min_real": args.min_real,
        "rows": int(len(X)),
        "features": int(X.shape[1]),
        "classes": classes,
        "test_size": args.test_size,
        "val_size": args.val_size,
        "random_state": args.random_state,
        "sample_size": args.sample_size,
        "prep_time_s": float(time.perf_counter() - start),
        "drop_columns": DROP_COLUMNS,
    }
    (out_dir / "run_config.json").write_text(json.dumps(config, indent=2) + "\n")

    print(json.dumps(config, indent=2))
    if args.dry_run:
        return

    X_train, X_test, y_train, y_test = split_data(
        X, y, test_size=args.test_size, random_state=args.random_state
    )

    metrics = []
    for model_name in requested_models:
        print(f"\n=== {model_name} ===", flush=True)
        trainer = TRAINERS[model_name]
        model_start = time.perf_counter()
        pred, artifact, extra = trainer(X_train, y_train, X_test, args)
        elapsed_s = time.perf_counter() - model_start
        metrics.append(
            write_report(
                model_name,
                y_test,
                pred,
                classes,
                elapsed_s,
                out_dir,
                extra=extra,
            )
        )
        if args.save_models:
            save_model(model_name, artifact, out_dir)
        print(metrics[-1], flush=True)

    pd.DataFrame(metrics).to_csv(out_dir / "summary_by_model.csv", index=False)
    build_wide_f1_table(out_dir, classes)


if __name__ == "__main__":
    main()
