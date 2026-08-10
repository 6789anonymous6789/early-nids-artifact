"""Train the raw-byte mini-Transformer on CIC-IDS2018 strict-mask data.

Default input:
    data/CIC-IDS2018/partial_flow/raw_bytes_pkt5_b256_strict/raw_bytes

The raw-byte parquets are produced by scripts/cic_ids2018/extract_raw_bytes.py.
Labels are Distrinet-improved labels, with Attempted collapsed to the base
class and classes filtered by minimum real-flow count.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2017.byte_transformer import ByteTransformer  # noqa: E402


def collapse_attempted(label: str) -> str:
    return label.replace(" - Attempted", "")


def is_attempted(label: str) -> bool:
    return label.endswith(" - Attempted")


class RawByteFlowDataset(Dataset):
    def __init__(
        self,
        x: np.ndarray,
        pad_mask: np.ndarray,
        t_rel: np.ndarray,
        y: np.ndarray,
    ) -> None:
        self.x = x
        self.pad_mask = pad_mask
        self.t_rel = t_rel
        self.y = y

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.x[idx]).to(torch.uint8),
            torch.from_numpy(self.pad_mask[idx]).to(torch.bool),
            torch.from_numpy(self.t_rel[idx]).to(torch.float32),
            int(self.y[idx]),
        )


def load_raw_byte_flows(
    raw_bytes_dir: Path,
    max_pkts: int,
    n_bytes: int,
    min_real_per_class: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    frames = []
    for path in sorted(raw_bytes_dir.glob("*.parquet")):
        df = pd.read_parquet(
            path,
            columns=[
                "flow_id",
                "pkt_order",
                "timestamp_rel",
                "raw_bytes",
                "found",
                "Label_Distrinet",
            ],
            engine="pyarrow",
        )
        df["day"] = path.stem
        frames.append(df)
        print(f"  loaded {path.name}: {len(df):,} packet rows", flush=True)

    if not frames:
        raise FileNotFoundError(f"No parquet files found in {raw_bytes_dir}")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["found"]].copy()
    df["flow_key"] = df["day"].astype(str) + "_" + df["flow_id"].astype("int64").astype(str)
    df["y"] = df["Label_Distrinet"].astype(str).map(collapse_attempted)
    df["is_attempted"] = df["Label_Distrinet"].astype(str).map(is_attempted)

    per_flow = df.drop_duplicates("flow_key")[["flow_key", "y", "is_attempted"]]
    class_counts = (
        per_flow.groupby("y", dropna=False)["is_attempted"]
        .agg(total="size", real=lambda s: int((~s).sum()), attempted=lambda s: int(s.sum()))
        .reset_index()
        .rename(columns={"y": "class"})
        .sort_values(["real", "total"], ascending=False)
    )
    class_counts["keep"] = class_counts["real"] >= min_real_per_class
    keep_classes = sorted(class_counts.loc[class_counts["keep"], "class"].tolist())
    print(f"Classes kept ({len(keep_classes)}): {keep_classes}", flush=True)

    df = df[df["y"].isin(keep_classes)].copy()
    df = df[df["pkt_order"] < max_pkts].copy()
    flow_keys = sorted(df["flow_key"].unique().tolist())
    flow_to_pos = {flow_key: i for i, flow_key in enumerate(flow_keys)}
    n_flows = len(flow_keys)
    print(f"Building tensors for {n_flows:,} flows ...", flush=True)

    x = np.zeros((n_flows, max_pkts, n_bytes), dtype=np.uint8)
    pad_mask = np.ones((n_flows, max_pkts), dtype=bool)
    t_rel = np.zeros((n_flows, max_pkts), dtype=np.float32)
    y_per_flow = np.empty(n_flows, dtype=object)

    flow_pos = df["flow_key"].map(flow_to_pos).to_numpy()
    pkt_order = df["pkt_order"].to_numpy()
    raw_bytes = df["raw_bytes"].to_numpy()
    t_values = df["timestamp_rel"].to_numpy(dtype=np.float32)

    for i in range(len(df)):
        pos = int(flow_pos[i])
        pkt = int(pkt_order[i])
        if pkt >= max_pkts:
            continue
        arr = np.frombuffer(raw_bytes[i], dtype=np.uint8)
        length = min(len(arr), n_bytes)
        x[pos, pkt, :length] = arr[:length]
        pad_mask[pos, pkt] = False
        t_rel[pos, pkt] = t_values[i]

    first_labels = df.drop_duplicates("flow_key").set_index("flow_key")["y"]
    for flow_key, pos in flow_to_pos.items():
        y_per_flow[pos] = first_labels.loc[flow_key]

    return x, pad_mask, t_rel, y_per_flow, keep_classes, class_counts


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    model.eval()
    losses = []
    preds = []
    targets = []
    with torch.no_grad():
        for x_bytes, pad_mask, t_rel, y_b in loader:
            x_bytes = x_bytes.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            t_rel = t_rel.to(device, non_blocking=True)
            y_b = torch.as_tensor(y_b, device=device, dtype=torch.long)
            logits = model(x_bytes, pad_mask, t_rel)
            loss = criterion(logits, y_b)
            losses.append(loss.item() * x_bytes.size(0))
            preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
            targets.append(y_b.detach().cpu().numpy())

    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    loss_mean = float(sum(losses) / max(1, len(target)))
    macro_f1 = float(f1_score(target, pred, average="macro", zero_division=0))
    acc = float(accuracy_score(target, pred))
    return loss_mean, macro_f1, acc, pred, target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-bytes-dir",
        type=Path,
        default=Path("data/CIC-IDS2018/partial_flow/raw_bytes_pkt5_b256_strict/raw_bytes"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/cic_ids2018/byte_transformer_ablation/d64_pkt5_strict"),
    )
    parser.add_argument("--max-pkts", type=int, default=5)
    parser.add_argument("--n-bytes", type=int, default=256)
    parser.add_argument("--min-real", type=int, default=100)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--eval-batch-size", type=int, default=16384)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--val-size", type=float, default=0.20)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "byte_transformer"
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}", flush=True)

    start = time.perf_counter()
    x, pad_mask, t_rel, y_str, classes, class_counts = load_raw_byte_flows(
        args.raw_bytes_dir,
        max_pkts=args.max_pkts,
        n_bytes=args.n_bytes,
        min_real_per_class=args.min_real,
    )
    class_counts.to_csv(args.out_dir / "class_counts.csv", index=False)
    load_time = time.perf_counter() - start
    print(
        f"Loaded tensors: X={x.shape}, pad_mask={pad_mask.shape}, "
        f"t_rel={t_rel.shape}, load_time={load_time:.1f}s",
        flush=True,
    )

    label_encoder = LabelEncoder().fit(np.array(classes))
    y = label_encoder.transform(y_str.astype(str))
    n_classes = len(classes)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=args.test_size,
        stratify=y,
        random_state=args.random_state,
    )
    train_idx, val_idx = train_test_split(
        train_idx,
        test_size=args.val_size,
        stratify=y[train_idx],
        random_state=args.random_state,
    )
    print(
        f"Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}",
        flush=True,
    )

    train_ds = RawByteFlowDataset(x[train_idx], pad_mask[train_idx], t_rel[train_idx], y[train_idx])
    val_ds = RawByteFlowDataset(x[val_idx], pad_mask[val_idx], t_rel[val_idx], y[val_idx])
    test_ds = RawByteFlowDataset(x[test_idx], pad_mask[test_idx], t_rel[test_idx], y[test_idx])

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = ByteTransformer(
        n_classes=n_classes,
        n_bytes=args.n_bytes,
        max_pkts=args.max_pkts,
        d_model=args.d_model,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(device)
    n_params = model.num_parameters()
    print(f"Model params: {n_params:,}", flush=True)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(n_classes),
        y=y[train_idx],
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state = None
    best_epoch = 0
    best_val_f1 = -1.0
    stale = 0
    history = []
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        train_loss_sum = 0.0
        train_count = 0
        for x_bytes, pmask, t_batch, y_batch in train_loader:
            x_bytes = x_bytes.to(device, non_blocking=True)
            pmask = pmask.to(device, non_blocking=True)
            t_batch = t_batch.to(device, non_blocking=True)
            y_batch = torch.as_tensor(y_batch, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_bytes, pmask, t_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * x_bytes.size(0)
            train_count += x_bytes.size(0)

        scheduler.step()
        train_loss = float(train_loss_sum / max(1, train_count))
        val_loss, val_f1, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        epoch_seconds = time.perf_counter() - epoch_start
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_macro_f1": val_f1,
                "val_accuracy": val_acc,
                "epoch_seconds": epoch_seconds,
            }
        )
        print(
            f"epoch {epoch:>2}/{args.epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_f1={val_f1:.4f} val_acc={val_acc:.4f} "
            f"lr={scheduler.get_last_lr()[0]:.2e} ({epoch_seconds:.0f}s)",
            flush=True,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    training_time = time.perf_counter() - train_start
    if best_state is not None:
        model.load_state_dict(best_state)

    test_loss, test_macro_f1, test_acc, test_pred, test_target = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )
    test_weighted_f1 = float(f1_score(test_target, test_pred, average="weighted", zero_division=0))
    target_str = label_encoder.inverse_transform(test_target)
    pred_str = label_encoder.inverse_transform(test_pred)

    report = classification_report(
        target_str,
        pred_str,
        labels=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(model_dir / "classification_report.csv")
    cm = confusion_matrix(target_str, pred_str, labels=list(label_encoder.classes_))
    pd.DataFrame(cm, index=list(label_encoder.classes_), columns=list(label_encoder.classes_)).to_csv(
        model_dir / "confusion_matrix.csv"
    )
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)

    metrics = {
        "model": "byte_transformer",
        "accuracy": test_acc,
        "macro_f1": test_macro_f1,
        "weighted_f1": test_weighted_f1,
        "test_loss": test_loss,
        "best_val_macro_f1": best_val_f1,
        "best_epoch": int(best_epoch),
        "params": int(n_params),
        "load_time_s": float(load_time),
        "training_time_s": float(training_time),
        "n_classes": int(n_classes),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "classes": list(label_encoder.classes_),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (args.out_dir / "run_config.json").write_text(json.dumps(metrics["args"], indent=2) + "\n")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "classes": list(label_encoder.classes_),
            "args": metrics["args"],
        },
        model_dir / "model.pt",
    )

    print("\n=== TEST RESULTS ===", flush=True)
    print(f"accuracy={test_acc:.6f}", flush=True)
    print(f"macro_f1={test_macro_f1:.6f}", flush=True)
    print(f"weighted_f1={test_weighted_f1:.6f}", flush=True)
    print(f"best_epoch={best_epoch}", flush=True)
    print(f"written={model_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
