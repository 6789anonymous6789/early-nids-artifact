"""Train the mini-Transformer on CIC-IDS2017 raw-byte flow sequences.

Usage:
    python scripts/cic_ids2017/train_byte_transformer.py \
        --raw-bytes-dir data/CIC-IDS2017/partial_flow/raw_bytes_pkt5_b256/raw_bytes \
        --out-dir outputs/cic_ids2017/11class_distrinet_byte_transformer_pkt5_b256
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
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2017.byte_transformer import ByteTransformer  # noqa: E402
from src.cic_ids2017.raw_byte_dataset import RawByteFlowDataset, load_raw_byte_flows  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--raw-bytes-dir",
        type=Path,
        default=Path("data/CIC-IDS2017/partial_flow/raw_bytes_pkt5_b256/raw_bytes"),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/cic_ids2017/11class_distrinet_byte_transformer_pkt5_b256"),
    )
    p.add_argument("--max-pkts", type=int, default=5)
    p.add_argument("--n-bytes", type=int, default=256)
    p.add_argument("--d-model", type=int, default=16)
    p.add_argument("--nhead", type=int, default=4)
    p.add_argument("--dim-feedforward", type=int, default=32)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--val-size", type=float, default=0.20)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--device", type=str, default=None,
                   help="cuda or cpu. default: auto-detect")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.out_dir / "byte_transformer"
    model_dir.mkdir(parents=True, exist_ok=True)

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # === Load data ===
    print(f"Loading raw-byte parquets from {args.raw_bytes_dir} ...")
    X, pad_mask, t_rel, y_str, classes = load_raw_byte_flows(
        args.raw_bytes_dir,
        max_pkts=args.max_pkts,
        n_bytes=args.n_bytes,
    )
    print(f"  X: {X.shape}  pad_mask: {pad_mask.shape}  t_rel: {t_rel.shape}  y: {y_str.shape}")

    le = LabelEncoder().fit(np.array(classes))
    y = le.transform(y_str.astype(str))
    n_classes = len(classes)
    print(f"  Classes ({n_classes}): {le.classes_.tolist()}")

    # === Splits ===
    print(f"Splitting train/test {1 - args.test_size:.2f}/{args.test_size:.2f} (seed {args.random_state})")
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
    print(f"  train: {len(train_idx):,}   val: {len(val_idx):,}   test: {len(test_idx):,}")

    train_ds = RawByteFlowDataset(X[train_idx], pad_mask[train_idx], t_rel[train_idx], y[train_idx])
    val_ds = RawByteFlowDataset(X[val_idx], pad_mask[val_idx], t_rel[val_idx], y[val_idx])
    test_ds = RawByteFlowDataset(X[test_idx], pad_mask[test_idx], t_rel[test_idx], y[test_idx])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size * 2, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    # === Model ===
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
    print(f"Model: ByteTransformer  params: {n_params:,}")

    # === Loss ===
    class_weights = compute_class_weight(class_weight="balanced",
                                         classes=np.arange(n_classes),
                                         y=y[train_idx])
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)
    print(f"  class weights: {dict(zip(le.classes_.tolist(), class_weights.round(3).tolist()))}")
    criterion = nn.CrossEntropyLoss(weight=class_weights_t)

    # === Optim + scheduler ===
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # === Training loop ===
    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    stale = 0
    history = []

    t0_total = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        t0 = time.perf_counter()
        model.train()
        train_loss = 0.0
        n_train = 0
        for x_bytes, pmask, t, y_b in train_loader:
            x_bytes = x_bytes.to(device, non_blocking=True)
            pmask = pmask.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            y_b = torch.as_tensor(y_b, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x_bytes, pmask, t)
            loss = criterion(logits, y_b)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x_bytes.size(0)
            n_train += x_bytes.size(0)
        train_loss /= max(1, n_train)

        # Validation
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0.0
        n_val = 0
        with torch.no_grad():
            for x_bytes, pmask, t, y_b in val_loader:
                x_bytes = x_bytes.to(device, non_blocking=True)
                pmask = pmask.to(device, non_blocking=True)
                t = t.to(device, non_blocking=True)
                y_b = torch.as_tensor(y_b, device=device, dtype=torch.long)
                logits = model(x_bytes, pmask, t)
                loss = criterion(logits, y_b)
                val_loss += loss.item() * x_bytes.size(0)
                n_val += x_bytes.size(0)
                val_preds.append(logits.argmax(-1).cpu().numpy())
                val_targets.append(y_b.cpu().numpy())
        val_loss /= max(1, n_val)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_targets, val_preds)
        scheduler.step()

        elapsed = time.perf_counter() - t0
        print(
            f"epoch {epoch:>2}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_f1={val_f1:.4f}  val_acc={val_acc:.4f}  "
            f"lr={scheduler.get_last_lr()[0]:.2e}  ({elapsed:.0f}s)",
            flush=True,
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_f1,
            "val_accuracy": val_acc,
            "epoch_seconds": elapsed,
        })

        if val_f1 > best_val_f1:
            best_val_f1 = float(val_f1)
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                print(f"Early-stopping at epoch {epoch} (no improvement for {stale} epochs)")
                break
    train_elapsed = time.perf_counter() - t0_total

    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"Best val macro-F1: {best_val_f1:.4f} @ epoch {best_epoch}")

    # === Final eval on test ===
    model.eval()
    test_preds = []
    test_targets = []
    with torch.no_grad():
        for x_bytes, pmask, t, y_b in test_loader:
            x_bytes = x_bytes.to(device, non_blocking=True)
            pmask = pmask.to(device, non_blocking=True)
            t = t.to(device, non_blocking=True)
            y_b = torch.as_tensor(y_b, device=device, dtype=torch.long)
            logits = model(x_bytes, pmask, t)
            test_preds.append(logits.argmax(-1).cpu().numpy())
            test_targets.append(y_b.cpu().numpy())
    test_preds = np.concatenate(test_preds)
    test_targets = np.concatenate(test_targets)

    test_acc = accuracy_score(test_targets, test_preds)
    test_macro_f1 = f1_score(test_targets, test_preds, average="macro", zero_division=0)
    test_weighted_f1 = f1_score(test_targets, test_preds, average="weighted", zero_division=0)

    # Map labels back to strings for the report
    test_targets_str = le.inverse_transform(test_targets)
    test_preds_str = le.inverse_transform(test_preds)

    rep = classification_report(
        test_targets_str, test_preds_str,
        labels=list(le.classes_),
        output_dict=True, zero_division=0,
    )
    pd.DataFrame(rep).T.to_csv(model_dir / "classification_report.csv")

    cm = confusion_matrix(test_targets_str, test_preds_str, labels=list(le.classes_))
    pd.DataFrame(cm, index=list(le.classes_), columns=list(le.classes_)).to_csv(
        model_dir / "confusion_matrix.csv"
    )

    metrics = {
        "model": "byte_transformer",
        "accuracy": float(test_acc),
        "macro_f1": float(test_macro_f1),
        "weighted_f1": float(test_weighted_f1),
        "best_val_macro_f1": float(best_val_f1),
        "best_epoch": int(best_epoch),
        "params": int(n_params),
        "training_time_s": float(train_elapsed),
        "n_classes": int(n_classes),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(test_idx)),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)

    # Save model
    torch.save({
        "state_dict": model.state_dict(),
        "classes": le.classes_.tolist(),
        "args": metrics["args"],
    }, model_dir / "model.pt")

    print(f"\n=== TEST RESULTS ===")
    print(f"  accuracy:    {test_acc:.4f}")
    print(f"  macro_f1:    {test_macro_f1:.4f}")
    print(f"  weighted_f1: {test_weighted_f1:.4f}")
    print(f"\nWritten artifacts to: {model_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
