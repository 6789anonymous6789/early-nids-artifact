"""Shared train/eval loop for CIC-IoT2023 raw-byte models (BiGRU / Transformer).

Factored out so ``scripts/cic_iot2023/train_byte_bigru.py`` and
``train_byte_transformer.py`` only differ in the model they build. Mirrors the
2018 byte-trainer loop (balanced CE, AdamW + cosine, early stopping on val
macro-F1) but loads via ``src.cic_iot2023.raw_byte_loading.load_raw_byte_flows``
(label_mode raw=34-class / group=8-class).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.cic_iot2023.raw_byte_loading import load_raw_byte_flows


class RawByteFlowDataset(Dataset):
    def __init__(self, x, pad_mask, t_rel, y):
        self.x, self.pad_mask, self.t_rel, self.y = x, pad_mask, t_rel, y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.x[idx]).to(torch.uint8),
            torch.from_numpy(self.pad_mask[idx]).to(torch.bool),
            torch.from_numpy(self.t_rel[idx]).to(torch.float32),
            int(self.y[idx]),
        )


def evaluate(model, loader, criterion, device):
    model.eval()
    losses, preds, targets = [], [], []
    with torch.no_grad():
        for x_bytes, pad_mask, t_rel, y_b in loader:
            x_bytes = x_bytes.to(device, non_blocking=True)
            pad_mask = pad_mask.to(device, non_blocking=True)
            t_rel = t_rel.to(device, non_blocking=True)
            y_b = torch.as_tensor(y_b, device=device, dtype=torch.long)
            logits = model(x_bytes, pad_mask, t_rel)
            losses.append(criterion(logits, y_b).item() * x_bytes.size(0))
            preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
            targets.append(y_b.detach().cpu().numpy())
    pred = np.concatenate(preds)
    target = np.concatenate(targets)
    return (float(sum(losses) / max(1, len(target))),
            float(f1_score(target, pred, average="macro", zero_division=0)),
            float(accuracy_score(target, pred)), pred, target)


def run_training(model_name: str, build_model: Callable[[int], nn.Module], args) -> int:
    """build_model(n_classes) -> nn.Module. `args` is the argparse Namespace."""
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_dir = out_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}", flush=True)

    start = time.perf_counter()
    x, pad_mask, t_rel, y_str, classes, class_counts = load_raw_byte_flows(
        args.raw_bytes_dir, max_pkts=args.max_pkts, n_bytes=args.n_bytes,
        min_real_per_class=args.min_real, label_mode=args.label_mode,
    )
    class_counts.to_csv(out_dir / "class_counts.csv", index=False)
    load_time = time.perf_counter() - start
    print(f"Loaded: X={x.shape}, load_time={load_time:.1f}s", flush=True)

    label_encoder = LabelEncoder().fit(np.array(classes))
    y = label_encoder.transform(y_str.astype(str))
    n_classes = len(classes)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=args.test_size, stratify=y, random_state=args.random_state)
    train_idx, val_idx = train_test_split(
        train_idx, test_size=args.val_size, stratify=y[train_idx], random_state=args.random_state)
    print(f"Split: train={len(train_idx):,}, val={len(val_idx):,}, test={len(test_idx):,}", flush=True)

    pin = device.type == "cuda"
    mk = lambda idx, bs, sh: DataLoader(  # noqa: E731
        RawByteFlowDataset(x[idx], pad_mask[idx], t_rel[idx], y[idx]),
        batch_size=bs, shuffle=sh, num_workers=args.num_workers, pin_memory=pin)
    train_loader = mk(train_idx, args.batch_size, True)
    val_loader = mk(val_idx, args.eval_batch_size, False)
    test_loader = mk(test_idx, args.eval_batch_size, False)

    model = build_model(n_classes).to(device)
    n_params = model.num_parameters()
    print(f"Model params: {n_params:,}", flush=True)

    class_weights = compute_class_weight("balanced", classes=np.arange(n_classes), y=y[train_idx])
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state, best_epoch, best_val_f1, stale, history = None, 0, -1.0, 0, []
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        ep_start = time.perf_counter()
        model.train()
        tl_sum, tl_cnt = 0.0, 0
        for x_bytes, pmask, t_batch, y_batch in train_loader:
            x_bytes = x_bytes.to(device, non_blocking=True)
            pmask = pmask.to(device, non_blocking=True)
            t_batch = t_batch.to(device, non_blocking=True)
            y_batch = torch.as_tensor(y_batch, device=device, dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x_bytes, pmask, t_batch), y_batch)
            loss.backward()
            optimizer.step()
            tl_sum += loss.item() * x_bytes.size(0)
            tl_cnt += x_bytes.size(0)
        scheduler.step()
        train_loss = float(tl_sum / max(1, tl_cnt))
        val_loss, val_f1, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                        "val_macro_f1": val_f1, "val_accuracy": val_acc,
                        "epoch_seconds": time.perf_counter() - ep_start})
        print(f"epoch {epoch:>2}/{args.epochs} train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_f1={val_f1:.4f} val_acc={val_acc:.4f} "
              f"({time.perf_counter()-ep_start:.0f}s)", flush=True)
        if val_f1 > best_val_f1:
            best_val_f1, best_epoch = val_f1, epoch
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
        model, test_loader, criterion, device)
    test_weighted_f1 = float(f1_score(test_target, test_pred, average="weighted", zero_division=0))
    target_str = label_encoder.inverse_transform(test_target)
    pred_str = label_encoder.inverse_transform(test_pred)
    cls_list = list(label_encoder.classes_)

    pd.DataFrame(classification_report(target_str, pred_str, labels=cls_list,
                 output_dict=True, zero_division=0)).T.to_csv(model_dir / "classification_report.csv")
    pd.DataFrame(confusion_matrix(target_str, pred_str, labels=cls_list),
                 index=cls_list, columns=cls_list).to_csv(model_dir / "confusion_matrix.csv")
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)

    metrics = {
        "model": model_name, "label_mode": args.label_mode,
        "accuracy": test_acc, "macro_f1": test_macro_f1, "weighted_f1": test_weighted_f1,
        "test_loss": test_loss, "best_val_macro_f1": best_val_f1, "best_epoch": int(best_epoch),
        "params": int(n_params), "load_time_s": float(load_time),
        "training_time_s": float(training_time), "n_classes": int(n_classes),
        "n_train": int(len(train_idx)), "n_val": int(len(val_idx)), "n_test": int(len(test_idx)),
        "classes": cls_list,
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (out_dir / "run_config.json").write_text(json.dumps(metrics["args"], indent=2) + "\n")
    torch.save({"state_dict": model.state_dict(), "classes": cls_list, "args": metrics["args"]},
               model_dir / "model.pt")

    print("\n=== TEST RESULTS ===", flush=True)
    print(f"accuracy={test_acc:.6f} macro_f1={test_macro_f1:.6f} "
          f"weighted_f1={test_weighted_f1:.6f} best_epoch={best_epoch}", flush=True)
    print(f"written={model_dir}", flush=True)
    return 0


def add_common_args(parser, default_out: str) -> None:
    parser.add_argument("--raw-bytes-dir", type=Path,
                        default=Path("data/CIC-IoT2023/partial_flow/raw_bytes_pkt5_b256_strict/raw_bytes"))
    parser.add_argument("--out-dir", type=Path, default=Path(default_out))
    parser.add_argument("--label-mode", choices=["raw", "group"], default="group",
                        help="raw=34-class, group=8-class (default).")
    parser.add_argument("--max-pkts", type=int, default=5)
    parser.add_argument("--n-bytes", type=int, default=256)
    parser.add_argument("--min-real", type=int, default=100)
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
