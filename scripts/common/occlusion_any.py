#!/usr/bin/env python3
"""Region-occlusion attribution for any dataset and either raw-byte architecture.

Generalizes scripts/cic_ids2017/byte_attribution.py so the same 22-region sweep
can be run on the BiGRU as well as the Transformer, and on all four testbeds,
reusing the checkpoints already on disk. Occlusion is inference only: the model
is held fixed and one byte region at a time is zeroed in the test set.

The test split is reproduced with the same seed and test fraction used at
training, so the baseline reported here reproduces the training-time macro-F1.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

REPO_ROOT = Path(os.environ.get("EARLYNIDS_ROOT",
                                Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2017.byte_bigru import ByteBiGRU  # noqa: E402
from src.cic_ids2017.byte_transformer import ByteTransformer  # noqa: E402

# Same regions as the original 2017 sweep, so results are directly comparable.
DEFAULT_REGIONS = [
    ("ip_ver_ihl",      0,   1,   "version+IHL"),
    ("ip_tos",          1,   2,   "type of service"),
    ("ip_total_len",    2,   4,   "IP total length"),
    ("ip_id",           4,   6,   "IP identification"),
    ("ip_flags_frag",   6,   8,   "IP flags + frag offset"),
    ("ip_ttl",          8,   9,   "TTL - OS fingerprint"),
    ("ip_proto",        9,   10,  "protocol"),
    ("ip_checksum",     10,  12,  "IP header checksum"),
    ("ip_src",          12,  16,  "src IP (already zeroed)"),
    ("ip_dst",          16,  20,  "dst IP (already zeroed)"),
    ("tcp_sport",       20,  22,  "src port (already zeroed)"),
    ("tcp_dport",       22,  24,  "dst port (already zeroed)"),
    ("tcp_seq",         24,  28,  "TCP sequence number"),
    ("tcp_ack",         28,  32,  "TCP ack number"),
    ("tcp_off_flags",   32,  34,  "data offset + flags"),
    ("tcp_window",      34,  36,  "TCP window - tool fingerprint"),
    ("tcp_checksum",    36,  38,  "TCP checksum"),
    ("tcp_urgent",      38,  40,  "TCP urgent ptr"),
    ("tcp_options",     40,  60,  "TCP options"),
    ("payload_60_128",  60,  128, "payload chunk 1"),
    ("payload_128_192", 128, 192, "payload chunk 2"),
    ("payload_192_256", 192, 256, "payload chunk 3"),
]


def load_dataset(dataset: str, raw_bytes_dir: Path, max_pkts: int, n_bytes: int,
                 label_mode: str):
    if dataset == "cic_ids2017":
        from src.cic_ids2017.raw_byte_dataset import load_raw_byte_flows
        out = load_raw_byte_flows(raw_bytes_dir, max_pkts=max_pkts, n_bytes=n_bytes)
    elif dataset == "toniot":
        from src.toniot.raw_byte_loading import load_raw_byte_flows
        out = load_raw_byte_flows(raw_bytes_dir, max_pkts, n_bytes, 100)
    elif dataset == "cic_iot2023":
        from src.cic_iot2023.raw_byte_loading import load_raw_byte_flows
        out = load_raw_byte_flows(raw_bytes_dir, max_pkts, n_bytes, 100,
                                  label_mode=label_mode)
    elif dataset == "cic_ids2018":
        sys.path.insert(0, str(REPO_ROOT / "scripts" / "cic_ids2018"))
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "t2018", REPO_ROOT / "scripts" / "cic_ids2018" / "train_byte_transformer.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out = mod.load_raw_byte_flows(raw_bytes_dir, max_pkts, n_bytes, 100)
    else:
        raise SystemExit(f"dataset sconosciuto: {dataset}")
    return out[0], out[1], out[2], out[3], out[4]


def build_model(arch: str, ckpt, device):
    classes = ckpt["classes"]
    a = ckpt["args"]
    if arch == "transformer":
        m = ByteTransformer(
            n_classes=len(classes),
            n_bytes=int(a.get("n_bytes", 256)),
            max_pkts=int(a.get("max_pkts", 5)),
            d_model=int(a.get("d_model", 64)),
            nhead=int(a.get("nhead", 4)),
            dim_feedforward=int(a.get("dim_feedforward", 128)),
            n_layers=int(a.get("n_layers", 2)),
            dropout=float(a.get("dropout", 0.1)),
        )
    else:
        m = ByteBiGRU(
            n_classes=len(classes),
            n_bytes=int(a.get("n_bytes", 256)),
            max_pkts=int(a.get("max_pkts", 5)),
            d_model=int(a.get("d_model", 64)),
            hidden_size=int(a.get("hidden_size", 64)),
            n_layers=int(a.get("n_layers", 1)),
            dropout=float(a.get("dropout", 0.1)),
        )
    m = m.to(device)
    m.load_state_dict(ckpt["state_dict"])
    m.eval()
    return m, classes


def predict(model, X, pad_mask, t_rel, batch_size, device):
    n = len(X)
    preds = np.empty(n, dtype=np.int64)
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sl = slice(i, min(i + batch_size, n))
            xb = torch.from_numpy(X[sl]).to(device, non_blocking=True)
            pm = torch.from_numpy(pad_mask[sl]).to(device, non_blocking=True)
            tb = torch.from_numpy(t_rel[sl]).to(device, non_blocking=True)
            preds[sl] = model(xb, pm, tb).argmax(-1).cpu().numpy()
    return preds


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", required=True,
                   choices=["cic_ids2017", "cic_ids2018", "toniot", "cic_iot2023"])
    p.add_argument("--arch", required=True, choices=["transformer", "bigru"])
    p.add_argument("--model-path", type=Path, required=True)
    p.add_argument("--raw-bytes-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--max-pkts", type=int, default=5)
    p.add_argument("--n-bytes", type=int, default=256)
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--label-mode", default="raw")
    p.add_argument("--tcp-only", action="store_true",
                   help="restrict the evaluation to flows whose first packet is TCP")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = torch.load(args.model_path, map_location=device, weights_only=False)
    model, classes = build_model(args.arch, ckpt, device)

    X, pad_mask, t_rel, y_str, classes_loaded = load_dataset(
        args.dataset, args.raw_bytes_dir, args.max_pkts, args.n_bytes, args.label_mode)

    le = LabelEncoder().fit(np.array(classes))
    y = le.transform(y_str.astype(str))
    _, test_idx = train_test_split(np.arange(len(y)), test_size=args.test_size,
                                   stratify=y, random_state=args.random_state)
    X_test, pm_test = X[test_idx], pad_mask[test_idx]
    t_test, y_test = t_rel[test_idx], y[test_idx]

    # Protocol composition of the evaluated population, read from the IP header
    # of the first packet. Meaningful only when the protocol byte is visible,
    # i.e. under the basic mask.
    proto = X_test[:, 0, 9].astype(int)
    comp = {"tcp": int((proto == 6).sum()), "udp": int((proto == 17).sum()),
            "other": int(((proto != 6) & (proto != 17)).sum())}
    if args.tcp_only:
        keep = proto == 6
        X_test, pm_test, t_test, y_test = (X_test[keep], pm_test[keep],
                                           t_test[keep], y_test[keep])

    base_preds = predict(model, X_test, pm_test, t_test, args.batch_size, device)
    base_macro = f1_score(y_test, base_preds, average="macro", zero_division=0)
    base_rep = classification_report(y_test, base_preds, labels=list(range(len(classes))),
                                     target_names=classes, output_dict=True, zero_division=0)
    base_per_class = {c: base_rep[c]["f1-score"] for c in classes}
    print(f"baseline macro-F1: {base_macro:.4f}  su {len(X_test):,} flussi  comp={comp}",
          flush=True)

    rows, per_class_rows = [], []
    for label, s, e, note in DEFAULT_REGIONS:
        if e > args.n_bytes:
            continue
        X_occ = X_test.copy()
        X_occ[:, :, s:e] = 0
        preds = predict(model, X_occ, pm_test, t_test, args.batch_size, device)
        macro = f1_score(y_test, preds, average="macro", zero_division=0)
        rep = classification_report(y_test, preds, labels=list(range(len(classes))),
                                    target_names=classes, output_dict=True, zero_division=0)
        rows.append({"region": label, "start": s, "end": e, "n_bytes": e - s,
                     "macro_f1_occluded": macro, "delta_macro_f1": macro - base_macro,
                     "drop": base_macro - macro, "note": note})
        line = {"region": label}
        for c in classes:
            line[c] = rep[c]["f1-score"] - base_per_class[c]
        per_class_rows.append(line)
        print(f"  {label:18s} macro={macro:.4f}  drop={base_macro - macro:+.4f}", flush=True)

    df = pd.DataFrame(rows).sort_values("drop", ascending=False)
    df.to_csv(args.out_dir / "region_occlusion.csv", index=False)
    pd.DataFrame(per_class_rows).set_index("region").to_csv(
        args.out_dir / "region_occlusion_per_class.csv")
    (args.out_dir / "summary.json").write_text(json.dumps({
        "dataset": args.dataset, "arch": args.arch,
        "baseline_macro_f1": float(base_macro),
        "baseline_per_class": base_per_class,
        "model_path": str(args.model_path),
        "raw_bytes_dir": str(args.raw_bytes_dir),
        "n_test_flows": int(len(X_test)),
        "tcp_only": bool(args.tcp_only),
        "protocol_composition_full_test": comp,
        "max_pkts": args.max_pkts,
    }, indent=2))
    print(df[["region", "drop"]].head(8).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
