"""PyTorch dataset for the raw-byte sequential pipeline (CIC-IDS2017).

Loads the per-packet parquets written by `extract_raw_bytes.py` and produces,
for each flow, a fixed-size tensor of shape (max_pkts, n_bytes).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

DAYS = [
    "Monday-WorkingHours",
    "Tuesday-WorkingHours",
    "Wednesday-workingHours",
    "Thursday-WorkingHours",
    "Friday-WorkingHours",
]


def collapse_attempted(label: str) -> str:
    return label.replace(" - Attempted", "")


def is_attempted(label: str) -> bool:
    return label.endswith(" - Attempted")


def load_raw_byte_flows(
    raw_bytes_dir: Path,
    max_pkts: int = 5,
    n_bytes: int = 256,
    min_real_per_class: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Read all day parquets, return per-flow tensors + labels.

    Returns
    -------
    X : np.uint8, shape (N, max_pkts, n_bytes)
    pad_mask : np.bool_, shape (N, max_pkts) — True for padding (no packet)
    t_rel : np.float32, shape (N, max_pkts)
    y : np.object_, shape (N,) — string class labels
    classes : list[str] — sorted unique kept classes
    """
    frames = []
    for day in DAYS:
        p = Path(raw_bytes_dir) / f"{day}.parquet"
        if not p.exists():
            print(f"  skip {day} (file missing)")
            continue
        df = pd.read_parquet(p)
        df["day"] = day  # disambiguate per-day flow_ids
        frames.append(df)
        print(f"  loaded {day}: {len(df):,} rows")
    if not frames:
        raise FileNotFoundError(f"no parquets in {raw_bytes_dir}")
    df = pd.concat(frames, ignore_index=True)
    # Build a globally-unique flow key
    df["flow_key"] = df["day"].astype(str) + "_" + df["flow_id"].astype("int64").astype(str)
    print(f"Total per-packet rows: {len(df):,}")

    # Determine classes: collapse Attempted, keep classes with enough real samples.
    df["y"] = df["Label_Distrinet"].astype(str).apply(collapse_attempted)
    df["is_attempted"] = df["Label_Distrinet"].astype(str).apply(is_attempted)
    # Per-flow is determined by flow_id; check counts on flow_id (one per flow)
    per_flow = df.drop_duplicates("flow_key")[["flow_key", "y", "is_attempted"]]
    counts = per_flow.groupby("y", dropna=False)["is_attempted"].agg(
        total="size",
        real=lambda s: int((~s).sum()),
    )
    counts["kept"] = counts["real"] >= min_real_per_class
    keep_classes = sorted(counts.index[counts["kept"]].tolist())
    print(f"Classes kept ({len(keep_classes)}): {keep_classes}")

    df = df[df["y"].isin(keep_classes)].copy()
    df = df[df["found"]].copy()  # drop packets we couldn't recover from PCAP
    print(f"After class+found filter: {len(df):,} packet rows")

    flow_keys = sorted(df["flow_key"].unique().tolist())
    flow_to_pos = {fk: i for i, fk in enumerate(flow_keys)}
    n_flows = len(flow_keys)
    print(f"Building tensors for {n_flows:,} flows ...")

    X = np.zeros((n_flows, max_pkts, n_bytes), dtype=np.uint8)
    pad_mask = np.ones((n_flows, max_pkts), dtype=bool)  # default True = padding
    t_rel = np.zeros((n_flows, max_pkts), dtype=np.float32)
    y_per_flow = np.empty(n_flows, dtype=object)

    flow_pos = df["flow_key"].map(flow_to_pos).to_numpy()
    pkt_order = df["pkt_order"].to_numpy()
    raw_b = df["raw_bytes"].to_numpy()
    t_arr = df["timestamp_rel"].to_numpy()

    for i in range(len(df)):
        fp = int(flow_pos[i])
        po = int(pkt_order[i])
        if po >= max_pkts:
            continue
        b = raw_b[i]
        if isinstance(b, (bytes, bytearray)):
            arr = np.frombuffer(b, dtype=np.uint8)
            L = min(len(arr), n_bytes)
            X[fp, po, :L] = arr[:L]
        pad_mask[fp, po] = False
        t_rel[fp, po] = t_arr[i]

    # Assign per-flow label from any packet of the flow (they share label)
    label_first = df.drop_duplicates("flow_key").set_index("flow_key")["y"]
    for fk, pos in flow_to_pos.items():
        y_per_flow[pos] = label_first.loc[fk]

    return X, pad_mask, t_rel, y_per_flow, keep_classes


class RawByteFlowDataset(Dataset):
    """Wraps numpy tensors as a torch Dataset."""

    def __init__(
        self,
        X: np.ndarray,
        pad_mask: np.ndarray,
        t_rel: np.ndarray,
        y_idx: np.ndarray,
    ):
        assert len(X) == len(pad_mask) == len(t_rel) == len(y_idx)
        self.X = X
        self.pad_mask = pad_mask
        self.t_rel = t_rel
        self.y = y_idx

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return (
            torch.from_numpy(self.X[i]).to(torch.uint8),
            torch.from_numpy(self.pad_mask[i]).to(torch.bool),
            torch.from_numpy(self.t_rel[i]).to(torch.float32),
            int(self.y[i]),
        )
