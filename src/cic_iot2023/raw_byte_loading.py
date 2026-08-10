"""Load CIC-IoT2023 raw-byte parquets into (x, pad_mask, t_rel, y) tensors.

Mirrors the loader in ``scripts/cic_ids2018/train_byte_transformer.py`` but:
  - reads the ``Label`` column (folder-derived class), no ``Label_Distrinet`` /
    ``Attempted`` machinery;
  - ``label_mode`` selects 34-class (raw) or 8-class (group via CLASS_TO_GROUP).

Returns the same tuple contract as the 2018 loader so the existing
``RawByteFlowDataset`` and training loops can be reused unchanged.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.cic_iot2023.labeling import to_group


def load_raw_byte_flows(
    raw_bytes_dir: Path,
    max_pkts: int,
    n_bytes: int,
    min_real_per_class: int,
    label_mode: str = "raw",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], pd.DataFrame]:
    frames = []
    for path in sorted(Path(raw_bytes_dir).glob("*.parquet")):
        df = pd.read_parquet(
            path,
            columns=["flow_id", "pkt_order", "timestamp_rel", "raw_bytes", "found", "Label"],
            engine="pyarrow",
        )
        df["src_class"] = path.stem  # one file == one class, keeps flow_id unique
        frames.append(df)
        print(f"  loaded {path.name}: {len(df):,} packet rows", flush=True)

    if not frames:
        raise FileNotFoundError(f"No parquet files found in {raw_bytes_dir}")

    df = pd.concat(frames, ignore_index=True)
    df = df[df["found"]].copy()
    df["flow_key"] = df["src_class"].astype(str) + "_" + df["flow_id"].astype("int64").astype(str)

    if label_mode == "group":
        df["y"] = df["Label"].astype(str).map(to_group)
    elif label_mode == "raw":
        df["y"] = df["Label"].astype(str)
    else:
        raise ValueError(f"unknown label_mode={label_mode!r}; expected 'raw' or 'group'")

    per_flow = df.drop_duplicates("flow_key")[["flow_key", "y"]]
    class_counts = (
        per_flow.groupby("y", dropna=False)["flow_key"]
        .size().reset_index(name="flows")
        .rename(columns={"y": "class"})
        .sort_values("flows", ascending=False)
    )
    class_counts["keep"] = class_counts["flows"] >= min_real_per_class
    keep_classes = sorted(class_counts.loc[class_counts["keep"], "class"].tolist())
    print(f"Classes kept ({len(keep_classes)}): {keep_classes}", flush=True)

    df = df[df["y"].isin(keep_classes)].copy()
    df = df[df["pkt_order"] < max_pkts].copy()
    flow_keys = sorted(df["flow_key"].unique().tolist())
    flow_to_pos = {fk: i for i, fk in enumerate(flow_keys)}
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
    for fk, pos in flow_to_pos.items():
        y_per_flow[pos] = first_labels.loc[fk]

    return x, pad_mask, t_rel, y_per_flow, keep_classes, class_counts
