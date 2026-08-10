"""Extract raw packet bytes for the first K packets of each flow on CIC-IoT2023.

Mirrors ``scripts/cic_ids2018/extract_raw_bytes.py`` but:
  - partition unit is the attack-class folder (label == folder name);
  - flow selection reads the per-class packet_base directly (no Distrinet labels);
  - per-class subsampling reuses ``pcap_extracted_flow_pipeline.subsample_flow_ids``
    with the SAME random_state, so the byte models and the RF flow-feature models
    see the identical flow subset.

Two masks are produced for the leakage study (deck slides 10/14):
  - basic  : zero Eth (stripped) + IP src/dst + L4 ports          -> raw_bytes_pkt{K}_b{N}
  - strict : also zero IP version/TOS/ID/Protocol/Options + TCP Options
             (Abo-Alian 2026 schema)                              -> raw_bytes_pkt{K}_b{N}_strict

Output: data/CIC-IoT2023/partial_flow/raw_bytes_pkt{K}_b{N}[_strict]/raw_bytes/<class>.parquet
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.utils import RawPcapReader, RawPcapNgReader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.cic_iot2023.labeling import RAW_CLASSES, get_cap  # noqa: E402
from src.cic_iot2023.pcap_extracted_flow_pipeline import (  # noqa: E402
    PARTIAL_FLOW_ROOT,
    PCAP_ROOT,
    packet_base_path,
    subsample_flow_ids,
)

ETH_HDR_LEN = 14
ETH_TYPE_IP = 0x0800
ETH_TYPE_VLAN = 0x8100
TCP_PROTO = 6
UDP_PROTO = 17


def _strip_eth(buf: bytes) -> bytes | None:
    if len(buf) < ETH_HDR_LEN:
        return None
    eth_type = struct.unpack("!H", buf[12:14])[0]
    offset = ETH_HDR_LEN
    if eth_type == ETH_TYPE_VLAN and len(buf) >= ETH_HDR_LEN + 4:
        eth_type = struct.unpack("!H", buf[16:18])[0]
        offset = ETH_HDR_LEN + 4
    if eth_type != ETH_TYPE_IP:
        return None
    return buf[offset:]


def _mask_basic(ip_bytes: bytes) -> bytes:
    """Zero IP src/dst + L4 ports only (keeps the leaky fields: TTL, IP ID, TCP options)."""
    if len(ip_bytes) < 20:
        return ip_bytes
    buf = bytearray(ip_bytes)
    ihl = (buf[0] & 0x0F) * 4
    if ihl < 20 or ihl > 60:
        return bytes(buf)
    proto = buf[9]
    for i in range(12, 20):
        if i < len(buf):
            buf[i] = 0
    if proto in (TCP_PROTO, UDP_PROTO) and ihl + 4 <= len(buf):
        for i in range(ihl, ihl + 4):
            buf[i] = 0
    return bytes(buf)


def _mask_strict(ip_bytes: bytes) -> bytes:
    """Strict mask (Abo-Alian 2026): basic + zero IP version/TOS/ID/Protocol/Options + TCP Options."""
    if len(ip_bytes) < 20:
        return ip_bytes
    buf = bytearray(ip_bytes)
    ihl = (buf[0] & 0x0F) * 4
    if ihl < 20 or ihl > 60:
        return bytes(buf)
    proto = buf[9]
    buf[0] = buf[0] & 0x0F          # zero IP version, keep IHL
    buf[1] = 0                       # TOS
    buf[4] = 0; buf[5] = 0           # IP Identification
    buf[9] = 0                       # Protocol
    for i in range(12, 20):          # src/dst IP
        if i < len(buf):
            buf[i] = 0
    if ihl > 20:                     # IP Options
        for i in range(20, min(ihl, len(buf))):
            buf[i] = 0
    if proto in (TCP_PROTO, UDP_PROTO):
        if ihl + 4 <= len(buf):      # L4 ports
            for i in range(ihl, ihl + 4):
                buf[i] = 0
        if proto == TCP_PROTO and ihl + 20 <= len(buf):
            tcp_off = ((buf[ihl + 12] >> 4) & 0x0F) * 4
            if 20 < tcp_off <= 60 and ihl + tcp_off <= len(buf):
                for i in range(ihl + 20, min(ihl + tcp_off, len(buf))):
                    buf[i] = 0
    return bytes(buf)


MASKS = {"basic": _mask_basic, "strict": _mask_strict}


def _select_flows(cls: str, k: int, cap: int | None, random_state: int) -> pd.DataFrame:
    """First-K packets of the (capped) flows for one class, with PCAP locations."""
    cols = ["capture_file", "capture_packet_index", "flow_id", "packet_index", "timestamp"]
    pdf = pd.read_parquet(packet_base_path(cls), columns=cols, engine="pyarrow")

    if cap is not None:
        all_ids = pdf["flow_id"].unique()
        keep = subsample_flow_ids(all_ids, cap, random_state)
        if len(keep) < len(all_ids):
            pdf = pdf[pdf["flow_id"].isin(set(keep.tolist()))].copy()
            print(f"  subsampled flows {len(all_ids):,} -> {len(keep):,} (cap={cap})", flush=True)

    pdf = pdf.sort_values(["flow_id", "packet_index"], kind="mergesort")
    pdf["pkt_order"] = pdf.groupby("flow_id", sort=False).cumcount()
    pdf = pdf[pdf["pkt_order"] < k].copy()

    first_ts = pdf.groupby("flow_id", sort=False)["timestamp"].transform("min")
    pdf["timestamp_rel"] = (pdf["timestamp"] - first_ts).astype("float32")
    pdf["capture_packet_index"] = pdf["capture_packet_index"].astype("int64")
    return pdf[["capture_file", "capture_packet_index", "flow_id", "pkt_order",
                "timestamp_rel"]].reset_index(drop=True)


def _open_pcap(p: Path):
    with open(p, "rb") as fh:
        magic = fh.read(4)
    if magic == b"\x0a\x0d\x0d\x0a":
        return RawPcapNgReader(str(p))
    return RawPcapReader(str(p))


def _extract_one(cls: str, k: int, n_bytes: int, mask_name: str, cap: int | None,
                 random_state: int, force: bool) -> dict:
    mask_fn = MASKS[mask_name]
    tag = f"raw_bytes_pkt{k}_b{n_bytes}" + ("_strict" if mask_name == "strict" else "")
    out_dir = PARTIAL_FLOW_ROOT / tag / "raw_bytes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{cls}.parquet"
    if out_path.exists() and not force:
        print(f"[{cls}/{tag}] exists, skipping", flush=True)
        return {"class": cls, "tag": tag, "skipped": True}

    t0 = time.perf_counter()
    sel = _select_flows(cls, k, cap, random_state)
    if len(sel) == 0:
        print(f"[{cls}/{tag}] nothing to extract", flush=True)
        return {"class": cls, "tag": tag, "n_rows": 0}

    raw_bytes_col = np.zeros((len(sel), n_bytes), dtype=np.uint8)
    real_len_col = np.zeros(len(sel), dtype=np.int32)
    found_mask = np.zeros(len(sel), dtype=bool)

    pcap_class_dir = PCAP_ROOT / cls
    n_extracted = 0
    for cap_file, sub in sel.groupby("capture_file", sort=False):
        pcap_path = pcap_class_dir / cap_file
        if not pcap_path.exists():
            print(f"  WARN: missing PCAP {pcap_path}", flush=True)
            continue
        idx_to_pos = {int(i): int(p) for i, p in zip(sub["capture_packet_index"].values,
                                                      sub.index.values)}
        max_idx = max(idx_to_pos)
        try:
            reader = _open_pcap(pcap_path)
        except Exception as e:
            print(f"  WARN: can't open {pcap_path.name}: {e}", flush=True)
            continue
        ip_index = 0
        try:
            with reader as r:
                for pkt_tuple in r:
                    ip_bytes = _strip_eth(pkt_tuple[0])
                    if ip_bytes is None:
                        continue
                    cur_idx = ip_index
                    ip_index += 1
                    pos = idx_to_pos.get(cur_idx)
                    if pos is None:
                        if cur_idx > max_idx:
                            break
                        continue
                    masked = mask_fn(ip_bytes)
                    real_len_col[pos] = len(masked)
                    if real_len_col[pos] >= n_bytes:
                        raw_bytes_col[pos] = np.frombuffer(masked[:n_bytes], dtype=np.uint8)
                    else:
                        raw_bytes_col[pos, :real_len_col[pos]] = np.frombuffer(masked, dtype=np.uint8)
                    found_mask[pos] = True
                    n_extracted += 1
        except Exception as e:
            print(f"  WARN: parse error on {pcap_path.name} at ip_index={ip_index}: {e}", flush=True)

    out_df = pd.DataFrame({
        "flow_id": sel["flow_id"].astype("int64"),
        "pkt_order": sel["pkt_order"].astype("int8"),
        "capture_packet_index": sel["capture_packet_index"].astype("int64"),
        "timestamp_rel": sel["timestamp_rel"].astype("float32"),
        "raw_bytes": [raw_bytes_col[i].tobytes() for i in range(len(sel))],
        "real_length": real_len_col.astype("int16"),
        "found": found_mask,
        "Label": cls,
    })
    out_df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    elapsed = time.perf_counter() - t0
    print(f"[{cls}/{tag}] wrote {len(out_df):,} rows, found={int(found_mask.sum()):,}, "
          f"{elapsed:.0f}s -> {out_path}", flush=True)
    return {"class": cls, "tag": tag, "n_rows": int(len(out_df)),
            "n_found": int(found_mask.sum()), "elapsed_s": elapsed}


def _extract_class(cls, max_packets, n_bytes, mask_names, cap_arg, random_state, force):
    """One class, all requested masks. Module-level so it can run in a process pool
    (cross-class parallelism — extract_raw_bytes re-reads the pcaps, so concurrency
    across classes fills the cores)."""
    cap = get_cap(cls) if cap_arg == -1 else (None if cap_arg == 0 else cap_arg)
    return [_extract_one(cls, max_packets, n_bytes, m, cap, random_state, force)
            for m in mask_names]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classes", nargs="+", default=list(RAW_CLASSES))
    p.add_argument("--max-packets", type=int, default=5)
    p.add_argument("--n-bytes", type=int, default=256)
    p.add_argument("--mask", choices=["basic", "strict", "both"], default="both")
    p.add_argument("--cap", type=int, default=-1,
                   help="Per-class flow cap (-1=class default, 0=keep all). Use the SAME "
                        "value as the partial-flow build (e.g. 100000) so the byte models "
                        "and the RF flow-feature models see the identical flow subset.")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--class-workers", type=int, default=None,
                   help="Classes to process concurrently (default: min(cpu-2, 8)).")
    args = p.parse_args()

    mask_names = ["basic", "strict"] if args.mask == "both" else [args.mask]
    n_cw = args.class_workers or max(1, min((os.cpu_count() or 2) - 2, 8))
    n_cw = max(1, min(n_cw, len(args.classes)))
    print(f"Extracting raw bytes: {len(args.classes)} classes, masks={mask_names}, "
          f"cap={args.cap}, {n_cw} concurrent class-workers", flush=True)

    t0 = time.perf_counter()
    if n_cw == 1:
        for cls in args.classes:
            _extract_class(cls, args.max_packets, args.n_bytes, mask_names, args.cap,
                           args.random_state, args.force)
    else:
        with ProcessPoolExecutor(max_workers=n_cw) as ex:
            futs = {ex.submit(_extract_class, cls, args.max_packets, args.n_bytes,
                              mask_names, args.cap, args.random_state, args.force): cls
                    for cls in args.classes}
            for fut in as_completed(futs):
                cls = futs[fut]
                try:
                    fut.result()
                    print(f"[done] {cls}", flush=True)
                except Exception as e:  # keep going across classes
                    print(f"[{cls}] FAILED: {e}", flush=True)
    print(f"\nTotal time: {time.perf_counter()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
