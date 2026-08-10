"""Extract raw bytes (strict mask) for the first K packets of each flow on CIC-IDS2018.

Differences from 2017 version:
- packet_base already contains flow_id (no need to re-run group_into_flows).
- 441 capture_file per day (one PCAP per VM in pcap_extracted/<day>/<capture_file>).
- Subsampling per class:
  BENIGN cap 2M, top-4 attacks (DoS Hulk, DDoS-HOIC, DDoS-LOIC-HTTP, Botnet Ares)
  cap 100k, all rare classes kept intact. Total: ~2.62M flows.

Output: data/CIC-IDS2018/partial_flow/raw_bytes_pkt{K}_b{N}_strict/raw_bytes/<day>.parquet

Strict mask aligned with Abo-Alian 2026:
removes IP version/TOS/ID/Protocol/Options + IP src/dst + TCP/UDP src/dst port + TCP Options.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.utils import RawPcapReader, RawPcapNgReader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_ROOT = REPO_ROOT / "data" / "CIC-IDS2018"
PCAP_DIR = DATA_ROOT / "pcap_extracted"
PACKET_BASE = DATA_ROOT / "partial_flow" / "packet_base"
DISTRINET_LABELS = DATA_ROOT / "partial_flow" / "distrinet_labels"
OUT_ROOT = DATA_ROOT / "partial_flow"

DAYS = [
    "Wednesday-14-02-2018",
    "Thursday-15-02-2018",
    "Friday-16-02-2018",
    "Tuesday-20-02-2018",
    "Wednesday-21-02-2018",
    "Thursday-22-02-2018",
    "Friday-23-02-2018",
    "Wednesday-28-02-2018",
    "Thursday-01-03-2018",
    "Friday-02-03-2018",
]

ETH_HDR_LEN = 14
ETH_TYPE_IP = 0x0800
ETH_TYPE_VLAN = 0x8100
TCP_PROTO = 6
UDP_PROTO = 17

# Subsampling caps, applied PER DAY (reproducible with random_state). Matches
# Subsampling budget: BENIGN 200k/day (~2M total over
# the 10 days), top-4 attacks ≤ 100k, all other (rare) classes kept intact. Total ~2.62M.
# Caps for the rare classes are set to 100k; since their full counts are
# below 100k they're effectively no-ops, but keeping the entries here
# documents intent + protects against future label re-runs.
DEFAULT_CAPS = {
    "BENIGN": 200_000,
    "DoS Hulk": 100_000,
    "DDoS-HOIC": 100_000,
    "DDoS-LOIC-HTTP": 100_000,
    "Botnet Ares": 100_000,
    "SSH-BruteForce": 100_000,
    "Infiltration - NMAP Portscan": 100_000,
    "DoS GoldenEye": 100_000,
    "DoS Slowloris": 100_000,
    "DDoS-LOIC-UDP": 100_000,
    "Web Attack - Brute Force": 100_000,
    "Web Attack - XSS": 100_000,
}
# Classes with <100 real flows are excluded (consistent with 2017 ≥100 filter)
EXCLUDED = {
    "Infiltration - Communication Victim Attacker",
    "Infiltration - Dropbox Download",
    "Web Attack - SQL",
    "FTP-BruteForce",  # 100% attempted, no real
}


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


def _mask_ip_strict(ip_bytes: bytes) -> bytes:
    """Strict mask: remove IP version/TOS/ID/Protocol/IP-Options + IP src/dst + TCP-Options + L4 ports."""
    if len(ip_bytes) < 20:
        return ip_bytes
    buf = bytearray(ip_bytes)
    ihl = (buf[0] & 0x0F) * 4
    if ihl < 20 or ihl > 60:
        return bytes(buf)
    proto = buf[9]

    buf[0] = buf[0] & 0x0F          # zero IP version high nibble, keep IHL
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
        if ihl + 4 <= len(buf):      # L4 src/dst ports
            for i in range(ihl, ihl + 4):
                buf[i] = 0
        if proto == TCP_PROTO and ihl + 20 <= len(buf):
            tcp_off = ((buf[ihl + 12] >> 4) & 0x0F) * 4
            if 20 < tcp_off <= 60 and ihl + tcp_off <= len(buf):
                for i in range(ihl + 20, min(ihl + tcp_off, len(buf))):
                    buf[i] = 0
    return bytes(buf)


def _mask_ip_basic(ip_bytes: bytes) -> bytes:
    """Basic mask: zero IP src/dst + L4 ports only (keeps the leaky fields: TTL, IP ID, TCP Options)."""
    if len(ip_bytes) < 20:
        return ip_bytes
    buf = bytearray(ip_bytes)
    ihl = (buf[0] & 0x0F) * 4
    if ihl < 20 or ihl > 60:
        return bytes(buf)
    proto = buf[9]
    for i in range(12, 20):          # src/dst IP
        if i < len(buf):
            buf[i] = 0
    if proto in (TCP_PROTO, UDP_PROTO) and ihl + 4 <= len(buf):  # L4 ports
        for i in range(ihl, ihl + 4):
            buf[i] = 0
    return bytes(buf)


MASKS = {"basic": _mask_ip_basic, "strict": _mask_ip_strict}


def _select_flows_for_day(day: str, k: int, caps: dict, random_state: int,
                          anchor_flow_ids: set | None = None) -> pd.DataFrame:
    """Pick first K packets for flows we want to keep (after class filter + subsampling).

    If anchor_flow_ids is given, the per-class cap/subsample is bypassed and exactly
    those flows are kept, mirroring an existing extraction flow-for-flow (the committed
    caps do not necessarily reproduce older data on disk, so anchoring is the safe path
    for a basic-vs-strict comparison that must share flows)."""
    print(f"[{day}] loading distrinet labels", flush=True)
    reg = pd.read_parquet(DISTRINET_LABELS / f"{day}.parquet",
                          columns=["flow_id", "Label_Distrinet", "Attempted_Category", "matched"])
    reg = reg[reg["matched"]].copy()
    reg["y"] = reg["Label_Distrinet"].astype(str).str.replace(" - Attempted", "", regex=False)
    reg["is_attempt"] = reg["Label_Distrinet"].astype(str).str.endswith(" - Attempted")
    reg = reg[~reg["y"].isin(EXCLUDED)].copy()

    print(f"  matched flows after class filter: {len(reg):,}", flush=True)

    if anchor_flow_ids is not None:
        reg_ids = set(reg["flow_id"].tolist())
        keep_set = anchor_flow_ids & reg_ids
        print(f"  anchored: {len(keep_set):,} kept of {len(anchor_flow_ids):,} requested "
              f"({len(anchor_flow_ids) - len(keep_set):,} not in current labels)", flush=True)
    else:
        # Per-class subsample
        rng = np.random.RandomState(random_state)
        keep_flow_ids = []
        for cls, sub in reg.groupby("y"):
            cap = caps.get(cls, 100_000)
            if len(sub) > cap:
                idx = rng.choice(sub.index.values, size=cap, replace=False)
                keep_flow_ids.append(reg.loc[idx, "flow_id"].values)
            else:
                keep_flow_ids.append(sub["flow_id"].values)
        keep_flow_ids = np.concatenate(keep_flow_ids) if keep_flow_ids else np.array([], dtype=np.int64)
        print(f"  kept after subsample: {len(keep_flow_ids):,}", flush=True)
        keep_set = set(keep_flow_ids.tolist())

    # Build label map per flow_id (as int dict for speed)
    label_map = reg.set_index("flow_id")["Label_Distrinet"]
    attempt_map = reg.set_index("flow_id")["Attempted_Category"]

    # Load packet_base for these flows (filter on flow_id) — read all then filter
    print(f"[{day}] loading packet_base + filtering by kept flow_ids", flush=True)
    pkt_cols = ["capture_file", "capture_packet_index", "flow_id", "packet_index", "timestamp"]
    pdf = pd.read_parquet(PACKET_BASE / f"{day}.parquet", columns=pkt_cols)
    pdf = pdf[pdf["flow_id"].isin(keep_set)].copy()
    print(f"  packets in kept flows: {len(pdf):,}", flush=True)

    # Take first K packets per flow (already ordered by packet_index in 2018)
    pdf = pdf.sort_values(["flow_id", "packet_index"], kind="mergesort")
    pdf["pkt_order"] = pdf.groupby("flow_id", sort=False).cumcount()
    pdf = pdf[pdf["pkt_order"] < k].copy()
    print(f"  first-{k} packets: {len(pdf):,}", flush=True)

    # timestamp_rel
    first_ts = pdf.groupby("flow_id", sort=False)["timestamp"].transform("min")
    pdf["timestamp_rel"] = (pdf["timestamp"] - first_ts).astype("float32")

    # Attach Label / Attempted_Category by flow_id
    pdf["Label_Distrinet"] = pdf["flow_id"].map(label_map)
    pdf["Attempted_Category"] = pdf["flow_id"].map(attempt_map).astype("Int8")
    pdf["capture_packet_index"] = pdf["capture_packet_index"].astype("int64")
    return pdf[
        ["capture_file", "capture_packet_index", "flow_id", "pkt_order", "timestamp_rel",
         "Label_Distrinet", "Attempted_Category"]
    ].reset_index(drop=True)


def _open_pcap(p: Path):
    with open(p, "rb") as fh:
        magic = fh.read(4)
    if magic == b"\x0a\x0d\x0d\x0a":
        return RawPcapNgReader(str(p))
    return RawPcapReader(str(p))


def _extract_for_day(day: str, k: int, n_bytes: int, caps: dict,
                     random_state: int, mask_name: str = "strict",
                     anchor_dir: Path | None = None, force: bool = False) -> dict:
    mask_fn = MASKS[mask_name]
    suffix = "_strict" if mask_name == "strict" else ""
    tag = f"raw_bytes_pkt{k}_b{n_bytes}{suffix}"
    out_dir = OUT_ROOT / tag / "raw_bytes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{day}.parquet"
    if out_path.exists() and not force:
        print(f"[{day}] {out_path.name} exists, skipping", flush=True)
        return {"day": day, "skipped": True, "path": str(out_path)}

    t0 = time.perf_counter()
    anchor_ids = None
    if anchor_dir is not None:
        ap = anchor_dir / f"{day}.parquet"
        anchor_ids = set(pd.read_parquet(ap, columns=["flow_id"])["flow_id"].unique().tolist())
    sel = _select_flows_for_day(day, k, caps, random_state, anchor_flow_ids=anchor_ids)
    if len(sel) == 0:
        print(f"[{day}] nothing to extract", flush=True)
        return {"day": day, "skipped": False, "n_rows": 0}

    raw_bytes_col = np.zeros((len(sel), n_bytes), dtype=np.uint8)
    real_len_col = np.zeros(len(sel), dtype=np.int32)
    found_mask = np.zeros(len(sel), dtype=bool)

    # Group selection rows by capture_file → for each PCAP we collect a single pass
    print(f"[{day}] iterating over {sel['capture_file'].nunique():,} PCAPs", flush=True)
    pcap_day_dir = PCAP_DIR / day
    n_extracted = 0
    progress_print = 0

    for cap, sub in sel.groupby("capture_file", sort=False):
        pcap_path = pcap_day_dir / cap
        if not pcap_path.exists():
            print(f"  WARN: missing PCAP {pcap_path}", flush=True)
            continue
        # Build per-capture lookup: capture_packet_index -> position in raw_bytes_col
        idx_to_pos = {int(idx): pos for idx, pos in zip(sub["capture_packet_index"].values,
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
                    buf = pkt_tuple[0]
                    ip_bytes = _strip_eth(buf)
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

        if n_extracted - progress_print >= 100_000:
            now = time.perf_counter()
            print(f"  [{day}] extracted {n_extracted:,}/{len(sel):,} after {now-t0:.0f}s", flush=True)
            progress_print = n_extracted

    raw_bytes_objs = [raw_bytes_col[i].tobytes() for i in range(len(sel))]
    out_df = pd.DataFrame({
        "flow_id": sel["flow_id"].astype("int64"),
        "pkt_order": sel["pkt_order"].astype("int8"),
        "capture_packet_index": sel["capture_packet_index"].astype("int32"),
        "timestamp_rel": sel["timestamp_rel"].astype("float32"),
        "raw_bytes": raw_bytes_objs,
        "real_length": real_len_col.astype("int16"),
        "found": found_mask,
        "Label_Distrinet": sel["Label_Distrinet"].astype("string"),
        "Attempted_Category": sel["Attempted_Category"].astype("Int8"),
    })
    out_df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)
    elapsed = time.perf_counter() - t0
    print(f"[{day}] wrote {out_path.name}: {len(out_df):,} rows, "
          f"{out_path.stat().st_size/1e6:.1f} MB, {elapsed:.0f}s, "
          f"found={int(found_mask.sum()):,}/{len(out_df):,}", flush=True)
    return {"day": day, "skipped": False, "n_rows": int(len(out_df)),
            "n_found": int(found_mask.sum()), "elapsed_s": elapsed}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-packets", type=int, default=5)
    p.add_argument("--n-bytes", type=int, default=256)
    p.add_argument("--days", nargs="+", default=None)
    p.add_argument("--mask", choices=["basic", "strict", "both"], default="strict")
    p.add_argument("--anchor-flows-from", type=Path, default=None,
                   help="Directory of an existing extraction (<dir>/<day>.parquet); keep exactly "
                        "those flow_ids instead of re-running the per-class cap subsample.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()
    days = args.days or DAYS
    caps = DEFAULT_CAPS
    mask_names = ["basic", "strict"] if args.mask == "both" else [args.mask]

    t0 = time.perf_counter()
    for d in days:
        for mask_name in mask_names:
            _extract_for_day(d, args.max_packets, args.n_bytes, caps, args.random_state,
                             mask_name, args.anchor_flows_from, args.force)
    print(f"\nTotal time: {time.perf_counter()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
