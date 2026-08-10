"""Extract raw bytes for the first K packets of each flow on CIC-IDS2017.

Reuses the flow_id assignment from packet_base (same as the baseline RF) and
joins back to the original PCAPs to recover the first N bytes of each packet.

Output: data/CIC-IDS2017/partial_flow/raw_bytes_pkt{K}_b{N}/raw_bytes/<day>.parquet

For each kept packet we store:
  flow_id (int64), pkt_order (int8 0..K-1), capture_packet_index (int32),
  timestamp_rel (float32, seconds since first packet of flow),
  raw_bytes (binary, exactly N bytes — zero-padded if shorter),
  real_length (int16, length of IP packet before truncation/padding),
  Label_Distrinet, Attempted_Category, matched (carried from registry)

Header masking: src_ip, dst_ip, src_port, dst_port are zeroed (after stripping
Ethernet) so the model cannot rely on identifiers.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scapy.utils import RawPcapNgReader, RawPcapReader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2017.distrinet_labeling import load_registry  # noqa: E402
from src.cic_ids2017.flow_pipeline import DAYS, load_packet_base  # noqa: E402

DATA_ROOT = REPO_ROOT / "data" / "CIC-IDS2017"
PCAP_DIR = DATA_ROOT / "original" / "PCAPs"
PARTIAL_FLOW_ROOT = DATA_ROOT / "partial_flow"

ETH_HDR_LEN = 14
ETH_TYPE_IP = 0x0800
ETH_TYPE_VLAN = 0x8100
TCP_PROTO = 6
UDP_PROTO = 17


def _select_first_k_pkts_per_flow(day: str, k: int) -> pd.DataFrame:
    """Return a DataFrame with the rows of packet_base that are the first K
    packets of each matched flow, augmented with flow_id, pkt_order, distrinet
    columns."""
    print(f"[{day}] loading packet_base + grouping into flows", flush=True)
    flows = load_packet_base(day)
    print(f"  packets after grouping: {len(flows):,}", flush=True)

    print(f"[{day}] joining Distrinet registry", flush=True)
    reg = load_registry(day)[["flow_id", "Label_Distrinet", "Attempted_Category", "matched"]]
    flows = flows.merge(reg, on="flow_id", how="left")
    flows["matched"] = flows["matched"].fillna(False).astype("bool")
    flows = flows[flows["matched"]].copy()
    print(f"  matched packets: {len(flows):,}", flush=True)

    # Within each flow_id, keep the first K packets ordered by timestamp.
    flows = flows.sort_values(["flow_id", "timestamp", "capture_packet_index"], kind="mergesort")
    flows["pkt_order"] = flows.groupby("flow_id", sort=False).cumcount()
    flows = flows[flows["pkt_order"] < k].copy()
    print(f"  first-{k} packets: {len(flows):,}", flush=True)

    # Compute timestamp_rel (seconds since first pkt of flow)
    first_ts = flows.groupby("flow_id", sort=False)["timestamp"].transform("min")
    flows["timestamp_rel"] = (flows["timestamp"] - first_ts).astype("float32")

    return flows[
        [
            "capture_file",
            "capture_packet_index",
            "flow_id",
            "pkt_order",
            "timestamp_rel",
            "Label_Distrinet",
            "Attempted_Category",
        ]
    ].reset_index(drop=True)


def _mask_ip_packet(ip_bytes: bytes) -> bytes:
    """Zero out src_ip, dst_ip, and TCP/UDP ports in an IPv4 packet (basic mask)."""
    if len(ip_bytes) < 20:
        return ip_bytes
    buf = bytearray(ip_bytes)

    # IP header length is encoded in the lower nibble of byte 0 (in 32-bit words)
    ihl = (buf[0] & 0x0F) * 4
    if ihl < 20 or ihl > 60:
        return bytes(buf)

    proto = buf[9]

    # Zero src_ip (offsets 12-15) and dst_ip (16-19)
    for i in range(12, 20):
        if i < len(buf):
            buf[i] = 0

    # Zero src_port and dst_port for TCP/UDP
    if proto in (TCP_PROTO, UDP_PROTO):
        # Ports are at the start of the L4 header (offsets ihl..ihl+3)
        if ihl + 4 <= len(buf):
            buf[ihl + 0] = 0
            buf[ihl + 1] = 0
            buf[ihl + 2] = 0
            buf[ihl + 3] = 0

    return bytes(buf)


def _mask_ip_packet_strict(ip_bytes: bytes) -> bytes:
    """Strict mask: aligned with the image-based NIDS paper (Abo-Alian 2026).

    Removed (zeroed): Ethernet (already stripped), IP version, IP TOS,
    IP Identification, IP Protocol, IP src/dst, IP Options, TCP src/dst port,
    TCP Options. Kept: IHL, Total Length, DF/MF/Frag Offset, TTL,
    IP Header Checksum, TCP Sequence, TCP Ack, TCP Data Offset, TCP Flags,
    TCP Window, TCP Checksum, TCP Urgent, payload.
    """
    if len(ip_bytes) < 20:
        return ip_bytes
    buf = bytearray(ip_bytes)

    ihl = (buf[0] & 0x0F) * 4
    if ihl < 20 or ihl > 60:
        return bytes(buf)
    proto = buf[9]

    # IP version (high nibble of byte 0) — zero, keep IHL nibble
    buf[0] = buf[0] & 0x0F
    # IP TOS (byte 1)
    buf[1] = 0
    # IP Identification (bytes 4-5)
    buf[4] = 0
    buf[5] = 0
    # IP Protocol (byte 9) — masked AFTER reading proto above
    buf[9] = 0
    # Src IP (12-15), Dst IP (16-19)
    for i in range(12, 20):
        if i < len(buf):
            buf[i] = 0
    # IP Options (bytes 20..ihl-1, if ihl > 20)
    if ihl > 20:
        for i in range(20, min(ihl, len(buf))):
            buf[i] = 0

    # L4 masking
    if proto in (TCP_PROTO, UDP_PROTO):
        if ihl + 4 <= len(buf):
            for i in range(ihl, ihl + 4):
                buf[i] = 0
        if proto == TCP_PROTO and ihl + 20 <= len(buf):
            # TCP data offset (high nibble of byte ihl+12), in 32-bit words
            tcp_data_offset = ((buf[ihl + 12] >> 4) & 0x0F) * 4
            if 20 < tcp_data_offset <= 60 and ihl + tcp_data_offset <= len(buf):
                for i in range(ihl + 20, min(ihl + tcp_data_offset, len(buf))):
                    buf[i] = 0

    return bytes(buf)


def _strip_eth(eth_bytes: bytes) -> bytes | None:
    """Strip Ethernet (and possibly VLAN) header, return IPv4 packet bytes or None."""
    if len(eth_bytes) < ETH_HDR_LEN:
        return None
    eth_type = struct.unpack("!H", eth_bytes[12:14])[0]
    offset = ETH_HDR_LEN
    if eth_type == ETH_TYPE_VLAN and len(eth_bytes) >= ETH_HDR_LEN + 4:
        # 802.1Q tag: 2 bytes TCI + 2 bytes inner ethertype
        eth_type = struct.unpack("!H", eth_bytes[16:18])[0]
        offset = ETH_HDR_LEN + 4
    if eth_type != ETH_TYPE_IP:
        return None
    return eth_bytes[offset:]


def _extract_for_day(day: str, k: int, n_bytes: int, force: bool = False, strict_mask: bool = False) -> dict:
    tag = f"raw_bytes_pkt{k}_b{n_bytes}"
    if strict_mask:
        tag += "_strict"
    mask_fn = _mask_ip_packet_strict if strict_mask else _mask_ip_packet
    out_dir = PARTIAL_FLOW_ROOT / tag / "raw_bytes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{day}.parquet"

    if out_path.exists() and not force:
        print(f"[{day}] {out_path.name} exists, skipping", flush=True)
        return {"day": day, "skipped": True, "path": str(out_path)}

    t0 = time.perf_counter()
    selection = _select_first_k_pkts_per_flow(day, k)

    # Build a lookup: capture_packet_index -> list of row positions (one row per
    # selected (flow, pkt_order)).
    # For CIC-IDS2017 each capture_packet_index belongs to a single flow, so the
    # lookup is one-to-one (faster).
    wanted_idx = selection.set_index("capture_packet_index", drop=False)

    pcap_path = PCAP_DIR / f"{day}.pcap"
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")
    print(f"[{day}] reading {pcap_path.name} ({pcap_path.stat().st_size / 1e9:.1f} GB)", flush=True)

    raw_bytes_col = np.zeros((len(selection), n_bytes), dtype=np.uint8)
    real_len_col = np.zeros(len(selection), dtype=np.int32)
    found_mask = np.zeros(len(selection), dtype=bool)
    pos_lookup: dict[int, int] = {
        int(idx): pos for pos, idx in enumerate(selection["capture_packet_index"].values)
    }

    ip_index = 0  # index after IP filter, mirrors tshark's behaviour
    n_seen_ip = 0
    n_extracted = 0
    n_pcap_packets = 0

    last_progress = t0

    # Auto-detect pcap vs pcapng by magic bytes
    with open(pcap_path, "rb") as fh:
        magic = fh.read(4)
    if magic == b"\x0a\x0d\x0d\x0a":
        reader_cls = RawPcapNgReader
    else:
        reader_cls = RawPcapReader

    with reader_cls(str(pcap_path)) as reader:
        for pkt_tuple in reader:
            buf = pkt_tuple[0]
            n_pcap_packets += 1
            ip_bytes = _strip_eth(buf)
            if ip_bytes is None:
                continue
            cur_idx = ip_index
            ip_index += 1
            n_seen_ip += 1
            pos = pos_lookup.get(cur_idx)
            if pos is None:
                continue
            masked = mask_fn(ip_bytes)
            real_len = len(masked)
            if real_len >= n_bytes:
                raw_bytes_col[pos] = np.frombuffer(masked[:n_bytes], dtype=np.uint8)
            else:
                raw_bytes_col[pos, :real_len] = np.frombuffer(masked, dtype=np.uint8)
            real_len_col[pos] = real_len
            found_mask[pos] = True
            n_extracted += 1
            if n_extracted % 500_000 == 0:
                now = time.perf_counter()
                print(
                    f"  [{day}] extracted {n_extracted:,}/{len(selection):,} "
                    f"({100 * n_extracted / len(selection):.1f}%) "
                    f"after {n_pcap_packets:,} pcap pkts "
                    f"in {now - t0:.0f}s",
                    flush=True,
                )
                last_progress = now

    if not found_mask.all():
        n_missing = int((~found_mask).sum())
        print(
            f"[{day}] WARNING: {n_missing:,}/{len(selection):,} selected packets "
            f"not found in PCAP (likely PCAP truncation)",
            flush=True,
        )

    # Convert raw bytes to bytes objects for parquet binary column
    raw_bytes_objs = [raw_bytes_col[i].tobytes() for i in range(len(selection))]
    out_df = pd.DataFrame({
        "flow_id": selection["flow_id"].astype("int64"),
        "pkt_order": selection["pkt_order"].astype("int8"),
        "capture_packet_index": selection["capture_packet_index"].astype("int32"),
        "timestamp_rel": selection["timestamp_rel"].astype("float32"),
        "raw_bytes": raw_bytes_objs,
        "real_length": real_len_col.astype("int16"),
        "found": found_mask,
        "Label_Distrinet": selection["Label_Distrinet"].astype("string"),
        "Attempted_Category": selection["Attempted_Category"].astype("Int8"),
    })

    out_df.to_parquet(out_path, engine="pyarrow", compression="zstd", index=False)

    elapsed = time.perf_counter() - t0
    size_mb = out_path.stat().st_size / 1e6
    print(
        f"[{day}] wrote {out_path.name}: {len(out_df):,} rows, "
        f"{size_mb:.1f} MB, {elapsed:.0f}s",
        flush=True,
    )
    return {
        "day": day,
        "skipped": False,
        "path": str(out_path),
        "n_rows": int(len(out_df)),
        "n_missing": int((~found_mask).sum()),
        "n_pcap_packets": int(n_pcap_packets),
        "n_ip_packets": int(n_seen_ip),
        "elapsed_s": float(elapsed),
        "size_mb": float(size_mb),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--max-packets", type=int, default=5)
    p.add_argument("--n-bytes", type=int, default=256)
    p.add_argument("--days", nargs="+", default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--strict-mask", action="store_true",
                   help="Aggressively mask IP version/TOS/ID/Protocol/Options + TCP Options "
                        "(image-based-NIDS-paper-equivalent).")
    args = p.parse_args()
    days = args.days or DAYS

    t0 = time.perf_counter()
    for d in days:
        _extract_for_day(d, args.max_packets, args.n_bytes, args.force, strict_mask=args.strict_mask)
    print(f"\nTotal time: {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
