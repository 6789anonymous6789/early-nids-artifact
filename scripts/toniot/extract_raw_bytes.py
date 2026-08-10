"""Extract raw packet bytes for the first K packets of each kept flow on ToN-IoT.

Models ``scripts/cic_iot2023/extract_raw_bytes.py`` but ToN-IoT labels are
per-flow (the pcap folder is mixed normal+attack, NOT the label), so the design
differs in two ways:

  1. **Flow set + labels come from the assembled dataset.** The byte models must
     train on the IDENTICAL flows/labels the RF flow-feature models used. Those
     are fixed by ``pct_100/dataset.parquet`` (the seeded, per-class capped,
     ground-truth-labeled pool). We read its ``flow_id -> label`` map and select
     exactly those flows — no re-derivation that could drift from the RF subset.
     The kept flow_id set is identical across packet cuts (a flow exists in every
     cut), so pct_100's set also covers packet_abs_{3,4,5}.

  2. **Single pcap pass, both masks.** ToN-IoT pcaps total ~72 GiB, so re-reading
     them per mask (as the CIC-IoT2023 script does) is wasteful. Each pcap is read
     ONCE here and both masks are applied to every packet:
       - basic  : zero Eth(stripped)+IP src/dst+L4 ports        -> raw_bytes_pkt{K}_b{N}
       - strict : also zero IP version/TOS/ID/Protocol/Options+TCP Options
                  (Abo-Alian 2026)                              -> raw_bytes_pkt{K}_b{N}_strict

Extract ONCE at the maximum K; train any K' <= K via ``--max-pkts`` on the byte
trainer. The loader truncates ``pkt_order < max_pkts`` and ``timestamp_rel`` is
first-packet-relative (identical for every K), so a K'-cut read from the K=5
extraction is bit-identical to a separate K' extraction.

Output: data/ToN-IoT/partial_flow/raw_bytes_pkt{K}_b{N}[_strict]/raw_bytes/<scenario>.parquet
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

from src.toniot.labeling import ATTACK_CLASSES  # noqa: E402
from src.toniot.pcap_extracted_flow_pipeline import (  # noqa: E402
    PARTIAL_FLOW_ROOT,
    _scenario_tag,
    dataset_path,
    list_scenarios,
    packet_base_path,
    scenario_pcap_dir,
)

ETH_HDR_LEN = 14
ETH_TYPE_IP = 0x0800
ETH_TYPE_VLAN = 0x8100
TCP_PROTO = 6
UDP_PROTO = 17

# Link-layer types (libpcap DLT). ToN-IoT pcaps mix Ethernet and Linux cooked
# capture (SLL v1) — a census of all pcaps shows only these two. tshark dissects
# both transparently (so the packet_base is correct for every pcap); the raw-byte
# re-reader must strip the right link header or the IP byte offset is wrong and
# capture_packet_index alignment collapses (SLL pcaps gave ~0% found otherwise).
DLT_EN10MB = 1        # Ethernet
DLT_LINUX_SLL = 113   # Linux cooked-mode capture v1 (16-byte header)
SLL_HDR_LEN = 16

_CLASS_ARR = np.array(ATTACK_CLASSES, dtype=object)


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


def _strip_sll(buf: bytes) -> bytes | None:
    """Strip a Linux cooked-mode (SLL v1) header. Layout: pkt_type(2) ARPHRD(2)
    lladdr_len(2) lladdr(8) protocol(2) | payload. IP starts at offset 16."""
    if len(buf) < SLL_HDR_LEN:
        return None
    proto = struct.unpack("!H", buf[14:16])[0]
    offset = SLL_HDR_LEN
    if proto == ETH_TYPE_VLAN and len(buf) >= SLL_HDR_LEN + 4:
        proto = struct.unpack("!H", buf[18:20])[0]
        offset = SLL_HDR_LEN + 4
    if proto != ETH_TYPE_IP:
        return None
    return buf[offset:]


def _strip_link(buf: bytes, linktype: int) -> bytes | None:
    """Return the IP-onward bytes for an IPv4 frame, or None for non-IPv4.

    Branches on the pcap's link-layer type so Ethernet and SLL pcaps both yield
    the IP layer at the correct offset (the stored bytes are IP-onward in both
    cases, matching the 2018/2023 byte datasets)."""
    if linktype == DLT_LINUX_SLL:
        return _strip_sll(buf)
    return _strip_eth(buf)  # DLT_EN10MB and any unexpected type fall back to Ethernet


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


def _basic_tag(k: int, n_bytes: int) -> str:
    return f"raw_bytes_pkt{k}_b{n_bytes}"


def _strict_tag(k: int, n_bytes: int) -> str:
    return f"raw_bytes_pkt{k}_b{n_bytes}_strict"


def _out_path(tag: str, scenario: str) -> Path:
    return PARTIAL_FLOW_ROOT / tag / "raw_bytes" / f"{_scenario_tag(scenario)}.parquet"


def _load_label_lookup() -> tuple[np.ndarray, np.ndarray]:
    """flow_id -> class-code lookup from the assembled pct_100 dataset (the seeded,
    capped, ground-truth-labeled flow set the RF models trained on).

    Returns ``(sorted_ids, codes)`` so a worker can map its scenario's flow_ids
    with ``np.searchsorted``. ``codes`` index into ``ATTACK_CLASSES``.
    """
    p = dataset_path(None)  # pct_100/dataset.parquet
    if not p.exists():
        raise FileNotFoundError(
            f"assembled dataset not found: {p}\nRun the partial-flow pipeline with "
            "--assemble first (it defines the kept+labeled flow set)."
        )
    df = pd.read_parquet(p, columns=["flow_id", "label_encoded"], engine="pyarrow")
    ids = df["flow_id"].to_numpy(dtype=np.int64)
    codes = df["label_encoded"].to_numpy(dtype=np.int16)
    order = np.argsort(ids, kind="stable")
    return ids[order], codes[order]


def _select_flows(scenario: str, k: int, sorted_ids: np.ndarray,
                  codes: np.ndarray) -> pd.DataFrame:
    """First-K packets of this scenario's kept flows, with pcap locations + class code."""
    cols = ["capture_file", "capture_packet_index", "flow_id", "packet_index", "timestamp"]
    pdf = pd.read_parquet(packet_base_path(scenario), columns=cols, engine="pyarrow")

    fid = pdf["flow_id"].to_numpy(dtype=np.int64)
    pos = np.clip(np.searchsorted(sorted_ids, fid), 0, len(sorted_ids) - 1)
    in_set = sorted_ids[pos] == fid
    if not in_set.any():
        return pdf.iloc[:0].assign(pkt_order=np.array([], dtype="int64"),
                                   timestamp_rel=np.array([], dtype="float32"),
                                   _code=np.array([], dtype="int16"))
    pdf = pdf[in_set].copy()
    pdf["_code"] = codes[pos[in_set]].astype("int16")

    pdf = pdf.sort_values(["flow_id", "packet_index"], kind="mergesort")
    pdf["pkt_order"] = pdf.groupby("flow_id", sort=False).cumcount()
    pdf = pdf[pdf["pkt_order"] < k].copy()

    first_ts = pdf.groupby("flow_id", sort=False)["timestamp"].transform("min")
    pdf["timestamp_rel"] = (pdf["timestamp"] - first_ts).astype("float32")
    pdf["capture_packet_index"] = pdf["capture_packet_index"].astype("int64")
    return pdf[["capture_file", "capture_packet_index", "flow_id", "pkt_order",
                "timestamp_rel", "_code"]].reset_index(drop=True)


def _open_pcap(p: Path):
    with open(p, "rb") as fh:
        magic = fh.read(4)
    if magic == b"\x0a\x0d\x0d\x0a":
        return RawPcapNgReader(str(p))
    return RawPcapReader(str(p))


def _extract_scenario(scenario: str, k: int, n_bytes: int, force: bool) -> dict:
    """One scenario: select kept flows, re-read its pcaps ONCE, apply both masks,
    write the basic + strict raw-byte parquets."""
    basic_path = _out_path(_basic_tag(k, n_bytes), scenario)
    strict_path = _out_path(_strict_tag(k, n_bytes), scenario)
    if basic_path.exists() and strict_path.exists() and not force:
        print(f"[{scenario}] both outputs exist, skipping", flush=True)
        return {"scenario": scenario, "skipped": True}
    basic_path.parent.mkdir(parents=True, exist_ok=True)
    strict_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    sorted_ids, codes = _load_label_lookup()
    sel = _select_flows(scenario, k, sorted_ids, codes)
    if len(sel) == 0:
        print(f"[{scenario}] no kept flows, nothing to extract", flush=True)
        return {"scenario": scenario, "n_rows": 0}

    n = len(sel)
    raw_basic = np.zeros((n, n_bytes), dtype=np.uint8)
    raw_strict = np.zeros((n, n_bytes), dtype=np.uint8)
    real_len_col = np.zeros(n, dtype=np.int32)
    found_mask = np.zeros(n, dtype=bool)

    pcap_dir = scenario_pcap_dir(scenario)
    n_extracted = 0
    for cap_file, sub in sel.groupby("capture_file", sort=False):
        pcap_path = pcap_dir / cap_file
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
        # Link type can be per-packet (pcapng carries it on each packet's
        # metadata, one entry per interface) or file-level (classic pcap exposes
        # reader.linktype). Many ToN-IoT pcaps are pcapng+SLL, where the reader
        # has NO .linktype attribute, so per-packet meta.linktype is essential —
        # falling back to Ethernet there miscounts ip_index and craters found.
        file_linktype = getattr(reader, "linktype", None)
        ip_index = 0
        try:
            with reader as r:
                for pkt_tuple in r:
                    lt = getattr(pkt_tuple[1], "linktype", None)
                    if lt is None:
                        lt = file_linktype if file_linktype is not None else DLT_EN10MB
                    ip_bytes = _strip_link(pkt_tuple[0], lt)
                    if ip_bytes is None:
                        continue
                    cur_idx = ip_index
                    ip_index += 1
                    pos = idx_to_pos.get(cur_idx)
                    if pos is None:
                        if cur_idx > max_idx:
                            break
                        continue
                    mb = _mask_basic(ip_bytes)
                    ms = _mask_strict(ip_bytes)
                    real_len_col[pos] = len(mb)  # masks preserve length
                    take = min(len(mb), n_bytes)
                    raw_basic[pos, :take] = np.frombuffer(mb[:take], dtype=np.uint8)
                    raw_strict[pos, :take] = np.frombuffer(ms[:take], dtype=np.uint8)
                    found_mask[pos] = True
                    n_extracted += 1
        except Exception as e:
            print(f"  WARN: parse error on {pcap_path.name} at ip_index={ip_index}: {e}",
                  flush=True)

    labels = _CLASS_ARR[sel["_code"].to_numpy()]
    base = {
        "flow_id": sel["flow_id"].astype("int64"),
        "pkt_order": sel["pkt_order"].astype("int8"),
        "capture_packet_index": sel["capture_packet_index"].astype("int64"),
        "timestamp_rel": sel["timestamp_rel"].astype("float32"),
        "real_length": real_len_col.astype("int16"),
        "found": found_mask,
        "Label": pd.array(labels, dtype="string"),
    }
    basic_df = pd.DataFrame({**base,
                             "raw_bytes": [raw_basic[i].tobytes() for i in range(n)]})
    strict_df = pd.DataFrame({**base,
                              "raw_bytes": [raw_strict[i].tobytes() for i in range(n)]})
    cols = ["flow_id", "pkt_order", "capture_packet_index", "timestamp_rel",
            "raw_bytes", "real_length", "found", "Label"]
    basic_df[cols].to_parquet(basic_path, engine="pyarrow", compression="zstd", index=False)
    strict_df[cols].to_parquet(strict_path, engine="pyarrow", compression="zstd", index=False)

    elapsed = time.perf_counter() - t0
    found = int(found_mask.sum())
    print(f"[{scenario}] wrote {n:,} rows, found={found:,} ({100.0*found/n:.1f}%), "
          f"{elapsed:.0f}s -> {basic_path.parent.parent.name}/+strict", flush=True)
    return {"scenario": scenario, "n_rows": int(n), "n_found": found,
            "found_pct": 100.0 * found / n, "elapsed_s": elapsed}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="+", default=None,
                   help="Scenario folders to process (default: all).")
    p.add_argument("--max-packets", type=int, default=5,
                   help="Max packets per flow to extract (train smaller K via --max-pkts).")
    p.add_argument("--n-bytes", type=int, default=256)
    p.add_argument("--force", action="store_true")
    p.add_argument("--scenario-workers", type=int, default=None,
                   help="Scenarios to process concurrently (default: min(cpu-2, 6)). "
                        "Each worker holds its scenario's byte arrays in RAM.")
    args = p.parse_args()

    scenarios = args.scenarios or list_scenarios()
    n_sw = args.scenario_workers or max(1, min((os.cpu_count() or 2) - 2, 6))
    n_sw = max(1, min(n_sw, len(scenarios)))
    print(f"Extracting raw bytes: {len(scenarios)} scenarios, K={args.max_packets}, "
          f"n_bytes={args.n_bytes}, masks=[basic,strict], {n_sw} concurrent workers",
          flush=True)
    print(f"scenarios: {scenarios}", flush=True)

    t0 = time.perf_counter()
    results = []
    if n_sw == 1:
        for sc in scenarios:
            results.append(_extract_scenario(sc, args.max_packets, args.n_bytes, args.force))
    else:
        with ProcessPoolExecutor(max_workers=n_sw) as ex:
            futs = {ex.submit(_extract_scenario, sc, args.max_packets, args.n_bytes,
                              args.force): sc for sc in scenarios}
            for fut in as_completed(futs):
                sc = futs[fut]
                try:
                    results.append(fut.result())
                    print(f"[done] {sc}", flush=True)
                except Exception as e:  # keep going across scenarios
                    print(f"[{sc}] FAILED: {e}", flush=True)

    total_rows = sum(r.get("n_rows", 0) for r in results)
    total_found = sum(r.get("n_found", 0) for r in results)
    print(f"\nTotal: {total_rows:,} rows, found={total_found:,} "
          f"({100.0*total_found/max(1,total_rows):.1f}%), {time.perf_counter()-t0:.0f}s",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
