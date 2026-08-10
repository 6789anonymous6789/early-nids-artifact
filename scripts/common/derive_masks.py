#!/usr/bin/env python3
"""Derive new header-mask variants from an existing basic-mask raw_bytes dir.

The basic-mask parquet already carries the full 256 bytes with addresses and
ports zeroed, so any stricter mask is a further zeroing of the same buffers and
does not require re-reading the PCAPs. Offsets are parsed per packet (IHL from
byte 0, TCP data offset from byte ihl+12), so variable-length headers are
handled rather than assumed away.

Variants
--------
occ_topK   cumulative zeroing of the header regions ranked by the occlusion
           sweep run on the basic-mask model:
             1 IP checksum        bytes 10-11
             2 TCP options        ihl+20 .. ihl+data_offset
             3 IP TTL             byte 8
             4 IP Identification  bytes 4-5
             5 TCP window         ihl+14 .. ihl+15
csum_fix   leaves every field in place but RECOMPUTES the IPv4 header checksum
           over the masked header, so the field remains a valid checksum while
           carrying no information about the bytes the basic mask zeroed.
strict     reproduces the published strict mask (Abo-Alian et al.) from basic,
           used to check this derivation against the existing strict extraction.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

CHUNK = 200_000
TCP, UDP = 6, 17

# Header regions ordered by the macro-F1 drop they cause in the occlusion sweep
# on the basic-mask model. Payload chunks are excluded on purpose: this is a
# header mask.
REGIONS = ["ip_checksum", "tcp_options", "ip_ttl", "ip_id", "tcp_window"]


def _col_mask(n_rows: int, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    """Boolean (n_rows, 256) mask selecting [start, end) on each row."""
    cols = np.arange(256, dtype=np.int32)[None, :]
    return (cols >= start[:, None]) & (cols < end[:, None])


def ip_checksum(a: np.ndarray, ihl: np.ndarray) -> np.ndarray:
    """One's complement IPv4 header checksum, computed with the field zeroed."""
    n = a.shape[0]
    out = np.zeros((n, 2), dtype=np.uint8)
    for length in np.unique(ihl):
        if length < 20 or length > 60 or length % 4:
            continue
        sel = ihl == length
        head = a[sel, :length].astype(np.uint32).copy()
        head[:, 10] = 0
        head[:, 11] = 0
        words = head[:, 0::2] * 256 + head[:, 1::2]
        total = words.sum(axis=1)
        while np.any(total > 0xFFFF):
            total = (total & 0xFFFF) + (total >> 16)
        chk = (~total) & 0xFFFF
        out[sel, 0] = (chk >> 8).astype(np.uint8)
        out[sel, 1] = (chk & 0xFF).astype(np.uint8)
    return out


def transform(a: np.ndarray, variant: str, stats: dict) -> np.ndarray:
    ihl = ((a[:, 0] & 0x0F) * 4).astype(np.int32)
    proto = a[:, 9].astype(np.int32)
    sane = (ihl >= 20) & (ihl <= 60)
    stats["rows"] = stats.get("rows", 0) + int(a.shape[0])
    stats["rows_sane"] = stats.get("rows_sane", 0) + int(sane.sum())

    if variant == "csum_fix":
        new = ip_checksum(a, ihl)
        changed = np.any(new != a[:, 10:12], axis=1) & sane
        stats["checksum_changed"] = stats.get("checksum_changed", 0) + int(changed.sum())
        a[sane, 10:12] = new[sane]
        return a

    if variant in ("strict", "strict_csum"):
        if variant == "strict_csum":            # published scheme plus the checksum
            a[sane, 10] = 0
            a[sane, 11] = 0
        a[sane, 0] = a[sane, 0] & 0x0F          # IP version, keep IHL
        a[sane, 1] = 0                           # TOS
        a[sane, 4] = 0                           # Identification
        a[sane, 5] = 0
        a[sane, 9] = 0                           # protocol
        opts_ip = sane & (ihl > 20)
        if opts_ip.any():                        # IP options
            idx = np.where(opts_ip)[0]
            m = _col_mask(len(idx), np.full(len(idx), 20, np.int32), ihl[idx])
            sub = a[idx]
            sub[m] = 0
            a[idx] = sub
        _zero_tcp(a, ihl, proto, sane, stats, options=True, window=False)
        return a

    k = int(variant.replace("occ_top", ""))
    todo = REGIONS[:k]

    if "ip_checksum" in todo:
        a[sane, 10] = 0
        a[sane, 11] = 0
    if "ip_ttl" in todo:
        a[sane, 8] = 0
    if "ip_id" in todo:
        a[sane, 4] = 0
        a[sane, 5] = 0
    _zero_tcp(a, ihl, proto, sane, stats,
              options="tcp_options" in todo, window="tcp_window" in todo)
    return a


def _zero_tcp(a, ihl, proto, sane, stats, options: bool, window: bool) -> None:
    if not (options or window):
        return
    is_tcp = sane & (proto == TCP)
    if not is_tcp.any():
        return
    idx = np.where(is_tcp)[0]
    base = ihl[idx]
    doff_byte = a[idx, np.minimum(base + 12, 255)]
    doff = (((doff_byte >> 4) & 0x0F).astype(np.int32)) * 4
    stats["tcp_rows"] = stats.get("tcp_rows", 0) + len(idx)

    if window:
        m = _col_mask(len(idx), np.minimum(base + 14, 256), np.minimum(base + 16, 256))
        sub = a[idx]
        sub[m] = 0
        a[idx] = sub

    if options:
        has = (doff > 20) & (doff <= 60)
        if has.any():
            jdx = idx[has]
            m = _col_mask(len(jdx),
                          np.minimum(base[has] + 20, 256),
                          np.minimum(base[has] + doff[has], 256))
            sub = a[jdx]
            sub[m] = 0
            a[jdx] = sub
            stats["tcp_with_options"] = stats.get("tcp_with_options", 0) + int(has.sum())


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--in-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--variant", required=True)
    p.add_argument("--limit-files", type=int, default=0)
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stats: dict = {}
    t0 = time.perf_counter()
    files = sorted(args.in_dir.glob("*.parquet"))
    if args.limit_files:
        files = files[: args.limit_files]
    for src in files:
        dst = args.out_dir / src.name
        pf = pq.ParquetFile(src)
        writer = None
        for batch in pf.iter_batches(batch_size=CHUNK):
            tb = pa.Table.from_batches([batch])
            blobs = tb.column("raw_bytes").to_pylist()
            arr = np.frombuffer(b"".join(blobs), dtype=np.uint8)
            arr = arr.reshape(len(blobs), 256).copy()
            arr = transform(arr, args.variant, stats)
            col = pa.array([bytes(r) for r in arr], type=pa.binary())
            tb = tb.set_column(tb.schema.get_field_index("raw_bytes"), "raw_bytes", col)
            if writer is None:
                writer = pq.ParquetWriter(dst, tb.schema, compression="snappy")
            writer.write_table(tb)
        if writer:
            writer.close()
        print(f"  {src.name}", flush=True)

    stats["variant"] = args.variant
    stats["seconds"] = round(time.perf_counter() - t0, 1)
    (args.out_dir.parent / f"derive_{args.variant}_stats.json").write_text(
        json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
