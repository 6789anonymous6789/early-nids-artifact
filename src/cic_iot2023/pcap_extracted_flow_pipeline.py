"""pcap → packet_base → partial-flow features for CIC-IoT2023.

Mirrors ``src/cic_ids2018/pcap_extracted_flow_pipeline.py`` but the partition
unit is the **attack-class folder** (not "days"), and labels come for free from
the folder name (``src.cic_iot2023.labeling.label_from_pcap_dir``).

Per the supervisor deck, the early-detection axis is **packet caps** (5/4/3
packets), NOT time cuts — so this module only builds ``pct_100`` and
``packet_abs_{K}`` outputs.

Output layout under ``data/CIC-IoT2023/partial_flow/``:

    packet_base/<class>.parquet              # per-class grouped packet rows (all packets)
    pct_100/raw_flow/<class>.parquet         # full-flow CICFlow features, labeled
    packet_abs_{K}/raw_flow/<class>.parquet  # first-K-packet flow features, labeled
    manifests/<class>.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.cic_ids2018.pcap_extracted_flow_pipeline import (
    ACTIVITY_TIMEOUT_SECONDS,
    DEFAULT_MAX_WORKERS,
    RAW_FEATURE_COLUMNS,
    SCHEMA_VERSION,
    aggregate_flow_features,
)
from src.cic_ids2018.pcap_packet_extraction import extract_packets, group_into_flows
from src.cic_iot2023.labeling import RAW_CLASSES, get_cap

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_ROOT = _REPO_ROOT / "data" / "CIC-IoT2023"
PCAP_ROOT = _DATA_ROOT / "original" / "PCAP"
PARTIAL_FLOW_ROOT = _DATA_ROOT / "partial_flow"
PACKET_BASE_DIR = PARTIAL_FLOW_ROOT / "packet_base"
MANIFEST_DIR = PARTIAL_FLOW_ROOT / "manifests"

# Probe the first pcap reading flow_cap * this many packets. Pure volumetric floods are
# dense from the start, so this reaches the cap in one bounded read (~seconds, bounded
# memory) instead of parsing 20-30M-packet files (~35 min, OOM risk).
_PACKET_PROBE_FACTOR = 10


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------


def class_pcap_dir(cls: str) -> Path:
    return PCAP_ROOT / cls


def list_class_pcaps(cls: str) -> list[Path]:
    d = class_pcap_dir(cls)
    if not d.exists():
        raise FileNotFoundError(f"PCAP directory missing: {d}")
    return sorted(d.glob("*.pcap"))


def packet_base_path(cls: str) -> Path:
    return PACKET_BASE_DIR / f"{cls}.parquet"


def raw_flow_parquet_path(cls: str, max_packets: int | None = None) -> Path:
    tag = "pct_100" if max_packets is None else f"packet_abs_{max_packets}"
    return PARTIAL_FLOW_ROOT / tag / "raw_flow" / f"{cls}.parquet"


def manifest_path(cls: str) -> Path:
    return MANIFEST_DIR / f"{cls}.json"


# ---------------------------------------------------------------------------
# Per-pcap worker
# ---------------------------------------------------------------------------


def _process_single_pcap_to_temp(pcap_path_str: str, active_timeout: float,
                                 tmp_dir_str: str, max_packets: int | None = None) -> dict:
    """Worker: extract + group one PCAP into a temp parquet (same shape as 2018)."""
    pcap_path = Path(pcap_path_str)
    tmp_dir = Path(tmp_dir_str)
    temp_output_path = tmp_dir / f"{pcap_path.name}.parquet"

    packets = extract_packets(pcap_path, source_id=pcap_path.name, max_packets=max_packets)
    if packets.empty:
        return {"capture_file": pcap_path.name, "temp_parquet_path": None,
                "packet_rows": 0, "flow_rows": 0}

    grouped = group_into_flows(packets, active_timeout=active_timeout)
    grouped["flow_total_packets"] = (
        grouped.groupby("flow_id")["packet_index"].transform("max") + 1
    )
    grouped.to_parquet(temp_output_path, engine="pyarrow", compression="snappy", index=False)
    return {"capture_file": pcap_path.name, "temp_parquet_path": str(temp_output_path),
            "packet_rows": int(len(grouped)), "flow_rows": int(grouped["flow_id"].nunique())}


def _resolve_max_workers(max_workers: int | None, n_pcaps: int) -> int:
    if n_pcaps <= 0:
        return 1
    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        return max(1, min(DEFAULT_MAX_WORKERS, cpu_count, n_pcaps))
    return max(1, min(int(max_workers), cpu_count, n_pcaps))


def _append_dataframe(writer: pq.ParquetWriter | None, output_path: Path,
                      df: pd.DataFrame) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
    writer.write_table(table)
    return writer


# ---------------------------------------------------------------------------
# Class-level packet base
# ---------------------------------------------------------------------------


def build_class_packet_base(cls: str, *, force: bool = False,
                            max_workers: int | None = None,
                            flow_cap: int | None = None,
                            max_files: int | None = None,
                            probe_packets: int | None = None) -> dict:
    """Build the per-class packet-base parquet (one row per packet, with flow_id).

    flow_id is offset per-pcap so it stays unique within the class file.

    ``flow_cap`` enables adaptive early-stop: parse the first pcap to estimate
    flows-per-file, then parse only as many more pcaps as needed to reach ~1.3x
    the cap (re-checking after each batch). Volumetric floods (~250k flows/file)
    usually stop after one file; classes below the cap parse all their files.
    ``flow_cap=None`` parses every pcap (no early-stop).
    """
    pcaps = list_class_pcaps(cls)
    if max_files is not None and max_files > 0:
        # Hard ceiling: low-flow-density floods (few flows, many pkts/flow) never reach
        # flow_cap, so without this they would parse all ~50GB of files. We accept fewer
        # flows for such classes (the cap is a MAXIMUM).
        pcaps = pcaps[:max_files]
    output_path = packet_base_path(cls)
    tmp_path = output_path.with_suffix(".tmp.parquet")

    if output_path.exists() and not force:
        mf = manifest_path(cls)
        if mf.exists():
            return json.loads(mf.read_text())

    if tmp_path.exists():
        tmp_path.unlink()
    tmp_dir = PARTIAL_FLOW_ROOT / "_tmp" / cls
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    worker_count = _resolve_max_workers(max_workers, len(pcaps))
    print(f"[{cls}] processing up to {len(pcaps)} PCAPs with {worker_count} workers "
          f"(flow_cap={flow_cap}) ...", flush=True)
    t0 = time.perf_counter()
    results_by_capture: dict[str, dict] = {}

    def _run_batch(batch: list[Path], mp: int | None = None) -> None:
        wc = _resolve_max_workers(max_workers, len(batch))
        if wc == 1:
            for pcap in batch:
                results_by_capture[pcap.name] = _process_single_pcap_to_temp(
                    str(pcap), ACTIVITY_TIMEOUT_SECONDS, str(tmp_dir), mp)
                print(f"[{cls}] extract {len(results_by_capture)}/{len(pcaps)}: {pcap.name}",
                      flush=True)
        else:
            with ProcessPoolExecutor(max_workers=wc) as ex:
                futures = {ex.submit(_process_single_pcap_to_temp, str(pcap),
                                     ACTIVITY_TIMEOUT_SECONDS, str(tmp_dir), mp): pcap
                           for pcap in batch}
                for fut in as_completed(futures):
                    pcap = futures[fut]
                    results_by_capture[pcap.name] = fut.result()
                    print(f"[{cls}] extract {len(results_by_capture)}/{len(pcaps)}: {pcap.name}",
                          flush=True)

    def _flows_done(done: list[Path]) -> int:
        return sum(int(results_by_capture[p.name].get("flow_rows", 0)) for p in done)

    def _full_parse_early_stop() -> list[Path]:
        # Full parse (no packet cap), adding files until ~1.3x the flow cap.
        proc = list(pcaps[:1])
        pending = list(pcaps[1:])
        _run_batch(proc, None)
        target = int(flow_cap * 1.3) if flow_cap else None
        while pending and target is not None and _flows_done(proc) < target:
            done_now = _flows_done(proc)
            per_file = max(1.0, done_now / len(proc))
            need = max(1, int(np.ceil((target - done_now) / per_file)))
            batch, pending = pending[:need], pending[need:]
            _run_batch(batch, None)
            proc += batch
        if pending:
            print(f"[{cls}] early-stop: parsed {len(proc)}/{len(pcaps)} pcaps "
                  f"(~{_flows_done(proc):,} flows >= target {target:,}); "
                  f"skipped {len(pending)}", flush=True)
        return proc

    if probe_packets is not None and probe_packets > 0:
        # Bounded mode: read only file0 up to probe_packets packets and use the result
        # as-is. For ultra-low-flow-density volumetric floods (e.g. ICMP_Flood at ~2300
        # pkt/flow) reaching flow_cap would need tens of GB; the cap is a MAXIMUM, so we
        # accept fewer flows. No early-stop, no full-parse fallback.
        processed = list(pcaps[:1])
        _run_batch(processed, probe_packets)
        print(f"[{cls}] bounded read: <= {probe_packets:,} packets of file0 -> "
              f"{_flows_done(processed):,} flows (1/{len(pcaps)} pcaps)", flush=True)
    elif flow_cap is None:
        processed = list(pcaps)
        _run_batch(processed, None)
    else:
        # Fast path: probe the first pcap reading only flow_cap*FACTOR packets. Pure
        # volumetric floods are dense from the start, so this yields >= flow_cap flows
        # in seconds with bounded memory (vs parsing a 26M-packet file for 35 min). If
        # the flood density is NOT in the first packets (e.g. a capture that starts with
        # setup traffic, like Mirai-greeth), fall back to a full file-level early-stop
        # — those classes have far fewer packets, so the full parse is cheap anyway.
        probe_n = int(flow_cap * _PACKET_PROBE_FACTOR)
        processed = list(pcaps[:1])
        _run_batch(processed, probe_n)
        if _flows_done(processed) >= flow_cap:
            print(f"[{cls}] packet-cap fast path: {_flows_done(processed):,} flows from "
                  f"~{probe_n:,} packets of 1/{len(pcaps)} pcaps", flush=True)
        else:
            results_by_capture.clear()
            processed = _full_parse_early_stop()

    print(f"[{cls}] merging into {tmp_path.name} ...", flush=True)
    writer = None
    total_packet_rows = 0
    total_flow_rows = 0
    flow_offset = 0
    try:
        for pcap in processed:
            r = results_by_capture.get(pcap.name)
            if r is None:
                continue
            tpath = r.get("temp_parquet_path")
            if tpath is None:
                continue
            grouped = pd.read_parquet(tpath, engine="pyarrow")
            if grouped.empty:
                continue
            grouped["flow_id"] = grouped["flow_id"].astype("int64") + flow_offset
            flow_offset = int(grouped["flow_id"].max()) + 1
            writer = _append_dataframe(writer, tmp_path, grouped)
            total_packet_rows += int(len(grouped))
            total_flow_rows += int(grouped["flow_id"].nunique())
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError(f"No IP packets extracted for class {cls}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(output_path)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "class": cls,
        "source": {"pcap_dir": str(class_pcap_dir(cls)),
                   "pcaps_total": len(pcaps),
                   "pcaps_processed": [p.name for p in processed],
                   "flow_cap": flow_cap},
        "build": {"active_timeout_seconds": ACTIVITY_TIMEOUT_SECONDS,
                  "flow_grouping_scope": "per_capture_file",
                  "max_workers": int(worker_count)},
        "labeling": {"label_source": "pcap_directory_name", "label": cls,
                     "labels_available": True, "training_ready": True},
        "artifacts": {"packet_base_parquet": str(output_path)},
        "stats": {"packet_rows": int(total_packet_rows),
                  "packet_base_flow_rows": int(total_flow_rows),
                  "elapsed_seconds": time.perf_counter() - t0},
    }
    manifest_path(cls).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(cls).write_text(json.dumps(manifest, indent=2))
    print(f"[{cls}] packet-base done: packet_rows={total_packet_rows:,}, "
          f"flows={total_flow_rows:,}, {time.perf_counter()-t0:.0f}s", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# Class-level labeled raw-flow output (full flow or first-K-packet cut)
# ---------------------------------------------------------------------------


def subsample_flow_ids(flow_ids: np.ndarray, cap: int | None,
                       random_state: int = 42) -> np.ndarray:
    """Pick up to `cap` flow_ids (seeded). cap=None -> keep all."""
    flow_ids = np.asarray(flow_ids)
    if cap is None or len(flow_ids) <= cap:
        return flow_ids
    rng = np.random.RandomState(random_state)
    return rng.choice(flow_ids, size=cap, replace=False)


def build_class_raw_flow(cls: str, *, max_packets: int | None = None,
                         cap: int | None = -1, random_state: int = 42,
                         force: bool = False, max_workers: int | None = None) -> dict:
    """Aggregate the class packet-base into labeled flow features.

    max_packets=None -> full flow (pct_100); else keep first `max_packets` packets.
    cap=-1 uses the class default from labeling.get_cap; cap=None keeps all flows;
    any int caps the number of flows (seeded subsample).
    """
    out_path = raw_flow_parquet_path(cls, max_packets)
    if out_path.exists() and not force:
        return {"output_path": str(out_path), "cached": True}

    if cap == -1:
        cap = get_cap(cls)

    pkt_base = packet_base_path(cls)
    if not pkt_base.exists():
        build_class_packet_base(cls, force=False, max_workers=max_workers, flow_cap=cap)

    cols = None
    filters = None
    if max_packets is not None:
        filters = [("packet_index", "<", max_packets)]
    print(f"[{cls}] reading packet-base (max_packets={max_packets}) ...", flush=True)
    packets = pd.read_parquet(pkt_base, engine="pyarrow", columns=cols, filters=filters)

    if cap is not None:
        all_ids = packets["flow_id"].unique()
        keep = subsample_flow_ids(all_ids, cap, random_state)
        if len(keep) < len(all_ids):
            packets = packets[packets["flow_id"].isin(set(keep.tolist()))].copy()
            print(f"[{cls}] subsampled flows {len(all_ids):,} -> {len(keep):,} (cap={cap})",
                  flush=True)

    print(f"[{cls}] aggregating {len(packets):,} packet rows ...", flush=True)
    flows = aggregate_flow_features(packets, day=cls)
    flows["Label"] = cls
    flows = flows[RAW_FEATURE_COLUMNS].copy()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    flows.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"[{cls}] wrote {len(flows):,} flows -> {out_path}", flush=True)
    return {"output_path": str(out_path), "n_flows": int(len(flows)), "cached": False}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classes", nargs="+", default=list(RAW_CLASSES),
                   help="Which class folders to process (default: all 34).")
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-flows", action="store_true",
                   help="Stop after building the packet-base parquet.")
    p.add_argument("--max-packets", type=int, default=None,
                   help="First-K-packet cut (default None = full flow / pct_100).")
    p.add_argument("--cap", type=int, default=-1,
                   help="Per-class flow cap (-1=class default, 0=keep all).")
    args = p.parse_args()
    cap = None if args.cap == 0 else args.cap

    for cls in args.classes:
        print(f"\n=== {cls} ===", flush=True)
        build_class_packet_base(cls, force=args.force, max_workers=args.max_workers)
        if args.skip_flows:
            continue
        build_class_raw_flow(cls, max_packets=args.max_packets, cap=cap,
                             force=args.force, max_workers=args.max_workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
