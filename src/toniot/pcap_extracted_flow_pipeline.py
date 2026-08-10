"""pcap → packet_base → partial-flow features for ToN-IoT.

Mirrors ``src/cic_iot2023/pcap_extracted_flow_pipeline.py`` but ToN-IoT differs
in two ways that drive the design:

  1. **Mixed-traffic scenarios.** Each pcap lives under a scenario folder
     (``normal_DDoS``, ``normal_scanning``, ...) that contains BOTH benign and
     attack packets, so the folder name is NOT the flow label. Labels come from
     the official per-connection ground truth via ``src.toniot.labeling`` (exact
     canonical 5-tuple + time, validated: ~73-89% coverage, L0 only).

  2. **Label-first + global caps.** Because labels are per-flow (not per-folder),
     per-class subsampling can only happen AFTER labeling, on the pooled set of
     all scenarios. So extraction does NOT early-stop by class; each scenario is
     fully parsed, every flow is labeled, and caps are applied at assembly time
     (``assemble_dataset``) with a seeded per-class subsample. The kept flow_id
     set is identical across packet cuts (a flow exists in every cut as long as
     it has >=1 packet), so the same seed yields the same flows for pct_100 and
     each packet_abs_K — keeping the cuts directly comparable.

Output layout under ``data/ToN-IoT/partial_flow/``:

    packet_base/<scenario>.parquet              # per-packet rows, globally-unique flow_id
    pct_100/raw_flow/<scenario>.parquet         # full-flow features, labeled (uncapped)
    packet_abs_{K}/raw_flow/<scenario>.parquet  # first-K-packet features, labeled (uncapped)
    pct_100/dataset.parquet                     # pooled + globally capped (assemble step)
    packet_abs_{K}/dataset.parquet
    manifests/<scenario>.json
"""

from __future__ import annotations

import argparse
import hashlib
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
from src.toniot.labeling import (
    ATTACK_CLASSES,
    NORMAL_LABEL,
    GtIndex,
    build_gt_index,
    coverage_summary,
    get_cap,
    label_flows,
    load_gt,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_ROOT = _REPO_ROOT / "data" / "ToN-IoT"
PCAP_ROOT = (
    _DATA_ROOT / "original" / "Raw_datasets" / "network_data"
    / "Network_dataset_pcaps"
)
PARTIAL_FLOW_ROOT = _DATA_ROOT / "partial_flow"
PACKET_BASE_DIR = PARTIAL_FLOW_ROOT / "packet_base"
MANIFEST_DIR = PARTIAL_FLOW_ROOT / "manifests"

# ToN-IoT label/feature columns: the shared 2018 raw features minus the 2018
# "Label"/"label_encoded" (we attach ToN-IoT's own), plus our labeling columns.
_BASE_FEATURE_COLUMNS = [c for c in RAW_FEATURE_COLUMNS
                        if c not in ("Label", "label_encoded")]
TONIOT_FLOW_COLUMNS = _BASE_FEATURE_COLUMNS + [
    "Label_ToNIoT", "label_encoded", "matched", "match_level",
]


# ---------------------------------------------------------------------------
# Scenario discovery + path helpers
# ---------------------------------------------------------------------------


def list_scenarios() -> list[str]:
    """All scenario folders holding pcaps (normal_pcaps + every attack folder)."""
    scenarios: list[str] = []
    normal = PCAP_ROOT / "normal_pcaps"
    if normal.is_dir() and any(normal.glob("*.pcap")):
        scenarios.append("normal_pcaps")
    attack_root = PCAP_ROOT / "normal_attack_pcaps"
    if attack_root.is_dir():
        for d in sorted(attack_root.iterdir()):
            if d.is_dir() and any(d.glob("*.pcap")):
                scenarios.append(f"normal_attack_pcaps/{d.name}")
    return scenarios


def scenario_pcap_dir(scenario: str) -> Path:
    return PCAP_ROOT / scenario


def list_scenario_pcaps(scenario: str) -> list[Path]:
    d = scenario_pcap_dir(scenario)
    if not d.exists():
        raise FileNotFoundError(f"scenario dir missing: {d}")
    return sorted(d.glob("*.pcap"))


def _scenario_tag(scenario: str) -> str:
    """Filesystem-safe leaf tag for a scenario path (drops the parent prefix)."""
    return scenario.split("/")[-1]


def packet_base_path(scenario: str) -> Path:
    return PACKET_BASE_DIR / f"{_scenario_tag(scenario)}.parquet"


def raw_flow_parquet_path(scenario: str, max_packets: int | None = None) -> Path:
    tag = "pct_100" if max_packets is None else f"packet_abs_{max_packets}"
    return PARTIAL_FLOW_ROOT / tag / "raw_flow" / f"{_scenario_tag(scenario)}.parquet"


def dataset_path(max_packets: int | None = None) -> Path:
    tag = "pct_100" if max_packets is None else f"packet_abs_{max_packets}"
    return PARTIAL_FLOW_ROOT / tag / "dataset.parquet"


def manifest_path(scenario: str) -> Path:
    return MANIFEST_DIR / f"{_scenario_tag(scenario)}.json"


def _flow_id_offset(scenario: str) -> int:
    """Deterministic per-scenario flow_id offset so flow_ids are globally unique
    across scenarios. Derived from a hash of the scenario tag, spaced 1e12 apart
    (no scenario reaches 1e12 flows)."""
    h = hashlib.sha1(_scenario_tag(scenario).encode()).hexdigest()
    return (int(h[:8], 16) % 9000) * 1_000_000_000_000 + 1_000_000_000_000


# ---------------------------------------------------------------------------
# Per-pcap worker
# ---------------------------------------------------------------------------


def _process_single_pcap_to_temp(pcap_path_str: str, active_timeout: float,
                                 tmp_dir_str: str, max_packets: int | None = None) -> dict:
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


def _decode_dictionary_columns(table: pa.Table) -> pa.Table:
    """Cast any dictionary-encoded columns to their plain value type.

    pandas restores object/category columns inconsistently across parquet files
    (``flow_src_ip`` comes back as ``dictionary<large_string>`` for some pcaps,
    plain ``large_string`` for others). A ParquetWriter fixes its schema from the
    first table, so a later dictionary-typed table would raise "Table schema does
    not match". Decoding to the value type makes every appended table uniform.
    """
    new_fields = []
    changed = False
    for field in table.schema:
        if pa.types.is_dictionary(field.type):
            new_fields.append(pa.field(field.name, field.type.value_type))
            changed = True
        else:
            new_fields.append(field)
    if not changed:
        return table
    return table.cast(pa.schema(new_fields))


def _append_dataframe(writer: pq.ParquetWriter | None, output_path: Path,
                      df: pd.DataFrame) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(df, preserve_index=False)
    table = _decode_dictionary_columns(table)
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
    else:
        # Conform later tables to the writer's established schema (handles
        # large_string vs string and column-order drift across pcaps).
        names = [f.name for f in writer.schema]
        table = table.select(names).cast(writer.schema)
    writer.write_table(table)
    return writer


# ---------------------------------------------------------------------------
# Scenario-level packet base (full parse — no class early-stop)
# ---------------------------------------------------------------------------


def build_scenario_packet_base(scenario: str, *, force: bool = False,
                               max_workers: int | None = None,
                               max_packets_per_pcap: int | None = None,
                               max_files: int | None = None) -> dict:
    """Build the per-scenario packet-base parquet (one row per packet, globally
    unique flow_id). Parses every pcap in the scenario (mixed normal+attack);
    ``max_packets_per_pcap`` / ``max_files`` exist only for smoke tests."""
    pcaps = list_scenario_pcaps(scenario)
    if max_files is not None and max_files > 0:
        pcaps = pcaps[:max_files]
    output_path = packet_base_path(scenario)
    tmp_path = output_path.with_suffix(".tmp.parquet")

    if output_path.exists() and not force:
        mf = manifest_path(scenario)
        if mf.exists():
            return json.loads(mf.read_text())

    if tmp_path.exists():
        tmp_path.unlink()
    tmp_dir = PARTIAL_FLOW_ROOT / "_tmp" / _scenario_tag(scenario)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    worker_count = _resolve_max_workers(max_workers, len(pcaps))
    print(f"[{scenario}] processing {len(pcaps)} PCAPs with {worker_count} workers ...",
          flush=True)
    t0 = time.perf_counter()
    results_by_capture: dict[str, dict] = {}

    if worker_count == 1:
        for pcap in pcaps:
            results_by_capture[pcap.name] = _process_single_pcap_to_temp(
                str(pcap), ACTIVITY_TIMEOUT_SECONDS, str(tmp_dir), max_packets_per_pcap)
            print(f"[{scenario}] extract {len(results_by_capture)}/{len(pcaps)}: {pcap.name}",
                  flush=True)
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as ex:
            futures = {ex.submit(_process_single_pcap_to_temp, str(pcap),
                                 ACTIVITY_TIMEOUT_SECONDS, str(tmp_dir),
                                 max_packets_per_pcap): pcap for pcap in pcaps}
            for fut in as_completed(futures):
                pcap = futures[fut]
                results_by_capture[pcap.name] = fut.result()
                print(f"[{scenario}] extract {len(results_by_capture)}/{len(pcaps)}: {pcap.name}",
                      flush=True)

    print(f"[{scenario}] merging into {tmp_path.name} ...", flush=True)
    writer = None
    total_packet_rows = 0
    total_flow_rows = 0
    flow_offset = _flow_id_offset(scenario)
    try:
        for pcap in pcaps:
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
        raise RuntimeError(f"No IP packets extracted for scenario {scenario}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(output_path)
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scenario": scenario,
        "source": {"pcap_dir": str(scenario_pcap_dir(scenario)),
                   "pcaps_total": len(pcaps),
                   "pcaps_processed": [p.name for p in pcaps],
                   "max_packets_per_pcap": max_packets_per_pcap},
        "build": {"active_timeout_seconds": ACTIVITY_TIMEOUT_SECONDS,
                  "flow_grouping_scope": "per_capture_file",
                  "max_workers": int(worker_count)},
        "labeling": {"label_source": "official_ground_truth_5tuple",
                     "labels_available": True, "training_ready": True},
        "artifacts": {"packet_base_parquet": str(output_path)},
        "stats": {"packet_rows": int(total_packet_rows),
                  "packet_base_flow_rows": int(total_flow_rows),
                  "elapsed_seconds": time.perf_counter() - t0},
    }
    manifest_path(scenario).parent.mkdir(parents=True, exist_ok=True)
    manifest_path(scenario).write_text(json.dumps(manifest, indent=2))
    print(f"[{scenario}] packet-base done: packet_rows={total_packet_rows:,}, "
          f"flows={total_flow_rows:,}, {time.perf_counter()-t0:.0f}s", flush=True)
    return manifest


# ---------------------------------------------------------------------------
# Scenario-level labeled raw flows (full flow or first-K-packet cut), uncapped
# ---------------------------------------------------------------------------


def build_scenario_raw_flow(scenario: str, *, max_packets: int | None = None,
                            force: bool = False, max_workers: int | None = None,
                            gt_index: GtIndex | None = None) -> dict:
    """Aggregate the scenario packet-base into labeled flow features (uncapped).

    max_packets=None -> full flow (pct_100); else keep first ``max_packets``
    packets per flow. Labels come from the official ground truth.
    """
    out_path = raw_flow_parquet_path(scenario, max_packets)
    if out_path.exists() and not force:
        return {"output_path": str(out_path), "cached": True}

    pkt_base = packet_base_path(scenario)
    if not pkt_base.exists():
        build_scenario_packet_base(scenario, force=False, max_workers=max_workers)

    filters = None
    if max_packets is not None:
        filters = [("packet_index", "<", max_packets)]
    print(f"[{scenario}] reading packet-base (max_packets={max_packets}) ...", flush=True)
    packets = pd.read_parquet(pkt_base, engine="pyarrow", filters=filters)

    print(f"[{scenario}] aggregating {len(packets):,} packet rows ...", flush=True)
    flows = aggregate_flow_features(packets, day=_scenario_tag(scenario))

    if gt_index is None:
        gt_index = build_gt_index()
    labeled = label_flows(flows, gt_index=gt_index)
    cov = coverage_summary(labeled)
    print(f"[{scenario}] coverage: {cov['pct_matched']:.1f}% matched "
          f"({cov['n_matched']:,}/{cov['n_total']:,})", flush=True)

    labeled = labeled[TONIOT_FLOW_COLUMNS].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labeled.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    print(f"[{scenario}] wrote {len(labeled):,} flows -> {out_path}", flush=True)
    return {"output_path": str(out_path), "n_flows": int(len(labeled)),
            "coverage": cov, "cached": False}


# ---------------------------------------------------------------------------
# Dataset assembly: pool all scenarios + global per-class caps (seeded)
# ---------------------------------------------------------------------------


def assemble_dataset(*, max_packets: int | None = None, random_state: int = 42,
                     caps: dict[str, int] | None = None,
                     scenarios: list[str] | None = None) -> dict:
    """Pool every scenario's labeled raw flows for one cut, apply global
    per-class caps with a seeded subsample, write ``<tag>/dataset.parquet``.

    The kept flow_id set is computed from class membership (identical across
    cuts), so the same seed selects the same flows for pct_100 and each
    packet_abs_K — keeping cuts comparable.
    """
    scenarios = scenarios or list_scenarios()
    frames = []
    for sc in scenarios:
        p = raw_flow_parquet_path(sc, max_packets)
        if not p.exists():
            print(f"[assemble] WARN missing {p}, skipping {sc}", flush=True)
            continue
        frames.append(pd.read_parquet(p, engine="pyarrow"))
    if not frames:
        raise RuntimeError("no scenario raw-flow parquets found to assemble")
    pool = pd.concat(frames, ignore_index=True)
    print(f"[assemble] pooled {len(pool):,} flows from {len(frames)} scenarios", flush=True)

    rng = np.random.RandomState(random_state)
    keep_idx = []
    for cls, grp in pool.groupby("Label_ToNIoT", sort=False):
        cap = get_cap(cls, caps)
        if cap is None or len(grp) <= cap:
            keep_idx.append(grp.index.to_numpy())
        else:
            keep_idx.append(rng.choice(grp.index.to_numpy(), size=cap, replace=False))
            print(f"[assemble] cap {cls}: {len(grp):,} -> {cap:,}", flush=True)
    keep = np.concatenate(keep_idx)
    capped = pool.loc[keep].reset_index(drop=True)

    out_path = dataset_path(max_packets)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    capped.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)
    dist = {str(k): int(v) for k, v in capped["Label_ToNIoT"].value_counts().items()}
    print(f"[assemble] wrote {len(capped):,} flows -> {out_path}\n  dist={dist}", flush=True)
    return {"output_path": str(out_path), "n_flows": int(len(capped)),
            "label_distribution": dist}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scenarios", nargs="+", default=None,
                   help="Scenario folders to process (default: all).")
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-flows", action="store_true",
                   help="Stop after building the packet-base parquet.")
    p.add_argument("--packet-cuts", nargs="+", type=int, default=[3, 4, 5],
                   help="First-K-packet cuts to build besides pct_100 (default 3 4 5).")
    p.add_argument("--no-pct100", action="store_true",
                   help="Skip the full-flow (pct_100) cut.")
    p.add_argument("--assemble", action="store_true",
                   help="After per-scenario builds, pool + cap into dataset.parquet per cut.")
    p.add_argument("--max-packets-per-pcap", type=int, default=None,
                   help="Smoke-test cap on packets read per pcap.")
    p.add_argument("--max-files", type=int, default=None,
                   help="Smoke-test cap on pcaps per scenario.")
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    scenarios = args.scenarios or list_scenarios()
    print(f"[main] {len(scenarios)} scenarios: {scenarios}", flush=True)

    cuts: list[int | None] = ([] if args.no_pct100 else [None]) + list(args.packet_cuts)

    gt_index = None
    if not args.skip_flows:
        print("[main] building GT index once ...", flush=True)
        gt_index = build_gt_index()
    for sc in scenarios:
        print(f"\n=== {sc} ===", flush=True)
        build_scenario_packet_base(sc, force=args.force, max_workers=args.max_workers,
                                   max_packets_per_pcap=args.max_packets_per_pcap,
                                   max_files=args.max_files)
        if args.skip_flows:
            continue
        for mp in cuts:
            build_scenario_raw_flow(sc, max_packets=mp, force=args.force,
                                    max_workers=args.max_workers, gt_index=gt_index)

    if args.assemble and not args.skip_flows:
        for mp in cuts:
            print(f"\n=== assemble cut={mp} ===", flush=True)
            assemble_dataset(max_packets=mp, random_state=args.random_state,
                             scenarios=scenarios)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
