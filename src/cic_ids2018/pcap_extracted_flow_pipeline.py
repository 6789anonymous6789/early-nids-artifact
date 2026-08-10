"""
Build packet-base and partial-flow datasets directly from CIC-IDS2018 raw PCAP archives.

The current target is the 100% observation baseline:
  - one packet-base Parquet per day
  - one raw flow-view Parquet per day under partial_flow/pct_100/

The pipeline is already ratio-aware so future lower-percentage datasets can be
generated from the retained packet-base artifacts without re-extracting raw PCAPs.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import os
import re
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.cic_ids2018.pcap_flow_labeling import day_has_attack_profiles, label_flows, load_attack_profiles
from src.cic_ids2018.pcap_packet_extraction import extract_packets, group_into_flows


_REPO_ROOT = Path(__file__).parent.parent.parent
_DATA_ROOT = _REPO_ROOT / "data" / "CIC-IDS2018"
RAW_ROOT = _DATA_ROOT / "original" / "raw"
EXTRACTED_ROOT = _DATA_ROOT / "pcap_extracted"
PARTIAL_FLOW_ROOT = _DATA_ROOT / "partial_flow"
PACKET_BASE_DIR = PARTIAL_FLOW_ROOT / "packet_base"
MANIFEST_DIR = PARTIAL_FLOW_ROOT / "manifests"
BY_PCAP_ROOT = PARTIAL_FLOW_ROOT / "by_pcap"
_CAPTURE_HOST_IP_RE = re.compile(r"(\d{1,3}(?:\.\d{1,3}){3})")
_CAPTURE_MULTIPART_RE = re.compile(
    r"\d{1,3}(?:\.\d{1,3}){3}\s*(?:-| )?\s*part\s*\d+$",
    re.IGNORECASE,
)

SCHEMA_VERSION = 1
ACTIVITY_TIMEOUT_SECONDS = 120.0
ACTIVE_IDLE_THRESHOLD_SECONDS = 5.0
UNLABELED_LABEL_ENCODED = -1
DEFAULT_MAX_WORKERS = max(1, min((os.cpu_count() or 1) - 1, 12))

DAY_ALIASES = {
    "Thuesday-20-02-2018": "Tuesday-20-02-2018",
    "Tuesday-20-02-2018": "Tuesday-20-02-2018",
}

RAW_FEATURE_COLUMNS = [
    "day",
    "capture_file",
    "flow_id",
    "flow_src_ip",
    "flow_dst_ip",
    "flow_src_port",
    "flow_dst_port",
    "flow_start_ts",
    "flow_end_ts",
    "Dst Port",
    "Protocol",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Fwd Pkt Len Max",
    "Fwd Pkt Len Min",
    "Fwd Pkt Len Mean",
    "Fwd Pkt Len Std",
    "Bwd Pkt Len Max",
    "Bwd Pkt Len Min",
    "Bwd Pkt Len Mean",
    "Bwd Pkt Len Std",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Tot",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Tot",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Fwd URG Flags",
    "Fwd Pkts/s",
    "Bwd Pkts/s",
    "Pkt Len Max",
    "Pkt Len Std",
    "Pkt Len Var",
    "FIN Flag Cnt",
    "SYN Flag Cnt",
    "RST Flag Cnt",
    "PSH Flag Cnt",
    "ACK Flag Cnt",
    "URG Flag Cnt",
    "Down/Up Ratio",
    "Pkt Size Avg",
    "Init Fwd Win Byts",
    "Init Bwd Win Byts",
    "Fwd Act Data Pkts",
    "Fwd Seg Size Min",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",
    "Label",
    "label_encoded",
]


def normalize_day_name(day: str) -> str:
    """Map project aliases onto the raw-PCAP directory names."""
    return DAY_ALIASES.get(day, day)


def profile_day_name(day: str) -> str:
    """Map raw-PCAP day names onto attack-profile day names."""
    normalized = normalize_day_name(day)
    if normalized == "Tuesday-20-02-2018":
        return "Thuesday-20-02-2018"
    return normalized


def observation_tag(observation_ratio: float) -> str:
    pct = int(round(observation_ratio * 100))
    return f"pct_{pct:03d}"


CUT_MODES = ("packet_pct", "time_abs")


def _format_seconds_tag(seconds: float) -> str:
    """Render an absolute-time cut value into a compact directory label.

    Examples: 0.5 -> 500ms, 1.0 -> 1s, 2.5 -> 2500ms, 10.0 -> 10s.
    """
    if seconds <= 0:
        raise ValueError(f"time_abs cut_value must be > 0, got {seconds}")
    if float(seconds).is_integer():
        return f"{int(seconds)}s"
    ms = int(round(seconds * 1000))
    return f"{ms}ms"


def cut_tag(cut_mode: str, cut_value: float) -> str:
    """Directory label for a given (cut_mode, cut_value) spec.

    - packet_pct + ratio  -> "pct_XXX"  (back-compat with observation_tag)
    - time_abs   + seconds -> "time_abs_{Ns|Nms}"

    Note: time_pct (truncate at X% of full-flow duration) was removed — it is an
    oracle cut that requires knowing the flow's total duration, so it cannot be
    realized online and overstates early-detection performance. Use packet-count
    or absolute-time cuts (deployable) instead.
    """
    if cut_mode == "packet_pct":
        return observation_tag(cut_value)
    if cut_mode == "time_abs":
        return f"time_abs_{_format_seconds_tag(float(cut_value))}"
    raise ValueError(f"unknown cut_mode={cut_mode!r}; expected one of {CUT_MODES}")


def packet_base_path(day: str) -> Path:
    return PACKET_BASE_DIR / f"{normalize_day_name(day)}.parquet"


def raw_flow_parquet_path(day: str, observation_ratio: float = 1.0) -> Path:
    return PARTIAL_FLOW_ROOT / observation_tag(observation_ratio) / "raw_flow" / (
        f"{normalize_day_name(day)}.parquet"
    )


def raw_flow_csv_path(day: str, observation_ratio: float = 1.0) -> Path:
    return PARTIAL_FLOW_ROOT / observation_tag(observation_ratio) / "raw_flow_csv" / (
        f"{normalize_day_name(day)}.csv"
    )


def cut_raw_flow_parquet_path(day: str, cut_mode: str, cut_value: float) -> Path:
    return PARTIAL_FLOW_ROOT / cut_tag(cut_mode, cut_value) / "raw_flow" / (
        f"{normalize_day_name(day)}.parquet"
    )


def cut_raw_flow_csv_path(day: str, cut_mode: str, cut_value: float) -> Path:
    return PARTIAL_FLOW_ROOT / cut_tag(cut_mode, cut_value) / "raw_flow_csv" / (
        f"{normalize_day_name(day)}.csv"
    )


def manifest_path(day: str) -> Path:
    return MANIFEST_DIR / f"{normalize_day_name(day)}.json"


def pcap_packet_base_path(day: str, capture_file: str) -> Path:
    return BY_PCAP_ROOT / normalize_day_name(day) / "packet_base" / f"{capture_file}.parquet"


def pcap_raw_flow_parquet_path(
    day: str,
    capture_file: str,
    observation_ratio: float = 1.0,
) -> Path:
    return (
        BY_PCAP_ROOT
        / normalize_day_name(day)
        / observation_tag(observation_ratio)
        / "raw_flow"
        / f"{capture_file}.parquet"
    )


def pcap_raw_flow_csv_path(
    day: str,
    capture_file: str,
    observation_ratio: float = 1.0,
) -> Path:
    return (
        BY_PCAP_ROOT
        / normalize_day_name(day)
        / observation_tag(observation_ratio)
        / "raw_flow_csv"
        / f"{capture_file}.csv"
    )


def pcap_manifest_path(day: str, capture_file: str) -> Path:
    return BY_PCAP_ROOT / normalize_day_name(day) / "manifests" / f"{capture_file}.json"


def validate_observation_ratio(observation_ratio: float) -> None:
    if not (0.0 < observation_ratio <= 1.0):
        raise ValueError(f"observation_ratio must be in (0, 1], got {observation_ratio}")


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"

    total_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _progress_timing(completed: int, total: int, start_time: float) -> tuple[str, str]:
    elapsed_seconds = max(time.perf_counter() - start_time, 0.0)
    if completed <= 0 or total <= 0 or elapsed_seconds <= 0:
        eta_seconds = None
    else:
        rate = completed / elapsed_seconds
        eta_seconds = (total - completed) / rate if rate > 0 else None

    return _format_duration(elapsed_seconds), _format_duration(eta_seconds)


def _progress_label(prefix: str, completed: int, total: int, start_time: float) -> str:
    elapsed_label, eta_label = _progress_timing(completed, total, start_time)
    return (
        f"[{prefix} {completed}/{total} | "
        f"elapsed={elapsed_label} eta={eta_label}]"
    )


def resolve_day_dir(day: str) -> Path:
    day_dir = RAW_ROOT / normalize_day_name(day)
    if not day_dir.exists():
        raise FileNotFoundError(f"Raw day directory not found: {day_dir}")
    return day_dir


def resolve_raw_archive(day: str) -> Path:
    day_dir = resolve_day_dir(day)
    for candidate in ("pcap.zip", "pcap.rar"):
        path = day_dir / candidate
        if path.exists():
            return path
    raise FileNotFoundError(f"No PCAP archive found in {day_dir}")


def _run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}): {result.stderr[:500]}")
    return result


def _is_capture_member(member: str) -> bool:
    """
    Return True for archive members that look like actual packet captures.

    CIC-IDS2018 host captures are mostly extensionless file names whose final
    suffix is just the last numeric IP octet (for example `.111`). We keep those,
    plus standard capture extensions, and skip obvious junk such as `.lnk`.
    """
    suffix = Path(member).suffix.lower()
    if not suffix:
        return True
    if suffix in {".pcap", ".pcapng", ".cap"}:
        return True
    if _CAPTURE_MULTIPART_RE.search(Path(member).name):
        return True
    return suffix[1:].isdigit()


def _capture_host_ip_from_name(capture_name: str) -> str | None:
    match = _CAPTURE_HOST_IP_RE.search(capture_name)
    if match is None:
        return None
    try:
        return str(ipaddress.ip_address(match.group(1)))
    except ValueError:
        return None


def _dedup_host_sort_key(host_ip: str) -> tuple[int, int, int, int]:
    return tuple(int(part) for part in host_ip.split("."))


def _canonical_capture_host_for_flow(
    flow_src_ip: str,
    flow_dst_ip: str,
    capture_host_ips: set[str],
) -> str | None:
    participating_hosts = sorted(
        {ip for ip in (flow_src_ip, flow_dst_ip) if ip in capture_host_ips},
        key=_dedup_host_sort_key,
    )
    if len(participating_hosts) <= 1:
        return None
    return participating_hosts[0]


def _deduplicate_cross_capture_flows(
    grouped_packets: pd.DataFrame,
    capture_host_ips: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Drop duplicate internal-to-internal flows seen from multiple host captures.

    The day-level packet base keeps only the capture whose host IP is the lower-IP
    endpoint among the participating captured hosts. External-facing flows are kept
    because only one captured internal host participates.
    """
    if grouped_packets.empty or "capture_file" not in grouped_packets.columns:
        return grouped_packets.copy(), {
            "dropped_flow_rows": 0,
            "dropped_packet_rows": 0,
        }

    capture_files = grouped_packets["capture_file"].dropna().unique()
    capture_file = str(capture_files[0]) if len(capture_files) else ""
    capture_host_ip = _capture_host_ip_from_name(capture_file)
    if capture_host_ip is None or capture_host_ip not in capture_host_ips:
        return grouped_packets.copy(), {
            "dropped_flow_rows": 0,
            "dropped_packet_rows": 0,
        }

    flow_first = grouped_packets.loc[
        grouped_packets["packet_index"] == 0,
        ["flow_id", "flow_src_ip", "flow_dst_ip"],
    ].copy()
    flow_first["canonical_capture_host_ip"] = [
        _canonical_capture_host_for_flow(flow_src_ip, flow_dst_ip, capture_host_ips)
        for flow_src_ip, flow_dst_ip in zip(
            flow_first["flow_src_ip"],
            flow_first["flow_dst_ip"],
            strict=False,
        )
    ]

    dropped_flow_ids = flow_first.loc[
        flow_first["canonical_capture_host_ip"].notna()
        & (flow_first["canonical_capture_host_ip"] != capture_host_ip),
        "flow_id",
    ]

    if dropped_flow_ids.empty:
        return grouped_packets.copy(), {
            "dropped_flow_rows": 0,
            "dropped_packet_rows": 0,
        }

    keep_mask = ~grouped_packets["flow_id"].isin(dropped_flow_ids.to_numpy())
    deduped_packets = grouped_packets.loc[keep_mask].copy()
    return deduped_packets, {
        "dropped_flow_rows": int(dropped_flow_ids.nunique()),
        "dropped_packet_rows": int((~keep_mask).sum()),
    }


def list_pcap_members(day: str) -> list[str]:
    archive_path = resolve_raw_archive(day)
    if archive_path.suffix == ".zip":
        result = _run_command(["unzip", "-Z1", str(archive_path)])
    elif archive_path.suffix == ".rar":
        result = _run_command(["unrar", "lb", str(archive_path)])
    else:
        raise ValueError(f"Unsupported archive format: {archive_path}")

    members = []
    for line in result.stdout.splitlines():
        member = line.strip()
        if not member or member.endswith("/"):
            continue
        if not member.startswith("pcap/"):
            continue
        if not _is_capture_member(member):
            continue
        members.append(member)
    return sorted(members)


def ensure_day_pcaps_extracted(day: str, force: bool = False) -> list[Path]:
    """
    Extract all PCAP members for a day into data/CIC-IDS2018/pcap_extracted/<day>/.

    The extraction cache is outside original/raw/ so the raw archives stay untouched.
    """
    day_name = normalize_day_name(day)
    archive_path = resolve_raw_archive(day_name)
    members = list_pcap_members(day_name)
    extracted_dir = EXTRACTED_ROOT / day_name
    extracted_dir.mkdir(parents=True, exist_ok=True)

    expected_paths = [extracted_dir / Path(member).name for member in members]
    missing = [path for path in expected_paths if not path.exists()]

    if force or missing:
        print(f"Extracting {len(members)} PCAPs for {day_name} from {archive_path.name} ...")
        if archive_path.suffix == ".zip":
            overwrite_flag = "-o" if force else "-n"
            _run_command([
                "unzip",
                "-qq",
                "-j",
                overwrite_flag,
                str(archive_path),
                "pcap/*",
                "-d",
                str(extracted_dir),
            ])
        elif archive_path.suffix == ".rar":
            overwrite_flag = "-o+" if force else "-o-"
            _run_command([
                "unrar",
                "e",
                "-idq",
                overwrite_flag,
                str(archive_path),
                "pcap/*",
                str(extracted_dir),
            ])

    still_missing = [str(path) for path in expected_paths if not path.exists()]
    if still_missing:
        raise FileNotFoundError(
            f"Missing extracted PCAPs for {day_name}: {still_missing[:5]}"
            + (" ..." if len(still_missing) > 5 else "")
        )
    return expected_paths


def ensure_capture_pcap_extracted(day: str, capture_file: str, force: bool = False) -> Path:
    """
    Extract one PCAP member for a day into the extracted cache and return its path.

    This is the direct single-PCAP entry point used when we want to build artifacts
    from one capture file instead of processing the whole day.
    """
    day_name = normalize_day_name(day)
    archive_path = resolve_raw_archive(day_name)
    member_name = f"pcap/{capture_file}"
    members = list_pcap_members(day_name)
    if member_name not in members:
        raise FileNotFoundError(f"PCAP member not found for {day_name}: {capture_file}")

    extracted_dir = EXTRACTED_ROOT / day_name
    extracted_dir.mkdir(parents=True, exist_ok=True)
    output_path = extracted_dir / capture_file

    if output_path.exists() and not force:
        return output_path

    print(f"Extracting {capture_file} for {day_name} from {archive_path.name} ...")
    if archive_path.suffix == ".zip":
        overwrite_flag = "-o" if force else "-n"
        _run_command([
            "unzip",
            "-qq",
            "-j",
            overwrite_flag,
            str(archive_path),
            member_name,
            "-d",
            str(extracted_dir),
        ])
    elif archive_path.suffix == ".rar":
        overwrite_flag = "-o+" if force else "-o-"
        _run_command([
            "unrar",
            "e",
            "-idq",
            overwrite_flag,
            str(archive_path),
            member_name,
            str(extracted_dir),
        ])

    if not output_path.exists():
        raise FileNotFoundError(f"Extracted PCAP not found: {output_path}")
    return output_path


def _resolve_max_workers(max_workers: int | None, n_pcaps: int) -> int:
    if n_pcaps <= 0:
        return 1

    cpu_count = os.cpu_count() or 1
    if max_workers is None:
        workers = DEFAULT_MAX_WORKERS
    else:
        workers = max_workers

    return max(1, min(int(workers), cpu_count, n_pcaps))


def _pcap_temp_output_path(tmp_dir: Path, pcap_path: Path) -> Path:
    return tmp_dir / f"{pcap_path.name}.parquet"


def _process_single_pcap_to_temp(
    pcap_path_str: str,
    active_timeout: float,
    tmp_dir_str: str,
) -> dict:
    """
    Worker-side extraction path for one PCAP.

    The grouped packet rows are written to a temporary Parquet file so the parent
    process does not need to receive large DataFrames over IPC.
    """
    pcap_path = Path(pcap_path_str)
    tmp_dir = Path(tmp_dir_str)
    temp_output_path = _pcap_temp_output_path(tmp_dir, pcap_path)

    packets = extract_packets(pcap_path, source_id=pcap_path.name)
    if packets.empty:
        return {
            "capture_file": pcap_path.name,
            "temp_parquet_path": None,
            "packet_rows": 0,
            "flow_rows": 0,
        }

    grouped_packets = group_into_flows(packets, active_timeout=active_timeout)
    grouped_packets["flow_total_packets"] = grouped_packets.groupby("flow_id")["packet_index"].transform("max") + 1
    grouped_packets.to_parquet(temp_output_path, engine="pyarrow", compression="snappy", index=False)

    return {
        "capture_file": pcap_path.name,
        "temp_parquet_path": str(temp_output_path),
        "packet_rows": int(len(grouped_packets)),
        "flow_rows": int(grouped_packets["flow_id"].nunique()),
    }


def _extract_day_pcaps_to_temp(
    day_name: str,
    extracted_pcaps: list[Path],
    max_workers: int | None,
) -> tuple[dict[str, dict], Path, int]:
    tmp_dir = PARTIAL_FLOW_ROOT / "_tmp" / day_name
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    worker_count = _resolve_max_workers(max_workers, len(extracted_pcaps))
    results_by_capture: dict[str, dict] = {}
    extract_start = time.perf_counter()

    if worker_count == 1:
        print(f"Processing {len(extracted_pcaps)} PCAPs for {day_name} with 1 worker ...")
        for idx, pcap_path in enumerate(extracted_pcaps, start=1):
            results_by_capture[pcap_path.name] = _process_single_pcap_to_temp(
                str(pcap_path),
                ACTIVITY_TIMEOUT_SECONDS,
                str(tmp_dir),
            )
            print(
                f"{_progress_label('extract', idx, len(extracted_pcaps), extract_start)} "
                f"{day_name}: {pcap_path.name}"
            )
        return results_by_capture, tmp_dir, worker_count

    print(
        f"Processing {len(extracted_pcaps)} PCAPs for {day_name} "
        f"with {worker_count} workers ..."
    )
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_pcap = {
            executor.submit(
                _process_single_pcap_to_temp,
                str(pcap_path),
                ACTIVITY_TIMEOUT_SECONDS,
                str(tmp_dir),
            ): pcap_path
            for pcap_path in extracted_pcaps
        }

        completed = 0
        for future in as_completed(future_to_pcap):
            pcap_path = future_to_pcap[future]
            completed += 1
            results_by_capture[pcap_path.name] = future.result()
            print(
                f"{_progress_label('extract', completed, len(extracted_pcaps), extract_start)} "
                f"{day_name}: {pcap_path.name}"
            )

    return results_by_capture, tmp_dir, worker_count


def _append_dataframe(writer: pq.ParquetWriter | None, output_path: Path, df: pd.DataFrame) -> pq.ParquetWriter:
    table = pa.Table.from_pandas(df, preserve_index=False)
    if writer is None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = pq.ParquetWriter(output_path, table.schema, compression="snappy")
    writer.write_table(table)
    return writer


def _flow_duration_us(flow_start_ts: pd.Series, flow_end_ts: pd.Series) -> pd.Series:
    return ((flow_end_ts - flow_start_ts).clip(lower=0.0) * 1e6).astype("float64")


def _safe_rate(numerator: pd.Series, duration_us: pd.Series) -> pd.Series:
    numerator = numerator.astype("float64")
    duration_us = duration_us.astype("float64")
    rate = pd.Series(np.zeros(len(numerator), dtype="float64"), index=numerator.index)
    positive_mask = duration_us > 0
    rate.loc[positive_mask] = numerator.loc[positive_mask] * 1e6 / duration_us.loc[positive_mask]
    zero_duration_nonzero_num = (~positive_mask) & (numerator > 0)
    rate.loc[zero_duration_nonzero_num] = np.inf
    return rate


def _empty_iat_stats(flow_index: pd.Index) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "mean": np.zeros(len(flow_index), dtype="float64"),
            "std": np.zeros(len(flow_index), dtype="float64"),
            "max": np.zeros(len(flow_index), dtype="float64"),
            "min": np.zeros(len(flow_index), dtype="float64"),
            "sum": np.zeros(len(flow_index), dtype="float64"),
        },
        index=flow_index,
    )


def truncate_packet_base(packet_df: pd.DataFrame, observation_ratio: float) -> pd.DataFrame:
    """Retain only the first ceil(N * observation_ratio) packets of each flow."""
    validate_observation_ratio(observation_ratio)
    if observation_ratio >= 1.0:
        return packet_df.copy()

    df = packet_df.copy()
    packets_per_flow = df.groupby("flow_id")["packet_index"].transform("max") + 1
    keep_packets = np.maximum(
        np.ceil(packets_per_flow.to_numpy(dtype="float64") * observation_ratio).astype("int32"),
        1,
    )
    return df.loc[df["packet_index"].to_numpy() < keep_packets].copy()


def _enforce_first_packet_per_flow(df: pd.DataFrame, keep_mask: np.ndarray) -> np.ndarray:
    """Guarantee at least one retained packet per flow: keep packet_index==0 when empty.

    Mirrors the ``min=1`` behaviour of ``truncate_packet_base`` so time-based cuts
    stay compatible with downstream aggregation (which requires ≥1 pkt per flow_id).
    """
    first_pkt_mask = df["packet_index"].to_numpy() == 0
    return keep_mask | first_pkt_mask


def truncate_packet_base_by_time_abs(
    packet_df: pd.DataFrame, seconds: float
) -> pd.DataFrame:
    """Retain packets whose timestamp is within ``seconds`` of the flow's first packet."""
    if seconds <= 0:
        raise ValueError(f"seconds must be > 0, got {seconds}")
    if packet_df.empty:
        return packet_df.copy()

    df = packet_df.copy()
    flow_start = df.groupby("flow_id")["timestamp"].transform("min").to_numpy()
    elapsed = df["timestamp"].to_numpy() - flow_start
    keep_mask = elapsed <= float(seconds)
    keep_mask = _enforce_first_packet_per_flow(df, keep_mask)
    return df.loc[keep_mask].copy()


def truncate_packet_base_by_cut(
    packet_df: pd.DataFrame, cut_mode: str, cut_value: float
) -> pd.DataFrame:
    """Dispatch to the right truncation function based on ``cut_mode``.

    Note: ``time_pct`` (truncate at X% of full-flow duration) was removed as an
    oracle cut — the threshold depends on the flow's total duration, which is
    unknown online. Only deployable cuts remain (packet-count and absolute-time).
    """
    if cut_mode == "packet_pct":
        return truncate_packet_base(packet_df, cut_value)
    if cut_mode == "time_abs":
        return truncate_packet_base_by_time_abs(packet_df, cut_value)
    raise ValueError(f"unknown cut_mode={cut_mode!r}; expected one of {CUT_MODES}")


def aggregate_flow_features(
    packet_df: pd.DataFrame,
    day: str,
    active_idle_threshold_seconds: float = ACTIVE_IDLE_THRESHOLD_SECONDS,
) -> pd.DataFrame:
    """
    Aggregate packet-base rows into a CICFlow-style raw feature subset.

    Flow grouping is intentionally per capture_file. The day-level outputs concatenate
    these per-capture flows into a single file without merging flows across hosts.
    """
    if packet_df.empty:
        return pd.DataFrame(columns=RAW_FEATURE_COLUMNS)

    df = packet_df.sort_values(["flow_id", "packet_index"]).reset_index(drop=True).copy()
    flow_groups = df.groupby("flow_id", sort=False)
    flow_index = flow_groups.size().index

    first_packets = flow_groups.first()
    last_timestamps = flow_groups["timestamp"].max()
    total_packets = flow_groups.size().reindex(flow_index).fillna(0).astype("int32")
    flow_duration_us = _flow_duration_us(first_packets["timestamp"], last_timestamps)

    fwd = df.loc[df["direction"] == 1].copy()
    bwd = df.loc[df["direction"] == 0].copy()

    fwd_groups = fwd.groupby("flow_id", sort=False) if not fwd.empty else None
    bwd_groups = bwd.groupby("flow_id", sort=False) if not bwd.empty else None

    tot_fwd_pkts = (
        fwd_groups.size().reindex(flow_index).fillna(0).astype("int32")
        if fwd_groups is not None
        else pd.Series(0, index=flow_index, dtype="int32")
    )
    tot_bwd_pkts = (
        bwd_groups.size().reindex(flow_index).fillna(0).astype("int32")
        if bwd_groups is not None
        else pd.Series(0, index=flow_index, dtype="int32")
    )

    if fwd_groups is not None:
        fwd_payload = fwd_groups["payload_len"]
        totlen_fwd = fwd_payload.sum().reindex(flow_index).fillna(0).astype("float64")
        fwd_pkt_len_max = fwd_payload.max().reindex(flow_index).fillna(0).astype("float64")
        fwd_pkt_len_min = fwd_payload.min().reindex(flow_index).fillna(0).astype("float64")
        fwd_pkt_len_mean = fwd_payload.mean().reindex(flow_index).fillna(0).astype("float64")
        fwd_pkt_len_std = fwd_payload.std(ddof=0).reindex(flow_index).fillna(0).astype("float64")
        fwd_act_data_pkts = (
            (fwd["payload_len"] > 0).astype("int32").groupby(fwd["flow_id"]).sum()
            .reindex(flow_index).fillna(0).astype("int32")
        )
        fwd_seg_size_min = (
            fwd_groups["ip_hdr_len"].min().reindex(flow_index).fillna(0).astype("float64")
        )
        fwd_psh_flags = (
            ((fwd["tcp_flags"] & 0x08) != 0).astype("int32").groupby(fwd["flow_id"]).max()
            .reindex(flow_index).fillna(0).astype("int32")
        )
        fwd_urg_flags = (
            ((fwd["tcp_flags"] & 0x20) != 0).astype("int32").groupby(fwd["flow_id"]).max()
            .reindex(flow_index).fillna(0).astype("int32")
        )
        init_fwd_win = (
            fwd_groups["tcp_win"].first().reindex(flow_index).fillna(0).astype("int32")
        )
    else:
        totlen_fwd = pd.Series(0.0, index=flow_index)
        fwd_pkt_len_max = pd.Series(0.0, index=flow_index)
        fwd_pkt_len_min = pd.Series(0.0, index=flow_index)
        fwd_pkt_len_mean = pd.Series(0.0, index=flow_index)
        fwd_pkt_len_std = pd.Series(0.0, index=flow_index)
        fwd_act_data_pkts = pd.Series(0, index=flow_index, dtype="int32")
        fwd_seg_size_min = pd.Series(0.0, index=flow_index)
        fwd_psh_flags = pd.Series(0, index=flow_index, dtype="int32")
        fwd_urg_flags = pd.Series(0, index=flow_index, dtype="int32")
        init_fwd_win = pd.Series(0, index=flow_index, dtype="int32")

    if bwd_groups is not None:
        bwd_payload = bwd_groups["payload_len"]
        totlen_bwd = bwd_payload.sum().reindex(flow_index).fillna(0).astype("float64")
        bwd_pkt_len_max = bwd_payload.max().reindex(flow_index).fillna(0).astype("float64")
        bwd_pkt_len_min = bwd_payload.min().reindex(flow_index).fillna(0).astype("float64")
        bwd_pkt_len_mean = bwd_payload.mean().reindex(flow_index).fillna(0).astype("float64")
        bwd_pkt_len_std = bwd_payload.std(ddof=0).reindex(flow_index).fillna(0).astype("float64")
        init_bwd_win = (
            bwd_groups["tcp_win"].first().reindex(flow_index).fillna(0).astype("int32")
        )
    else:
        totlen_bwd = pd.Series(0.0, index=flow_index)
        bwd_pkt_len_max = pd.Series(0.0, index=flow_index)
        bwd_pkt_len_min = pd.Series(0.0, index=flow_index)
        bwd_pkt_len_mean = pd.Series(0.0, index=flow_index)
        bwd_pkt_len_std = pd.Series(0.0, index=flow_index)
        init_bwd_win = pd.Series(0, index=flow_index, dtype="int32")

    all_payload = flow_groups["payload_len"]
    pkt_len_max = all_payload.max().reindex(flow_index).fillna(0).astype("float64")
    pkt_len_std = all_payload.std(ddof=0).reindex(flow_index).fillna(0).astype("float64")
    pkt_len_var = all_payload.var(ddof=0).reindex(flow_index).fillna(0).astype("float64")
    pkt_size_avg = all_payload.mean().reindex(flow_index).fillna(0).astype("float64")

    flow_iat_us = df.groupby("flow_id")["timestamp"].diff() * 1e6
    flow_iat_mean = flow_iat_us.groupby(df["flow_id"]).mean().reindex(flow_index).fillna(0).astype("float64")
    flow_iat_std = flow_iat_us.groupby(df["flow_id"]).std(ddof=0).reindex(flow_index).fillna(0).astype("float64")
    flow_iat_max = flow_iat_us.groupby(df["flow_id"]).max().reindex(flow_index).fillna(0).astype("float64")
    flow_iat_min = flow_iat_us.groupby(df["flow_id"]).min().reindex(flow_index).fillna(0).astype("float64")

    # Forward IAT features
    if not fwd.empty:
        fwd_iat_us = fwd.groupby("flow_id")["timestamp"].diff() * 1e6
        fwd_iat_stats = pd.DataFrame(
            {
                "mean": fwd_iat_us.groupby(fwd["flow_id"]).mean(),
                "std": fwd_iat_us.groupby(fwd["flow_id"]).std(ddof=0),
                "max": fwd_iat_us.groupby(fwd["flow_id"]).max(),
                "min": fwd_iat_us.groupby(fwd["flow_id"]).min(),
                "sum": fwd_iat_us.groupby(fwd["flow_id"]).sum(),
            }
        ).reindex(flow_index).fillna(0).astype("float64")
    else:
        fwd_iat_stats = _empty_iat_stats(flow_index)

    if not bwd.empty:
        bwd_iat_us = bwd.groupby("flow_id")["timestamp"].diff() * 1e6
        bwd_iat_stats = pd.DataFrame(
            {
                "mean": bwd_iat_us.groupby(bwd["flow_id"]).mean(),
                "std": bwd_iat_us.groupby(bwd["flow_id"]).std(ddof=0),
                "max": bwd_iat_us.groupby(bwd["flow_id"]).max(),
                "min": bwd_iat_us.groupby(bwd["flow_id"]).min(),
                "sum": bwd_iat_us.groupby(bwd["flow_id"]).sum(),
            }
        ).reindex(flow_index).fillna(0).astype("float64")
    else:
        bwd_iat_stats = _empty_iat_stats(flow_index)

    df["flow_iat_us"] = flow_iat_us.fillna(0.0)
    split_mask = df["flow_iat_us"] > (active_idle_threshold_seconds * 1e6)
    df["active_segment"] = split_mask.groupby(df["flow_id"]).cumsum().astype("int32")
    active_segments = (
        df.groupby(["flow_id", "active_segment"], sort=False)["timestamp"]
        .agg(["min", "max"])
    )
    active_durations = ((active_segments["max"] - active_segments["min"]).clip(lower=0.0) * 1e6)
    active_by_flow = active_durations.groupby(level=0)
    active_mean = active_by_flow.mean().reindex(flow_index).fillna(0).astype("float64")
    active_std = active_by_flow.std(ddof=0).reindex(flow_index).fillna(0).astype("float64")
    active_max = active_by_flow.max().reindex(flow_index).fillna(0).astype("float64")
    active_min = active_by_flow.min().reindex(flow_index).fillna(0).astype("float64")
    idle_by_flow = df.loc[split_mask].groupby("flow_id")["flow_iat_us"]
    idle_mean = idle_by_flow.mean().reindex(flow_index).fillna(0).astype("float64")
    idle_std = idle_by_flow.std(ddof=0).reindex(flow_index).fillna(0).astype("float64")
    idle_max = idle_by_flow.max().reindex(flow_index).fillna(0).astype("float64")
    idle_min = idle_by_flow.min().reindex(flow_index).fillna(0).astype("float64")

    fin_flag_cnt = (
        ((df["tcp_flags"] & 0x01) != 0).astype("int32").groupby(df["flow_id"]).sum()
        .reindex(flow_index).fillna(0).astype("int32")
    )
    syn_flag_cnt = (
        ((df["tcp_flags"] & 0x02) != 0).astype("int32").groupby(df["flow_id"]).sum()
        .reindex(flow_index).fillna(0).astype("int32")
    )
    rst_flag_cnt = (
        ((df["tcp_flags"] & 0x04) != 0).astype("int32").groupby(df["flow_id"]).sum()
        .reindex(flow_index).fillna(0).astype("int32")
    )
    psh_flag_cnt = (
        ((df["tcp_flags"] & 0x08) != 0).astype("int32").groupby(df["flow_id"]).sum()
        .reindex(flow_index).fillna(0).astype("int32")
    )
    ack_flag_cnt = (
        ((df["tcp_flags"] & 0x10) != 0).astype("int32").groupby(df["flow_id"]).sum()
        .reindex(flow_index).fillna(0).astype("int32")
    )
    urg_flag_cnt = (
        ((df["tcp_flags"] & 0x20) != 0).astype("int32").groupby(df["flow_id"]).sum()
        .reindex(flow_index).fillna(0).astype("int32")
    )

    total_bytes = totlen_fwd + totlen_bwd
    flow_byts_per_s = _safe_rate(total_bytes, flow_duration_us)
    flow_pkts_per_s = _safe_rate(total_packets, flow_duration_us)
    fwd_pkts_per_s = _safe_rate(tot_fwd_pkts, flow_duration_us)
    bwd_pkts_per_s = _safe_rate(tot_bwd_pkts, flow_duration_us)

    down_up_ratio = pd.Series(np.zeros(len(flow_index), dtype="float64"), index=flow_index)
    valid_ratio = tot_fwd_pkts > 0
    down_up_ratio.loc[valid_ratio] = (
        tot_bwd_pkts.loc[valid_ratio].astype("float64")
        / tot_fwd_pkts.loc[valid_ratio].astype("float64")
    )

    protocol = first_packets["protocol"].reindex(flow_index).fillna(0).astype("int32")
    init_fwd_win = init_fwd_win.where(protocol == 6, -1).astype("int32")
    init_bwd_win = init_bwd_win.where(protocol == 6, -1).astype("int32")

    flow_df = pd.DataFrame(index=flow_index)
    flow_df["day"] = normalize_day_name(day)
    flow_df["capture_file"] = first_packets["capture_file"].reindex(flow_index)
    flow_df["flow_id"] = flow_index.astype("int64")
    flow_df["flow_src_ip"] = first_packets["flow_src_ip"].reindex(flow_index)
    flow_df["flow_dst_ip"] = first_packets["flow_dst_ip"].reindex(flow_index)
    flow_df["flow_src_port"] = first_packets["flow_src_port"].reindex(flow_index).fillna(0).astype("int32")
    flow_df["flow_dst_port"] = first_packets["flow_dst_port"].reindex(flow_index).fillna(0).astype("int32")
    flow_df["flow_start_ts"] = first_packets["timestamp"].reindex(flow_index).astype("float64")
    flow_df["flow_end_ts"] = last_timestamps.reindex(flow_index).astype("float64")
    flow_df["Dst Port"] = first_packets["flow_dst_port"].reindex(flow_index).fillna(0).astype("int32")
    flow_df["Protocol"] = protocol
    flow_df["Flow Duration"] = flow_duration_us
    flow_df["Tot Fwd Pkts"] = tot_fwd_pkts
    flow_df["Tot Bwd Pkts"] = tot_bwd_pkts
    flow_df["TotLen Fwd Pkts"] = totlen_fwd
    flow_df["TotLen Bwd Pkts"] = totlen_bwd
    flow_df["Fwd Pkt Len Max"] = fwd_pkt_len_max
    flow_df["Fwd Pkt Len Min"] = fwd_pkt_len_min
    flow_df["Fwd Pkt Len Mean"] = fwd_pkt_len_mean
    flow_df["Fwd Pkt Len Std"] = fwd_pkt_len_std
    flow_df["Bwd Pkt Len Max"] = bwd_pkt_len_max
    flow_df["Bwd Pkt Len Min"] = bwd_pkt_len_min
    flow_df["Bwd Pkt Len Mean"] = bwd_pkt_len_mean
    flow_df["Bwd Pkt Len Std"] = bwd_pkt_len_std
    flow_df["Flow Byts/s"] = flow_byts_per_s
    flow_df["Flow Pkts/s"] = flow_pkts_per_s
    flow_df["Flow IAT Mean"] = flow_iat_mean
    flow_df["Flow IAT Std"] = flow_iat_std
    flow_df["Flow IAT Max"] = flow_iat_max
    flow_df["Flow IAT Min"] = flow_iat_min
    flow_df["Fwd IAT Tot"] = fwd_iat_stats["sum"]
    flow_df["Fwd IAT Mean"] = fwd_iat_stats["mean"]
    flow_df["Fwd IAT Std"] = fwd_iat_stats["std"]
    flow_df["Fwd IAT Max"] = fwd_iat_stats["max"]
    flow_df["Fwd IAT Min"] = fwd_iat_stats["min"]
    flow_df["Bwd IAT Tot"] = bwd_iat_stats["sum"]
    flow_df["Bwd IAT Mean"] = bwd_iat_stats["mean"]
    flow_df["Bwd IAT Std"] = bwd_iat_stats["std"]
    flow_df["Bwd IAT Max"] = bwd_iat_stats["max"]
    flow_df["Bwd IAT Min"] = bwd_iat_stats["min"]
    flow_df["Fwd PSH Flags"] = fwd_psh_flags
    flow_df["Fwd URG Flags"] = fwd_urg_flags
    flow_df["Fwd Pkts/s"] = fwd_pkts_per_s
    flow_df["Bwd Pkts/s"] = bwd_pkts_per_s
    flow_df["Pkt Len Max"] = pkt_len_max
    flow_df["Pkt Len Std"] = pkt_len_std
    flow_df["Pkt Len Var"] = pkt_len_var
    flow_df["FIN Flag Cnt"] = fin_flag_cnt
    flow_df["SYN Flag Cnt"] = syn_flag_cnt
    flow_df["RST Flag Cnt"] = rst_flag_cnt
    flow_df["PSH Flag Cnt"] = psh_flag_cnt
    flow_df["ACK Flag Cnt"] = ack_flag_cnt
    flow_df["URG Flag Cnt"] = urg_flag_cnt
    flow_df["Down/Up Ratio"] = down_up_ratio
    flow_df["Pkt Size Avg"] = pkt_size_avg
    flow_df["Init Fwd Win Byts"] = init_fwd_win
    flow_df["Init Bwd Win Byts"] = init_bwd_win
    flow_df["Fwd Act Data Pkts"] = fwd_act_data_pkts
    flow_df["Fwd Seg Size Min"] = fwd_seg_size_min
    flow_df["Active Mean"] = active_mean
    flow_df["Active Std"] = active_std
    flow_df["Active Max"] = active_max
    flow_df["Active Min"] = active_min
    flow_df["Idle Mean"] = idle_mean
    flow_df["Idle Std"] = idle_std
    flow_df["Idle Max"] = idle_max
    flow_df["Idle Min"] = idle_min
    flow_df["Label"] = pd.Series(pd.NA, index=flow_index, dtype="string")
    flow_df["label_encoded"] = UNLABELED_LABEL_ENCODED

    return flow_df[RAW_FEATURE_COLUMNS].reset_index(drop=True)


def _known_day_issues(day: str, profiles: dict) -> list[str]:
    profile_day = profile_day_name(day)
    issues = []

    excluded_reason = profiles.get("days_excluded", {}).get(profile_day)
    if excluded_reason:
        issues.append(excluded_reason)

    for attack in profiles.get("attacks", []):
        if attack["day"] != profile_day:
            continue
        note = attack.get("known_issues")
        if note:
            issues.append(f"{attack['label']}: {note}")
    return issues


def _read_manifest(day: str) -> dict:
    path = manifest_path(day)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _write_manifest(day: str, payload: dict) -> None:
    path = manifest_path(day)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _write_pcap_manifest(day: str, capture_file: str, payload: dict) -> None:
    path = pcap_manifest_path(day, capture_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=False)


def _base_manifest(day: str, profiles: dict) -> dict:
    day_name = normalize_day_name(day)
    raw_archive = resolve_raw_archive(day_name)
    profile_day = profile_day_name(day_name)
    labels_available = day_has_attack_profiles(profile_day, profiles)
    return {
        "schema_version": SCHEMA_VERSION,
        "day": day_name,
        "profile_day": profile_day,
        "source": {
            "raw_archive_path": str(raw_archive),
            "archive_format": raw_archive.suffix.lstrip("."),
            "pcap_members_processed": list_pcap_members(day_name),
        },
        "build": {
            "active_timeout_seconds": ACTIVITY_TIMEOUT_SECONDS,
            "active_idle_threshold_seconds": ACTIVE_IDLE_THRESHOLD_SECONDS,
            "flow_grouping_scope": "per_capture_file",
            "cross_capture_dedup_applied": True,
            "cross_capture_dedup_method": "canonical_capture_host_ip",
        },
        "labeling": {
            "label_source": (
                str(_DATA_ROOT / "attack_profiles.json") if labels_available else None
            ),
            "labels_available": labels_available,
            "training_ready": labels_available,
            "known_issues": _known_day_issues(day_name, profiles),
            "unlabeled_reason": (
                None if labels_available else "No attack profiles configured for this day."
            ),
        },
        "artifacts": {},
        "stats": {},
    }


def _pcap_manifest(day: str, capture_file: str, source_pcap_path: Path, profiles: dict) -> dict:
    day_name = normalize_day_name(day)
    profile_day = profile_day_name(day_name)
    labels_available = day_has_attack_profiles(profile_day, profiles)
    return {
        "schema_version": SCHEMA_VERSION,
        "day": day_name,
        "profile_day": profile_day,
        "capture_file": capture_file,
        "source": {
            "pcap_path": str(source_pcap_path),
        },
        "build": {
            "active_timeout_seconds": ACTIVITY_TIMEOUT_SECONDS,
            "active_idle_threshold_seconds": ACTIVE_IDLE_THRESHOLD_SECONDS,
            "flow_grouping_scope": "single_capture_file",
            "cross_capture_dedup_applied": False,
        },
        "labeling": {
            "label_source": (
                str(_DATA_ROOT / "attack_profiles.json") if labels_available else None
            ),
            "labels_available": labels_available,
            "training_ready": labels_available,
            "known_issues": _known_day_issues(day_name, profiles),
            "unlabeled_reason": (
                None if labels_available else "No attack profiles configured for this day."
            ),
        },
        "artifacts": {},
        "stats": {},
    }


def build_day_packet_base(
    day: str,
    force: bool = False,
    max_workers: int | None = None,
) -> dict:
    """
    Build the canonical packet-base artifact for a single day.

    The packet-base Parquet stores all grouped packet rows for the day while keeping
    flow IDs unique across capture files by offsetting per-PCAP flow IDs.
    """
    day_name = normalize_day_name(day)
    output_path = packet_base_path(day_name)
    tmp_path = output_path.with_suffix(".tmp.parquet")

    profiles = load_attack_profiles()
    extracted_pcaps = ensure_day_pcaps_extracted(day_name, force=False)

    if output_path.exists() and not force:
        manifest = _read_manifest(day_name)
        if manifest:
            return manifest

    if tmp_path.exists():
        tmp_path.unlink()

    writer = None
    total_packet_rows = 0
    total_flow_rows = 0
    current_flow_offset = 0
    dedup_dropped_packet_rows = 0
    dedup_dropped_flow_rows = 0
    results_by_capture = {}
    tmp_dir = None
    day_start = time.perf_counter()
    worker_count = 1
    capture_host_ips = {
        host_ip
        for pcap_path in extracted_pcaps
        if (host_ip := _capture_host_ip_from_name(pcap_path.name)) is not None
    }

    try:
        results_by_capture, tmp_dir, worker_count = _extract_day_pcaps_to_temp(
            day_name=day_name,
            extracted_pcaps=extracted_pcaps,
            max_workers=max_workers,
        )

        print(f"Merging grouped packet tables into {tmp_path.name} ...")
        merge_start = time.perf_counter()
        for idx, pcap_path in enumerate(extracted_pcaps, start=1):
            result = results_by_capture[pcap_path.name]
            temp_parquet_path = result["temp_parquet_path"]
            if temp_parquet_path is None:
                continue

            grouped_packets = pd.read_parquet(temp_parquet_path, engine="pyarrow")
            grouped_packets, dedup_stats = _deduplicate_cross_capture_flows(
                grouped_packets,
                capture_host_ips=capture_host_ips,
            )
            dedup_dropped_packet_rows += dedup_stats["dropped_packet_rows"]
            dedup_dropped_flow_rows += dedup_stats["dropped_flow_rows"]
            if grouped_packets.empty:
                continue

            grouped_packets["flow_id"] = grouped_packets["flow_id"].astype("int64") + current_flow_offset
            current_flow_offset = int(grouped_packets["flow_id"].max()) + 1

            writer = _append_dataframe(writer, tmp_path, grouped_packets)
            total_packet_rows += int(len(grouped_packets))
            total_flow_rows += int(grouped_packets["flow_id"].nunique())

            if idx == 1 or idx == len(extracted_pcaps) or idx % 25 == 0:
                print(f"{_progress_label('merge', idx, len(extracted_pcaps), merge_start)} {day_name}")
    finally:
        if writer is not None:
            writer.close()

    if writer is None:
        raise RuntimeError(f"No IP packets extracted for {day_name}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.replace(output_path)

    manifest = _base_manifest(day_name, profiles)
    manifest["artifacts"]["packet_base_parquet"] = str(output_path)
    manifest["build"]["max_workers"] = int(worker_count)
    manifest["stats"]["packet_rows"] = int(total_packet_rows)
    manifest["stats"]["packet_base_flow_rows"] = int(total_flow_rows)
    manifest["stats"]["cross_capture_dedup_dropped_packet_rows"] = int(dedup_dropped_packet_rows)
    manifest["stats"]["cross_capture_dedup_dropped_flow_rows"] = int(dedup_dropped_flow_rows)
    _write_manifest(day_name, manifest)

    if tmp_dir is not None and tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    print(
        f"Completed packet base for {day_name} "
        f"in {_format_duration(time.perf_counter() - day_start)} "
        f"(packet_rows={total_packet_rows}, flows={total_flow_rows}, "
        f"dedup_dropped_flows={dedup_dropped_flow_rows})"
    )
    return manifest


def build_capture_partial_flow(
    day: str,
    capture_file: str,
    observation_ratio: float = 1.0,
    force: bool = False,
    write_csv: bool = False,
) -> dict:
    """
    Build packet-base and raw flow-view artifacts directly from one extracted PCAP.

    This bypasses the whole-day merge path and is intended for quick inspection or
    targeted PCAP pipeline debugging on a single source capture.
    """
    validate_observation_ratio(observation_ratio)
    day_name = normalize_day_name(day)
    pcap_path = ensure_capture_pcap_extracted(day_name, capture_file, force=False)

    packet_path = pcap_packet_base_path(day_name, capture_file)
    raw_parquet_path = pcap_raw_flow_parquet_path(day_name, capture_file, observation_ratio)
    raw_csv_path = pcap_raw_flow_csv_path(day_name, capture_file, observation_ratio)

    if packet_path.exists() and raw_parquet_path.exists() and not force:
        manifest_path_ = pcap_manifest_path(day_name, capture_file)
        if manifest_path_.exists():
            with open(manifest_path_) as f:
                return json.load(f)

    packets = extract_packets(pcap_path, source_id=capture_file)
    if packets.empty:
        raise RuntimeError(f"No IP packets extracted for {pcap_path}")

    packet_df = group_into_flows(packets, active_timeout=ACTIVITY_TIMEOUT_SECONDS)
    packet_df["flow_total_packets"] = packet_df.groupby("flow_id")["packet_index"].transform("max") + 1
    packet_path.parent.mkdir(parents=True, exist_ok=True)
    packet_df.to_parquet(packet_path, engine="pyarrow", compression="snappy", index=False)

    truncated_packet_df = truncate_packet_base(packet_df, observation_ratio)
    flow_df = aggregate_flow_features(truncated_packet_df, day=day_name)

    profiles = load_attack_profiles()
    profile_day = profile_day_name(day_name)
    if day_has_attack_profiles(profile_day, profiles):
        label_df = label_flows(truncated_packet_df, day=profile_day, profiles=profiles).rename(
            columns={"label": "Label"}
        )
        label_df["Label"] = label_df["Label"].astype("string")
    else:
        label_df = pd.DataFrame(
            {
                "flow_id": flow_df["flow_id"].astype("int64"),
                "Label": pd.Series("Benign", index=range(len(flow_df)), dtype="string"),
                "label_encoded": 0,
            }
        )

    flow_df = flow_df.drop(columns=["Label", "label_encoded"]).merge(
        label_df,
        on="flow_id",
        how="left",
    )
    flow_df["Label"] = flow_df["Label"].astype("string")
    flow_df["label_encoded"] = flow_df["label_encoded"].fillna(UNLABELED_LABEL_ENCODED).astype("int32")
    flow_df = flow_df[RAW_FEATURE_COLUMNS]

    raw_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    flow_df.to_parquet(raw_parquet_path, engine="pyarrow", compression="snappy", index=False)

    raw_csv_written = None
    if write_csv:
        raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        flow_df.to_csv(raw_csv_path, index=False)
        raw_csv_written = str(raw_csv_path)

    manifest = _pcap_manifest(day_name, capture_file, pcap_path, profiles)
    manifest["build"]["observation_ratio"] = float(observation_ratio)
    manifest["artifacts"]["packet_base_parquet"] = str(packet_path)
    manifest["artifacts"][f"{observation_tag(observation_ratio)}_raw_flow_parquet"] = str(raw_parquet_path)
    manifest["artifacts"][f"{observation_tag(observation_ratio)}_raw_flow_csv"] = raw_csv_written
    manifest["stats"]["packet_rows"] = int(len(packet_df))
    manifest["stats"]["packet_base_flow_rows"] = int(packet_df["flow_id"].nunique())
    manifest["stats"][f"{observation_tag(observation_ratio)}_flow_rows"] = int(len(flow_df))
    manifest["stats"][f"{observation_tag(observation_ratio)}_label_counts"] = {
        str(label): int(count)
        for label, count in flow_df["Label"].dropna().value_counts().to_dict().items()
    }
    manifest["stats"][f"{observation_tag(observation_ratio)}_unlabeled_flows"] = int(
        flow_df["Label"].isna().sum()
    )
    _write_pcap_manifest(day_name, capture_file, manifest)
    return manifest


def build_day_partial_flow(
    day: str,
    observation_ratio: float = 1.0,
    force: bool = False,
    write_csv: bool = False,
    max_workers: int | None = None,
    *,
    cut_mode: str | None = None,
    cut_value: float | None = None,
) -> dict:
    """Build one day-level raw flow-view from the retained packet-base artifact.

    Back-compat: when ``cut_mode`` is None we fall back to packet-percentage cuts
    driven by ``observation_ratio`` — existing ``pct_XXX`` outputs are unchanged.
    When ``cut_mode`` is provided (``time_abs``), ``cut_value`` is
    required and the output directory is named after :func:`cut_tag`.
    """
    if cut_mode is None:
        cut_mode = "packet_pct"
        cut_value = float(observation_ratio)
    else:
        if cut_value is None:
            raise ValueError("cut_value is required when cut_mode is set")
        cut_value = float(cut_value)
        if cut_mode == "packet_pct":
            validate_observation_ratio(cut_value)
            observation_ratio = cut_value
    if cut_mode == "packet_pct":
        validate_observation_ratio(cut_value)

    tag = cut_tag(cut_mode, cut_value)
    day_name = normalize_day_name(day)
    flow_build_start = time.perf_counter()
    packet_manifest = build_day_packet_base(day_name, force=force, max_workers=max_workers)

    if cut_mode == "packet_pct":
        raw_parquet_path = raw_flow_parquet_path(day_name, cut_value)
        raw_csv_path = raw_flow_csv_path(day_name, cut_value)
    else:
        raw_parquet_path = cut_raw_flow_parquet_path(day_name, cut_mode, cut_value)
        raw_csv_path = cut_raw_flow_csv_path(day_name, cut_mode, cut_value)

    if raw_parquet_path.exists() and not force:
        manifest = _read_manifest(day_name)
        if manifest.get("artifacts", {}).get(f"{tag}_raw_flow_parquet"):
            return manifest

    packet_df = pd.read_parquet(packet_base_path(day_name), engine="pyarrow")
    truncated_packet_df = truncate_packet_base_by_cut(packet_df, cut_mode, cut_value)
    flow_df = aggregate_flow_features(truncated_packet_df, day=day_name)

    profiles = load_attack_profiles()
    profile_day = profile_day_name(day_name)
    if day_has_attack_profiles(profile_day, profiles):
        label_df = label_flows(truncated_packet_df, day=profile_day, profiles=profiles).rename(
            columns={"label": "Label"}
        )
        label_df["Label"] = label_df["Label"].astype("string")
    else:
        label_df = pd.DataFrame(
            {
                "flow_id": flow_df["flow_id"].astype("int64"),
                "Label": pd.Series("Benign", index=range(len(flow_df)), dtype="string"),
                "label_encoded": 0,
            }
        )

    flow_df = flow_df.drop(columns=["Label", "label_encoded"]).merge(
        label_df,
        on="flow_id",
        how="left",
    )
    flow_df["Label"] = flow_df["Label"].astype("string")
    flow_df["label_encoded"] = flow_df["label_encoded"].fillna(UNLABELED_LABEL_ENCODED).astype("int32")
    flow_df = flow_df[RAW_FEATURE_COLUMNS]

    raw_parquet_path.parent.mkdir(parents=True, exist_ok=True)
    flow_df.to_parquet(raw_parquet_path, engine="pyarrow", compression="snappy", index=False)

    raw_csv_written = None
    if write_csv:
        raw_csv_path.parent.mkdir(parents=True, exist_ok=True)
        flow_df.to_csv(raw_csv_path, index=False)
        raw_csv_written = str(raw_csv_path)

    manifest = packet_manifest if packet_manifest else _base_manifest(day_name, profiles)
    manifest["build"]["cut_mode"] = cut_mode
    manifest["build"]["cut_value"] = float(cut_value)
    if cut_mode == "packet_pct":
        manifest["build"]["observation_ratio"] = float(cut_value)
    manifest["artifacts"][f"{tag}_raw_flow_parquet"] = str(raw_parquet_path)
    manifest["artifacts"][f"{tag}_raw_flow_csv"] = raw_csv_written
    manifest["stats"][f"{tag}_flow_rows"] = int(len(flow_df))
    manifest["stats"][f"{tag}_label_counts"] = {
        str(label): int(count)
        for label, count in flow_df["Label"].dropna().value_counts().to_dict().items()
    }
    manifest["stats"][f"{tag}_unlabeled_flows"] = int(
        flow_df["Label"].isna().sum()
    )
    _write_manifest(day_name, manifest)
    print(
        f"Completed raw flow view for {day_name} [{tag}] "
        f"in {_format_duration(time.perf_counter() - flow_build_start)} "
        f"(rows={len(flow_df)})"
    )
    return manifest


def build_all_days(
    observation_ratio: float = 1.0,
    force: bool = False,
    write_csv: bool = False,
    max_workers: int | None = None,
) -> dict[str, dict]:
    """Build packet-base and raw flow-view artifacts for every raw day directory."""
    manifests = {}
    day_names = sorted(day_dir.name for day_dir in RAW_ROOT.iterdir() if day_dir.is_dir())
    day_pcap_counts = {day_name: len(list_pcap_members(day_name)) for day_name in day_names}
    total_days = len(day_names)
    total_pcaps = sum(day_pcap_counts.values())
    completed_pcaps = 0
    overall_start = time.perf_counter()

    for day_index, day_name in enumerate(day_names, start=1):
        overall_elapsed, overall_eta = _progress_timing(
            completed_pcaps,
            total_pcaps,
            overall_start,
        )
        print(
            f"[day {day_index}/{total_days} | elapsed={overall_elapsed} eta={overall_eta}] "
            f"{day_name} ({day_pcap_counts[day_name]} PCAPs)"
        )

        day_start = time.perf_counter()
        manifests[day_name] = build_day_partial_flow(
            day=day_name,
            observation_ratio=observation_ratio,
            force=force,
            write_csv=write_csv,
            max_workers=max_workers,
        )

        completed_pcaps += day_pcap_counts[day_name]
        overall_elapsed, overall_eta = _progress_timing(
            completed_pcaps,
            total_pcaps,
            overall_start,
        )
        print(
            f"[day {day_index}/{total_days} complete | "
            f"day_elapsed={_format_duration(time.perf_counter() - day_start)} "
            f"overall_elapsed={overall_elapsed} overall_eta={overall_eta}] {day_name}"
        )
    return manifests


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build PCAP-derived packet-base and partial-flow datasets for CIC-IDS2018."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    day_parser = subparsers.add_parser("build-day", help="Build one day-level packet base and flow-view.")
    day_parser.add_argument("day", help="Day directory name, e.g. Wednesday-14-02-2018")
    day_parser.add_argument("--pct", type=float, default=1.0, help="Observation ratio in (0, 1].")
    day_parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    day_parser.add_argument("--write-csv", action="store_true", help="Also export raw flow CSVs.")
    day_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Parallel PCAP workers (default: auto, currently {DEFAULT_MAX_WORKERS}).",
    )

    pcap_parser = subparsers.add_parser(
        "build-pcap",
        help="Build packet base and flow-view for one extracted PCAP file.",
    )
    pcap_parser.add_argument("day", help="Day directory name, e.g. Wednesday-14-02-2018")
    pcap_parser.add_argument("capture_file", help="Extracted PCAP member name, e.g. UCAP172.31.69.25")
    pcap_parser.add_argument("--pct", type=float, default=1.0, help="Observation ratio in (0, 1].")
    pcap_parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    pcap_parser.add_argument("--write-csv", action="store_true", help="Also export raw flow CSVs.")

    all_parser = subparsers.add_parser("build-all", help="Build all day-level packet bases and flow-views.")
    all_parser.add_argument("--pct", type=float, default=1.0, help="Observation ratio in (0, 1].")
    all_parser.add_argument("--force", action="store_true", help="Overwrite existing artifacts.")
    all_parser.add_argument("--write-csv", action="store_true", help="Also export raw flow CSVs.")
    all_parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help=f"Parallel PCAP workers (default: auto, currently {DEFAULT_MAX_WORKERS}).",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "build-day":
        build_day_partial_flow(
            day=args.day,
            observation_ratio=args.pct,
            force=args.force,
            write_csv=args.write_csv,
            max_workers=args.workers,
        )
    elif args.command == "build-pcap":
        build_capture_partial_flow(
            day=args.day,
            capture_file=args.capture_file,
            observation_ratio=args.pct,
            force=args.force,
            write_csv=args.write_csv,
        )
    elif args.command == "build-all":
        build_all_days(
            observation_ratio=args.pct,
            force=args.force,
            write_csv=args.write_csv,
            max_workers=args.workers,
        )
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
