"""
PCAP → per-packet flow data extraction pipeline.

Uses tshark for fast packet field extraction, then groups packets into
bidirectional flows using 5-tuple + active timeout (matching CICFlowMeter).

Usage:
    from src.cic_ids2018.pcap_packet_extraction import extract_packets, group_into_flows

    packets_df = extract_packets(pcap_path)
    flows_df = group_into_flows(packets_df, active_timeout=120.0)
"""

import io
import subprocess
from pathlib import Path

import numba as nb
import numpy as np
import pandas as pd


# ── tshark field spec ────────────────────────────────────────────────
# Each tuple: (tshark_field, column_name, pandas_dtype)
TSHARK_FIELDS = [
    ("frame.time_epoch", "timestamp", "float64"),
    ("ip.src", "src_ip", "str"),
    ("ip.dst", "dst_ip", "str"),
    ("ip.proto", "protocol", "Int32"),
    ("ip.hdr_len", "ip_hdr_len", "Int32"),
    ("tcp.srcport", "tcp_srcport", "Int32"),
    ("tcp.dstport", "tcp_dstport", "Int32"),
    ("tcp.hdr_len", "tcp_hdr_len", "Int32"),
    ("tcp.len", "tcp_payload_len", "Int32"),
    ("udp.srcport", "udp_srcport", "Int32"),
    ("udp.dstport", "udp_dstport", "Int32"),
    ("udp.length", "udp_len", "Int32"),
    ("ip.len", "ip_len", "Int32"),          # IP total length (header + payload)
    ("tcp.flags", "tcp_flags_hex", "str"),   # hex string e.g. "0x0012"
    ("ip.ttl", "ttl", "Int32"),
    ("tcp.window_size_value", "tcp_win", "Int32"),
]

TSHARK_FIELD_NAMES = [f[0] for f in TSHARK_FIELDS]
COLUMN_NAMES = [f[1] for f in TSHARK_FIELDS]

_NONFATAL_TSHARK_PATTERNS = (
    "appears to have been cut short in the middle of a packet",
    "cut short in the middle of a packet",
    "appears to be damaged or corrupt",
)


def _is_nonfatal_tshark_error(stderr: str, stdout: str) -> bool:
    if not stdout.strip():
        return False
    stderr_lower = stderr.lower()
    return any(pattern in stderr_lower for pattern in _NONFATAL_TSHARK_PATTERNS)


def extract_packets(
    pcap_path: Path,
    display_filter: str = "ip",
    source_id: str | None = None,
    max_packets: int | None = None,
) -> pd.DataFrame:
    """
    Extract per-packet fields from a PCAP file using tshark.

    Parameters
    ----------
    pcap_path : Path
        Path to the PCAP file.
    display_filter : str
        Wireshark display filter (default: "ip" to keep only IP packets,
        excluding ARP/other non-IP which CICFlowMeter also mishandled).

    source_id : str, optional
        Stable identifier for the PCAP source. Defaults to the file name.

    Returns
    -------
    pd.DataFrame with one row per packet, columns matching COLUMN_NAMES plus:
        capture_file, capture_packet_index, payload_len
    """
    pcap_path = Path(pcap_path)
    if not pcap_path.exists():
        raise FileNotFoundError(f"PCAP not found: {pcap_path}")
    if source_id is None:
        source_id = pcap_path.name

    cmd = [
        "tshark",
        "-n",                            # no name resolution (faster; we use numeric fields)
        "-r", str(pcap_path),
    ]
    if max_packets is not None and max_packets > 0:
        # Stop after reading max_packets (counted BEFORE -Y); bounds work on huge
        # captures when only a flow subsample is kept downstream.
        cmd += ["-c", str(int(max_packets))]
    cmd += [
        "-Y", display_filter,           # display filter: IP packets only
        "-T", "fields",                  # tab-separated field output
        "-E", "separator=\t",
        "-E", "header=y",
        "-E", "quote=n",
        "-E", "occurrence=f",            # first occurrence if field repeats
    ]
    for field_name in TSHARK_FIELD_NAMES:
        cmd.extend(["-e", field_name])

    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=3600,
    )
    if result.returncode != 0:
        if _is_nonfatal_tshark_error(result.stderr, result.stdout):
            print(
                f"WARNING: tshark reported a recoverable PCAP parse error in {pcap_path}. "
                "Using the packets extracted before the parser failure."
            )
        else:
            raise RuntimeError(f"tshark failed: {result.stderr[:500]}")

    # Parse tshark TSV output
    df = pd.read_csv(
        io.StringIO(result.stdout),
        sep="\t",
        names=COLUMN_NAMES,
        header=0,
        na_values=[""],
        low_memory=False,
    )

    # Drop packets missing IP addresses (malformed)
    df = df.dropna(subset=["src_ip", "dst_ip"]).reset_index(drop=True)

    # Unify TCP/UDP port columns into src_port, dst_port
    df["src_port"] = df["tcp_srcport"].fillna(df["udp_srcport"]).fillna(0).astype("int32")
    df["dst_port"] = df["tcp_dstport"].fillna(df["udp_dstport"]).fillna(0).astype("int32")
    df.drop(columns=["tcp_srcport", "tcp_dstport", "udp_srcport", "udp_dstport"], inplace=True)

    # Parse TCP flags hex → integer (0 for non-TCP)
    df["tcp_flags"] = (
        df["tcp_flags_hex"]
        .apply(lambda x: int(x, 16) if isinstance(x, str) and x.startswith("0x") else 0)
        .astype("int32")
    )
    df.drop(columns=["tcp_flags_hex"], inplace=True)

    # Fill remaining NaNs
    df["ip_len"] = df["ip_len"].fillna(0).astype("int32")
    df["ip_hdr_len"] = df["ip_hdr_len"].fillna(0).astype("int32")
    df["tcp_hdr_len"] = df["tcp_hdr_len"].fillna(0).astype("int32")
    df["tcp_payload_len"] = df["tcp_payload_len"].fillna(0).astype("int32")
    df["udp_len"] = df["udp_len"].fillna(0).astype("int32")
    df["ttl"] = df["ttl"].fillna(0).astype("int32")
    df["tcp_win"] = df["tcp_win"].fillna(0).astype("int32")
    df["protocol"] = df["protocol"].fillna(0).astype("int32")

    udp_payload = (df["udp_len"] - 8).clip(lower=0)
    other_payload = (df["ip_len"] - df["ip_hdr_len"]).clip(lower=0)
    df["payload_len"] = np.where(
        df["protocol"] == 6,
        df["tcp_payload_len"],
        np.where(df["protocol"] == 17, udp_payload, other_payload),
    ).astype("int32")

    df.drop(columns=["tcp_payload_len", "udp_len", "tcp_hdr_len"], inplace=True)

    # Sort by timestamp
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["capture_file"] = source_id
    df["capture_packet_index"] = np.arange(len(df), dtype="int64")

    return df


# ── Numba-accelerated flow assignment ───────────────────────────────

_TCP_FIN = np.int32(0x01)
_TCP_RST = np.int32(0x04)
_TCP_FIN_RST = np.int32(_TCP_FIN | _TCP_RST)


@nb.njit(cache=True)
def _assign_flows_numba(
    key_codes: np.ndarray,       # int64 — factorized canonical key per packet
    timestamps: np.ndarray,      # float64
    src_ip_codes: np.ndarray,    # int64 — factorized src_ip
    src_ports: np.ndarray,       # int32
    tcp_flags: np.ndarray,       # int32
    protocols: np.ndarray,       # int32
    active_timeout: float,
):
    """
    Assign flow IDs, directions, and packet indices.

    Flow termination conditions:
    1. Active timeout: (ts - flow_start_ts) > active_timeout  (max-duration cap)
    2. TCP FIN/RST flag on a TCP packet

    Returns (flow_ids, directions, pkt_indices, init_ip_codes, init_ports).
    init_ip_codes/init_ports are per-packet arrays holding the initiator
    of each packet's flow (for direction and later flow_src/dst lookup).
    """
    n = len(key_codes)
    flow_ids = np.empty(n, dtype=np.int64)
    directions = np.empty(n, dtype=np.int8)
    pkt_indices = np.empty(n, dtype=np.int32)
    init_ip_codes = np.empty(n, dtype=np.int64)
    init_ports = np.empty(n, dtype=np.int32)

    flow_counter = np.int64(0)
    current_key = np.int64(-1)
    flow_start_ts = 0.0
    flow_terminated = False
    cur_init_ip = np.int64(-1)
    cur_init_port = np.int32(0)
    pkt_idx = np.int32(0)
    fwd_fin_seen = False
    bwd_fin_seen = False

    fin_flag = np.int32(0x01)
    rst_flag = np.int32(0x04)

    for i in range(n):
        k = key_codes[i]
        ts = timestamps[i]
        sip = src_ip_codes[i]
        sp = src_ports[i]
        fl = tcp_flags[i]
        proto = protocols[i]

        new_key = k != current_key
        active_expired = (not new_key) and (ts - flow_start_ts) > active_timeout
        start_new = new_key or active_expired or flow_terminated

        if start_new:
            current_key = k
            flow_start_ts = ts
            flow_terminated = False
            # Engelen WTMC 2021 §III-A.2: when active_timeout splits a long TCP
            # connection (same 5-tuple), the next sub-flow MUST inherit the
            # initiator from the previous one. Otherwise direction flips when
            # the timeout boundary lands between a request and its response,
            # breaking IP-based labeling and flow_id↔Distrinet matching.
            if new_key:
                cur_init_ip = sip
                cur_init_port = sp
            pkt_idx = np.int32(0)
            fwd_fin_seen = False
            bwd_fin_seen = False
            fid = flow_counter
            flow_counter += 1
        else:
            pkt_idx += np.int32(1)
            fid = flow_ids[i - 1]

        flow_ids[i] = fid
        is_fwd = (sip == cur_init_ip and sp == cur_init_port)
        directions[i] = np.int8(1) if is_fwd else np.int8(0)
        pkt_indices[i] = pkt_idx
        init_ip_codes[i] = cur_init_ip
        init_ports[i] = cur_init_port

        # Per RFC 793: RST aborts immediately. FIN is half-close: connection ends
        # only when BOTH directions have FIN'd. CICFlowMeter's "first FIN ends
        # flow" creates ~26% of TCP appendix flows (Engelen WTMC 2021).
        if proto == 6:
            if fl & rst_flag:
                flow_terminated = True
            elif fl & fin_flag:
                if is_fwd:
                    fwd_fin_seen = True
                else:
                    bwd_fin_seen = True
                if fwd_fin_seen and bwd_fin_seen:
                    flow_terminated = True

    return flow_ids, directions, pkt_indices, init_ip_codes, init_ports


def group_into_flows(
    packets_df: pd.DataFrame,
    active_timeout: float = 120.0,
) -> pd.DataFrame:
    """
    Group packets into bidirectional flows matching CICFlowMeter's logic.

    Flow termination rules (from CICFlowMeter FlowGenerator.java):
    1. Active timeout: flow is expired when (current_ts - flow_start_ts) > active_timeout.
       This is a max-duration cap.
    2. TCP FIN: flow is terminated when a FIN flag is observed.
    3. TCP RST: flow is terminated when a RST flag is observed.

    Parameters
    ----------
    packets_df : pd.DataFrame
        Output of extract_packets().
    active_timeout : float
        Maximum flow duration in seconds (default: 120s).

    Returns
    -------
    pd.DataFrame with columns:
        flow_id, packet_index, timestamp, ip_len, direction (1=fwd, 0=bwd),
        tcp_flags, ttl, tcp_win, protocol, src_ip, dst_ip, src_port, dst_port,
        flow_src_ip, flow_dst_ip, flow_src_port, flow_dst_port
    """
    df = packets_df.copy()

    # ── Vectorized canonical flow key ──────────────────────────────
    # Swap so (lower_ip, lower_port) is always side A
    swap = (
        (df["src_ip"] > df["dst_ip"])
        | ((df["src_ip"] == df["dst_ip"]) & (df["src_port"] > df["dst_port"]))
    )
    df["_ip_a"] = df["src_ip"].where(~swap, df["dst_ip"])
    df["_ip_b"] = df["dst_ip"].where(~swap, df["src_ip"])
    df["_port_a"] = df["src_port"].where(~swap, df["dst_port"])
    df["_port_b"] = df["dst_port"].where(~swap, df["src_port"])

    # Sort by canonical key components + timestamp
    sort_cols = ["_ip_a", "_ip_b", "_port_a", "_port_b", "protocol", "timestamp"]
    if "capture_packet_index" in df.columns:
        sort_cols.append("capture_packet_index")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    # Factorize the composite canonical key into integer codes for numba
    composite_key = (
        df["_ip_a"].astype(str) + "|"
        + df["_ip_b"].astype(str) + "|"
        + df["_port_a"].astype(str) + "|"
        + df["_port_b"].astype(str) + "|"
        + df["protocol"].astype(str)
    )
    key_codes, _ = pd.factorize(composite_key, sort=False)

    # Factorize src_ip for fast comparison inside numba
    all_ips = pd.Categorical(df["src_ip"])
    src_ip_codes = all_ips.codes.astype(np.int64)

    # ── Run numba-accelerated flow assignment ──────────────────────
    flow_ids, directions, pkt_indices, init_ip_codes, init_ports = _assign_flows_numba(
        key_codes.astype(np.int64),
        df["timestamp"].to_numpy(dtype=np.float64),
        src_ip_codes,
        df["src_port"].to_numpy(dtype=np.int32),
        df["tcp_flags"].to_numpy(dtype=np.int32),
        df["protocol"].to_numpy(dtype=np.int32),
        active_timeout,
    )

    df["flow_id"] = flow_ids
    df["direction"] = directions
    df["packet_index"] = pkt_indices

    # ── Record flow initiator from numba-preserved state ────────────
    # init_ip_codes / init_ports are preserved across active_timeout splits
    # (Engelen WTMC 2021), so they always point to the ORIGINAL connection
    # initiator even for sub-flows whose first packet is a server response.
    # Using packet_index==0 directly here would silently flip direction.
    df["flow_src_ip"] = pd.Categorical.from_codes(init_ip_codes, all_ips.categories)
    df["flow_src_port"] = init_ports.astype(np.int32)
    a_is_src = df["_ip_a"].astype(str) == df["flow_src_ip"].astype(str)
    df["flow_dst_ip"] = df["_ip_b"].where(a_is_src, df["_ip_a"])
    df["flow_dst_port"] = df["_port_b"].where(a_is_src, df["_port_a"]).astype(np.int32)

    # Clean up temporary columns
    df.drop(columns=["_ip_a", "_ip_b", "_port_a", "_port_b"], inplace=True)

    # Final column order
    cols = []
    if "capture_file" in df.columns:
        cols.append("capture_file")
    if "capture_packet_index" in df.columns:
        cols.append("capture_packet_index")
    cols.extend([
        "flow_id", "packet_index", "timestamp", "ip_len", "ip_hdr_len", "payload_len", "direction",
        "tcp_flags", "ttl", "tcp_win", "protocol",
        "src_ip", "dst_ip", "src_port", "dst_port",
        "flow_src_ip", "flow_dst_ip", "flow_src_port", "flow_dst_port",
    ])
    return df[cols].reset_index(drop=True)


def extract_and_group(
    pcap_path: Path,
    active_timeout: float = 120.0,
    display_filter: str = "ip",
    source_id: str | None = None,
) -> pd.DataFrame:
    """Convenience: extract packets + group into flows in one call."""
    packets = extract_packets(pcap_path, display_filter=display_filter, source_id=source_id)
    return group_into_flows(packets, active_timeout=active_timeout)


def save_flows_parquet(flows_df: pd.DataFrame, output_path: Path) -> None:
    """Save flow-level packet data to Parquet."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flows_df.to_parquet(output_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Saved {len(flows_df):,} packet rows ({flows_df['flow_id'].nunique():,} flows) → {output_path}")
