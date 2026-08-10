"""
IP-based flow labeling using CIC-IDS2018 attack profiles.

Labels flows by matching (src_ip, dst_ip) against documented
(attacker_ip, victim_ip) pairs within the attack time window.

Usage:
    from src.cic_ids2018.pcap_flow_labeling import load_attack_profiles, label_flows

    profiles = load_attack_profiles()
    labeled_df = label_flows(flows_df, day="Wednesday-14-02-2018", profiles=profiles)
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ATTACK_PROFILES_PATH = Path(__file__).parent.parent.parent / "data" / "CIC-IDS2018" / "attack_profiles.json"


def load_attack_profiles(path: Path = ATTACK_PROFILES_PATH) -> dict:
    """Load attack profile metadata from JSON."""
    with open(path) as f:
        return json.load(f)


def _parse_time(day_str: str, time_str: str) -> float:
    """
    Parse a day + time string into a Unix timestamp.

    The CIC-IDS2018 PCAPs use EST (UTC-5) timestamps.
    Day format: 'Wednesday-14-02-2018', time format: 'HH:MM'.
    """
    # Extract date from day string (e.g., '14-02-2018' from 'Wednesday-14-02-2018')
    parts = day_str.split("-")
    date_str = "-".join(parts[1:])  # '14-02-2018'
    dt = datetime.strptime(f"{date_str} {time_str}", "%d-%m-%Y %H:%M")
    # Convert EST to Unix timestamp (EST = UTC-5, so UTC = EST + 5h)
    est_offset = timedelta(hours=5)
    utc_dt = dt + est_offset
    return utc_dt.replace(tzinfo=timezone.utc).timestamp()


def _get_day_attacks(day: str, profiles: dict) -> list[dict]:
    """Get all attack entries for a specific day."""
    return [a for a in profiles["attacks"] if a["day"] == day]


def day_has_attack_profiles(day: str, profiles: dict) -> bool:
    """Return True when attack profile metadata exists for the day."""
    return len(_get_day_attacks(day, profiles)) > 0


def label_flows(
    flows_df: pd.DataFrame,
    day: str,
    profiles: dict,
    time_margin_seconds: float = 300.0,
) -> pd.DataFrame:
    """
    Label flows using IP + optional port matching against attack profiles.

    Parameters
    ----------
    flows_df : pd.DataFrame
        Output of group_into_flows() — must have columns:
        flow_id, flow_src_ip, flow_dst_ip, timestamp, packet_index.
        When attack profiles specify ``victim_ports``, the DataFrame must
        also include flow_src_port and flow_dst_port.
    day : str
        Day identifier, e.g. 'Wednesday-14-02-2018'.
    profiles : dict
        Attack profiles loaded from attack_profiles.json.
    time_margin_seconds : float
        Extra margin (seconds) around documented attack windows to account
        for timing imprecision. Default 300s (5 min).

    Returns
    -------
    pd.DataFrame with one row per flow_id, columns: flow_id, label, label_encoded.
    """
    attacks = _get_day_attacks(day, profiles)
    if not attacks:
        raise ValueError(
            f"No attack profiles configured for {day}. "
            "Handle this day explicitly rather than defaulting all flows to Benign."
        )

    # Build list of (attacker_ip_set, victim_ip_set, victim_port_set, start_ts, end_ts, label)
    attack_rules = []
    for a in attacks:
        start_ts = _parse_time(day, a["start_time"]) - time_margin_seconds
        end_ts = _parse_time(day, a["end_time"]) + time_margin_seconds
        attacker_set = set(a["attacker_ips"])
        victim_set = set(a["victim_ips"])
        victim_ports = a.get("victim_ports")  # None or list of ints
        victim_port_set = set(victim_ports) if victim_ports else None
        attack_rules.append((attacker_set, victim_set, victim_port_set, start_ts, end_ts, a["label"]))

    # Determine whether port columns are needed
    any_port_filter = any(vps is not None for _, _, vps, _, _, _ in attack_rules)

    # Get first packet per flow (defines flow start time, IPs, and ports)
    first_pkt_cols = ["flow_id", "flow_src_ip", "flow_dst_ip", "timestamp"]
    if any_port_filter:
        for pcol in ("flow_src_port", "flow_dst_port"):
            if pcol in flows_df.columns:
                first_pkt_cols.append(pcol)
    flow_first = (
        flows_df.loc[flows_df["packet_index"] == 0, first_pkt_cols]
        .copy()
        .reset_index(drop=True)
    )

    # Vectorized labeling: apply each attack rule as a broadcast mask
    labels = np.full(len(flow_first), "Benign", dtype=object)
    src = flow_first["flow_src_ip"].to_numpy(dtype=object)
    dst = flow_first["flow_dst_ip"].to_numpy(dtype=object)
    ts = flow_first["timestamp"].to_numpy(dtype=np.float64)
    has_ports = "flow_src_port" in flow_first.columns and "flow_dst_port" in flow_first.columns
    if has_ports:
        src_port = flow_first["flow_src_port"].to_numpy(dtype=np.int32)
        dst_port = flow_first["flow_dst_port"].to_numpy(dtype=np.int32)
    already_labeled = np.zeros(len(flow_first), dtype=bool)

    for attacker_set, victim_set, victim_port_set, start_ts, end_ts, label in attack_rules:
        # IP match: (src in attackers AND dst in victims) OR reverse
        src_is_attacker = np.isin(src, list(attacker_set))
        dst_is_victim = np.isin(dst, list(victim_set))
        dst_is_attacker = np.isin(dst, list(attacker_set))
        src_is_victim = np.isin(src, list(victim_set))
        ip_match = (src_is_attacker & dst_is_victim) | (dst_is_attacker & src_is_victim)

        # Port match: when victim_ports is specified, require the victim-side port
        # to be in the allowed set. The victim port is dst_port when src is attacker,
        # or src_port when dst is attacker (reverse direction).
        if victim_port_set is not None and has_ports:
            port_list = np.array(list(victim_port_set), dtype=np.int32)
            fwd_port_match = src_is_attacker & np.isin(dst_port, port_list)
            rev_port_match = dst_is_attacker & np.isin(src_port, port_list)
            port_match = fwd_port_match | rev_port_match
            ip_match = ip_match & port_match

        # Time window match
        time_match = (ts >= start_ts) & (ts <= end_ts)

        # Apply label where matched and not already labeled (first match wins)
        match_mask = ip_match & time_match & ~already_labeled
        labels[match_mask] = label
        already_labeled |= match_mask

    label_df = pd.DataFrame({"flow_id": flow_first["flow_id"], "label": labels})

    # Load label encoding from label_mapping.json
    mapping_path = (
        Path(__file__).parent.parent.parent
        / "data" / "CIC-IDS2018" / "parquet" / "full_flow_processed" / "label_mapping.json"
    )
    with open(mapping_path) as f:
        mapping = json.load(f)

    label_to_int = mapping["label_to_int"]
    label_df["label_encoded"] = label_df["label"].map(label_to_int).fillna(-1).astype(int)

    # Warn about unknown labels
    unknown = label_df[label_df["label_encoded"] == -1]
    if len(unknown) > 0:
        print(f"WARNING: {len(unknown)} flows with unknown labels: "
              f"{unknown['label'].unique().tolist()}")

    return label_df


def label_summary(label_df: pd.DataFrame) -> pd.DataFrame:
    """Print and return label distribution summary."""
    summary = (
        label_df.groupby("label")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary["pct"] = (summary["count"] / summary["count"].sum() * 100).round(2)
    print(summary.to_string(index=False))
    return summary
