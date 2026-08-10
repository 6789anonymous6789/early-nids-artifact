"""DEPRECATED — kept temporarily for historical reference.

This is the legacy IP+time-window labeler for CIC-IDS2017. It has been replaced
by `distrinet_labeling.py`, which joins our flow_id to the gold-standard
Distrinet CNS 2022 CSV labels (https://intrusion-detection.distrinet-research.be/CNS2022/).

The IP+time approach is structurally weaker (cannot disambiguate overlapping
attacks like Wed 2017 Slowloris/Slowhttptest, no `-Attempted` semantic split).
Slated for deletion once the Distrinet-based pipeline is validated end-to-end.

----- ORIGINAL DOCSTRING BELOW -----

IP-based flow labeling for CIC-IDS2017 using the Distrinet CNS 2022 audit.

Same algorithm as the 2018 module: label by matching (src, dst) IPs against the
documented (attacker, victim) pairs within the corrected time windows.

Differences vs 2018:
- Time format is HH:MM:SS (UTC) instead of HH:MM (EST).
- Day strings are WORKINGHOURS-style (e.g. 'Tuesday-WorkingHours'), so the
  date->PCAP date mapping is hard-coded here.
- Label set is 2017-specific (FTP-Patator, DoS Hulk, Heartbleed, PortScan, ...);
  see data/CIC-IDS2017/label_mapping.json.
- Supports attacks with empty victim_ports (Infiltration, Bot, PortScan), where
  matching is IP+time only.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
ATTACK_PROFILES_PATH = _REPO_ROOT / "data" / "CIC-IDS2017" / "attack_profiles.json"
LABEL_MAPPING_PATH = _REPO_ROOT / "data" / "CIC-IDS2017" / "label_mapping.json"

# Sharafaldin 2018 ICISSP: capture week was Mon 3 July 2017 - Fri 7 July 2017.
DAY_TO_DATE = {
    "Monday-WorkingHours":    "2017-07-03",
    "Tuesday-WorkingHours":   "2017-07-04",
    "Wednesday-workingHours": "2017-07-05",  # note: original CIC typo kept
    "Thursday-WorkingHours":  "2017-07-06",
    "Friday-WorkingHours":    "2017-07-07",
}


def load_attack_profiles(path: Path = ATTACK_PROFILES_PATH) -> dict:
    with open(path) as f:
        return json.load(f)


def _parse_time(day_str: str, time_str: str) -> float:
    """Distrinet CNS 2022 windows are already in UTC: just combine day + time."""
    if day_str not in DAY_TO_DATE:
        raise ValueError(f"Unknown day stem: {day_str!r}. Known: {list(DAY_TO_DATE)}")
    date_str = DAY_TO_DATE[day_str]
    dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M:%S")
    return dt.replace(tzinfo=timezone.utc).timestamp()


def _get_day_attacks(day: str, profiles: dict) -> list[dict]:
    return [a for a in profiles["attacks"] if a["day"] == day]


def day_has_attack_profiles(day: str, profiles: dict) -> bool:
    return len(_get_day_attacks(day, profiles)) > 0


def label_flows(
    flows_df: pd.DataFrame,
    day: str,
    profiles: dict,
    time_margin_seconds: float = 300.0,
) -> pd.DataFrame:
    """Label flows by (attacker, victim) IP pair and time window.

    Parameters match the 2018 interface. Monday has no attacks → caller must
    short-circuit to a 'Benign' label before reaching here (day_has_attack_profiles
    returns False for Monday).
    """
    attacks = _get_day_attacks(day, profiles)
    if not attacks:
        raise ValueError(
            f"No attack profiles configured for {day}. "
            "Callers must handle attack-free days (e.g. Monday) explicitly."
        )

    attack_rules = []
    for a in attacks:
        start_ts = _parse_time(day, a["start_time"]) - time_margin_seconds
        end_ts = _parse_time(day, a["end_time"]) + time_margin_seconds
        attacker_set = set(a["attacker_ips"])
        victim_set = set(a["victim_ips"])
        victim_ports = a.get("victim_ports") or []
        victim_port_set = set(victim_ports) if victim_ports else None
        attack_rules.append((attacker_set, victim_set, victim_port_set, start_ts, end_ts, a["label"]))

    # Narrowest-window-wins: when multiple rules match the same flow (e.g. Wed
    # 2017 DoS Slowloris/Slowhttptest/Hulk/GoldenEye all share attacker→victim:80),
    # the rule with the tightest time window claims it first. Avoids the silent
    # ordering dependency where a wide-window rule (Slowloris) shadows a nested
    # narrow-window rule (Slowhttptest).
    attack_rules.sort(key=lambda r: r[4] - r[3])

    any_port_filter = any(vps is not None for _, _, vps, _, _, _ in attack_rules)

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
        src_is_attacker = np.isin(src, list(attacker_set))
        dst_is_victim = np.isin(dst, list(victim_set))
        dst_is_attacker = np.isin(dst, list(attacker_set))
        src_is_victim = np.isin(src, list(victim_set))
        ip_match = (src_is_attacker & dst_is_victim) | (dst_is_attacker & src_is_victim)

        if victim_port_set is not None and has_ports:
            port_list = np.array(list(victim_port_set), dtype=np.int32)
            fwd_port_match = src_is_attacker & np.isin(dst_port, port_list)
            rev_port_match = dst_is_attacker & np.isin(src_port, port_list)
            port_match = fwd_port_match | rev_port_match
            ip_match = ip_match & port_match

        time_match = (ts >= start_ts) & (ts <= end_ts)
        match_mask = ip_match & time_match & ~already_labeled
        labels[match_mask] = label
        already_labeled |= match_mask

    label_df = pd.DataFrame({"flow_id": flow_first["flow_id"], "label": labels})

    with open(LABEL_MAPPING_PATH) as f:
        mapping = json.load(f)
    label_to_int = mapping["label_to_int"]
    label_df["label_encoded"] = label_df["label"].map(label_to_int).fillna(-1).astype(int)

    unknown = label_df[label_df["label_encoded"] == -1]
    if len(unknown) > 0:
        print(f"WARNING: {len(unknown)} flows with unknown labels: "
              f"{unknown['label'].unique().tolist()}")
    return label_df


def label_summary(label_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        label_df.groupby("label")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    summary["pct"] = (summary["count"] / summary["count"].sum() * 100).round(2)
    print(summary.to_string(index=False))
    return summary
