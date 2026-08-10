"""Labeling + class taxonomy for CIC-IoT2023.

Unlike CIC-IDS2017/2018 (Distrinet 5-tuple matching) or UNSW-NB15 (GT CSV),
CIC-IoT2023 labels are *free*: each PCAP lives under a directory whose name is
the attack class, e.g. ``PCAP/DDoS-SYN_Flood/DDoS-SYN_Flood1.pcap``. So the
partition unit is the **class folder** (not "days") and ``label_from_pcap_dir``
is a trivial path lookup.

Two label granularities are exposed, matching the settings reported by the
dataset paper (Neto et al., 2023, *CICIoT2023*):

  - 34-class: 33 attack types + Benign (the raw folder names).
  - 8-class:  7 attack categories + Benign, via ``CLASS_TO_GROUP``.

Binary (Benign vs. attack) is derivable from either.

Per-class subsampling caps (``DEFAULT_CAPS``) mirror the 2018 methodology:
cap Benign + the dominant
volumetric flood classes (DDoS/DoS/Mirai), keep the rare classes intact.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# 34-class taxonomy (raw PCAP directory names == labels)
# ---------------------------------------------------------------------------

BENIGN_CLASS = "Benign_Final"

RAW_CLASSES: tuple[str, ...] = (
    "Backdoor_Malware",
    "Benign_Final",
    "BrowserHijacking",
    "CommandInjection",
    "DDoS-ACK_Fragmentation",
    "DDoS-HTTP_Flood",
    "DDoS-ICMP_Flood",
    "DDoS-ICMP_Fragmentation",
    "DDoS-PSHACK_Flood",
    "DDoS-RSTFINFlood",
    "DDoS-SlowLoris",
    "DDoS-SYN_Flood",
    "DDoS-SynonymousIP_Flood",
    "DDoS-TCP_Flood",
    "DDoS-UDP_Flood",
    "DDoS-UDP_Fragmentation",
    "DictionaryBruteForce",
    "DNS_Spoofing",
    "DoS-HTTP_Flood",
    "DoS-SYN_Flood",
    "DoS-TCP_Flood",
    "DoS-UDP_Flood",
    "Mirai-greeth_flood",
    "Mirai-greip_flood",
    "Mirai-udpplain",
    "MITM-ArpSpoofing",
    "Recon-HostDiscovery",
    "Recon-OSScan",
    "Recon-PingSweep",
    "Recon-PortScan",
    "SqlInjection",
    "Uploading_Attack",
    "VulnerabilityScan",
    "XSS",
)

# ---------------------------------------------------------------------------
# 8-class grouping (7 attack categories + Benign) — Neto et al., 2023
# ---------------------------------------------------------------------------

CLASS_TO_GROUP: dict[str, str] = {
    # DDoS (12)
    "DDoS-ACK_Fragmentation": "DDoS",
    "DDoS-HTTP_Flood": "DDoS",
    "DDoS-ICMP_Flood": "DDoS",
    "DDoS-ICMP_Fragmentation": "DDoS",
    "DDoS-PSHACK_Flood": "DDoS",
    "DDoS-RSTFINFlood": "DDoS",
    "DDoS-SlowLoris": "DDoS",
    "DDoS-SYN_Flood": "DDoS",
    "DDoS-SynonymousIP_Flood": "DDoS",
    "DDoS-TCP_Flood": "DDoS",
    "DDoS-UDP_Flood": "DDoS",
    "DDoS-UDP_Fragmentation": "DDoS",
    # DoS (4)
    "DoS-HTTP_Flood": "DoS",
    "DoS-SYN_Flood": "DoS",
    "DoS-TCP_Flood": "DoS",
    "DoS-UDP_Flood": "DoS",
    # Mirai (3)
    "Mirai-greeth_flood": "Mirai",
    "Mirai-greip_flood": "Mirai",
    "Mirai-udpplain": "Mirai",
    # Recon (5)
    "Recon-HostDiscovery": "Recon",
    "Recon-OSScan": "Recon",
    "Recon-PingSweep": "Recon",
    "Recon-PortScan": "Recon",
    "VulnerabilityScan": "Recon",
    # Spoofing (2)
    "DNS_Spoofing": "Spoofing",
    "MITM-ArpSpoofing": "Spoofing",
    # Web-based (6)
    "Backdoor_Malware": "Web",
    "BrowserHijacking": "Web",
    "CommandInjection": "Web",
    "SqlInjection": "Web",
    "Uploading_Attack": "Web",
    "XSS": "Web",
    # BruteForce (1)
    "DictionaryBruteForce": "BruteForce",
    # Benign (1)
    "Benign_Final": "Benign",
}

GROUPS: tuple[str, ...] = (
    "Benign", "DDoS", "DoS", "Mirai", "Recon", "Spoofing", "Web", "BruteForce",
)

# ---------------------------------------------------------------------------
# Per-class subsampling caps (flows kept per class, applied with random_state)
# ---------------------------------------------------------------------------

DEFAULT_BENIGN_CAP = 2_000_000
DEFAULT_FLOOD_CAP = 100_000

# Volumetric / dominant classes capped at 100k; everything else uncapped
# (kept intact) so rare Web/BruteForce/Spoofing/Recon classes survive.
DEFAULT_CAPS: dict[str, int] = {BENIGN_CLASS: DEFAULT_BENIGN_CAP}
for _cls, _grp in CLASS_TO_GROUP.items():
    if _grp in ("DDoS", "DoS", "Mirai"):
        DEFAULT_CAPS[_cls] = DEFAULT_FLOOD_CAP


def get_cap(cls: str, caps: dict[str, int] | None = None) -> int | None:
    """Return the subsampling cap for a class, or None if it should be kept intact."""
    caps = DEFAULT_CAPS if caps is None else caps
    return caps.get(cls)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def label_from_pcap_dir(pcap_path: str | Path) -> str:
    """Class label for a PCAP = name of its parent directory."""
    return Path(pcap_path).parent.name


def to_group(cls: str) -> str:
    """Map a 34-class label to its 8-class group."""
    return CLASS_TO_GROUP[cls]


def is_benign(cls: str) -> bool:
    return cls == BENIGN_CLASS
