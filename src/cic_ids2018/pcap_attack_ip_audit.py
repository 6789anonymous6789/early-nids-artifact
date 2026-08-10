"""
Audit: verify attacker/victim/C2 IP presence in packet_base extractions.

Reproduces the empirical claims cited in
data/CIC-IDS2018/attack_profiles.json (days_excluded, removed_attacks)
and in data/CIC-IDS2018/README.md. For each claim, counts packets in
the corresponding day's packet_base.parquet matching the IP (and,
where relevant, a time window or a destination port) and checks the
result against the expected assertion.

Usage:
    conda run -n earlynids python3 -m src.pcap_attack_ip_audit
    conda run -n earlynids python3 -m src.pcap_attack_ip_audit --csv audit.csv

Exits non-zero if any assertion fails, so CI / notebooks can depend
on the claims staying true as the packet_base is rebuilt.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.dataset as ds

_REPO_ROOT = Path(__file__).parent.parent.parent
_PACKET_BASE_DIR = _REPO_ROOT / "data" / "CIC-IDS2018" / "partial_flow" / "packet_base"

# CIC-IDS2018 captures were recorded on AWS in EST (UTC-5, no DST).
_EST = timezone(timedelta(hours=-5))

# Common Distrinet DDoS attacker set (Tue-20 LOIC + Wed-21 HOIC/LOIC-UDP).
_DDOS_ATTACKERS = [
    "18.218.115.60",
    "18.219.9.1",
    "18.219.32.43",
    "18.218.55.126",
    "52.14.136.135",
    "18.219.5.43",
    "18.216.200.189",
    "18.218.229.235",
    "18.218.11.51",
    "18.216.24.42",
]


def _est_window_to_epoch(day: str, start_hm: str, end_hm: str) -> tuple[float, float]:
    """Convert 'HH:MM'..'HH:MM' EST for a given Day-dd-mm-YYYY to Unix seconds."""
    _, dd, mm, yyyy = day.split("-")
    base = datetime(int(yyyy), int(mm), int(dd), tzinfo=_EST)
    sh, sm = map(int, start_hm.split(":"))
    eh, em = map(int, end_hm.split(":"))
    start = base.replace(hour=sh, minute=sm)
    end = base.replace(hour=eh, minute=em)
    return start.timestamp(), end.timestamp()


def _count_packets(
    day: str,
    ip: str,
    *,
    window: tuple[float, float] | None = None,
    port: int | None = None,
    port_side: str = "either",
) -> int:
    """Count packets in <day>.parquet where src_ip == ip OR dst_ip == ip.

    Optional filters:
      window     — (start_ts, end_ts) Unix seconds.
      port       — match on src_port/dst_port/either side depending on port_side.
      port_side  — 'src', 'dst', or 'either'.
    """
    path = _PACKET_BASE_DIR / f"{day}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)

    dataset = ds.dataset(path, format="parquet")
    ip_filter = (pc.field("src_ip") == ip) | (pc.field("dst_ip") == ip)
    filt = ip_filter
    if window is not None:
        start, end = window
        filt = filt & (pc.field("timestamp") >= start) & (pc.field("timestamp") <= end)
    if port is not None:
        if port_side == "src":
            filt = filt & (pc.field("src_port") == port)
        elif port_side == "dst":
            filt = filt & (pc.field("dst_port") == port)
        else:
            filt = filt & ((pc.field("src_port") == port) | (pc.field("dst_port") == port))
    return dataset.count_rows(filter=filt)


def _count_external_packets(day: str, ip: str) -> int:
    """Packets where `ip` is one endpoint AND the other endpoint is NOT 172.31.x.x."""
    path = _PACKET_BASE_DIR / f"{day}.parquet"
    dataset = ds.dataset(path, format="parquet")
    internal_prefix = "172.31."
    src_internal = pc.starts_with(pc.field("src_ip"), internal_prefix)
    dst_internal = pc.starts_with(pc.field("dst_ip"), internal_prefix)
    as_src = (pc.field("src_ip") == ip) & (~dst_internal)
    as_dst = (pc.field("dst_ip") == ip) & (~src_internal)
    return dataset.count_rows(filter=(as_src | as_dst))


# ── Audit plan ────────────────────────────────────────────────────────
# Each entry is a dict describing one claim and how to verify it.
# Supported kinds:
#   "absent"          — expect count == 0
#   "present_min"     — expect count >= min_count
#   "port_exclusive"  — expect count_on_port == total_count (all traffic on a single port)
#   "external_zero"   — expect zero packets to/from any non-172.31.x.x endpoint


def _build_plan() -> list[dict]:
    plan: list[dict] = []

    # ── Fri-16 SlowHTTPTest: 13.59.126.31 present, all on port 21 ──
    plan.append({
        "category": "removed_attack",
        "day": "Friday-16-02-2018",
        "label": "DoS attacks-SlowHTTPTest",
        "claim": "13.59.126.31 is present but 100% of its traffic is on TCP port 21 (FTP), "
                 "confirming Distrinet's finding that SlowHTTPTest was misfired at FTP.",
        "kind": "port_exclusive",
        "ip": "13.59.126.31",
        "port": 21,
        "min_total": 100_000,
    })

    # ── Wed-21 LOIC-UDP + HOIC: all 10 attackers absent ──
    loic_window = _est_window_to_epoch("Wednesday-21-02-2018", "14:08", "14:43")
    hoic_window = _est_window_to_epoch("Wednesday-21-02-2018", "18:11", "19:05")

    for attacker in _DDOS_ATTACKERS:
        plan.append({
            "category": "removed_attack",
            "day": "Wednesday-21-02-2018",
            "label": "DDOS attack-LOIC-UDP+HOIC",
            "claim": f"Distrinet attacker {attacker} has 0 packets on Wed-21.",
            "kind": "absent",
            "ip": attacker,
        })

    # Wed-21 victim: present but no flood in either window.
    plan.append({
        "category": "removed_attack",
        "day": "Wednesday-21-02-2018",
        "label": "HOIC victim window",
        "claim": "Victim 172.31.69.25 has 0 packets in the HOIC window 18:11-19:05 EST.",
        "kind": "absent_in_window",
        "ip": "172.31.69.25",
        "window": hoic_window,
    })
    plan.append({
        "category": "removed_attack",
        "day": "Wednesday-21-02-2018",
        "label": "LOIC-UDP victim window",
        "claim": "Victim 172.31.69.25 traffic in the LOIC-UDP window 14:08-14:43 EST "
                 "is limited to background scan noise (<5,000 packets, no flood).",
        "kind": "bounded_in_window",
        "ip": "172.31.69.25",
        "window": loic_window,
        "max_count": 5_000,
    })

    # ── Wed-28 Infiltration ──
    plan.append({
        "category": "removed_attack",
        "day": "Wednesday-28-02-2018",
        "label": "Infilteration",
        "claim": "Infiltration victim 172.31.69.24 is present but has zero external traffic.",
        "kind": "external_zero",
        "ip": "172.31.69.24",
        "min_total": 10_000,
    })
    plan.append({
        "category": "removed_attack",
        "day": "Wednesday-28-02-2018",
        "label": "Infilteration C2",
        "claim": "Infiltration C2 13.58.225.34 is absent from Wed-28.",
        "kind": "absent",
        "ip": "13.58.225.34",
    })

    # ── Thu-01 Infiltration ──
    plan.append({
        "category": "removed_attack",
        "day": "Thursday-01-03-2018",
        "label": "Infilteration",
        "claim": "Infiltration victim 172.31.69.24 is absent from Thu-01.",
        "kind": "absent",
        "ip": "172.31.69.24",
    })
    plan.append({
        "category": "removed_attack",
        "day": "Thursday-01-03-2018",
        "label": "Infilteration C2",
        "claim": "Infiltration C2 13.58.225.34 is absent from Thu-01.",
        "kind": "absent",
        "ip": "13.58.225.34",
    })

    return plan


def _run_check(entry: dict) -> dict:
    kind = entry["kind"]
    day = entry["day"]
    ip = entry["ip"]
    result = {**entry, "count": None, "secondary": None, "passed": False}

    if kind == "absent":
        n = _count_packets(day, ip)
        result["count"] = n
        result["passed"] = n == 0

    elif kind == "absent_in_window":
        n = _count_packets(day, ip, window=entry["window"])
        result["count"] = n
        result["passed"] = n == 0

    elif kind == "bounded_in_window":
        n = _count_packets(day, ip, window=entry["window"])
        result["count"] = n
        result["passed"] = n <= entry["max_count"]

    elif kind == "port_exclusive":
        total = _count_packets(day, ip)
        on_port = _count_packets(day, ip, port=entry["port"])
        result["count"] = total
        result["secondary"] = on_port
        result["passed"] = (
            total >= entry["min_total"]
            and on_port == total
        )

    elif kind == "external_zero":
        total = _count_packets(day, ip)
        external = _count_external_packets(day, ip)
        result["count"] = total
        result["secondary"] = external
        result["passed"] = (total >= entry["min_total"]) and (external == 0)

    else:
        raise ValueError(f"unknown kind: {kind}")

    return result


def _format_row(r: dict) -> str:
    tag = "PASS" if r["passed"] else "FAIL"
    sec = "" if r["secondary"] is None else f"  (secondary={r['secondary']:,})"
    return (
        f"[{tag}] {r['day']:22s} {r['label']:30s} "
        f"kind={r['kind']:20s} ip={r['ip']:16s} count={r['count']:>12,}{sec}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None,
                        help="Optional path to write the audit results as CSV.")
    args = parser.parse_args()

    plan = _build_plan()
    results = [_run_check(entry) for entry in plan]

    for r in results:
        print(_format_row(r))

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print(f"\n{passed}/{total} checks passed")

    if args.csv is not None:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = ["category", "day", "label", "kind", "ip", "count",
                      "secondary", "passed", "claim"]
        with args.csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for r in results:
                writer.writerow({k: r.get(k) for k in fieldnames})
        print(f"Wrote {args.csv}")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
