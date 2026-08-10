#!/usr/bin/env python3
"""Emit the four result tables of Section V from results.json.

Prints one LaTeX tabular body per table, ready to paste into the section files.
Nothing here invents a number: every cell is looked up by key, and a key that is
missing prints MISSING rather than a plausible value.

Usage:  python3 make_tables.py results.json [4|5|6|7]
"""
import json
import sys

DSMAC = {"cic_ids2017": r"\dsA{}", "cic_ids2018": r"\dsB{}",
         "toniot": r"\dsC{}", "cic_iot2023": r"\dsD{}"}
CUTS = [("pct_100", "full"), ("packet_abs_5", "5 pkt"),
        ("packet_abs_4", "4 pkt"), ("packet_abs_3", "3 pkt")]
REGION_LABEL = {
    "tcp_options": "TCP options", "ip_checksum": "IP checksum",
    "tcp_window": "TCP window", "ip_ttl": "IP TTL", "ip_id": "IP Identification",
    "payload_60_128": "payload 60--128", "payload_128_192": "payload 128--192",
    "payload_192_256": "payload 192--256", "tcp_off_flags": "TCP data offset and flags",
    "ip_flags_frag": "IP flags and fragment offset", "tcp_ack": "TCP ack number",
    "tcp_seq": "TCP sequence number", "ip_total_len": "IP total length",
    "tcp_checksum": "TCP checksum", "ip_tos": "IP TOS", "ip_proto": "IP protocol",
    "ip_ver_ihl": "IP version and IHL", "tcp_urgent": "TCP urgent pointer",
}
# Table VII: display name -> configuration suffix under seeds5/
LADDER = [
    (r"\basicmask{}", "basic"),
    (r"\strictmask{}", "strict"),
    (r"\strictmask{} $+$ chksum", "strict_csum"),
    ("chksum recomputed", "csum_fix"),
    (None, None),                       # midrule
    ("$+$ IP checksum", "occ_top1"),
    ("$+$ TCP options", "occ_top2"),
    ("$+$ IP TTL", "occ_top3"),
    ("$+$ IP Identification", "occ_top4"),
    ("$+$ initial TCP window", "occ_top5"),
]

R = json.load(open(sys.argv[1]))
WANT = sys.argv[2] if len(sys.argv) > 2 else "all"


def pm(a, digits=4):
    """mean +- std, or MISSING. Standard deviation at one digit more."""
    if not a:
        return r"\textsc{missing}"
    return f"{a['mean']:.{digits}f}\\,$\\pm$\\,{a['std']:.{digits}f}"


def macro(key):
    return R["runs"].get(key, {}).get("macro_f1")


def per_class(key, cls):
    return R["runs"].get(key, {}).get("per_class", {}).get(cls)


def table4():
    """Per-class F1 on CIC-IDS2017, five columns."""
    cols = ["cic_ids2017/rf_pct_100", "cic_ids2017/rf_packet_abs_5",
            "cic_ids2017/rf_packet_abs_4", "cic_ids2017/rf_packet_abs_3",
            "cic_ids2017/rf_packet_abs_3_noInitWin"]
    order = ["BENIGN", "DoS Hulk", "Portscan", "DDoS", "Infiltration - Portscan",
             "Botnet", None, "DoS Slowhttptest", "DoS Slowloris", "DoS GoldenEye",
             "FTP-Patator", "SSH-Patator"]
    disp = {"Infiltration - Portscan": "Infiltration--Portscan"}
    print("% ---- Table IV: per-class F1, CIC-IDS2017 ----")
    for cls in order:
        if cls is None:
            print(r"\midrule")
            continue
        cells = " & ".join(pm(per_class(c, cls)) for c in cols)
        print(f"{disp.get(cls, cls)} & {cells} \\\\")


def table5():
    """Initial-window ablation on the four testbeds."""
    print("% ---- Table V: initial TCP window ----")
    for i, ds in enumerate(DSMAC):
        if i:
            print(r"\midrule")
        for j, (cut, lab) in enumerate(CUTS):
            name = DSMAC[ds] if j == 0 else "       "
            a = macro(f"{ds}/rf_{cut}")
            b = macro(f"{ds}/rf_{cut}_noInitWin")
            # The delta column reports the loss, so it carries a minus sign:
            # the paired difference is (all features - no initial window).
            d = R["deltas"].get(f"window/{ds}/{cut}")
            dcell = (f"$-{d['mean_points']:.1f}\\,\\pm\\,{d['std_points']:.1f}$"
                     if d else r"\textsc{missing}")
            print(f"{name} & {lab} & {pm(a)} & {pm(b)} & {dcell} \\\\")


def table6():
    """Byte-region occlusion across the thirteen sweeps."""
    rows = [("cic_ids2017", "transformer", 5, "Tx-64, 5"),
            ("cic_ids2017", "transformer", 3, "Tx-64, 3"),
            ("cic_ids2017", "bigru", 5, "GRU, 5"),
            ("cic_ids2017", "bigru", 3, "GRU, 3")]
    for ds in ("cic_ids2018", "toniot", "cic_iot2023"):
        rows += [(ds, "transformer", 5, "Tx-64, 5"), (ds, "transformer", 3, "Tx-64, 3"),
                 (ds, "bigru", 5, "GRU, 5")]
    regions = ["tcp_options", "ip_checksum", "tcp_window", "ip_ttl"]
    print("% ---- Table VI: byte-region occlusion ----")
    prev = None
    for ds, arch, pkt, lab in rows:
        if prev and prev != ds:
            print(r"\midrule")
        name = DSMAC[ds] if prev != ds else "       "
        prev = ds
        occ = R["occlusion"].get(f"{ds}/occl_{arch}_pkt{pkt}", {})
        cells = " & ".join(pm(occ.get("regions", {}).get(r), 3) for r in regions)
        ma = occ.get("most_affected")
        top = REGION_LABEL.get(ma["region"], ma["region"]) if ma else r"\textsc{missing}"
        if ma and ma["votes"] < ma["of"]:
            top += f"$^{{{ma['votes']}/{ma['of']}}}$"
        print(f"{name} & {lab} & {cells} & {top} \\\\")


def table7():
    """The derived-mask ladder on CIC-IDS2017.

    Architecture sits in the rows rather than the columns. Four columns of
    mean-and-deviation do not fit an IEEE column at any font size; two do, with
    room to spare.
    """
    print("% ---- Table VII: masks derived from the attribution ----")
    for label, suffix in LADDER:
        if label is None:
            print(r"\midrule")
            continue
        for arch, disp in (("transformer", "Tx-64"), ("bigru", "GRU")):
            name = label if arch == "transformer" else ""
            cells = " & ".join(
                pm(macro(f"cic_ids2017/{arch}_pkt{pkt}_{suffix}")) for pkt in (5, 3))
            print(f"{name:<26} & {disp:<5} & {cells} \\\\")


for name, fn in (("4", table4), ("5", table5), ("6", table6), ("7", table7)):
    if WANT in ("all", name):
        fn()
        print()
