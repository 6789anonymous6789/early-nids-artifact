#!/usr/bin/env python3
"""Paper figures at IEEE column width, as vector PDF.

Every value is read from results.json, the single aggregate produced by
scripts/common/collect_seeds.py over the five-seed runs. Nothing is transcribed
by hand, so a cell that appears in both a figure and a table cannot disagree
with itself. Bars and markers are means over five runs; error bars are the
sample standard deviation, and for a difference between two configurations the
standard deviation of the five paired differences.

Single column is 3.5 in, so type sits at 7-8 pt. Every series also carries a
marker and dash pattern and every bar category a hatch, so the figures still
read in greyscale.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.json")
if not os.path.exists(RESULTS):
    sys.exit(f"{RESULTS} not found: run collect_seeds.py on the server and copy it here.")
R = json.load(open(RESULTS))

# Categorical slots, used in fixed order and never cycled.
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

COL = 3.45          # IEEE single column, inches
COL2 = 7.10         # IEEE double column

plt.rcParams.update({
    "font.size": 7.5,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "axes.edgecolor": MUTED,
    "axes.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.labelcolor": INK,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "legend.frameon": False,
    "pdf.fonttype": 42,       # embed as TrueType, not Type 3: IEEE Xplore rejects Type 3
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})

MISSING = []


def mf(key):
    """(mean, std) of the macro-F1 of one configuration, (None, None) if absent."""
    a = R["runs"].get(key, {}).get("macro_f1")
    if not a:
        MISSING.append(key)
        return None, None
    return a["mean"], a["std"]


def series(keys):
    """Two aligned lists, means and standard deviations, with None for gaps."""
    out = [mf(k) if k else (None, None) for k in keys]
    return [m for m, _ in out], [s for _, s in out]


def delta(key):
    """(mean, std) of a paired difference, in macro-F1 points."""
    d = R["deltas"].get(key)
    if not d:
        MISSING.append("delta/" + key)
        return None, None
    return d["mean_points"], d["std_points"]


def errs(values):
    """Matplotlib wants no error bars at all rather than a list of Nones."""
    return None if all(v is None for v in values) else [v or 0.0 for v in values]


def bars(values):
    """Bar heights with gaps as NaN: ax.bar rejects None once yerr is given,
    and NaN is what makes it draw nothing for that slot."""
    return [float("nan") if v is None else v for v in values]


def finish(ax, ylabel=None, xlabel=None):
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)


def save(fig, name):
    out = os.path.join(HERE, name + ".pdf")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", os.path.basename(out))


# --------------------------------------------------------------------------
# Fig. 1 - both representations against the packet budget, on all four
# testbeds. Flow statistics from the random forest, raw bytes from the BiGRU
# and the d=64 Transformer under the basic mask. The raw-byte models have no
# complete-flow point by construction: they are defined on a packet budget.
# --------------------------------------------------------------------------
PANELS = [
    ("cic_ids2017", "CIC-IDS2017, 11 cl."),
    ("cic_ids2018", "CIC-IDS2018, 12 cl."),
    ("toniot",      "ToN-IoT, 10 cl."),
    ("cic_iot2023", "CIC-IoT2023, 34 cl."),
]
CUTS = ["pct_100", "packet_abs_5", "packet_abs_4", "packet_abs_3",
        "packet_abs_2", "packet_abs_1"]
PKTS = [None, 5, 4, 3, 2, 1]

# Which points each panel actually has. The stress cutoffs were only run where
# the earlier study ran them, so a gap here is a deliberate absence and not a
# job still queued: asking for the rest would report false holes.
HAVE = {
    "cic_ids2017": {"rf": {"pct_100", "packet_abs_5", "packet_abs_4",
                           "packet_abs_3", "packet_abs_2"},
                    "seq": {5, 4, 3, 2, 1}},
    "cic_ids2018": {"rf": {"pct_100", "packet_abs_5", "packet_abs_4", "packet_abs_3"},
                    "seq": {5, 4, 3}},
    "toniot":      {"rf": {"pct_100", "packet_abs_5", "packet_abs_4", "packet_abs_3"},
                    "seq": {5, 4, 3}},
    "cic_iot2023": {"rf": {"pct_100", "packet_abs_5", "packet_abs_4", "packet_abs_3"},
                    "seq": {5, 4, 3}},
}


def fig_threshold():
    LAB = ["full", "5", "4", "3", "2", "1"]
    fig, axes = plt.subplots(2, 2, figsize=(COL, 2.95), sharex=True, sharey=True)
    for ax, (ds, title) in zip(axes.ravel(), PANELS):
        have = HAVE[ds]
        rf, rf_e = series([f"{ds}/rf_{c}" if c in have["rf"] else None for c in CUTS])
        gru, gru_e = series([f"{ds}/bigru_pkt{n}_basic"
                             if n in have["seq"] else None for n in PKTS])
        tx, tx_e = series([f"{ds}/transformer_pkt{n}_basic"
                           if n in have["seq"] else None for n in PKTS])
        for vals, es, color, marker, ls, lab in (
            (rf,  rf_e,  INK,    "o", "-",  "random forest, flow stats"),
            (gru, gru_e, BLUE,   "s", "--", "BiGRU, raw bytes"),
            (tx,  tx_e,  ORANGE, "^", ":",  "Transformer, raw bytes"),
        ):
            xs = [i for i, v in enumerate(vals) if v is not None]
            ys = [v for v in vals if v is not None]
            ye = [es[i] or 0.0 for i in xs]
            # The markers used to be larger than the deviation they were meant to
            # let through: the median standard deviation on these curves is 0.0023
            # on an axis 0.68 tall, which a 3 pt marker covers entirely. Smaller
            # markers and wider caps let the points that do vary show it, the
            # five-packet BiGRU on \dsB{} above all, at 0.019.
            ax.errorbar(xs, ys, yerr=ye, color=color, marker=marker, ms=2.2, lw=1.1,
                        ls=ls, label=lab, zorder=3, elinewidth=0.7, capsize=1.7,
                        capthick=0.7)
        ax.set_title(title, fontsize=7, pad=2.5)
        ax.set_ylim(0.35, 1.03)
        ax.set_xlim(-0.35, 5.35)
        ax.set_xticks(range(6))
        ax.set_xticklabels(LAB)
        ax.grid(axis="y", zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    for ax in axes[:, 0]:
        ax.set_ylabel("macro-F1")

    fig.supxlabel("packets retained", fontsize=8, y=0.085)
    fig.tight_layout(pad=0.35, w_pad=0.9, h_pad=0.7, rect=(0, 0.125, 1, 1))
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.018), ncol=2,
               handlelength=2.0, columnspacing=1.0, labelspacing=0.25)
    save(fig, "fig_threshold")


# --------------------------------------------------------------------------
# Fig. 2 - occlusion of the d=64 Transformer at five packets on CIC-IDS2017,
# under the basic mask. All 22 regions were measured; the 13 with a
# non-negligible effect are shown.
# --------------------------------------------------------------------------
REGION_NAME = {
    "ip_checksum": "IP checksum", "tcp_options": "TCP options",
    "payload_60_128": "Payload 60–128", "ip_ttl": "TTL",
    "ip_id": "IP Identification", "payload_128_192": "Payload 128–192",
    "tcp_window": "TCP window", "ip_flags_frag": "IP flags/frag.",
    "tcp_ack": "TCP ack number", "tcp_off_flags": "TCP offset+flags",
    "payload_192_256": "Payload 192–256", "ip_total_len": "IP total length",
    "tcp_checksum": "TCP checksum", "ip_tos": "IP TOS", "ip_proto": "IP protocol",
    "tcp_urgent": "TCP urgent ptr", "ip_ver_ihl": "IP version+IHL",
    "tcp_seq": "TCP sequence number", "tcp_sport": "Src port (masked)",
    "tcp_dport": "Dst port (masked)", "ip_src": "Src IP (masked)",
    "ip_dst": "Dst IP (masked)",
}
REGION_CAT = {
    "ip_checksum": "derived",
    "tcp_options": "identifying", "ip_ttl": "identifying",
    "ip_id": "identifying", "tcp_window": "identifying",
    "payload_60_128": "payload", "payload_128_192": "payload",
    "payload_192_256": "payload",
}


def fig_occlusion():
    occ = R["occlusion"].get("cic_ids2017/occl_transformer_pkt5")
    if not occ:
        MISSING.append("occlusion/cic_ids2017/occl_transformer_pkt5")
        return
    rows = sorted(occ["regions"].items(), key=lambda kv: -kv[1]["mean"])[:13]
    style = {
        "derived":     (ORANGE, "//"),
        "identifying": (BLUE, ""),
        "payload":     (AQUA, "xx"),
        "other":       ("#b8b7b0", ".."),
    }

    # Narrower than COL on purpose: the y labels and the value annotations sit
    # outside the axes, and the tight bounding box has to land inside 3.45 in.
    # Error bars widened it past the column, hence 2.80 rather than 2.98.
    fig, ax = plt.subplots(figsize=(2.80, 2.55))
    top = max(r[1]["mean"] + r[1]["std"] for r in rows)
    for i, (region, a) in enumerate(rows):
        c, h = style[REGION_CAT.get(region, "other")]
        ax.barh(i, a["mean"], xerr=a["std"], color=c, hatch=h, edgecolor="white",
                lw=0.5, height=0.72, zorder=3,
                error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=1.2, capthick=0.6,
                              zorder=5))
        ax.text(a["mean"] + a["std"] + 0.012 * top, i, f"{a['mean']:.3f}",
                va="center", fontsize=6.5, color=INK)

    ax.set_yticks(list(range(len(rows))))
    ax.set_yticklabels([REGION_NAME.get(r, r) for r, _ in rows])
    ax.invert_yaxis()
    ax.set_xlim(0, top * 1.18)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    # Short label on purpose: the panel is narrower than a column so that the
    # region names fit outside the axes, and the longer wording ran past the
    # bounding box and lost its last letter. The caption carries the meaning.
    ax.set_xlabel("macro-F1 drop")

    handles = [
        Patch(fc=ORANGE, hatch="//", ec="white", label="IP checksum"),
        Patch(fc=BLUE, label="IP/TCP header field"),
        Patch(fc=AQUA, hatch="xx", ec="white", label="application payload"),
        Patch(fc="#b8b7b0", hatch="..", ec="white", label="other header field"),
    ]
    ax.legend(handles=handles, loc="lower right", handlelength=1.6, labelspacing=0.3)
    save(fig, "fig_occlusion")


# --------------------------------------------------------------------------
# Fig. 3 - basic against strict mask on CIC-IDS2017, with the two flow-feature
# references. The two stress cutoffs are included because the mask effect is
# largest there. The random forest has no one-packet point and its ablated
# variant none below three packets: the statistics are undefined.
# --------------------------------------------------------------------------
def fig_masks():
    cuts = ["1 pkt", "2 pkt", "3 pkt", "4 pkt", "5 pkt"]
    order = [1, 2, 3, 4, 5]
    A = "cic_ids2017"
    bigru_b, bigru_be = series([f"{A}/bigru_pkt{n}_basic" for n in order])
    bigru_s, bigru_se = series([f"{A}/bigru_pkt{n}_strict" for n in order])
    tx_b, tx_be = series([f"{A}/transformer_pkt{n}_basic" for n in order])
    tx_s, tx_se = series([f"{A}/transformer_pkt{n}_strict" for n in order])
    rf, rf_e = series([None] + [f"{A}/rf_packet_abs_{n}" for n in (2, 3, 4, 5)])
    rf_abl, rf_abl_e = series([None, None]
                              + [f"{A}/rf_packet_abs_{n}_noInitWin" for n in (3, 4, 5)])

    fig, ax = plt.subplots(figsize=(COL, 2.45))
    w = 0.19
    pos = list(range(len(cuts)))
    groups = [
        (bigru_b, bigru_be, -1.5 * w, BLUE, "", "BiGRU, basic"),
        (bigru_s, bigru_se, -0.5 * w, BLUE, "///", "BiGRU, strict"),
        (tx_b, tx_be, 0.5 * w, ORANGE, "", "Transformer, basic"),
        (tx_s, tx_se, 1.5 * w, ORANGE, "///", "Transformer, strict"),
    ]
    for vals, es, off, c, h, lab in groups:
        ax.bar([p + off for p in pos], bars(vals), width=w, color=c, hatch=h,
               edgecolor="white", lw=0.5, label=lab, zorder=3, yerr=errs(es),
               error_kw=dict(ecolor=INK, elinewidth=0.7, capsize=1.5,
                             capthick=0.7, zorder=5))

    for vals, es, color, lw, ls, marker, lab in (
        (rf, rf_e, INK, 1.1, "--", "D", "random forest"),
        (rf_abl, rf_abl_e, MUTED, 1.0, ":", "v", "RF without init. window"),
    ):
        xs = [i for i, v in enumerate(vals) if v is not None]
        # Same reason as in fig_threshold: the diamond and the triangle were wide
        # enough to swallow their own error bars, while the bars beside them show
        # theirs.
        ax.errorbar(xs, [vals[i] for i in xs], yerr=[es[i] or 0.0 for i in xs],
                    color=color, lw=lw, ls=ls, marker=marker, ms=2.6, label=lab,
                    zorder=4, elinewidth=0.7, capsize=1.6, capthick=0.7)

    ax.set_xticks(pos)
    ax.set_xticklabels(cuts)
    ax.set_ylim(0, 1.05)
    finish(ax, ylabel="macro-F1")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2,
              handlelength=1.5, columnspacing=1.0, labelspacing=0.25)
    save(fig, "fig_masks")


# --------------------------------------------------------------------------
# Fig. 4 - the three probes at three packets on the four testbeds, at single
# column width. Each drop is a paired difference: for every seed, the masked or
# ablated run minus the reference run on the same split. The bars are therefore
# the mean of the five paired differences and the error bars their standard
# deviation. Masks are measured on the d=64 Transformer, the ablation on the
# random forest, so the three are commensurable.
# --------------------------------------------------------------------------
SOURCES = {"cic_ids2017": "6 src", "cic_ids2018": "26 src",
           "toniot": "99 src", "cic_iot2023": "2899 src"}


def fig_probes():
    order = [ds for ds, _ in PANELS]
    names = {"cic_ids2017": "CIC-IDS2017", "cic_ids2018": "CIC-IDS2018",
             "toniot": "ToN-IoT", "cic_iot2023": "CIC-IoT2023"}
    probes = []
    for key, color, hatch, label in (
        ("probe/{}/strict", BLUE, "", "strict mask, raw bytes"),
        ("probe/{}/derived", ORANGE, "///", "derived mask, raw bytes"),
        ("window/{}/packet_abs_3", AQUA, "xx", "initial-window ablation, random forest"),
    ):
        vals, es = zip(*[delta(key.format(ds)) for ds in order])
        probes.append((list(vals), list(es), color, hatch, label))

    fig, ax = plt.subplots(figsize=(COL, 2.55))
    w = 0.26
    pos = list(range(len(order)))
    for i, (vals, es, color, hatch, label) in enumerate(probes):
        off = (i - 1) * w
        ax.bar([p + off for p in pos], bars(vals), width=w, color=color, hatch=hatch,
               edgecolor="white", lw=0.5, zorder=3, label=label, yerr=errs(es),
               error_kw=dict(ecolor=INK, elinewidth=0.6, capsize=1.4,
                             capthick=0.6, zorder=5))
        for p_, v, e in zip(pos, vals, es):
            if v is None:
                continue
            ax.text(p_ + off, v + (e or 0) + 1.0, f"{v:.0f}", ha="center",
                    fontsize=6, color=MUTED)

    ax.set_xticks(pos)
    # Two lines per tick: the testbed and its attack-source count, which is the
    # variable the panel is read against.
    ax.set_xticklabels([f"{names[d]}\n{SOURCES[d]}" for d in order], fontsize=6.2)
    ax.set_ylim(0, 52)
    finish(ax, ylabel="macro-F1 drop (points)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=1,
              handlelength=1.5, labelspacing=0.25)
    save(fig, "fig_probes")


if __name__ == "__main__":
    fig_threshold()
    fig_occlusion()
    fig_masks()
    fig_probes()
    if MISSING:
        print(f"\n{len(MISSING)} value(s) missing from results.json:")
        for k in sorted(set(MISSING)):
            print("   ", k)
        sys.exit(1)
