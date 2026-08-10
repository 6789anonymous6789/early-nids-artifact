#!/usr/bin/env python3
"""Aggregate the five-seed runs under outputs/*/seeds5 into one results.json.

Every number the paper reports comes from this file, so a table and a figure
that share a cell necessarily agree. Standard deviations are the sample
standard deviation over the five runs (ddof=1). Differences between two
configurations are paired: the seed drives the split as well as the
initialisation, so the difference is taken seed by seed and then averaged.
"""
import csv
import glob
import json
import os
import statistics as st

ROOT = os.environ.get("EARLYNIDS_ROOT",
                      os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "outputs")
DATASETS = ["cic_ids2017", "cic_ids2018", "toniot", "cic_iot2023"]
SEEDS = [42, 1, 2, 3, 4]


def agg(values_by_seed):
    """mean/std over the seeds present, keeping the raw values for auditing."""
    vals = [v for _, v in sorted(values_by_seed.items())]
    if not vals:
        return None
    return {
        "n": len(vals),
        "mean": st.mean(vals),
        "std": st.stdev(vals) if len(vals) > 1 else 0.0,
        "seeds": {str(k): v for k, v in sorted(values_by_seed.items())},
    }


def read_macro_f1(seed_dir):
    """macro-F1 of one run, whichever artifact this pipeline happens to write."""
    for pat, key in (("byte_*/metrics.json", "macro_f1"),
                     ("rf/metrics.json", "macro_f1"),
                     ("run_config.json", "f1_macro")):
        for f in glob.glob(os.path.join(seed_dir, pat)):
            try:
                return json.load(open(f))[key]
            except (KeyError, json.JSONDecodeError):
                pass
    return None


def read_per_class(seed_dir):
    """{class: f1} of one run, from whichever per-class artifact exists."""
    out = {}
    for f in glob.glob(os.path.join(seed_dir, "*/classification_report.csv")):
        for row in csv.DictReader(open(f)):
            name = row[""] if "" in row else row[list(row)[0]]
            if name in ("accuracy", "macro avg", "weighted avg", ""):
                continue
            out[name] = float(row["f1-score"])
        return out
    for f in glob.glob(os.path.join(seed_dir, "per_class_metrics.csv")):
        for row in csv.DictReader(open(f)):
            out[row["class"]] = float(row["f1"])
        return out
    return out


def collect_runs():
    runs = {}
    for ds in DATASETS:
        for cfg_dir in sorted(glob.glob(f"{OUT}/{ds}/seeds5/*")):
            cfg = os.path.basename(cfg_dir)
            if cfg.startswith("occl_"):
                continue
            macro, per_class = {}, {}
            for s in SEEDS:
                sd = f"{cfg_dir}/seed{s}"
                if not os.path.isdir(sd):
                    continue
                m = read_macro_f1(sd)
                if m is None:
                    continue
                macro[s] = m
                for cls, f1 in read_per_class(sd).items():
                    per_class.setdefault(cls, {})[s] = f1
            if not macro:
                continue
            runs[f"{ds}/{cfg}"] = {
                "macro_f1": agg(macro),
                "per_class": {c: agg(v) for c, v in sorted(per_class.items())},
            }
    return runs


def collect_occlusion():
    occ = {}
    for ds in DATASETS:
        for cfg_dir in sorted(glob.glob(f"{OUT}/{ds}/seeds5/occl_*")):
            cfg = os.path.basename(cfg_dir)
            baseline, regions, leaders = {}, {}, []
            for s in SEEDS:
                sd = f"{cfg_dir}/seed{s}"
                summ, csvf = f"{sd}/summary.json", f"{sd}/region_occlusion.csv"
                if not (os.path.exists(summ) and os.path.exists(csvf)):
                    continue
                baseline[s] = json.load(open(summ))["baseline_macro_f1"]
                rows = list(csv.DictReader(open(csvf)))
                for r in rows:
                    # delta is stored negative: report the drop as a positive loss
                    regions.setdefault(r["region"], {})[s] = -float(r["delta_macro_f1"])
                leaders.append(max(rows, key=lambda r: -float(r["delta_macro_f1"]))["region"])
            if not baseline:
                continue
            votes = {r: leaders.count(r) for r in set(leaders)}
            top = max(votes, key=votes.get)
            occ[f"{ds}/{cfg}"] = {
                "baseline": agg(baseline),
                "regions": {r: agg(v) for r, v in sorted(regions.items())},
                "most_affected": {"region": top, "votes": votes[top], "of": len(leaders),
                                  "all_votes": votes},
            }
    return occ


def paired(runs, key_a, key_b):
    """mean and std of (a - b) taken seed by seed, in macro-F1 points."""
    a = runs.get(key_a, {}).get("macro_f1")
    b = runs.get(key_b, {}).get("macro_f1")
    if not a or not b:
        return None
    common = sorted(set(a["seeds"]) & set(b["seeds"]))
    if not common:
        return None
    d = [(a["seeds"][s] - b["seeds"][s]) * 100 for s in common]
    return {"n": len(d), "mean_points": st.mean(d),
            "std_points": st.stdev(d) if len(d) > 1 else 0.0,
            "seeds": dict(zip(common, d))}


def main():
    runs = collect_runs()
    occ = collect_occlusion()

    deltas = {}
    # Table V: the initial-window ablation, on every testbed and cutoff.
    for ds in DATASETS:
        for cut in ("pct_100", "packet_abs_5", "packet_abs_4", "packet_abs_3"):
            p = paired(runs, f"{ds}/rf_{cut}", f"{ds}/rf_{cut}_noInitWin")
            if p:
                deltas[f"window/{ds}/{cut}"] = p
    # Figure 4: the two mask probes at three packets, on the d=64 Transformer.
    for ds in DATASETS:
        base = f"{ds}/transformer_pkt3_basic"
        for name, other in (("strict", "transformer_pkt3_strict"),
                            ("derived", "transformer_pkt3_occ_top5")):
            p = paired(runs, base, f"{ds}/{other}")
            if p:
                deltas[f"probe/{ds}/{name}"] = p

    res = {"runs": runs, "occlusion": occ, "deltas": deltas}
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.json")
    json.dump(res, open(path, "w"), indent=1, sort_keys=True)

    print(f"configurations: {len(runs)}   occlusion sweeps: {len(occ)}   deltas: {len(deltas)}")
    incomplete = [k for k, v in runs.items() if v["macro_f1"]["n"] < 5]
    print(f"incomplete (fewer than five seeds): {len(incomplete)}")
    for k in incomplete[:20]:
        print(f"   {k}  n={runs[k]['macro_f1']['n']}")
    print(f"written {path}")


if __name__ == "__main__":
    main()
