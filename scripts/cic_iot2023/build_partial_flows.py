"""Drive the CIC-IoT2023 partial-flow build over all 34 attack classes.

For each class: build the packet-base once, then the labeled flow-feature
parquets for the full flow (pct_100) and the first-K-packet cuts (packet_abs_5/4/3).
Per-class subsampling caps (labeling.DEFAULT_CAPS) are applied at the flow level
with a fixed random_state so every cut + the raw-byte extractor see the same flows.

NO time cuts (deck slide 5): the early-detection axis here is packet count.

Examples
--------
# smoke: one small class, all cuts
python scripts/cic_iot2023/build_partial_flows.py --classes Backdoor_Malware

# full build (heavy — flood classes are large), background-friendly
python scripts/cic_iot2023/build_partial_flows.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.cic_iot2023.labeling import RAW_CLASSES, get_cap  # noqa: E402
from src.cic_iot2023.pcap_extracted_flow_pipeline import (  # noqa: E402
    PARTIAL_FLOW_ROOT,
    build_class_packet_base,
    build_class_raw_flow,
)

DEFAULT_CUTS = [None, 5, 4, 3]  # None == full flow (pct_100)


def _build_one_class(cls, cuts, cap, random_state, force, skip_flows, max_files,
                     probe_packets):
    """Build one class end-to-end (packet_base + flow cuts). Module-level so it can be
    submitted to a process pool; uses max_workers=1 internally so there is no nested
    pool — cross-class parallelism happens in the driver instead (tshark is
    single-threaded, so filling cores means running many classes at once)."""
    flow_cap = get_cap(cls) if cap == -1 else cap
    pb = build_class_packet_base(cls, force=force, max_workers=1, flow_cap=flow_cap,
                                 max_files=max_files, probe_packets=probe_packets)
    entry = {"class": cls,
             "packet_base_flows": pb.get("stats", {}).get("packet_base_flow_rows"),
             "cuts": {}}
    if not skip_flows:
        for k in cuts:
            r = build_class_raw_flow(cls, max_packets=k, cap=cap,
                                     random_state=random_state, force=force, max_workers=1)
            entry["cuts"][("pct_100" if k is None else f"packet_abs_{k}")] = \
                r.get("n_flows", r.get("cached"))
    return entry


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--classes", nargs="+", default=list(RAW_CLASSES))
    p.add_argument("--cuts", nargs="+", type=int, default=None,
                   help="Packet-cap cuts (e.g. 5 4 3). 0 means full flow (pct_100). "
                        "Default: full + 5 4 3.")
    p.add_argument("--max-workers", type=int, default=None)
    p.add_argument("--cap", type=int, default=-1,
                   help="Per-class flow cap (-1=class default, 0=keep all).")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--force", action="store_true")
    p.add_argument("--skip-flows", action="store_true",
                   help="Only build packet_base (no flow features).")
    p.add_argument("--class-workers", type=int, default=None,
                   help="Classes to build concurrently (default: min(cpu-2, 10)). "
                        "tshark is single-threaded, so concurrency ACROSS classes is "
                        "what fills the cores; keep <= cores-2 on a shared box.")
    p.add_argument("--max-files-per-class", type=int, default=None,
                   help="Hard ceiling on pcap files parsed per class. Bounds cost for "
                        "low-flow-density floods that never reach the cap; fewer flows "
                        "is fine since the cap is a maximum.")
    p.add_argument("--probe-packets", type=int, default=None,
                   help="Bounded mode: read only file0 up to N packets and accept the "
                        "resulting flows as-is (for ultra-low-flow-density volumetric "
                        "floods where reaching the cap would need tens of GB).")
    args = p.parse_args()

    if args.cuts is None:
        cuts = DEFAULT_CUTS
    else:
        cuts = [None if c == 0 else c for c in args.cuts]
    cap = None if args.cap == 0 else args.cap

    n_cw = args.class_workers or max(1, min((os.cpu_count() or 2) - 2, 10))
    n_cw = max(1, min(n_cw, len(args.classes)))
    print(f"Building {len(args.classes)} classes with {n_cw} concurrent class-workers "
          f"(cap={cap})", flush=True)

    t0 = time.perf_counter()
    results = []
    if n_cw == 1:
        for cls in args.classes:
            print(f"\n=== {cls} ===", flush=True)
            try:
                results.append(_build_one_class(cls, cuts, cap, args.random_state,
                                                 args.force, args.skip_flows,
                                                 args.max_files_per_class,
                                                 args.probe_packets))
            except Exception as e:  # keep going across classes
                print(f"[{cls}] FAILED: {e}", flush=True)
                results.append({"class": cls, "error": str(e)})
    else:
        with ProcessPoolExecutor(max_workers=n_cw) as ex:
            futs = {ex.submit(_build_one_class, cls, cuts, cap, args.random_state,
                              args.force, args.skip_flows,
                              args.max_files_per_class,
                              args.probe_packets): cls for cls in args.classes}
            for fut in as_completed(futs):
                cls = futs[fut]
                try:
                    results.append(fut.result())
                    print(f"[done] {cls}", flush=True)
                except Exception as e:  # keep going across classes
                    print(f"[{cls}] FAILED: {e}", flush=True)
                    results.append({"class": cls, "error": str(e)})

    summary_path = PARTIAL_FLOW_ROOT / "build_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\nALL DONE in {time.perf_counter()-t0:.0f}s -> {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
