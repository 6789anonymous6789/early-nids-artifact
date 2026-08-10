#!/usr/bin/env python3
"""Generate the five-seed job manifests under seed_runs/.

Three TSV files, one job per line:  id <TAB> donefile <TAB> command
  jobs_byte.tsv  360 raw-byte trainings   (GPU queue)
  jobs_rf.tsv    160 random-forest fits   (CPU queue, pinned)
  jobs_occl.tsv   65 occlusion sweeps     (GPU queue, after jobs_byte)

Every run uses a fixed 20-epoch budget (--patience 20) so the five seeds are
comparable; --random-state drives both the split and the initialisation.
"""
import os
import sys

ROOT = os.environ.get(
    "EARLYNIDS_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.environ.get("EARLYNIDS_PYTHON", sys.executable)
OUT = f"{ROOT}/outputs"
RUNS = f"{ROOT}/seed_runs"
SEEDS = [42, 1, 2, 3, 4]

DS = {                       # key -> (data root, script dir)
    "cic_ids2017": ("CIC-IDS2017", "cic_ids2017"),
    "cic_ids2018": ("CIC-IDS2018", "cic_ids2018"),
    "toniot":      ("ToN-IoT",     "toniot"),
    "cic_iot2023": ("CIC-IoT2023", "cic_iot2023"),
}
ARCH_MOD = {"bigru": "byte_bigru", "transformer": "byte_transformer"}

# Architecture and optimisation flags copied verbatim from the reference run of
# each (dataset, architecture) pair, so the seeded runs differ from the paper's
# current numbers only in seed and epoch budget. The 2017 transformer needs the
# explicit --d-model 64: the script defaults to 16.
ARCH_FLAGS = {
    ("cic_ids2017", "transformer"):
        "--n-bytes 256 --d-model 64 --nhead 4 --dim-feedforward 128 --n-layers 2 "
        "--dropout 0.1 --batch-size 1024 --lr 0.001 --weight-decay 0.0001 "
        "--test-size 0.25 --val-size 0.2",
    ("cic_ids2017", "bigru"):
        "--n-bytes 256 --d-model 64 --hidden-size 64 --n-layers 1 "
        "--dropout 0.1 --batch-size 1024 --lr 0.001 --weight-decay 0.0001 "
        "--test-size 0.25 --val-size 0.2",
    ("cic_ids2018", "transformer"):
        "--n-bytes 256 --min-real 100 --d-model 64 --nhead 4 --dim-feedforward 128 "
        "--n-layers 2 --dropout 0.1 --batch-size 8192 --eval-batch-size 16384 "
        "--lr 0.001 --weight-decay 0.0001 --test-size 0.25 --val-size 0.2",
    ("cic_ids2018", "bigru"):
        "--n-bytes 256 --min-real 100 --d-model 64 --hidden-size 64 --n-layers 1 "
        "--dropout 0.1 --batch-size 8192 --eval-batch-size 16384 "
        "--lr 0.001 --weight-decay 0.0001 --test-size 0.25 --val-size 0.2",
    ("toniot", "transformer"):
        "--n-bytes 256 --min-real 100 --d-model 64 --nhead 4 --dim-feedforward 128 "
        "--n-layers 2 --dropout 0.1 --batch-size 8192 --eval-batch-size 16384 "
        "--lr 0.001 --weight-decay 0.0001 --test-size 0.25 --val-size 0.2",
    ("toniot", "bigru"):
        "--n-bytes 256 --min-real 100 --d-model 64 --hidden-size 64 --n-layers 1 "
        "--dropout 0.1 --batch-size 8192 --eval-batch-size 16384 "
        "--lr 0.001 --weight-decay 0.0001 --test-size 0.25 --val-size 0.2",
    ("cic_iot2023", "transformer"):
        "--label-mode raw --n-bytes 256 --min-real 100 --d-model 64 --nhead 4 "
        "--dim-feedforward 128 --n-layers 2 --dropout 0.1 --batch-size 32768 "
        "--eval-batch-size 32768 --lr 0.001 --weight-decay 0.0001 "
        "--test-size 0.25 --val-size 0.2",
    ("cic_iot2023", "bigru"):
        "--label-mode raw --n-bytes 256 --min-real 100 --d-model 64 --hidden-size 64 "
        "--n-layers 1 --dropout 0.1 --batch-size 65536 --eval-batch-size 131072 "
        "--lr 0.001 --weight-decay 0.0001 --test-size 0.25 --val-size 0.2",
}

byte, rf, occl = [], [], []


def bytes_dir(ds, pkt, variant, own_cutoff_dirs):
    """Path to the raw-byte directory. 2017 has one directory per cutoff;
    the other three keep only pkt5 and are truncated at load with --max-pkts."""
    root = DS[ds][0]
    n = pkt if own_cutoff_dirs else 5
    suffix = "" if variant == "basic" else f"_{variant}"
    return f"data/{root}/partial_flow/raw_bytes_pkt{n}_b256{suffix}/raw_bytes"


def add_byte(ds, arch, pkt, variant, own_cutoff_dirs):
    for s in SEEDS:
        cfg = f"{arch}_pkt{pkt}_{variant}"
        out = f"{OUT}/{ds}/seeds5/{cfg}/seed{s}"
        done = f"{out}/{ARCH_MOD[arch]}/metrics.json"
        cmd = (f"{PY} -u {ROOT}/scripts/{DS[ds][1]}/train_byte_{arch}.py"
               f" --raw-bytes-dir {ROOT}/{bytes_dir(ds, pkt, variant, own_cutoff_dirs)}"
               f" --out-dir {out} --max-pkts {pkt}"
               f" --epochs 20 --patience 20 --random-state {s}"
               f" {ARCH_FLAGS[(ds, arch)]}")
        byte.append((f"byte:{ds}:{cfg}:s{s}", done, cmd))


# ---- raw byte, CIC-IDS2017: 48 configurations -----------------------------
A_VARIANTS = ([("basic", n) for n in (1, 2, 3, 4, 5)]
              + [("strict", n) for n in (1, 2, 3, 4, 5)]
              + [("csum_fix", n) for n in (3, 5)]
              + [("strict_csum", n) for n in (3, 5)]
              + [(f"occ_top{k}", n) for k in (1, 2, 3, 4, 5) for n in (3, 5)])
for arch in ("transformer", "bigru"):
    for variant, pkt in A_VARIANTS:
        add_byte("cic_ids2017", arch, pkt, variant, own_cutoff_dirs=True)

# The capacity comparison of Section V-A quotes the d=16 and d=128 Transformers
# at five packets. Their feed-forward width scales with d, as in the runs those
# numbers came from.
for d, ff in ((16, 32), (128, 256)):
    for s in SEEDS:
        cfg = f"transformer_d{d}_pkt5_basic"
        out = f"{OUT}/cic_ids2017/seeds5/{cfg}/seed{s}"
        flags = (ARCH_FLAGS[("cic_ids2017", "transformer")]
                 .replace("--d-model 64", f"--d-model {d}")
                 .replace("--dim-feedforward 128", f"--dim-feedforward {ff}"))
        cmd = (f"{PY} -u {ROOT}/scripts/cic_ids2017/train_byte_transformer.py"
               f" --raw-bytes-dir {ROOT}/data/CIC-IDS2017/partial_flow/raw_bytes_pkt5_b256/raw_bytes"
               f" --out-dir {out} --max-pkts 5 --epochs 20 --patience 20"
               f" --random-state {s} {flags}")
        byte.append((f"byte:cic_ids2017:{cfg}:s{s}", f"{out}/byte_transformer/metrics.json", cmd))

# ---- raw byte, the other three testbeds: 8 configurations each ------------
for ds in ("toniot", "cic_ids2018", "cic_iot2023"):
    for arch in ("transformer", "bigru"):
        for pkt in (3, 4, 5):
            add_byte(ds, arch, pkt, "basic", own_cutoff_dirs=False)
    add_byte(ds, "transformer", 3, "strict", own_cutoff_dirs=False)
    add_byte(ds, "transformer", 3, "occ_top5", own_cutoff_dirs=False)

# ---- random forest: 4 testbeds x 4 cutoffs x {all, no initial window} -----
CUTS = ["pct_100", "packet_abs_5", "packet_abs_4", "packet_abs_3"]
WIN = ["Init Fwd Win Byts", "Init Bwd Win Byts"]

# Figure 1 also plots the random forest at two packets on CIC-IDS2017 and at two
# and one on CIC-IoT2023, cutoffs the Table V grid does not contain. Only the
# all-features variant is drawn there.
STRESS = {"cic_ids2017": ["packet_abs_2"], "cic_iot2023": ["packet_abs_2", "packet_abs_1"]}

for cut in CUTS:
    for nowin in (False, True):
        tag = f"rf_{cut}" + ("_noInitWin" if nowin else "")
        for s in SEEDS:
            for ds, script, done_rel, extra in (
                ("cic_ids2017", "cic_ids2017/run_distrinet_11class_baselines.py", "rf/metrics.json",
                 f"--data-dir {ROOT}/data/CIC-IDS2017/partial_flow/{cut}/raw_flow --out-dir {{out}}"
                 f" --models rf --min-real 100"
                 + (f' --extra-drop "{",".join(WIN)}"' if nowin else "")),
                ("cic_ids2018", "cic_ids2018/train_rf_rawbyte_subset.py", "run_config.json",
                 f"--data-dir {ROOT}/data/CIC-IDS2018/partial_flow/{cut}/raw_flow"
                 f" --output-dir {{out}} --min-real 100"
                 + (" --no-init-win" if nowin else "")),
                ("toniot", "toniot/train_rf_pct100.py", "run_config.json",
                 f"--cut {cut} --output-dir {{out}}"
                 + (' --drop-features "Init Fwd Win Byts" "Init Bwd Win Byts"' if nowin else "")),
                ("cic_iot2023", "cic_iot2023/train_rf_packetcap.py", "run_config.json",
                 f"--data-dir {ROOT}/data/CIC-IoT2023/partial_flow/{cut}/raw_flow"
                 f" --out-dir {{out}} --label-mode raw --min-rows 100"
                 + (" --no-init-win" if nowin else "")),
            ):
                out = f"{OUT}/{ds}/seeds5/{tag}/seed{s}"
                cmd = (f"{PY} -u {ROOT}/scripts/{script} "
                       + extra.format(out=out) + f" --random-state {s}")
                rf.append((f"rf:{ds}:{tag}:s{s}", f"{out}/{done_rel}", cmd))

for ds, cuts in STRESS.items():
    for cut in cuts:
        for s in SEEDS:
            out = f"{OUT}/{ds}/seeds5/rf_{cut}/seed{s}"
            if ds == "cic_ids2017":
                cmd = (f"{PY} -u {ROOT}/scripts/cic_ids2017/run_distrinet_11class_baselines.py"
                       f" --data-dir {ROOT}/data/CIC-IDS2017/partial_flow/{cut}/raw_flow"
                       f" --out-dir {out} --models rf --min-real 100 --random-state {s}")
                done = f"{out}/rf/metrics.json"
            else:
                cmd = (f"{PY} -u {ROOT}/scripts/cic_iot2023/train_rf_packetcap.py"
                       f" --data-dir {ROOT}/data/CIC-IoT2023/partial_flow/{cut}/raw_flow"
                       f" --out-dir {out} --label-mode raw --min-rows 100 --random-state {s}")
                done = f"{out}/run_config.json"
            rf.append((f"rf:{ds}:rf_{cut}:s{s}", done, cmd))

# ---- occlusion: the thirteen sweeps of Table VI ---------------------------
OCCL_ROWS = [("cic_ids2017", a, n) for a, n in
             (("transformer", 5), ("transformer", 3), ("bigru", 5), ("bigru", 3))]
for ds in ("toniot", "cic_ids2018", "cic_iot2023"):
    OCCL_ROWS += [(ds, "transformer", 5), (ds, "transformer", 3), (ds, "bigru", 5)]

for ds, arch, pkt in OCCL_ROWS:
    own = ds == "cic_ids2017"
    for s in SEEDS:
        model = f"{OUT}/{ds}/seeds5/{arch}_pkt{pkt}_basic/seed{s}/{ARCH_MOD[arch]}/model.pt"
        out = f"{OUT}/{ds}/seeds5/occl_{arch}_pkt{pkt}/seed{s}"
        cmd = (f"{PY} -u {ROOT}/scripts/common/occlusion_any.py"
               f" --dataset {ds} --arch {arch} --model-path {model}"
               f" --raw-bytes-dir {ROOT}/{bytes_dir(ds, pkt, 'basic', own)}"
               f" --out-dir {out} --max-pkts {pkt} --random-state {s}")
        occl.append((f"occl:{ds}:{arch}_pkt{pkt}:s{s}", f"{out}/summary.json", cmd))

os.makedirs(RUNS, exist_ok=True)
for name, jobs in (("jobs_byte.tsv", byte), ("jobs_rf.tsv", rf), ("jobs_occl.tsv", occl)):
    with open(f"{RUNS}/{name}", "w") as fh:
        for jid, done, cmd in jobs:
            fh.write(f"{jid}\t{done}\t{cmd}\n")
    print(f"{name}: {len(jobs)} jobs")

# Sanity: every input path the manifest names must already exist.
missing = set()
for _, _, cmd in byte + occl:
    for tok in cmd.split():
        if "/raw_bytes" in tok and not os.path.isdir(tok):
            missing.add(tok)
for _, _, cmd in rf:
    for tok in cmd.split():
        if tok.endswith("/raw_flow") and not os.path.isdir(tok):
            missing.add(tok)
print("missing input dirs:", len(missing))
for m in sorted(missing):
    print("  MISSING", m)
print("total jobs:", len(byte) + len(rf) + len(occl))
