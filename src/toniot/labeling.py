"""Label ToN-IoT network flows using the official Ground-Truth CSVs.

Unlike CIC-IoT2023 (label == pcap folder name), ToN-IoT pcaps live in
*mixed-traffic* scenario folders (e.g. ``normal_DDoS`` holds both benign and
DDoS packets), so the folder name is NOT the per-flow label. The dataset ships
a per-connection ground truth instead, under
``SecuityEvents_GroundTruth_datasets/SecurityEvents_Network_datasets/
GroundTruth_Network_*.csv`` with columns:

    ts, src_ip, src_port, dst_ip, dst_port, proto, type

Each row is one Zeek/Bro connection: ``ts`` is the connection start (epoch
seconds, integer-truncated), the 5-tuple is the *exact* connection 5-tuple
(ephemeral src_port included), and ``type`` is one of the nine attack classes
(``scanning, ddos, xss, password, injection, dos, backdoor, ransomware,
mitm``). Benign connections are NOT in the ground truth.

Matching uses a **direction-independent canonical 5-tuple key**
``(proto, {ipA:portA, ipB:portB})`` so a pcap flow matches its GT row
regardless of endpoint ordering. When a key recurs over time (port reuse) the
closest connection start within ``time_tolerance_s`` is taken; a key that is
unique in the GT (the common case — ephemeral 5-tuples) is accepted directly.

Flows with no GT match are labeled ``normal``.

Performance note: the GT has ~17.8M rows with mostly-unique 5-tuples. The match
is done with a single vectorized pandas **merge** against a prebuilt index
(``build_gt_index``), NOT a per-key Python dict — building a 17M-entry dict per
call OOMs. The pipeline builds the index once and reuses it across all
scenarios and packet cuts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GT_DIR = (
    _REPO_ROOT / "data" / "ToN-IoT" / "original"
    / "SecuityEvents_GroundTruth_datasets" / "SecurityEvents_Network_datasets"
)

# ToN-IoT network taxonomy: 9 attack classes + normal.
ATTACK_CLASSES = (
    "normal",
    "backdoor",
    "ddos",
    "dos",
    "injection",
    "mitm",
    "password",
    "ransomware",
    "scanning",
    "xss",
)

NORMAL_LABEL = "normal"
NORMAL_LABEL_ENCODED = 0  # ATTACK_CLASSES[0]
UNLABELED_LABEL_ENCODED = -1

# ---------------------------------------------------------------------------
# Per-class subsampling caps (applied GLOBALLY after labeling, mirroring the
# 2018/CIC-IoT2023 methodology). Caps are a MAXIMUM: rarer classes keep all
# their flows. The volumetric flood classes (ddos, dos) and the dominant
# scanning class are capped; normal is capped at the benign ceiling.
# ---------------------------------------------------------------------------

DEFAULT_NORMAL_CAP = 2_000_000
DEFAULT_FLOOD_CAP = 100_000

DEFAULT_CAPS: dict[str, int] = {
    "normal": DEFAULT_NORMAL_CAP,
    "ddos": DEFAULT_FLOOD_CAP,
    "dos": DEFAULT_FLOOD_CAP,
    "scanning": DEFAULT_FLOOD_CAP,
}


def get_cap(cls: str, caps: dict[str, int] | None = None) -> int | None:
    """Return the subsampling cap for a class, or None to keep it intact."""
    caps = DEFAULT_CAPS if caps is None else caps
    return caps.get(cls)


# Protocol name -> IANA IPv4 protocol number. The GT only ever uses these three.
_PROTO_NAME_TO_INT: dict[str, int] = {"icmp": 1, "tcp": 6, "udp": 17}


# ---------------------------------------------------------------------------
# Canonical 5-tuple key (direction-independent)
# ---------------------------------------------------------------------------


def _canonical_key(src_ip: np.ndarray, src_port: np.ndarray,
                   dst_ip: np.ndarray, dst_port: np.ndarray,
                   proto: np.ndarray) -> np.ndarray:
    """Vectorized ``proto|epLow|epHigh`` with endpoints sorted so the key is
    independent of flow direction. ``ep = "ip:port"``."""
    ep1 = np.char.add(np.char.add(src_ip.astype(str), ":"), src_port.astype(str))
    ep2 = np.char.add(np.char.add(dst_ip.astype(str), ":"), dst_port.astype(str))
    low = np.where(ep1 <= ep2, ep1, ep2)
    high = np.where(ep1 <= ep2, ep2, ep1)
    pfx = np.char.add(proto.astype(str), "|")
    return np.char.add(np.char.add(np.char.add(pfx, low), "|"), high)


# ---------------------------------------------------------------------------
# Ground-truth loading + index
# ---------------------------------------------------------------------------


@dataclass
class GtIndex:
    """Prebuilt, reusable ground-truth lookup for vectorized labeling.

    ``frame`` has columns ``key`` (canonical 5-tuple), ``gt_ts`` (float64),
    ``type`` (string), ``key_dup`` (bool — True if the key occurs >1 time, so a
    time gate must be applied; False means a unique 5-tuple accepted directly).
    """
    frame: pd.DataFrame


def load_gt(gt_dir: Path | str = GT_DIR) -> pd.DataFrame:
    """Load + concatenate all GroundTruth_Network_*.csv into one lookup frame
    with columns ``gt_ts, gt_proto, key, type``."""
    gt_dir = Path(gt_dir)
    files = sorted(gt_dir.glob("GroundTruth_Network_*.csv"))
    if not files:
        raise FileNotFoundError(f"no GroundTruth_Network_*.csv under {gt_dir}")

    frames = []
    for f in files:
        # utf-8-sig strips the leading BOM on the ``ts`` header.
        df = pd.read_csv(
            f, encoding="utf-8-sig",
            dtype={"src_ip": "string", "dst_ip": "string", "proto": "string",
                   "type": "string"},
            usecols=["ts", "src_ip", "src_port", "dst_ip", "dst_port", "proto", "type"],
        )
        frames.append(df)
    gt = pd.concat(frames, ignore_index=True)

    gt["proto"] = gt["proto"].str.strip().str.lower()
    gt["type"] = gt["type"].str.strip().str.lower()
    gt_proto = gt["proto"].map(_PROTO_NAME_TO_INT).fillna(-1).astype("int32")

    # Ports can be non-numeric/blank for ICMP rows; coerce to int with 0 fallback.
    src_port = pd.to_numeric(gt["src_port"], errors="coerce").fillna(0).astype("int64")
    dst_port = pd.to_numeric(gt["dst_port"], errors="coerce").fillna(0).astype("int64")

    key = _canonical_key(
        gt["src_ip"].to_numpy(), src_port.to_numpy(),
        gt["dst_ip"].to_numpy(), dst_port.to_numpy(),
        gt_proto.to_numpy(),
    )

    out = pd.DataFrame({
        "gt_ts": gt["ts"].astype("float64"),
        "gt_proto": gt_proto,
        "key": pd.Series(key, dtype="object"),
        "type": gt["type"].astype("string"),
    })
    out = out.dropna(subset=["gt_ts", "type"]).reset_index(drop=True)
    return out


def build_gt_index(gt_df: pd.DataFrame | None = None) -> GtIndex:
    """Build the reusable GT index ONCE. Reuse across all label_flows calls."""
    if gt_df is None:
        gt_df = load_gt()
    frame = gt_df[["key", "gt_ts", "type"]].copy()
    # key_dup: True where the canonical key appears more than once (needs time
    # gate); False where unique (accept directly).
    frame["key_dup"] = frame["key"].duplicated(keep=False)
    return GtIndex(frame=frame)


# ---------------------------------------------------------------------------
# Flow labeling (vectorized merge)
# ---------------------------------------------------------------------------


def label_flows(
    flows_df: pd.DataFrame,
    gt_df: pd.DataFrame | None = None,
    *,
    gt_index: GtIndex | None = None,
    time_tolerance_s: float = 5.0,
) -> pd.DataFrame:
    """Add ``Label_ToNIoT``, ``label_encoded``, ``matched`` and ``match_level``.

    Matching (L0, exact canonical 5-tuple, direction-independent): each flow is
    merged against GT rows sharing its key. Among candidates the closest
    connection start is taken; it is accepted if the key is unique in the GT
    (``key_dup==False``) or the time gap is within ``time_tolerance_s``.
    Unmatched flows are ``normal``.

    Pass a prebuilt ``gt_index`` (from ``build_gt_index``) to avoid re-indexing
    the 17.8M-row GT on every call — essential for the pipeline's 40 calls.

    ``flows_df`` must carry the columns produced by ``aggregate_flow_features``:
    ``flow_id, flow_src_ip, flow_dst_ip, flow_src_port, flow_dst_port,
    Protocol, flow_start_ts``.

    ``match_level``: 0 = normal (unmatched), 1 = L0 exact match.
    """
    needed = {"flow_id", "flow_src_ip", "flow_dst_ip", "flow_src_port",
              "flow_dst_port", "Protocol", "flow_start_ts"}
    missing = needed.difference(flows_df.columns)
    if missing:
        raise KeyError(f"flows_df missing required columns: {sorted(missing)}")

    if gt_index is None:
        gt_index = build_gt_index(gt_df)
    g = gt_index.frame

    out = flows_df.reset_index(drop=True).copy()
    out["_row"] = np.arange(len(out), dtype="int64")
    fkey = _canonical_key(
        out["flow_src_ip"].to_numpy(), out["flow_src_port"].to_numpy(),
        out["flow_dst_ip"].to_numpy(), out["flow_dst_port"].to_numpy(),
        out["Protocol"].to_numpy(),
    )
    left = pd.DataFrame({
        "_row": out["_row"].to_numpy(),
        "key": pd.Series(fkey, dtype="object"),
        "flow_start_ts": out["flow_start_ts"].to_numpy(dtype="float64"),
    })

    # Inner-merge flows against GT on the canonical key. Most GT keys are unique,
    # so the result is ~the number of matched flows — small and bounded.
    merged = left.merge(g, on="key", how="inner", copy=False)

    labels = np.full(len(out), NORMAL_LABEL, dtype=object)
    match_level = np.zeros(len(out), dtype="int8")

    if len(merged):
        gap = np.abs(merged["gt_ts"].to_numpy(dtype="float64")
                     - merged["flow_start_ts"].to_numpy(dtype="float64"))
        merged = merged.assign(_gap=gap)
        # Accept rule: unique key (not dup) OR within tolerance.
        ok = (~merged["key_dup"].to_numpy()) | (merged["_gap"].to_numpy() <= time_tolerance_s)
        merged = merged.loc[ok]
        if len(merged):
            # Keep the closest candidate per flow row.
            merged = merged.sort_values("_gap", kind="mergesort")
            merged = merged.drop_duplicates("_row", keep="first")
            rows = merged["_row"].to_numpy()
            labels[rows] = merged["type"].to_numpy().astype(object)
            match_level[rows] = 1

    out = out.drop(columns=["_row"])
    out["Label_ToNIoT"] = pd.Series(labels, index=out.index, dtype="string")
    out["matched"] = out["Label_ToNIoT"].ne(NORMAL_LABEL)
    out["match_level"] = pd.Series(match_level, index=out.index, dtype="int8")
    cls_to_id = {c: i for i, c in enumerate(ATTACK_CLASSES)}
    out["label_encoded"] = (
        out["Label_ToNIoT"].map(cls_to_id).fillna(NORMAL_LABEL_ENCODED).astype("int16")
    )
    return out


def coverage_summary(labeled_flows: pd.DataFrame) -> dict:
    n_total = int(len(labeled_flows))
    n_matched = int(labeled_flows["matched"].sum())
    return {
        "n_total": n_total,
        "n_matched": n_matched,
        "n_unmatched": n_total - n_matched,
        "pct_matched": (100.0 * n_matched / n_total) if n_total else 0.0,
        "label_counts": {
            str(k): int(v)
            for k, v in labeled_flows["Label_ToNIoT"].value_counts(dropna=False).items()
        },
    }
