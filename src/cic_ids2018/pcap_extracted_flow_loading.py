"""Load packet-base and raw flow-view artifacts produced by the PCAP pipeline."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


_REPO_ROOT = Path(__file__).parent.parent.parent
_DATA_ROOT = _REPO_ROOT / "data" / "CIC-IDS2018"
PARTIAL_FLOW_ROOT = _DATA_ROOT / "partial_flow"
PACKET_BASE_DIR = PARTIAL_FLOW_ROOT / "packet_base"
MANIFEST_DIR = PARTIAL_FLOW_ROOT / "manifests"

DAY_ALIASES = {
    "Thuesday-20-02-2018": "Tuesday-20-02-2018",
    "Tuesday-20-02-2018": "Tuesday-20-02-2018",
}


def normalize_day_name(day: str) -> str:
    return DAY_ALIASES.get(day, day)


def observation_tag(pct: int | float = 100) -> str:
    if pct > 1:
        value = int(round(float(pct)))
    else:
        value = int(round(float(pct) * 100))
    return f"pct_{value:03d}"


def packet_base_path(day: str) -> Path:
    return PACKET_BASE_DIR / f"{normalize_day_name(day)}.parquet"


def raw_flow_path(
    day: str,
    pct: int | float = 100,
    cut_dir: str | None = None,
) -> Path:
    """Path to the raw-flow parquet for ``day``.

    If ``cut_dir`` is provided it overrides the ``pct`` tag and is used as the
    ``partial_flow`` subdirectory name (e.g. ``"time_abs_1s"``, ``"packet_abs_3"``,
    or a custom ``"pct_050"``). Falls back to ``observation_tag(pct)`` otherwise.
    """
    tag = cut_dir if cut_dir is not None else observation_tag(pct)
    return PARTIAL_FLOW_ROOT / tag / "raw_flow" / f"{normalize_day_name(day)}.parquet"


def manifest_path(day: str) -> Path:
    return MANIFEST_DIR / f"{normalize_day_name(day)}.json"


def load_packet_base(day: str, columns: list[str] | None = None) -> pd.DataFrame:
    path = packet_base_path(day)
    if not path.exists():
        raise FileNotFoundError(f"Packet-base artifact not found: {path}")
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def load_partial_flow_raw(
    day: str,
    pct: int | float = 100,
    columns: list[str] | None = None,
    cut_dir: str | None = None,
) -> pd.DataFrame:
    path = raw_flow_path(day, pct=pct, cut_dir=cut_dir)
    if not path.exists():
        raise FileNotFoundError(f"Raw flow artifact not found: {path}")
    return pd.read_parquet(path, columns=columns, engine="pyarrow")


def load_day_manifest(day: str) -> dict:
    path = manifest_path(day)
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    with open(path) as f:
        return json.load(f)


def list_available_days(
    pct: int | float = 100,
    include_untrainable: bool = True,
    cut_dir: str | None = None,
) -> list[str]:
    tag = cut_dir if cut_dir is not None else observation_tag(pct)
    raw_dir = PARTIAL_FLOW_ROOT / tag / "raw_flow"
    if not raw_dir.exists():
        return []

    days = []
    for path in sorted(raw_dir.glob("*.parquet")):
        day = path.stem
        if include_untrainable:
            days.append(day)
            continue

        manifest = load_day_manifest(day)
        if manifest.get("labeling", {}).get("training_ready", False):
            days.append(day)
    return days


def load_partial_flow_days(
    days: list[str] | None = None,
    pct: int | float = 100,
    include_untrainable: bool = True,
    as_dict: bool = False,
    columns: list[str] | None = None,
) -> dict[str, pd.DataFrame] | pd.DataFrame:
    if days is None:
        days = list_available_days(pct=pct, include_untrainable=include_untrainable)
    else:
        days = [normalize_day_name(day) for day in days]
        if not include_untrainable:
            days = [
                day for day in days
                if load_day_manifest(day).get("labeling", {}).get("training_ready", False)
            ]

    day_frames = {day: load_partial_flow_raw(day, pct=pct, columns=columns) for day in days}
    if as_dict:
        return day_frames
    if not day_frames:
        return pd.DataFrame()
    return pd.concat(day_frames.values(), ignore_index=True)
