"""Minimal loader for CIC-IDS2017 raw_flow parquets.

Mirrors the subset of the 2018 loader that the training scripts actually use.
"""
from __future__ import annotations

from pathlib import Path
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_DATA_ROOT = _REPO_ROOT / "data" / "CIC-IDS2017"
PARTIAL_FLOW_ROOT = _DATA_ROOT / "partial_flow"


def raw_flow_path(day: str, cut_dir: str) -> Path:
    return PARTIAL_FLOW_ROOT / cut_dir / "raw_flow" / f"{day}.parquet"


def load_partial_flow_raw(day: str, cut_dir: str, columns: list[str] | None = None) -> pd.DataFrame:
    p = raw_flow_path(day, cut_dir)
    if not p.exists():
        raise FileNotFoundError(f"Raw flow not found: {p}")
    return pd.read_parquet(p, columns=columns, engine="pyarrow")
