"""Train the raw-byte BiGRU on CIC-IoT2023 (basic vs strict mask, 34/8-class).

Example
-------
python scripts/cic_iot2023/train_byte_bigru.py \
    --raw-bytes-dir data/CIC-IoT2023/partial_flow/raw_bytes_pkt5_b256_strict/raw_bytes \
    --label-mode group
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2017.byte_bigru import ByteBiGRU  # noqa: E402
from src.cic_iot2023.byte_training import add_common_args, run_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, default_out="outputs/cic_iot2023/byte_bigru/pkt5_strict")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=1)
    args = parser.parse_args()

    def build(n_classes: int) -> ByteBiGRU:
        return ByteBiGRU(n_classes=n_classes, n_bytes=args.n_bytes, max_pkts=args.max_pkts,
                         d_model=args.d_model, hidden_size=args.hidden_size,
                         n_layers=args.n_layers, dropout=args.dropout)

    return run_training("byte_bigru", build, args)


if __name__ == "__main__":
    raise SystemExit(main())
