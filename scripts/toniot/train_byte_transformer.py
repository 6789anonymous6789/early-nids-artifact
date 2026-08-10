"""Train the raw-byte mini-Transformer on ToN-IoT (basic vs strict mask, flat 10-class).

Extraction is done once at K=5; train a smaller cut by setting --max-pkts (the
loader truncates to the first --max-pkts packets).

Example
-------
python scripts/toniot/train_byte_transformer.py \
    --raw-bytes-dir data/ToN-IoT/partial_flow/raw_bytes_pkt5_b256_strict/raw_bytes \
    --max-pkts 5 --out-dir outputs/toniot/byte_transformer/pkt5_strict
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.cic_ids2017.byte_transformer import ByteTransformer  # noqa: E402
from src.toniot.byte_training import add_common_args, run_training  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, default_out="outputs/toniot/byte_transformer/pkt5_strict")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--dim-feedforward", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    args = parser.parse_args()

    def build(n_classes: int) -> ByteTransformer:
        return ByteTransformer(n_classes=n_classes, n_bytes=args.n_bytes, max_pkts=args.max_pkts,
                               d_model=args.d_model, nhead=args.nhead,
                               dim_feedforward=args.dim_feedforward,
                               n_layers=args.n_layers, dropout=args.dropout)

    return run_training("byte_transformer", build, args)


if __name__ == "__main__":
    raise SystemExit(main())
