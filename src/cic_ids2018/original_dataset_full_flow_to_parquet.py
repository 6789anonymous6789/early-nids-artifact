"""
original_dataset_full_flow_to_parquet.py

Convert CIC-IDS2018 full-flow CSVs to Parquet (snappy) for fast downstream loading.

Background
----------
CICFlowMeter-V3 exports flow features as CSV. Several known artefacts require cleaning
before the files are useful:

  1. Duplicate header rows — CICFlowMeter sometimes re-emits the column header mid-file.
     These rows are detected by checking whether the first-column value equals the column
     name, then dropped.

  2. String-typed numeric columns — reading with dtype=str preserves all values faithfully;
     non-numeric entries (the duplicate headers) are dropped before casting.

After cleaning, files are written as Parquet with Snappy compression via PyArrow.
Snappy offers fast decompression at moderate file size — the right tradeoff for iterative
ML experimentation where the same file is loaded many times.

Reference
---------
Sharafaldin, I., Lashkari, A.H., Ghorbani, A.A. (2018). Toward Generating a New
Intrusion Detection Dataset and Intrusion Traffic Characterization. ICISSP 2018.

Usage
-----
  # From the thesis/ root:
  python src/original_dataset_full_flow_to_parquet.py
  python src/original_dataset_full_flow_to_parquet.py --data-dir data/CIC-IDS2018/original/full_flow --out-dir data/CIC-IDS2018/parquet/full_flow_original
"""

import argparse
from pathlib import Path

import pandas as pd


def convert_csv_to_parquet(csv_path: Path, out_path: Path) -> dict:
    """Read one CSV, clean it, and write a Parquet file. Returns a stats dict."""
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # Drop duplicate header rows emitted by CICFlowMeter mid-file
    col0 = df.columns[0]
    dup_rows = (df[col0] == col0).sum()
    df = df[df[col0] != col0].reset_index(drop=True)

    # Cast all columns except Label to numeric; coerce errors to NaN (surfaced in EDA)
    feature_cols = [c for c in df.columns if c != "Label"]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, engine="pyarrow", compression="snappy", index=False)

    csv_bytes = csv_path.stat().st_size
    parquet_bytes = out_path.stat().st_size
    return {
        "file": csv_path.name,
        "rows": len(df),
        "columns": len(df.columns),
        "dup_header_rows_dropped": int(dup_rows),
        "csv_mb": round(csv_bytes / 1e6, 1),
        "parquet_mb": round(parquet_bytes / 1e6, 1),
        "compression_ratio": round(csv_bytes / parquet_bytes, 2),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Convert CIC-IDS2018 full-flow CSVs to Parquet (snappy)."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/CIC-IDS2018/original/full_flow"),
        help="Directory containing the 10 day CSVs (default: data/CIC-IDS2018/original/full_flow)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/CIC-IDS2018/parquet/full_flow_original"),
        help="Output directory for Parquet files (default: data/CIC-IDS2018/parquet/full_flow_original)",
    )
    args = parser.parse_args()

    csv_files = sorted(args.data_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {args.data_dir}")

    print(f"Found {len(csv_files)} CSV files in {args.data_dir}")
    print(f"Output directory: {args.out_dir}\n")

    records = []
    for csv_path in csv_files:
        print(f"  Converting {csv_path.name} ...", flush=True)
        out_path = args.out_dir / (csv_path.stem + ".parquet")
        info = convert_csv_to_parquet(csv_path, out_path)
        records.append(info)
        print(
            f"    {info['rows']:,} rows | "
            f"{info['csv_mb']:.0f} MB -> {info['parquet_mb']:.0f} MB "
            f"({info['compression_ratio']}x) | "
            f"dup headers dropped: {info['dup_header_rows_dropped']}"
        )

    manifest = pd.DataFrame(records)
    manifest_path = args.out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print(f"\nDone. Manifest written to {manifest_path}")
    print(manifest.to_string(index=False))


if __name__ == "__main__":
    main()
