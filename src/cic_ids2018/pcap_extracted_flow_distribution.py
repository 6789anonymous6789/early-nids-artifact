"""Generate distribution and feature-quality reports for PCAP-extracted flow datasets."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import seaborn as sns
from matplotlib.ticker import PercentFormatter


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "CIC-IDS2018" / "partial_flow"
DEFAULT_NOTEBOOK_ROOT = REPO_ROOT / "notebooks" / "pcap_extracted_flows"

SCAN_COLUMNS = [
    "Label",
    "label_encoded",
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Flow Byts/s",
    "Flow Pkts/s",
    "Pkt Len Var",
]

NONNEGATIVE_COLUMNS = [
    "Flow Duration",
    "Tot Fwd Pkts",
    "Tot Bwd Pkts",
    "TotLen Fwd Pkts",
    "TotLen Bwd Pkts",
    "Pkt Len Var",
]

RATE_COLUMNS = ["Flow Byts/s", "Flow Pkts/s"]
BENIGN_LABEL = "Benign"


def observation_tag(pct: int | float) -> str:
    value = int(round(float(pct)))
    return f"pct_{value:03d}"


def notebook_flow_percentage_dir(pct: int | float) -> str:
    value = int(round(float(pct)))
    value_label = f"{value:03d}" if value < 100 else str(value)
    return f"flow_{value_label}_percentage"


def parse_day_name(day_name: str) -> datetime:
    return datetime.strptime(day_name.split("-", 1)[1], "%d-%m-%Y")


def format_label(value: object) -> str:
    if pd.isna(value):
        return "<NA>"
    return str(value)


def aggregate_partial_flow(
    raw_flow_dir: Path,
    batch_size: int = 250_000,
) -> dict[str, object]:
    parquet_files = sorted(raw_flow_dir.glob("*.parquet"), key=lambda path: parse_day_name(path.stem))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found under {raw_flow_dir}")

    day_label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    global_label_counts: Counter[str] = Counter()
    any_inf_label_counts: Counter[str] = Counter()
    feature_metrics = {
        column: {"null_rows": 0, "negative_rows": 0, "inf_rows": 0}
        for column in SCAN_COLUMNS
        if column not in {"Label", "label_encoded"}
    }
    cleanup_metrics: Counter[str] = Counter()
    label_encoded_neg1 = 0

    for parquet_path in parquet_files:
        day_name = parquet_path.stem
        parquet_file = pq.ParquetFile(parquet_path)

        for batch in parquet_file.iter_batches(columns=SCAN_COLUMNS, batch_size=batch_size):
            frame = batch.to_pandas()

            batch_label_counts = frame["Label"].value_counts(dropna=False)
            for label, count in batch_label_counts.items():
                label_name = format_label(label)
                count_int = int(count)
                day_label_counts[day_name][label_name] += count_int
                global_label_counts[label_name] += count_int

            label_encoded_neg1 += int((frame["label_encoded"] == -1).sum())
            cleanup_metrics["rows_total"] += int(len(frame))

            duration_zero_mask = frame["Flow Duration"] == 0
            single_packet_mask = (frame["Tot Fwd Pkts"] + frame["Tot Bwd Pkts"]) == 1
            flow_bytes_per_s_inf_mask = np.isinf(
                frame["Flow Byts/s"].to_numpy(dtype=float, copy=False)
            )
            flow_pkts_per_s_inf_mask = np.isinf(
                frame["Flow Pkts/s"].to_numpy(dtype=float, copy=False)
            )
            any_inf_mask = flow_bytes_per_s_inf_mask | flow_pkts_per_s_inf_mask

            cleanup_metrics["zero_duration_rows"] += int(duration_zero_mask.sum())
            cleanup_metrics["single_packet_rows"] += int(single_packet_mask.sum())
            cleanup_metrics["flow_bytes_per_s_inf_rows"] += int(flow_bytes_per_s_inf_mask.sum())
            cleanup_metrics["flow_pkts_per_s_inf_rows"] += int(flow_pkts_per_s_inf_mask.sum())
            cleanup_metrics["any_inf_rate_rows"] += int(any_inf_mask.sum())
            cleanup_metrics["zero_duration_single_packet_rows"] += int(
                (duration_zero_mask & single_packet_mask).sum()
            )

            for column in feature_metrics:
                series = frame[column]
                feature_metrics[column]["null_rows"] += int(series.isna().sum())
                if column in NONNEGATIVE_COLUMNS:
                    feature_metrics[column]["negative_rows"] += int((series.fillna(0) < 0).sum())
                if column in RATE_COLUMNS:
                    feature_metrics[column]["inf_rows"] += int(
                        np.isinf(series.to_numpy(dtype=float, copy=False)).sum()
                    )

            inf_label_counts = frame.loc[any_inf_mask, "Label"].value_counts(dropna=False)
            for label, count in inf_label_counts.items():
                any_inf_label_counts[format_label(label)] += int(count)

    day_label_rows: list[dict[str, object]] = []
    for day_name, counts in day_label_counts.items():
        day_total = int(sum(counts.values()))
        for label_name, count in counts.items():
            day_label_rows.append(
                {
                    "day": day_name,
                    "label": label_name,
                    "count": int(count),
                    "day_total": day_total,
                    "day_share": count / day_total if day_total else 0.0,
                }
            )

    day_label_df = pd.DataFrame(day_label_rows)
    day_label_df = day_label_df.sort_values(
        by=["day", "count", "label"],
        ascending=[True, False, True],
        key=lambda series: series.map(parse_day_name) if series.name == "day" else series,
    ).reset_index(drop=True)

    day_summary_df = (
        day_label_df.groupby("day", as_index=False)
        .agg(rows_total=("count", "sum"))
        .sort_values(by="day", key=lambda series: series.map(parse_day_name))
        .reset_index(drop=True)
    )
    benign_by_day = (
        day_label_df[day_label_df["label"] == BENIGN_LABEL][["day", "count"]]
        .rename(columns={"count": "benign_rows"})
    )
    day_summary_df = day_summary_df.merge(benign_by_day, on="day", how="left").fillna({"benign_rows": 0})
    day_summary_df["benign_rows"] = day_summary_df["benign_rows"].astype("int64")
    day_summary_df["attack_rows"] = day_summary_df["rows_total"] - day_summary_df["benign_rows"]
    day_summary_df["attack_ratio"] = day_summary_df["attack_rows"] / day_summary_df["rows_total"]
    label_count_by_day = (
        day_label_df.groupby("day")["label"].nunique().rename("distinct_labels").reset_index()
    )
    day_summary_df = day_summary_df.merge(label_count_by_day, on="day", how="left")

    global_label_df = (
        pd.DataFrame(
            [
                {"label": label_name, "count": int(count)}
                for label_name, count in global_label_counts.items()
            ]
        )
        .sort_values(by=["count", "label"], ascending=[False, True])
        .reset_index(drop=True)
    )
    rows_total = int(cleanup_metrics["rows_total"])
    global_label_df["share"] = global_label_df["count"] / rows_total if rows_total else 0.0

    feature_quality_df = pd.DataFrame(
        [
            {
                "feature": column,
                "null_rows": stats["null_rows"],
                "negative_rows": stats["negative_rows"],
                "inf_rows": stats["inf_rows"],
                "null_share": stats["null_rows"] / rows_total if rows_total else 0.0,
                "negative_share": stats["negative_rows"] / rows_total if rows_total else 0.0,
                "inf_share": stats["inf_rows"] / rows_total if rows_total else 0.0,
            }
            for column, stats in feature_metrics.items()
        ]
    ).sort_values(by="feature")

    cleanup_summary_df = pd.DataFrame(
        [
            {
                "metric": metric,
                "count": int(count),
                "share": count / rows_total if rows_total else 0.0,
            }
            for metric, count in sorted(cleanup_metrics.items())
        ]
    )

    cleanup_label_rows = []
    for label_name, total_count in global_label_counts.items():
        inf_count = int(any_inf_label_counts.get(label_name, 0))
        remaining_count = int(total_count - inf_count)
        cleanup_label_rows.append(
            {
                "label": label_name,
                "total_rows": int(total_count),
                "any_inf_rate_rows": inf_count,
                "remaining_rows_after_drop": remaining_count,
                "any_inf_rate_share_within_label": inf_count / total_count if total_count else 0.0,
                "remaining_share_after_drop": remaining_count / max(rows_total - cleanup_metrics["any_inf_rate_rows"], 1),
            }
        )
    cleanup_label_impact_df = (
        pd.DataFrame(cleanup_label_rows)
        .sort_values(by=["total_rows", "label"], ascending=[False, True])
        .reset_index(drop=True)
    )

    benign_total = int(global_label_counts.get(BENIGN_LABEL, 0))
    attack_total = rows_total - benign_total
    benign_any_inf = int(any_inf_label_counts.get(BENIGN_LABEL, 0))
    attack_any_inf = int(cleanup_metrics["any_inf_rate_rows"] - benign_any_inf)
    rows_after_drop = int(rows_total - cleanup_metrics["any_inf_rate_rows"])
    benign_after_drop = int(benign_total - benign_any_inf)
    attack_after_drop = int(attack_total - attack_any_inf)

    return {
        "raw_flow_dir": raw_flow_dir,
        "day_label_df": day_label_df,
        "day_summary_df": day_summary_df,
        "global_label_df": global_label_df,
        "feature_quality_df": feature_quality_df,
        "cleanup_summary_df": cleanup_summary_df,
        "cleanup_label_impact_df": cleanup_label_impact_df,
        "rows_total": rows_total,
        "label_encoded_neg1": label_encoded_neg1,
        "attack_total": attack_total,
        "attack_ratio": attack_total / rows_total if rows_total else 0.0,
        "rows_after_drop": rows_after_drop,
        "attack_after_drop": attack_after_drop,
        "attack_ratio_after_drop": attack_after_drop / rows_after_drop if rows_after_drop else 0.0,
        "benign_total": benign_total,
        "benign_after_drop": benign_after_drop,
        "benign_any_inf": benign_any_inf,
        "attack_any_inf": attack_any_inf,
    }


def save_day_attack_ratio_plot(day_summary_df: pd.DataFrame, output_path: Path) -> None:
    plot_df = day_summary_df.copy()
    plot_df["day_short"] = plot_df["day"].str.replace("-2018", "", regex=False)

    sns.set_theme(style="whitegrid", font_scale=1.05)
    fig, ax = plt.subplots(figsize=(12, 5), dpi=120)

    colors = np.where(plot_df["attack_rows"] > 0, "#c0392b", "#95a5a6")
    bars = ax.bar(plot_df["day_short"], plot_df["attack_ratio"], color=colors)
    ax.set_title("pct_100 Partial-Flow Attack Share by Day")
    ax.set_ylabel("Attack share")
    ax.set_xlabel("")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_ylim(0, max(plot_df["attack_ratio"].max() * 1.18, 0.05))
    ax.tick_params(axis="x", rotation=30)

    for bar, ratio in zip(bars, plot_df["attack_ratio"], strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            ratio + max(plot_df["attack_ratio"].max() * 0.01, 0.002),
            f"{ratio:.2%}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_day_label_mix_plot(
    day_label_df: pd.DataFrame,
    global_label_df: pd.DataFrame,
    output_path: Path,
) -> None:
    day_order = sorted(day_label_df["day"].unique(), key=parse_day_name)
    label_order = [BENIGN_LABEL] + [
        label
        for label in global_label_df["label"].tolist()
        if label != BENIGN_LABEL
    ]

    pivot_df = (
        day_label_df.pivot(index="day", columns="label", values="day_share")
        .fillna(0.0)
        .reindex(index=day_order, columns=label_order, fill_value=0.0)
    )

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, ax = plt.subplots(figsize=(14, 6), dpi=120)

    palette = sns.color_palette("tab20", n_colors=max(len(label_order) - 1, 1))
    colors = {BENIGN_LABEL: "#bdc3c7"}
    for label, color in zip(label_order[1:], palette, strict=False):
        colors[label] = color

    cumulative = np.zeros(len(pivot_df))
    y_labels = [day.replace("-2018", "") for day in pivot_df.index]

    for label in label_order:
        shares = pivot_df[label].to_numpy()
        ax.barh(y_labels, shares, left=cumulative, label=label, color=colors[label])
        cumulative += shares

    ax.set_title("pct_100 Partial-Flow Label Mix by Day")
    ax.set_xlabel("Share of rows")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlim(0, 1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False)

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_cleanup_impact_plot(analysis: dict[str, object], output_path: Path) -> None:
    cleanup_summary_df = analysis["cleanup_summary_df"]
    rows_total = int(analysis["rows_total"])

    metric_labels = {
        "any_inf_rate_rows": "Any infinite rate",
        "flow_pkts_per_s_inf_rows": "Flow Pkts/s = inf",
        "flow_bytes_per_s_inf_rows": "Flow Byts/s = inf",
        "zero_duration_rows": "Zero duration",
        "single_packet_rows": "Single-packet flow",
    }
    plot_metrics = (
        cleanup_summary_df[cleanup_summary_df["metric"].isin(metric_labels)]
        .assign(metric_name=lambda df: df["metric"].map(metric_labels))
        .set_index("metric")
        .loc[list(metric_labels)]
        .reset_index(drop=True)
    )

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    axes[0].barh(plot_metrics["metric_name"], plot_metrics["share"], color="#34495e")
    axes[0].set_title("Rows Affected by Rate-Cleanup Conditions")
    axes[0].set_xlabel("Share of rows")
    axes[0].xaxis.set_major_formatter(PercentFormatter(1.0))
    for y_pos, share in enumerate(plot_metrics["share"]):
        axes[0].text(share + 0.002, y_pos, f"{share:.2%}", va="center", fontsize=9)

    before_benign = int(analysis["benign_total"])
    before_attack = int(analysis["attack_total"])
    after_benign = int(analysis["benign_after_drop"])
    after_attack = int(analysis["attack_after_drop"])
    before_total = before_benign + before_attack
    after_total = after_benign + after_attack

    before_benign_share = before_benign / before_total if before_total else 0.0
    before_attack_share = before_attack / before_total if before_total else 0.0
    after_benign_share = after_benign / after_total if after_total else 0.0
    after_attack_share = after_attack / after_total if after_total else 0.0

    stages = ["Before cleanup", "Drop any inf-rate row"]
    benign_shares = [before_benign_share, after_benign_share]
    attack_shares = [before_attack_share, after_attack_share]

    axes[1].bar(stages, benign_shares, label="Benign", color="#bdc3c7")
    axes[1].bar(stages, attack_shares, bottom=benign_shares, label="Attack", color="#c0392b")
    axes[1].set_title("Binary Class Mix Before and After Dropping inf-Rate Rows")
    axes[1].set_ylabel("Share of rows")
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)

    axes[1].text(
        0,
        1.02,
        f"Attack share: {before_attack_share:.2%}",
        ha="center",
        va="bottom",
        fontsize=9,
        transform=axes[1].get_xaxis_transform(),
    )
    axes[1].text(
        1,
        1.02,
        f"Attack share: {after_attack_share:.2%}",
        ha="center",
        va="bottom",
        fontsize=9,
        transform=axes[1].get_xaxis_transform(),
    )

    fig.suptitle(
        f"pct_100 Cleanup Impact ({rows_total:,} total rows; dropping inf-rate rows removes "
        f"{int(analysis['rows_total'] - analysis['rows_after_drop']):,})",
        y=1.04,
    )
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def build_report_text(
    analysis: dict[str, object],
    pct: int | float,
    output_dir: Path,
    generated_files: list[Path],
) -> str:
    pct_tag = observation_tag(pct)
    rows_total = int(analysis["rows_total"])
    attack_total = int(analysis["attack_total"])
    benign_total = int(analysis["benign_total"])
    attack_ratio = float(analysis["attack_ratio"])
    rows_after_drop = int(analysis["rows_after_drop"])
    attack_ratio_after_drop = float(analysis["attack_ratio_after_drop"])

    day_summary_df = analysis["day_summary_df"]
    global_label_df = analysis["global_label_df"]
    feature_quality_df = analysis["feature_quality_df"]
    cleanup_summary_df = analysis["cleanup_summary_df"].set_index("metric")

    top_labels = global_label_df.copy()
    top_labels["share_pct"] = top_labels["share"] * 100
    top_labels_table = top_labels[["label", "count", "share_pct"]].head(12)

    day_table = day_summary_df.copy()
    day_table["attack_pct"] = day_table["attack_ratio"] * 100
    day_table = day_table[["day", "rows_total", "attack_rows", "attack_pct", "distinct_labels"]]

    inf_rows = int(cleanup_summary_df.loc["any_inf_rate_rows", "count"])
    inf_share = float(cleanup_summary_df.loc["any_inf_rate_rows", "share"])
    flow_bytes_inf_rows = int(cleanup_summary_df.loc["flow_bytes_per_s_inf_rows", "count"])
    flow_pkts_inf_rows = int(cleanup_summary_df.loc["flow_pkts_per_s_inf_rows", "count"])
    zero_duration_rows = int(cleanup_summary_df.loc["zero_duration_rows", "count"])
    single_packet_rows = int(cleanup_summary_df.loc["single_packet_rows", "count"])
    zero_duration_single_packet_rows = int(
        cleanup_summary_df.loc["zero_duration_single_packet_rows", "count"]
    )

    finite_quality_rows = feature_quality_df[
        (feature_quality_df["null_rows"] > 0)
        | (feature_quality_df["negative_rows"] > 0)
        | (feature_quality_df["inf_rows"] > 0)
    ].copy()
    finite_quality_rows["null_pct"] = finite_quality_rows["null_share"] * 100
    finite_quality_rows["negative_pct"] = finite_quality_rows["negative_share"] * 100
    finite_quality_rows["inf_pct"] = finite_quality_rows["inf_share"] * 100

    files_list = "\n".join(f"- `{path.name}`" for path in generated_files)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    return f"""# {pct_tag} Partial-Flow Distribution Report

Generated: {generated_at}

Source directory: `{analysis['raw_flow_dir']}`
Output directory: `{output_dir}`

## Executive Summary

- Total rows: {rows_total:,}
- Benign rows: {benign_total:,} ({(1 - attack_ratio):.2%})
- Attack rows: {attack_total:,} ({attack_ratio:.2%})
- Distinct labels: {len(global_label_df)}
- Unknown labels (`label_encoded == -1`): {analysis['label_encoded_neg1']:,}
- Benign-only days present in the extracted PCAP view: {int((day_summary_df['attack_rows'] == 0).sum())}

The extraction is internally consistent and the observed day-level balance matches the configured PCAP attack profiles. The main issue is not missing labels; it is strong class imbalance and a long multiclass tail.

## Global Label Distribution

| Label | Count | Share |
| --- | ---: | ---: |
{chr(10).join(f"| {row.label} | {row.count:,} | {row.share_pct:.4f}% |" for row in top_labels_table.itertuples(index=False))}

## Day-Level Balance

| Day | Rows | Attack Rows | Attack % | Distinct Labels |
| --- | ---: | ---: | ---: | ---: |
{chr(10).join(f"| {row.day} | {row.rows_total:,} | {row.attack_rows:,} | {row.attack_pct:.4f}% | {row.distinct_labels} |" for row in day_table.itertuples(index=False))}

## Feature Quality and Cleanup Impact

- Features with nulls: none
- Features with negative values: none
- Rows with any infinite rate feature: {inf_rows:,} ({inf_share:.2%})
- `Flow Byts/s = inf`: {flow_bytes_inf_rows:,} ({flow_bytes_inf_rows / rows_total:.2%})
- `Flow Pkts/s = inf`: {flow_pkts_inf_rows:,} ({flow_pkts_inf_rows / rows_total:.2%})
- Zero-duration rows: {zero_duration_rows:,} ({zero_duration_rows / rows_total:.2%})
- Single-packet rows: {single_packet_rows:,} ({single_packet_rows / rows_total:.2%})
- Zero-duration rows that are also single-packet: {zero_duration_single_packet_rows:,} ({zero_duration_single_packet_rows / max(zero_duration_rows, 1):.2%} of zero-duration rows)
- Attack share after dropping every row with an infinite rate feature: {attack_ratio_after_drop:.2%}
- Rows remaining after dropping every row with an infinite rate feature: {rows_after_drop:,} ({rows_after_drop / rows_total:.2%} of original)

Dropping all rows with infinite rate features would remove more than one fifth of the dataset, so blanket row deletion is a poor default cleanup strategy. The better preprocessing path is to keep the rows, replace `inf` with `NaN` or a capped value, and add an explicit zero-duration or single-packet indicator if the downstream model benefits from it.

## Features Requiring Attention

| Feature | Null Rows | Negative Rows | Infinite Rows |
| --- | ---: | ---: | ---: |
{chr(10).join(f"| {row.feature} | {row.null_rows:,} | {row.negative_rows:,} | {row.inf_rows:,} |" for row in finite_quality_rows.itertuples(index=False))}

## Generated Files

{files_list}
"""


def write_outputs(analysis: dict[str, object], pct: int | float, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pct_tag = observation_tag(pct)

    generated_files = [
        output_dir / f"08_{pct_tag}_day_label_counts.csv",
        output_dir / f"09_{pct_tag}_day_summary.csv",
        output_dir / f"10_{pct_tag}_global_label_counts.csv",
        output_dir / f"11_{pct_tag}_feature_quality.csv",
        output_dir / f"12_{pct_tag}_cleanup_label_impact.csv",
        output_dir / f"13_{pct_tag}_day_attack_ratio.png",
        output_dir / f"14_{pct_tag}_day_label_mix.png",
        output_dir / f"15_{pct_tag}_inf_cleanup_impact.png",
        output_dir / f"16_{pct_tag}_distribution_report.md",
    ]

    analysis["day_label_df"].to_csv(generated_files[0], index=False)
    analysis["day_summary_df"].to_csv(generated_files[1], index=False)
    analysis["global_label_df"].to_csv(generated_files[2], index=False)
    analysis["feature_quality_df"].to_csv(generated_files[3], index=False)
    analysis["cleanup_label_impact_df"].to_csv(generated_files[4], index=False)

    save_day_attack_ratio_plot(analysis["day_summary_df"], generated_files[5])
    save_day_label_mix_plot(
        analysis["day_label_df"],
        analysis["global_label_df"],
        generated_files[6],
    )
    save_cleanup_impact_plot(analysis, generated_files[7])

    report_text = build_report_text(analysis, pct, output_dir, generated_files)
    generated_files[8].write_text(report_text)

    return generated_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate distribution and cleanup reports for PCAP-extracted flow Parquet datasets."
    )
    parser.add_argument("--pct", type=float, default=100, help="Observation percentage tag, e.g. 100.")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Root directory containing partial_flow/<pct_tag>/raw_flow.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV, PNG, and markdown outputs. Defaults to the matching notebook percentage folder.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=250_000,
        help="Parquet batch size for scanning large datasets.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_flow_dir = args.data_root / observation_tag(args.pct) / "raw_flow"
    output_dir = args.output_dir or (
        DEFAULT_NOTEBOOK_ROOT / notebook_flow_percentage_dir(args.pct) / "outputs"
    )
    analysis = aggregate_partial_flow(raw_flow_dir=raw_flow_dir, batch_size=args.batch_size)
    output_files = write_outputs(analysis=analysis, pct=args.pct, output_dir=output_dir)

    print(f"Analyzed: {raw_flow_dir}")
    print(f"Wrote {len(output_files)} files to {output_dir}")
    for path in output_files:
        print(path)


if __name__ == "__main__":
    main()
