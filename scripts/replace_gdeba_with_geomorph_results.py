#!/usr/bin/env python3
"""Replace legacy GDEBA+NR rows with GeoMorph-ST GDEBA+NR rows.

The replacement source is the independently recomputed pipeline-stage metrics:
`gdeba_pipeline_structure_results/gdeba_pipeline_stage_metrics.csv`, stage D.
Other algorithms in the five-method summary are preserved unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve()
RELEASE_ROOT = SCRIPT.parents[1]
SOTA_ADD = SCRIPT.parents[2]
PROJECT_ROOT = SCRIPT.parents[4]


def default_path(rel: str) -> Path:
    return PROJECT_ROOT / rel


def build_geomorph_summary(stage_metrics: pd.DataFrame) -> pd.DataFrame:
    source = stage_metrics[
        stage_metrics["stage_code"].eq("D")
        & stage_metrics["stage"].eq("GeoMorph-ST GDEBA+NR")
        & stage_metrics["Status"].eq("SUCCESS")
    ].copy()
    if source.empty:
        raise RuntimeError("No successful stage-D GeoMorph-ST GDEBA+NR rows found.")

    rows = []
    for dataset, part in source.groupby("dataset", sort=False):
        row = {"dataset": dataset, "algorithm": "GDEBA+NR", "n_cases": int(len(part))}
        metric_map = {
            "PCC": "PCC",
            "CI": "CI",
            "Pairwise_MAE": "Pairwise_MAE",
            "Mean_NN_Distance": "Mean_NN_Distance",
            "Runtime_s": "Runtime_s",
        }
        for out_prefix, src_col in metric_map.items():
            row[f"{out_prefix}_mean"] = part[src_col].mean()
            row[f"{out_prefix}_std"] = part[src_col].std(ddof=1) if len(part) > 1 else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def replace_rows(input_summary: Path, stage_metrics: Path, output_summary: Path, replacement_log: Path) -> None:
    original = pd.read_csv(input_summary)
    geomorph = build_geomorph_summary(pd.read_csv(stage_metrics))
    updated = original.copy()
    log_rows = []

    for _, row in geomorph.iterrows():
        mask = updated["dataset"].eq(row["dataset"]) & updated["algorithm"].eq("GDEBA+NR")
        if not mask.any():
            log_rows.append(
                {
                    "dataset": row["dataset"],
                    "status": "MISSING_TARGET_ROW",
                    "source_file": str(stage_metrics),
                    "message": "No GDEBA+NR row found in original summary.",
                }
            )
            continue
        old = updated.loc[mask].iloc[0].to_dict()
        for col in updated.columns:
            if col in row.index:
                updated.loc[mask, col] = row[col]
        log_rows.append(
            {
                "dataset": row["dataset"],
                "status": "REPLACED",
                "source_file": str(stage_metrics),
                "old_PCC_mean": old.get("PCC_mean"),
                "new_PCC_mean": row.get("PCC_mean"),
                "old_CI_mean": old.get("CI_mean"),
                "new_CI_mean": row.get("CI_mean"),
                "old_Pairwise_MAE_mean": old.get("Pairwise_MAE_mean"),
                "new_Pairwise_MAE_mean": row.get("Pairwise_MAE_mean"),
                "old_Mean_NN_Distance_mean": old.get("Mean_NN_Distance_mean"),
                "new_Mean_NN_Distance_mean": row.get("Mean_NN_Distance_mean"),
                "old_Runtime_s_mean": old.get("Runtime_s_mean"),
                "new_Runtime_s_mean": row.get("Runtime_s_mean"),
            }
        )

    output_summary.parent.mkdir(parents=True, exist_ok=True)
    replacement_log.parent.mkdir(parents=True, exist_ok=True)
    updated.to_csv(output_summary, index=False)
    pd.DataFrame(log_rows).to_csv(replacement_log, index=False)
    print(f"Wrote updated summary: {output_summary}")
    print(f"Wrote replacement log: {replacement_log}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--input-summary",
        type=Path,
        default=default_path("STMy_align/SOTA_ADD/results_spateo_affine/five_algorithm_basic_merfish_gpu/summary/five_algorithm_mean_summary.csv"),
    )
    p.add_argument(
        "--geomorph-stage-metrics",
        type=Path,
        default=default_path("STMy_align/SOTA_ADD/results_spateo_affine/gdeba_pipeline_structure_results/gdeba_pipeline_stage_metrics.csv"),
    )
    p.add_argument(
        "--output-summary",
        type=Path,
        default=RELEASE_ROOT / "outputs/five_algorithm_mean_summary_geomorph_st.csv",
    )
    p.add_argument(
        "--replacement-log",
        type=Path,
        default=RELEASE_ROOT / "outputs/gdeba_to_geomorph_replacement_log.csv",
    )
    args = p.parse_args()
    replace_rows(args.input_summary, args.geomorph_stage_metrics, args.output_summary, args.replacement_log)


if __name__ == "__main__":
    main()
