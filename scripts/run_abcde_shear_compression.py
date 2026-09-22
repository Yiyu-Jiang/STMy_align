#!/usr/bin/env python3
"""Create the compression/shear A/B/C/D/E ablation release table.

By default this script extracts the already recomputed GPU-rigid ABCDE table
and writes a release copy containing only compression and shear.  Use
`--rerun-upstream` to call the upstream experiment runner before extraction.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


SCRIPT = Path(__file__).resolve()
RELEASE_ROOT = SCRIPT.parents[1]
PROJECT_ROOT = SCRIPT.parents[4]
SOTA_ADD = SCRIPT.parents[2]


def maybe_rerun_upstream(args) -> None:
    if not args.rerun_upstream:
        return
    upstream = SOTA_ADD / "run_spateo_affine_abcde_basic_shear_compression_gpu.py"
    cmd = [
        sys.executable,
        str(upstream),
        "--output-dir",
        str(args.upstream_output_dir),
        "--scenes",
        "compression,shear",
        "--device",
        args.device,
        "--seed",
        str(args.seed),
    ]
    if args.force:
        cmd.append("--force")
    print("Running upstream:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def extract_table(source_summary: Path, output_summary: Path, paper_table: Path) -> None:
    df = pd.read_csv(source_summary)
    out = df[df["deformation"].isin(["compression", "shear"])].copy()
    if out.empty:
        raise RuntimeError(f"No compression/shear rows found in {source_summary}")
    output_summary.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_summary, index=False)

    cols = [
        "deformation",
        "method",
        "Rigid",
        "GDEBA",
        "Gate",
        "NR",
        "PCC",
        "CI",
        "Pairwise_MAE",
        "Mean_NN_Distance",
        "Runtime_s",
        "gate_status",
        "fallback_to_rigid",
        "coordinates_source",
        "Status",
        "failure_reason",
    ]
    out[cols].to_csv(paper_table, index=False)
    print(f"Wrote ABCDE summary: {output_summary}")
    print(f"Wrote paper table: {paper_table}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--source-summary",
        type=Path,
        default=PROJECT_ROOT / "STMy_align/SOTA_ADD/results_spateo_affine/abcde_basic_shear_compression_gpu_rerun/summary/ABCDE_basic_usage_shear_compression_gpu_metrics.csv",
    )
    p.add_argument("--output-summary", type=Path, default=RELEASE_ROOT / "outputs/abcde_shear_compression_summary.csv")
    p.add_argument("--paper-table", type=Path, default=RELEASE_ROOT / "outputs/abcde_shear_compression_paper_table.csv")
    p.add_argument("--rerun-upstream", action="store_true")
    p.add_argument(
        "--upstream-output-dir",
        type=Path,
        default=PROJECT_ROOT / "STMy_align/SOTA_ADD/results_spateo_affine/abcde_basic_shear_compression_gpu_rerun",
    )
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    maybe_rerun_upstream(args)
    extract_table(args.source_summary, args.output_summary, args.paper_table)


if __name__ == "__main__":
    main()
