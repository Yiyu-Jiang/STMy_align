#!/usr/bin/env python3
"""Generate the five-algorithm summary with GeoMorph-ST replacement.

This wrapper optionally reruns the upstream five-algorithm benchmark, then
replaces the legacy GDEBA+NR rows with GeoMorph-ST GDEBA+NR values.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT = Path(__file__).resolve()
RELEASE_ROOT = SCRIPT.parents[1]
PROJECT_ROOT = SCRIPT.parents[4]
SOTA_ADD = SCRIPT.parents[2]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rerun-upstream", action="store_true")
    p.add_argument(
        "--upstream-output-dir",
        type=Path,
        default=PROJECT_ROOT / "STMy_align/SOTA_ADD/results_spateo_affine/five_algorithm_basic_merfish_gpu",
    )
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    if args.rerun_upstream:
        upstream = SOTA_ADD / "run_five_algorithm_basic_merfish_gpu.py"
        cmd = [sys.executable, str(upstream), "--output-dir", str(args.upstream_output_dir), "--device", args.device]
        if args.force:
            cmd.append("--force")
        print("Running upstream:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    replace_script = RELEASE_ROOT / "scripts/replace_gdeba_with_geomorph_results.py"
    subprocess.run([sys.executable, str(replace_script)], check=True)


if __name__ == "__main__":
    main()
