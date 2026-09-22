#!/usr/bin/env python3
"""Small coordinate-level metric utility.

Example:
  python scripts/compute_metrics.py --aligned result.npz --aligned-key moving_nonrigid \
      --reference result.npz --reference-key reference
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.metrics import mean_nn_distance, pairwise_mae


def load_key(path: Path, key: str) -> np.ndarray:
    z = np.load(path)
    if key not in z.files:
        raise KeyError(f"{path} does not contain key {key}; available={z.files}")
    return z[key]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aligned", type=Path, required=True)
    p.add_argument("--aligned-key", required=True)
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--reference-key", required=True)
    args = p.parse_args()
    aligned = load_key(args.aligned, args.aligned_key)
    reference = load_key(args.reference, args.reference_key)
    print(
        json.dumps(
            {
                "Pairwise_MAE": pairwise_mae(aligned, reference),
                "Mean_NN_Distance": mean_nn_distance(aligned, reference),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
