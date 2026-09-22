"""Common lightweight metric utilities for GeoMorph-ST release tables.

The original experiments recompute biological metrics from AnnData objects.
This module provides small geometry utilities used by the release scripts and
serves as the documented interface for PCC/CI/MAE table columns.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def pairwise_mae(aligned: np.ndarray, reference: np.ndarray) -> float:
    """Mean nearest-neighbor distance from aligned moving points to reference.

    This matches the Pairwise_MAE convention used by the current release
    tables: for every aligned moving coordinate, find its nearest reference
    coordinate and average Euclidean distances.
    """
    aligned = np.asarray(aligned, dtype=float)[:, :2]
    reference = np.asarray(reference, dtype=float)[:, :2]
    if aligned.size == 0 or reference.size == 0:
        return float("nan")
    dist, _ = cKDTree(reference).query(aligned, k=1)
    return float(np.mean(dist))


def mean_nn_distance(aligned: np.ndarray, reference: np.ndarray) -> float:
    """Symmetric mean nearest-neighbor distance between two point clouds."""
    aligned = np.asarray(aligned, dtype=float)[:, :2]
    reference = np.asarray(reference, dtype=float)[:, :2]
    if aligned.size == 0 or reference.size == 0:
        return float("nan")
    d1, _ = cKDTree(reference).query(aligned, k=1)
    d2, _ = cKDTree(aligned).query(reference, k=1)
    return float((np.mean(d1) + np.mean(d2)) / 2.0)


def safe_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.mean(arr)) if arr.size else float("nan")
