#!/usr/bin/env python3
"""Summarize the requested GDEBA pipeline structure.

Pipeline:
    Spateo rigid
      -> GDEBA affine pre-registration
      -> Geometry/biology Gate
      -> Spateo morpho_align non-rigid refinement

The script reuses completed coordinates from:
  * basic_usage_notebook_gdeba_nr
  * results_gdeba_nr_spateo_multialign

It recomputes PCC/CI/MAE/Mean NN Distance from each stage coordinate instead
of copying table values, and writes a clean paper-facing summary.
"""
from __future__ import annotations

import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from summarize_consistent_morpho_align_protocol import (
    evaluate_coords,
    procrustes_transform,
)


ROOT = Path(__file__).resolve().parent


STAGE_DEFS = [
    ("A", "Spateo rigid", "rigid_coords", "Spateo rigid", "rigid"),
    ("B", "Spateo rigid + GDEBA affine pre-registration", "gdeba_raw_coords", "GDEBA affine raw", "gdeba"),
    ("C", "Spateo rigid + GDEBA affine + Gate", "gdeba_final_coords", "Gate output", "gate"),
    ("D", "Spateo rigid + GDEBA affine + Gate + Spateo morpho_align NR", "gdeba_nonrigid_coords", "GeoMorph-ST GDEBA+NR", "final"),
    ("E", "Spateo rigid + Spateo morpho_align NR", "spateo_original_nonrigid_coords", "Original Spateo NR", "spateo"),
]


def _runtime_from_json(path: Path) -> float:
    if not path.exists():
        return float("nan")
    return float(json.loads(path.read_text()).get("runtime_sec", np.nan))


def _gate_meta(path: Path) -> dict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    gate = data.get("GDEBA_gate") or data.get("GDEBA") or {}
    return gate if isinstance(gate, dict) else {}


def _basic_usage_rows() -> list[dict]:
    pair_dir = ROOT / "results_isbi_dual_gate" / "basic_usage_notebook_gdeba_nr" / "basic_usage_notebook" / "basic_usage_demo_1_to_basic_usage_demo_2"
    cache_dir = ROOT / "results_isbi_dual_gate" / "basic_usage_notebook_gdeba_nr" / "cache" / "basic_usage_notebook" / "basic_usage_demo_1_to_basic_usage_demo_2"
    coords = np.load(pair_dir / "coordinates.npz")
    moving = ad.read_h5ad(cache_dir / "moving.h5ad")
    reference = ad.read_h5ad(cache_dir / "reference.h5ad")
    reference_coords = np.asarray(coords["reference_coords"], dtype=np.float64)[:, :2]

    rigid_runtime = _runtime_from_json(pair_dir / "rigid.json")
    gdeba_runtime = _runtime_from_json(pair_dir / "gdeba_initialization.json")
    nr_runtime = _runtime_from_json(pair_dir / "gdeba_nonrigid.json")
    spateo_nr_runtime = _runtime_from_json(pair_dir / "spateo_original.json")
    gate = _gate_meta(pair_dir / "parameters.json")
    gate_status = "accepted" if gate.get("accepted") is True else ("rejected" if gate.get("accepted") is False else "unknown")

    runtime_by_stage = {
        "rigid": rigid_runtime,
        "gdeba": rigid_runtime + gdeba_runtime,
        "gate": rigid_runtime + gdeba_runtime,
        "final": rigid_runtime + gdeba_runtime + nr_runtime,
        "spateo": spateo_nr_runtime,
    }

    rows: list[dict] = []
    for code, method, key, stage, stage_key in STAGE_DEFS:
        q = evaluate_coords(stage, coords[key], reference_coords, moving, reference)
        rows.append(
            {
                "dataset": "basic_usage_demo",
                "pair_id": "basic_usage_demo_1_to_2",
                "stage_code": code,
                "method": method,
                "stage": stage,
                "coordinate_key": key,
                "PCC": q["PCC"],
                "CI": q["CI"],
                "MAE": q["MAE"],
                "Pairwise_MAE": q["Pairwise_MAE"],
                "Mean_NN_Distance": q["Mean_NN_Distance"],
                "Runtime_s": runtime_by_stage[stage_key],
                "gate_status": gate_status if code in ("C", "D") else "not_applicable",
                "fallback_to_rigid": bool(gate.get("accepted") is False) if code in ("C", "D") else False,
                "label_key": q["label_key"],
                "n_moving": q["n_moving"],
                "n_reference": q["n_reference"],
                "protocol": "basic_usage: GDEBA initialization followed by Spateo morpho_align non-rigid refinement",
                "Status": "SUCCESS",
                "failure_reason": "",
            }
        )
    return rows


def _merfish_rows() -> list[dict]:
    source_root = ROOT / "results_gdeba_nr_spateo_multialign"
    manifest = pd.read_csv(source_root / "pair_manifest.csv")
    tasks = manifest.to_dict("records")
    slice_order = [str(tasks[0]["moving_slice"])] + [str(task["reference_slice"]) for task in tasks]

    # Keep separate cumulative coordinate maps for each reported stage.  This
    # implements the same Procrustes accumulation rule used by
    # Spateo_multialign.ipynb for every stage coordinate.
    cumulative: dict[str, dict[str, np.ndarray]] = {key: {} for *_prefix, key, _stage, _stage_key in STAGE_DEFS}
    raw_loaded: dict[str, tuple[np.ndarray, ad.AnnData]] = {}

    def load_raw(slice_id: str, cache_dir: Path, role: str) -> tuple[np.ndarray, ad.AnnData]:
        sid = str(slice_id)
        if sid in raw_loaded:
            return raw_loaded[sid]
        a = ad.read_h5ad(cache_dir / f"{role}.h5ad")
        xy = np.asarray(a.obsm["spatial"], dtype=np.float64)[:, :2]
        raw_loaded[sid] = (xy, a)
        for key in cumulative:
            cumulative[key][sid] = xy.copy()
        return raw_loaded[sid]

    rows: list[dict] = []
    for task in tasks:
        pair_id = f'{task["moving_slice"]}_to_{task["reference_slice"]}'
        pair_dir = source_root / task["dataset"] / pair_id
        cache_dir = source_root / "cache" / task["dataset"] / pair_id
        coords = np.load(pair_dir / "coordinates.npz")
        moving_slice = str(task["moving_slice"])
        reference_slice = str(task["reference_slice"])
        raw_moving, moving = load_raw(moving_slice, cache_dir, "moving")
        _raw_ref, reference = load_raw(reference_slice, cache_dir, "reference")

        rigid_runtime = _runtime_from_json(pair_dir / "rigid.json")
        gdeba_runtime = _runtime_from_json(pair_dir / "gdeba_initialization.json")
        gd_nr_meta = json.loads((pair_dir / "gdeba_nonrigid.json").read_text())
        spateo_runtime = _runtime_from_json(pair_dir / "spateo_original.json")
        # If the GDEBA branch was rejected, the implementation reuses the
        # deterministic original-NR coordinates.  For a fair end-to-end runtime
        # of the stated algorithm structure, count the actual Spateo NR time.
        gd_nr_runtime = float(gd_nr_meta.get("runtime_sec", np.nan))
        if gd_nr_meta.get("shared_execution_due_to_identical_pipeline"):
            gd_nr_runtime = spateo_runtime
        params = json.loads((pair_dir / "parameters.json").read_text())
        accepted = bool(params.get("GDEBA_Accepted"))
        gate_status = "accepted" if accepted else "rejected"
        runtime_by_stage = {
            "rigid": rigid_runtime,
            "gdeba": rigid_runtime + gdeba_runtime,
            "gate": rigid_runtime + gdeba_runtime,
            "final": rigid_runtime + gdeba_runtime + gd_nr_runtime,
            "spateo": spateo_runtime,
        }

        for code, method, key, stage, stage_key in STAGE_DEFS:
            aligned_moving = np.asarray(coords[key], dtype=np.float64)[:, :2]
            r, t = procrustes_transform(raw_moving, aligned_moving)
            moving_index = slice_order.index(moving_slice)
            for previous_slice in slice_order[:moving_index]:
                cumulative[key][previous_slice] = cumulative[key][previous_slice] @ r.T + t
            cumulative[key][moving_slice] = aligned_moving.copy()
            moving_eval = cumulative[key][moving_slice]
            reference_eval = cumulative[key][reference_slice]
            q = evaluate_coords(stage, moving_eval, reference_eval, moving, reference)
            rows.append(
                {
                    "dataset": "MERFISH",
                    "pair_id": pair_id,
                    "moving_slice": moving_slice,
                    "reference_slice": reference_slice,
                    "stage_code": code,
                    "method": method,
                    "stage": stage,
                    "coordinate_key": key,
                    "PCC": q["PCC"],
                    "CI": q["CI"],
                    "MAE": q["MAE"],
                    "Pairwise_MAE": q["Pairwise_MAE"],
                    "Mean_NN_Distance": q["Mean_NN_Distance"],
                    "Runtime_s": runtime_by_stage[stage_key],
                    "gate_status": gate_status if code in ("C", "D") else "not_applicable",
                    "fallback_to_rigid": (not accepted) if code in ("C", "D") else False,
                    "label_key": q["label_key"],
                    "n_moving": q["n_moving"],
                    "n_reference": q["n_reference"],
                    "protocol": "MERFISH: adjacent-pair morpho_align/GDEBA with Procrustes cumulative global coordinates",
                    "Status": "SUCCESS",
                    "failure_reason": "",
                }
            )
    return rows


def main() -> None:
    out = ROOT / "results_spateo_affine" / "gdeba_pipeline_structure_results"
    out.mkdir(parents=True, exist_ok=True)
    rows = _basic_usage_rows() + _merfish_rows()
    df = pd.DataFrame(rows)
    df.to_csv(out / "gdeba_pipeline_stage_metrics.csv", index=False)
    summary = (
        df.groupby(["dataset", "stage_code", "method", "stage"], as_index=False)
        .agg(
            n_cases=("pair_id", "nunique"),
            PCC=("PCC", "mean"),
            CI=("CI", "mean"),
            MAE=("MAE", "mean"),
            Pairwise_MAE=("Pairwise_MAE", "mean"),
            Mean_NN_Distance=("Mean_NN_Distance", "mean"),
            Runtime_s=("Runtime_s", "mean"),
            fallback_rate=("fallback_to_rigid", "mean"),
        )
        .sort_values(["dataset", "stage_code"])
    )
    summary.to_csv(out / "gdeba_pipeline_stage_summary.csv", index=False)

    deltas = []
    for dataset, part in summary.groupby("dataset"):
        final = part[part["stage_code"].eq("D")].iloc[0]
        spateo = part[part["stage_code"].eq("E")].iloc[0]
        row = {"dataset": dataset, "n_cases": int(final["n_cases"])}
        for metric in ("PCC", "CI", "MAE", "Pairwise_MAE", "Mean_NN_Distance", "Runtime_s"):
            row[f"GeoMorph_ST_{metric}"] = float(final[metric])
            row[f"Spateo_{metric}"] = float(spateo[metric])
            row[f"delta_{metric}"] = float(final[metric] - spateo[metric])
            row[f"rel_delta_{metric}_pct"] = float((final[metric] - spateo[metric]) / abs(spateo[metric]) * 100) if float(spateo[metric]) else np.nan
        row["PCC_win"] = row["delta_PCC"] > 0
        row["CI_win"] = row["delta_CI"] > 0
        row["MAE_win"] = row["delta_MAE"] < 0
        row["Mean_NN_Distance_win"] = row["delta_Mean_NN_Distance"] < 0
        row["Runtime_win"] = row["delta_Runtime_s"] < 0
        row["win_count_main4"] = sum(bool(row[k]) for k in ("PCC_win", "CI_win", "MAE_win", "Mean_NN_Distance_win"))
        deltas.append(row)
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(out / "gdeba_pipeline_final_vs_spateo_delta.csv", index=False)

    readme = [
        "# GDEBA pipeline structure results",
        "",
        "Pipeline: Spateo rigid → GDEBA affine pre-registration → Gate → Spateo morpho_align non-rigid refinement.",
        "",
        "Files:",
        "- `gdeba_pipeline_stage_metrics.csv`: per dataset/pair/stage metrics.",
        "- `gdeba_pipeline_stage_summary.csv`: mean stage metrics.",
        "- `gdeba_pipeline_final_vs_spateo_delta.csv`: final GeoMorph-ST vs Original Spateo NR.",
        "",
        "Final delta table:",
        "",
        delta_df.to_csv(index=False),
    ]
    (out / "README.md").write_text("\n".join(readme))
    print(f"Wrote {out}")
    print(summary.to_string(index=False))
    print(delta_df.to_string(index=False))


if __name__ == "__main__":
    main()
