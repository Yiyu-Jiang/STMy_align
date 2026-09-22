#!/usr/bin/env python3
"""A-E ablation with the results_spateo_affine rigid+affine algorithm.

Scenarios:
  - basic_usage_demo
  - shear
  - compression

Methods:
  A. Rigid only
  B. Rigid + GDEBA (the robust affine module from results_spateo_affine)
  C. Rigid + GDEBA + Gate
  D. Rigid + GDEBA + Gate + NR
  E. Rigid + NR (Original Spateo)

Spateo rigid/nonrigid stages are run through spateo_nonrigid_worker on CUDA.
The affine/GDEBA step is deterministic CPU-side numerical refinement but is
timed as part of the GPU-enabled end-to-end run.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute_spateo_affine_metrics import evaluate as evaluate_with_mae
from evaluate_real_gdeba_pair import metrics, panels
from module_ablation_utils import load_pair
from run_gdeba_prefilter_sampling_optimization_shear_compression import dataframe_to_markdown, plot_runtime_and_metrics
from run_spateo_rigid_affine import affine_refine


SCENE_INPUTS = {
    "basic_usage_demo": ROOT.parent / "align_data" / "basic_usage_demo_1.h5ad",
    "collapse": ROOT / "aug_h5ad" / "basic_usage_demo_1_collapse.h5ad",
    "compression": ROOT / "aug_h5ad" / "basic_usage_demo_1_compression.h5ad",
    "local_defect": ROOT / "aug_h5ad" / "basic_usage_demo_1_local_defect.h5ad",
    "nonlinear_stretch": ROOT / "aug_h5ad" / "basic_usage_demo_1_nonlinear_stretch.h5ad",
    "shear": ROOT / "aug_h5ad" / "basic_usage_demo_1_shear.h5ad",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def merge_mae(q: dict, coords, target, raw, mx, rx, ml, rl) -> dict:
    extra = evaluate_with_mae(q.get("method", "stage"), coords, target, raw, mx, rx, ml, rl)
    q["MAE"] = extra["MAE"]
    q["Pairwise_MAE"] = extra["MAE"]
    q["Mean_NN_Distance"] = extra["Mean_NN_Distance"]
    return q


def run_worker(
    worker_python: str,
    moving_h5ad: Path,
    reference_h5ad: Path,
    output_npz: Path,
    device: str,
    seed: int,
    kind: str,
    input_npz: Path | None = None,
    input_key: str | None = None,
    reference_key: str = "reference_coords",
    force: bool = False,
    timeout_sec: int | None = None,
) -> tuple[bool, float, str]:
    if not force and output_npz.exists() and output_npz.with_suffix(".json").exists():
        meta = read_json(output_npz.with_suffix(".json"))
        ok = meta.get("status") == "SUCCESS"
        return ok, float(meta.get("runtime_sec", np.nan)), "" if ok else meta.get("error", "cached worker failed")
    cmd = [
        worker_python,
        str(ROOT / "spateo_nonrigid_worker.py"),
        "--moving", str(moving_h5ad),
        "--reference", str(reference_h5ad),
        "--output", str(output_npz),
        "--device", device,
        "--kind", kind,
        "--seed", str(seed),
    ]
    if input_npz is not None:
        cmd += [
            "--input-npz", str(input_npz),
            "--input-key", str(input_key),
            "--reference-npz", str(input_npz),
            "--reference-key", reference_key,
        ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "/ldap_shared/home/t_jyy/STAlign_project/spateo-release" + os.pathsep + env.get("PYTHONPATH", "")
    started = time.time()
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env, timeout=timeout_sec)
    except subprocess.TimeoutExpired as exc:
        output_npz.with_suffix(".worker.log").write_text(
            "COMMAND: " + " ".join(cmd) + "\n\nTIMEOUT:\n" + repr(exc)
        )
        return False, time.time() - started, f"worker_timeout_after_{timeout_sec}s; log={output_npz.with_suffix('.worker.log')}"
    wall = time.time() - started
    output_npz.with_suffix(".worker.log").write_text("COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr)
    if proc.returncode != 0:
        return False, wall, f"worker_returncode_{proc.returncode}; log={output_npz.with_suffix('.worker.log')}"
    meta = read_json(output_npz.with_suffix(".json"))
    if meta.get("status") != "SUCCESS":
        return False, float(meta.get("runtime_sec", wall)), meta.get("error", "worker status failed")
    return True, float(meta.get("runtime_sec", wall)), ""


def eval_coords(method: str, coords, target, raw, mx, rx, mp, rp, ml, rl, markers) -> dict:
    q = metrics(method, coords, target, mx, rx, mp, rp, ml, rl, markers)
    return merge_mae(q, coords, target, raw, mx, rx, ml, rl)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(ROOT / "results_spateo_affine" / "abcde_basic_shear_compression_gpu"))
    p.add_argument("--target-h5ad", default=str(ROOT / "basic_usage_demo_2.h5ad"))
    p.add_argument("--scenes", default="basic_usage_demo,shear,compression")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--worker-python", default="/ldap_shared/home/t_jyy/miniconda3/envs/stalign/bin/python")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir).resolve()
    for sub in ("summary", "figures", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    target_h5ad = Path(args.target_h5ad).resolve()
    scenes = [s.strip() for s in args.scenes.split(",") if s.strip()]
    rows, components, failures = [], [], []

    for scene in scenes:
        scene_out = out / scene
        scene_out.mkdir(parents=True, exist_ok=True)
        moving_h5ad = SCENE_INPUTS[scene].resolve()
        try:
            moving, reference, mx, rx, mp, rp, ml, rl, label_key, markers = load_pair(moving_h5ad, target_h5ad)
            raw = np.asarray(moving.obsm["spatial"])[:, :2]

            rigid_npz = scene_out / "A_rigid_gpu.npz"
            ok, rigid_runtime, reason = run_worker(
                args.worker_python, moving_h5ad, target_h5ad, rigid_npz, args.device, args.seed, "rigid", force=args.force
            )
            if not ok:
                raise RuntimeError(f"rigid worker failed: {reason}")
            rz = np.load(rigid_npz)
            rigid = rz["moving_rigid"]
            target = rz["reference"]

            t0 = time.time()
            affine_raw, A, t, selected_candidate, after_geom, before_geom, accepted, n_candidates = affine_refine(
                rigid, target, seed=args.seed
            )
            gdeba_runtime = time.time() - t0
            gated = affine_raw if accepted else rigid
            gate_runtime = 0.0
            input_npz = scene_out / "nr_inputs.npz"
            np.savez_compressed(input_npz, reference_coords=target, rigid_coords=rigid, gated_gdeba_coords=gated, raw_gdeba_coords=affine_raw)

            d_npz = scene_out / "D_gdeba_gate_nr_gpu.npz"
            ok_d, d_nr_runtime, d_reason = run_worker(
                args.worker_python,
                moving_h5ad,
                target_h5ad,
                d_npz,
                args.device,
                args.seed,
                "residual",
                input_npz=input_npz,
                input_key="gated_gdeba_coords",
                force=args.force,
            )
            e_npz = scene_out / "E_original_spateo_nr_gpu.npz"
            ok_e, e_nr_runtime, e_reason = run_worker(
                args.worker_python,
                moving_h5ad,
                target_h5ad,
                e_npz,
                args.device,
                args.seed,
                "residual",
                input_npz=input_npz,
                input_key="rigid_coords",
                force=args.force,
            )

            methods = [
                ("A: Rigid only", rigid, True, False, False, False, rigid_runtime, "own_algorithm_rigid_gpu", "not_applicable", False, "SUCCESS", ""),
                ("B: Rigid + GDEBA", affine_raw, True, True, False, False, rigid_runtime + gdeba_runtime, "own_algorithm_raw_gdeba_affine", "not_applied", False, "SUCCESS", ""),
                ("C: Rigid + GDEBA + Gate", gated, True, True, True, False, rigid_runtime + gdeba_runtime + gate_runtime, "own_algorithm_gated_gdeba_affine" if accepted else "own_algorithm_rigid_fallback_after_gate", "accepted" if accepted else "rejected", not accepted, "SUCCESS", ""),
            ]
            if ok_d:
                d_coords = np.load(d_npz)["moving_nonrigid"]
                methods.append(("D: Rigid + GDEBA + Gate + NR", d_coords, True, True, True, True, rigid_runtime + gdeba_runtime + gate_runtime + d_nr_runtime, "own_algorithm_gdeba_gate_nr_gpu", "accepted_then_nr" if accepted else "rejected_then_nr_from_rigid", not accepted, "SUCCESS", ""))
            else:
                methods.append(("D: Rigid + GDEBA + Gate + NR", np.full_like(rigid, np.nan), True, True, True, True, rigid_runtime + gdeba_runtime + gate_runtime + d_nr_runtime, "failed", "failed", not accepted, "FAILED", d_reason))
                failures.append({"deformation": scene, "method": "D", "failure_reason": d_reason})
            if ok_e:
                e_coords = np.load(e_npz)["moving_nonrigid"]
                methods.append(("E: Rigid + NR (Original Spateo)", e_coords, True, False, False, True, rigid_runtime + e_nr_runtime, "original_spateo_rigid_plus_nr_gpu", "not_applicable", False, "SUCCESS", ""))
            else:
                methods.append(("E: Rigid + NR (Original Spateo)", np.full_like(rigid, np.nan), True, False, False, True, rigid_runtime + e_nr_runtime, "failed", "failed", False, "FAILED", e_reason))
                failures.append({"deformation": scene, "method": "E", "failure_reason": e_reason})

            panel_items = []
            for method, coords, Rigid, GDEBA, Gate, NR, runtime, source, gate_status, fallback, status, failure in methods:
                if status == "SUCCESS":
                    q = eval_coords(method, coords, target, raw, mx, rx, mp, rp, ml, rl, markers)
                    panel_items.append((method.split(":")[0], coords))
                else:
                    q = {}
                rows.append(
                    {
                        "deformation": scene,
                        "method": method,
                        "Rigid": Rigid,
                        "GDEBA": GDEBA,
                        "Gate": Gate,
                        "NR": NR,
                        "PCC": q.get("PCC", np.nan),
                        "CI": q.get("CI", np.nan),
                        "Pairwise_MAE": q.get("Pairwise_MAE", np.nan),
                        "MAE": q.get("MAE", q.get("Pairwise_MAE", np.nan)),
                        "Mean_NN_Distance": q.get("Mean_NN_Distance", np.nan),
                        "Boundary_Error": q.get("Boundary_Error", np.nan),
                        "Edge_IoU": q.get("Edge_IoU", np.nan),
                        "Runtime_s": runtime,
                        "Rigid_runtime_gpu_s": rigid_runtime,
                        "GDEBA_runtime_s": gdeba_runtime if GDEBA else 0.0,
                        "Nonrigid_runtime_gpu_s": (d_nr_runtime if method.startswith("D:") else (e_nr_runtime if method.startswith("E:") else 0.0)),
                        "runtime_device": args.device,
                        "Gate_accepted": accepted if GDEBA else np.nan,
                        "gate_status": gate_status,
                        "fallback_to_rigid": fallback,
                        "coordinates_source": source,
                        "selected_candidate": selected_candidate,
                        "n_affine_candidates": n_candidates,
                        "Status": status,
                        "failure_reason": failure,
                    }
                )
            panels(out / "figures" / f"{scene}_ABCDE_overlay.png", panel_items, target)
            components.append(
                {
                    "deformation": scene,
                    "rigid_runtime_gpu_s": rigid_runtime,
                    "gdeba_affine_runtime_s": gdeba_runtime,
                    "D_nonrigid_runtime_gpu_s": d_nr_runtime,
                    "E_nonrigid_runtime_gpu_s": e_nr_runtime,
                    "gate_accepted": accepted,
                    "selected_candidate": selected_candidate,
                    "affine_matrix_A": A.tolist(),
                    "affine_translation_t": t.tolist(),
                }
            )
        except Exception as exc:
            failures.append({"deformation": scene, "method": "SCENE", "failure_reason": repr(exc)})

    df = pd.DataFrame(rows)
    if not df.empty:
        order = {
            "A: Rigid only": 0,
            "B: Rigid + GDEBA": 1,
            "C: Rigid + GDEBA + Gate": 2,
            "D: Rigid + GDEBA + Gate + NR": 3,
            "E: Rigid + NR (Original Spateo)": 4,
        }
        df["_order"] = df["method"].map(order)
        df = df.sort_values(["deformation", "_order"]).drop(columns="_order")
        df.to_csv(out / "summary" / "ABCED_basic_usage_shear_compression_gpu_metrics.csv", index=False)
        (out / "summary" / "ABCED_basic_usage_shear_compression_gpu_metrics.md").write_text(dataframe_to_markdown(df))
        # Backward-compatible, correctly spelled alias.
        df.to_csv(out / "summary" / "ABCDE_basic_usage_shear_compression_gpu_metrics.csv", index=False)
        (out / "summary" / "ABCDE_basic_usage_shear_compression_gpu_metrics.md").write_text(dataframe_to_markdown(df))
        plot_runtime_and_metrics(df[df["Status"].eq("SUCCESS")], out)
    pd.DataFrame(components).to_csv(out / "summary" / "runtime_components.csv", index=False)
    pd.DataFrame(failures).to_csv(out / "logs" / "failed_runs_log.csv", index=False)
    (out / "logs" / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "logs" / "environment_info.txt").write_text(json.dumps({"python": sys.version, "platform": platform.platform()}, indent=2))
    print("[DONE]", out)
    if not df.empty:
        print(df[["deformation", "method", "PCC", "CI", "Pairwise_MAE", "Mean_NN_Distance", "Runtime_s", "Status"]].to_string(index=False))


if __name__ == "__main__":
    main()
