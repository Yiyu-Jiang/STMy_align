#!/usr/bin/env python3
"""Compare GDEBA+NR, Original Spateo NR, SANTO, STalign and STAligner.

Datasets:
  - basic_usage_demo pair used by results_spateo_affine/abcde_basic...
  - MERFISH adjacent slice pairs used by results_spateo_affine/merfish...

GDEBA+NR and Original Spateo NR are read from the previously generated GPU
A/B/C/D/E coordinate/metric outputs.  SANTO, STalign and STAligner are run in
their own conda environments on CUDA and then evaluated with the same metric
functions used for GDEBA+NR/Spateo.
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

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from compute_spateo_affine_metrics import evaluate as evaluate_with_mae
from evaluate_real_gdeba_pair import metrics, panels
from module_ablation_utils import load_pair
from run_gdeba_prefilter_sampling_optimization_shear_compression import dataframe_to_markdown
from run_spateo_affine_abcde_merfish_multislice_gpu import prepare_pair_h5ads, safe_name


ENVS = {"SANTO": "santo", "STalign": "st_stalign", "STAligner": "env_STAligner"}
METHOD_LABELS = {
    "D: Rigid + GDEBA + Gate + NR": "GDEBA+NR",
    "E: Rigid + NR (Original Spateo)": "Original Spateo NR",
    "SANTO": "SANTO",
    "STalign": "STalign",
    "STAligner": "STAligner",
}
ORDER = {"GDEBA+NR": 0, "Original Spateo NR": 1, "SANTO": 2, "STalign": 3, "STAligner": 4}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def merge_mae(q: dict, coords, target, raw, mx, rx, ml, rl) -> dict:
    extra = evaluate_with_mae(q.get("method", "stage"), coords, target, raw, mx, rx, ml, rl)
    q["MAE"] = extra["MAE"]
    q["Pairwise_MAE"] = extra["MAE"]
    q["Mean_NN_Distance"] = extra["Mean_NN_Distance"]
    return q


def eval_coords(method: str, coords, target, raw, mx, rx, mp, rp, ml, rl, markers) -> dict:
    q = metrics(method, coords, target, mx, rx, mp, rp, ml, rl, markers)
    return merge_mae(q, coords, target, raw, mx, rx, ml, rl)


def build_cases(args, out: Path) -> list[dict]:
    cases: list[dict] = []
    cache = out / "cache" / "preprocessed_pairs"
    # basic_usage_demo: same input convention as the A/B/C/D/E GPU-rigid table.
    basic_moving = (ROOT.parent / "align_data" / "basic_usage_demo_1.h5ad").resolve()
    basic_reference = (ROOT / "basic_usage_demo_2.h5ad").resolve()
    bm, br, bmeta = prepare_pair_h5ads(
        basic_moving, basic_reference, cache / "basic_usage_demo", args.n_pcs, args.seed, force=args.force_preprocess
    )
    cases.append(
        {
            "dataset": "basic_usage_demo",
            "case": "basic_usage_demo",
            "pair_id": "basic_usage_demo",
            "moving_slice": "basic_usage_demo_1",
            "reference_slice": "basic_usage_demo_2",
            "moving_h5ad": bm,
            "reference_h5ad": br,
            "meta": bmeta,
            "d_npz": ROOT / "results_spateo_affine" / "abcde_basic_shear_compression_gpu_rerun" / "basic_usage_demo" / "D_gdeba_gate_nr_gpu.npz",
            "e_npz": ROOT / "results_spateo_affine" / "abcde_basic_shear_compression_gpu_rerun" / "basic_usage_demo" / "E_original_spateo_nr_gpu.npz",
            "d_json": ROOT / "results_spateo_affine" / "abcde_basic_shear_compression_gpu_rerun" / "basic_usage_demo" / "D_gdeba_gate_nr_gpu.json",
            "e_json": ROOT / "results_spateo_affine" / "abcde_basic_shear_compression_gpu_rerun" / "basic_usage_demo" / "E_original_spateo_nr_gpu.json",
            "d_runtime_components": ROOT / "results_spateo_affine" / "abcde_basic_shear_compression_gpu_rerun" / "summary" / "runtime_components.csv",
            "d_runtime_scene": "basic_usage_demo",
        }
    )

    manifest = pd.read_csv(args.merfish_manifest)
    if args.max_merfish_pairs and args.max_merfish_pairs > 0:
        manifest = manifest.head(args.max_merfish_pairs).copy()
    merfish_de = ROOT / "results_spateo_affine" / "merfish_multislice_gpu_rigid_abcde"
    for _, rec in manifest.iterrows():
        pair_id = f"{safe_name(rec['moving_slice'])}_to_{safe_name(rec['reference_slice'])}"
        moving, reference, meta = prepare_pair_h5ads(
            Path(rec["moving"]),
            Path(rec["reference"]),
            cache / pair_id,
            args.n_pcs,
            args.seed,
            force=args.force_preprocess,
        )
        cases.append(
            {
                "dataset": "MERFISH",
                "case": pair_id,
                "pair_id": pair_id,
                "moving_slice": rec["moving_slice"],
                "reference_slice": rec["reference_slice"],
                "moving_h5ad": moving,
                "reference_h5ad": reference,
                "meta": meta,
                "d_npz": merfish_de / "pairs" / pair_id / "D_gdeba_gate_nr_gpu.npz",
                "e_npz": merfish_de / "pairs" / pair_id / "E_original_spateo_nr_gpu.npz",
                "d_json": merfish_de / "pairs" / pair_id / "D_gdeba_gate_nr_gpu.json",
                "e_json": merfish_de / "pairs" / pair_id / "E_original_spateo_nr_gpu.json",
                "d_runtime_components": merfish_de / "summary" / "runtime_components.csv",
                "d_runtime_scene": pair_id,
            }
        )
    return cases


def de_runtime(case: dict, which: str) -> float:
    """Return end-to-end D/E runtime matching the A/B/C/D/E tables."""
    comp_path = Path(case["d_runtime_components"])
    if comp_path.exists():
        comp = pd.read_csv(comp_path)
        key = "deformation" if "deformation" in comp.columns else "pair_id"
        row = comp[comp[key].astype(str).eq(str(case["d_runtime_scene"]))]
        if not row.empty:
            r = row.iloc[0]
            if which == "D":
                return float(r["rigid_runtime_gpu_s"] + r["gdeba_affine_runtime_s"] + r["D_nonrigid_runtime_gpu_s"])
            return float(r["rigid_runtime_gpu_s"] + r["E_nonrigid_runtime_gpu_s"])
    meta = read_json(case["d_json" if which == "D" else "e_json"])
    return float(meta.get("runtime_sec", np.nan))


def add_de_rows(case: dict, rows: list[dict]) -> None:
    moving, reference, mx, rx, mp, rp, ml, rl, label_key, markers = load_pair(case["moving_h5ad"], case["reference_h5ad"])
    raw = np.asarray(moving.obsm["spatial"])[:, :2]
    target = np.asarray(reference.obsm["spatial"])[:, :2]
    for label, which, key in [
        ("GDEBA+NR", "D", "moving_nonrigid"),
        ("Original Spateo NR", "E", "moving_nonrigid"),
    ]:
        npz = Path(case["d_npz" if which == "D" else "e_npz"])
        if not npz.exists():
            rows.append(base_row(case, label, "FAILED", f"missing {npz}", np.nan))
            continue
        coords = np.load(npz)[key]
        q = eval_coords(label, coords, target, raw, mx, rx, mp, rp, ml, rl, markers)
        row = base_row(case, label, "SUCCESS", "", de_runtime(case, which))
        row.update(
            {
                "PCC": q.get("PCC"),
                "CI": q.get("CI"),
                "Pairwise_MAE": q.get("Pairwise_MAE"),
                "MAE": q.get("MAE"),
                "Mean_NN_Distance": q.get("Mean_NN_Distance"),
                "Boundary_Error": q.get("Boundary_Error"),
                "Edge_IoU": q.get("Edge_IoU"),
                "coordinates_source": "existing_gpu_gdeba_nr" if which == "D" else "existing_gpu_original_spateo_nr",
                "output_coords": str(npz),
                "label_key": label_key or "",
            }
        )
        rows.append(row)


def base_row(case: dict, algorithm: str, status: str, error: str, runtime: float) -> dict:
    meta = case.get("meta", {})
    return {
        "dataset": case["dataset"],
        "case": case["case"],
        "pair_id": case["pair_id"],
        "moving_slice": case["moving_slice"],
        "reference_slice": case["reference_slice"],
        "algorithm": algorithm,
        "PCC": np.nan,
        "CI": np.nan,
        "Pairwise_MAE": np.nan,
        "MAE": np.nan,
        "Mean_NN_Distance": np.nan,
        "Boundary_Error": np.nan,
        "Edge_IoU": np.nan,
        "Runtime_s": runtime,
        "runtime_device": "",
        "Status": status,
        "Error": error,
        "n_moving": meta.get("n_moving", np.nan),
        "n_reference": meta.get("n_reference", np.nan),
        "n_shared_genes": meta.get("n_shared_genes", np.nan),
        "n_pcs": meta.get("n_pcs", np.nan),
        "label_key": "",
        "output_coords": "",
        "log_file": "",
        "coordinates_source": "",
    }


def run_baseline(case: dict, algorithm: str, args, out: Path) -> dict:
    env_name = ENVS[algorithm]
    case_dir = out / "baselines" / case["dataset"] / case["case"] / algorithm
    case_dir.mkdir(parents=True, exist_ok=True)
    coords = case_dir / "aligned_moving_coords.npy"
    status = case_dir / "status.json"
    log = case_dir / "run.log"
    reuse_success = False
    if (not args.force_baselines) and coords.exists() and status.exists():
        cached = read_json(status)
        reuse_success = cached.get("Status") == "SUCCESS"
        if reuse_success:
            st = cached

    if not reuse_success:
        cmd = [
            "conda",
            "run",
            "-n",
            env_name,
            "python",
            str(ROOT / "sota_pair_baseline_worker.py"),
            "--method",
            algorithm,
            "--moving",
            str(case["moving_h5ad"]),
            "--reference",
            str(case["reference_h5ad"]),
            "--output",
            str(coords),
            "--status",
            str(status),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--santo-epochs",
            str(args.santo_epochs),
            "--stalign-niter",
            str(args.stalign_niter),
            "--staligner-epochs",
            str(args.staligner_epochs),
            "--train-subsample",
            str(args.train_subsample),
        ]
        env = os.environ.copy()
        env.update(
            {
                "MPLCONFIGDIR": f"/tmp/mpl_five_alg_{algorithm}",
                "NUMBA_CACHE_DIR": f"/tmp/numba_five_alg_{algorithm}",
            }
        )
        if algorithm != "STalign":
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
        log.write_text("COMMAND: " + " ".join(cmd) + "\n\nSTDOUT:\n" + proc.stdout + "\nSTDERR:\n" + proc.stderr)
        st = read_json(status) if status.exists() else {
            "Status": "FAILED",
            "failure_reason": f"worker exited {proc.returncode} without status",
            "runtime_sec": np.nan,
        }
    row = base_row(case, algorithm, st.get("Status", "FAILED"), st.get("failure_reason", ""), float(st.get("runtime_sec", np.nan)))
    row.update(
        {
            "runtime_device": args.device,
            "output_coords": str(coords) if coords.exists() else "",
            "log_file": str(log),
            "coordinates_source": f"{algorithm}_own_algorithm",
        }
    )
    if st.get("Status") != "SUCCESS":
        return row
    moving, reference, mx, rx, mp, rp, ml, rl, label_key, markers = load_pair(case["moving_h5ad"], case["reference_h5ad"])
    raw = np.asarray(moving.obsm["spatial"])[:, :2]
    target = np.asarray(reference.obsm["spatial"])[:, :2]
    xy = np.load(coords)
    q = eval_coords(algorithm, xy, target, raw, mx, rx, mp, rp, ml, rl, markers)
    row.update(
        {
            "PCC": q.get("PCC"),
            "CI": q.get("CI"),
            "Pairwise_MAE": q.get("Pairwise_MAE"),
            "MAE": q.get("MAE"),
            "Mean_NN_Distance": q.get("Mean_NN_Distance"),
            "Boundary_Error": q.get("Boundary_Error"),
            "Edge_IoU": q.get("Edge_IoU"),
            "label_key": label_key or "",
        }
    )
    return row


def plot_case(case: dict, rows: pd.DataFrame, out: Path) -> None:
    try:
        moving, reference, *_ = load_pair(case["moving_h5ad"], case["reference_h5ad"])
        target = np.asarray(reference.obsm["spatial"])[:, :2]
        items = []
        for _, r in rows[rows["case"].eq(case["case"]) & rows["Status"].eq("SUCCESS")].iterrows():
            p = Path(str(r["output_coords"]))
            if not p.exists():
                continue
            if p.suffix == ".npz":
                xy = np.load(p)["moving_nonrigid"]
            else:
                xy = np.load(p)
            items.append((str(r["algorithm"]), xy))
        if items:
            fig_path = out / "figures" / "overlays" / f"{case['case']}_five_algorithm_overlay.png"
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            panels(fig_path, items, target)
    except Exception:
        pass


def plot_summary(df: pd.DataFrame, out: Path) -> None:
    fig_dir = out / "figures" / "barplots"
    fig_dir.mkdir(parents=True, exist_ok=True)
    ok = df[df["Status"].eq("SUCCESS")].copy()
    ok["_order"] = ok["algorithm"].map(ORDER)
    for dataset, g0 in ok.groupby("dataset"):
        g = g0.groupby("algorithm", sort=False)[["PCC", "CI", "Pairwise_MAE", "Mean_NN_Distance", "Runtime_s"]].mean()
        g["_order"] = g.index.map(ORDER)
        g = g.sort_values("_order").drop(columns="_order")
        for col in ["PCC", "CI", "Pairwise_MAE", "Mean_NN_Distance", "Runtime_s"]:
            fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=300)
            ax.bar(range(len(g)), g[col].values, color=["#1b9e77", "#7570b3", "#e6ab02", "#d95f02", "#66a61e"])
            ax.set_xticks(range(len(g)))
            ax.set_xticklabels(g.index, rotation=25, ha="right")
            direction = "↑" if col in ("PCC", "CI") else "↓"
            ax.set_ylabel(f"{col} {direction}")
            ax.set_title(f"{dataset}: {col} comparison")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{dataset}_{col}_five_algorithm.png", bbox_inches="tight")
            fig.savefig(fig_dir / f"{dataset}_{col}_five_algorithm.pdf", bbox_inches="tight")
            plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output-dir", default=str(ROOT / "results_spateo_affine" / "five_algorithm_basic_merfish_gpu"))
    p.add_argument("--merfish-manifest", default=str(ROOT / "results_isbi_dual_gate" / "merfish_final" / "merfish_pair_manifest_final.csv"))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--n-pcs", type=int, default=30)
    p.add_argument("--max-merfish-pairs", type=int, default=0)
    p.add_argument("--santo-epochs", type=int, default=40)
    p.add_argument("--stalign-niter", type=int, default=60)
    p.add_argument("--staligner-epochs", type=int, default=120)
    p.add_argument("--train-subsample", type=int, default=1500)
    p.add_argument("--force-preprocess", action="store_true")
    p.add_argument("--force-baselines", action="store_true")
    args = p.parse_args()

    out = Path(args.output_dir).resolve()
    for sub in ("summary", "figures/overlays", "figures/barplots", "logs"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    cases = build_cases(args, out)
    rows: list[dict] = []
    for case in cases:
        add_de_rows(case, rows)
        for alg in ["SANTO", "STalign", "STAligner"]:
            started = time.time()
            row = run_baseline(case, alg, args, out)
            rows.append(row)
            print(case["dataset"], case["case"], alg, row["Status"], f"{time.time()-started:.1f}s", flush=True)
        partial = pd.DataFrame(rows)
        partial.to_csv(out / "summary" / "five_algorithm_pair_level_partial.csv", index=False)
    df = pd.DataFrame(rows)
    df["_order"] = df["algorithm"].map(ORDER)
    df = df.sort_values(["dataset", "case", "_order"]).drop(columns="_order")
    df.to_csv(out / "summary" / "five_algorithm_pair_level_metrics.csv", index=False)
    (out / "summary" / "five_algorithm_pair_level_metrics.md").write_text(dataframe_to_markdown(df))
    summary = (
        df[df["Status"].eq("SUCCESS")]
        .groupby(["dataset", "algorithm"], sort=False)
        .agg(
            n_cases=("case", "nunique"),
            PCC_mean=("PCC", "mean"),
            CI_mean=("CI", "mean"),
            Pairwise_MAE_mean=("Pairwise_MAE", "mean"),
            Mean_NN_Distance_mean=("Mean_NN_Distance", "mean"),
            Runtime_s_mean=("Runtime_s", "mean"),
            PCC_std=("PCC", "std"),
            CI_std=("CI", "std"),
            Pairwise_MAE_std=("Pairwise_MAE", "std"),
            Mean_NN_Distance_std=("Mean_NN_Distance", "std"),
            Runtime_s_std=("Runtime_s", "std"),
        )
        .reset_index()
    )
    summary["_order"] = summary["algorithm"].map(ORDER)
    summary = summary.sort_values(["dataset", "_order"]).drop(columns="_order")
    summary.to_csv(out / "summary" / "five_algorithm_mean_summary.csv", index=False)
    (out / "summary" / "five_algorithm_mean_summary.md").write_text(dataframe_to_markdown(summary))
    for case in cases:
        plot_case(case, df, out)
    plot_summary(df, out)
    failures = df[df["Status"].ne("SUCCESS")].copy()
    failures.to_csv(out / "logs" / "failed_runs_log.csv", index=False)
    # Backward-compatible retry manifest used by previous runs.  Always
    # rewrite it so stale FAILED rows from an earlier interrupted run are not
    # mistaken for current failures after a successful resume.
    failures.to_csv(out / "summary" / "failed_baseline_rows_to_retry.csv", index=False)
    (out / "logs" / "run_config.json").write_text(json.dumps(vars(args), indent=2))
    (out / "logs" / "environment_info.txt").write_text(json.dumps({"python": sys.version, "platform": platform.platform()}, indent=2))
    (out / "logs" / "command_history.txt").write_text(" ".join(sys.argv) + "\n")
    qc = [
        "# Five-algorithm basic/MERFISH GPU comparison quality check",
        "",
        f"- Cases: {df['case'].nunique()}",
        f"- Rows: {len(df)}",
        f"- Status counts: `{df['Status'].value_counts(dropna=False).to_dict()}`",
        f"- Failed rows: {len(failures)}",
        f"- Main metric NaN count: {int(df[['PCC','CI','Pairwise_MAE','Mean_NN_Distance','Runtime_s']].isna().sum().sum())}",
        "",
        "## Mean summary",
        dataframe_to_markdown(summary),
    ]
    (out / "summary" / "quality_check_report.md").write_text("\n".join(qc) + "\n")
    print("[DONE]", out)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
