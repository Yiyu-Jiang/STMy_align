#!/usr/bin/env python3
"""Spateo rigid baseline plus an independent robust affine refinement module."""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree, ConvexHull
from gdeba_adaptive_expression import (
    ExpressionGuidanceConfig,
    candidate_cosine,
    cluster_candidate_weight,
    score_spatial_expression_candidates,
)

ROOT = Path(__file__).resolve().parent
DEFAULT_OUT = ROOT / "results_spateo_affine"


def clean(a: ad.AnnData) -> ad.AnnData:
    a = a.copy()
    if sparse.issparse(a.X):
        a.X = a.X.copy(); a.X.data = np.nan_to_num(a.X.data); a.X.eliminate_zeros()
    else: a.X = np.nan_to_num(np.asarray(a.X))
    a.obsm["spatial"] = np.nan_to_num(np.asarray(a.obsm["spatial"], float)[:, :2])
    return a


def clean_for_h5ad(a: ad.AnnData) -> ad.AnnData:
    """Remove legacy/non-serializable plotting metadata before H5AD output."""
    a=a.copy()
    for key in list(a.uns):
        if key.endswith("_colors") or key == "log1p": a.uns.pop(key,None)
    return a


def prepare(moving_path: Path, target_path: Path):
    moving, target = ad.read_h5ad(moving_path), ad.read_h5ad(target_path)
    common = np.intersect1d(moving.var_names[~moving.var_names.duplicated()], target.var_names[~target.var_names.duplicated()])
    if len(common) < 2: raise ValueError("fewer than two unique common genes")
    return clean(moving[:, common]), clean(target[:, common]), len(common)


def geometry(source, target):
    ds = cKDTree(target).query(source)[0]; dt = cKDTree(source).query(target)[0]
    def boundary(x):
        try: return x[ConvexHull(x).vertices]
        except Exception: return x
    bs, bt = boundary(source), boundary(target)
    be = np.r_[cKDTree(bt).query(bs)[0], cKDTree(bs).query(bt)[0]].mean()
    z=np.vstack([source,target]);lo=z.min(0);span=np.maximum(z.max(0)-lo,1e-9)
    def cells(x): return {tuple(v) for v in np.clip(((x-lo)/span*95).astype(int),0,95)}
    a,b=cells(source),cells(target)
    return dict(symmetric_chamfer=float((ds.mean()+dt.mean())/2), boundary_error=float(be),
                edge_iou=float(len(a&b)/max(1,len(a|b))), hausdorff95=float(max(np.quantile(ds,.95),np.quantile(dt,.95))))


def fit_affine(source, destination, ridge=1e-4):
    sm, dm = source.mean(0), destination.mean(0); x=source-sm; y=destination-dm
    scale=max(float(np.mean(np.sum(x*x,axis=1))),1e-9)
    a=np.linalg.solve(x.T@x + ridge*scale*np.eye(2), x.T@y)
    t=dm-sm@a
    return a,t


def fit_affine_weighted(source, destination, weights=None, ridge=1e-4):
    """Weighted affine fit with the same x' = x @ A + t convention.

    When ``weights`` is None this follows ``fit_affine``.  The helper is used
    only by optional GDEBA variants, so the default production path remains
    unchanged.
    """
    if weights is None:
        return fit_affine(source, destination, ridge)
    w = np.asarray(weights, float).reshape(-1)
    if len(w) != len(source) or not np.isfinite(w).any() or np.nanmax(w) <= 0:
        return fit_affine(source, destination, ridge)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.maximum(w, 1e-8)
    w = w / np.mean(w)
    sm = np.average(source, axis=0, weights=w)
    dm = np.average(destination, axis=0, weights=w)
    x = source - sm
    y = destination - dm
    xw = x * w[:, None]
    scale = max(float(np.mean(np.sum(x * x, axis=1))), 1e-9)
    a = np.linalg.solve(x.T @ xw + ridge * scale * np.eye(2), xw.T @ y)
    t = dm - sm @ a
    return a, t


def _mad_scale(values, floor=1e-6):
    """Robust scale estimate used by optional trimmed Huber fitting."""
    v = np.asarray(values, float).reshape(-1)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float(floor)
    med = float(np.median(v))
    mad = float(np.median(np.abs(v - med)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < floor:
        positive = v[v > 0]
        scale = float(np.median(positive)) if positive.size else float(floor)
    return float(max(scale, floor))


def fit_affine_weighted_huber(source, destination, weights=None, ridge=1e-4, huber_delta="auto", huber_rounds=4):
    """Weighted Huber IRLS affine fit with row-vector convention.

    This helper is used only when ``use_trimmed_huber=True``.  The historical
    path still calls ``fit_affine_weighted`` directly.
    """
    source = np.asarray(source, float)
    destination = np.asarray(destination, float)
    if len(source) < 3:
        return fit_affine_weighted(source, destination, weights, ridge), {
            "huber_delta": np.nan,
            "huber_rounds": 0,
        }
    base_w = np.ones(len(source), dtype=float) if weights is None else np.asarray(weights, float).reshape(-1)
    if len(base_w) != len(source):
        base_w = np.ones(len(source), dtype=float)
    base_w = np.nan_to_num(base_w, nan=0.0, posinf=0.0, neginf=0.0)
    base_w = np.maximum(base_w, 1e-8)
    a, t = fit_affine_weighted(source, destination, base_w, ridge)
    residual = np.linalg.norm(source @ a + t - destination, axis=1)
    if huber_delta == "auto":
        delta = 1.345 * _mad_scale(residual)
    else:
        try:
            delta = float(huber_delta)
        except Exception:
            delta = 1.345 * _mad_scale(residual)
    delta = float(max(delta, 1e-6))
    rounds = max(int(huber_rounds), 0)
    for _ in range(rounds):
        residual = np.linalg.norm(source @ a + t - destination, axis=1)
        robust = np.ones_like(residual, dtype=float)
        mask = residual > delta
        robust[mask] = delta / np.maximum(residual[mask], 1e-12)
        a, t = fit_affine_weighted(source, destination, base_w * robust, ridge)
    residual_final = np.linalg.norm(source @ a + t - destination, axis=1)
    return (a, t), {
        "huber_delta": delta,
        "huber_rounds": rounds,
        "median_residual_huber_final": float(np.median(residual_final)) if len(residual_final) else np.nan,
        "mean_residual_huber_final": float(np.mean(residual_final)) if len(residual_final) else np.nan,
    }


def fit_affine_trimmed_huber(
    source,
    destination,
    weights=None,
    ridge=1e-4,
    trim_ratio=0.8,
    max_trim_iter=1,
    min_matches=30,
    huber_delta="auto",
    residual_metric="euclidean",
):
    """Optional two-stage trimmed Huber affine solver.

    Step 1 fits the original weighted affine model. Step 2 trims high-residual
    correspondences and refits using Huber IRLS. If trimming leaves too few
    matches, it falls back to the original fit and records the reason.
    """
    source = np.asarray(source, float)
    destination = np.asarray(destination, float)
    n = int(len(source))
    initial_a, initial_t = fit_affine_weighted(source, destination, weights, ridge)
    transformed = source @ initial_a + initial_t
    if residual_metric != "euclidean":
        residual_metric = "euclidean"
    residual_before = np.linalg.norm(transformed - destination, axis=1)
    diag = {
        "use_trimmed_huber": True,
        "trim_ratio": float(trim_ratio),
        "max_trim_iter": int(max_trim_iter),
        "min_matches": int(min_matches),
        "n_matches_before_trim": n,
        "n_matches_after_trim": n,
        "trim_kept_fraction": 1.0,
        "median_residual_before": float(np.median(residual_before)) if n else np.nan,
        "mean_residual_before": float(np.mean(residual_before)) if n else np.nan,
        "median_residual_after": np.nan,
        "mean_residual_after": np.nan,
        "huber_delta": np.nan,
        "fallback_reason": "",
    }
    if n < int(min_matches):
        diag["fallback_reason"] = "insufficient_matches_before_trim"
        diag["median_residual_after"] = diag["median_residual_before"]
        diag["mean_residual_after"] = diag["mean_residual_before"]
        return initial_a, initial_t, diag
    keep = np.arange(n)
    tr = float(np.clip(trim_ratio, 0.05, 1.0))
    trim_iters = max(int(max_trim_iter), 1)
    current_a, current_t = initial_a, initial_t
    for _ in range(trim_iters):
        residual = np.linalg.norm(source @ current_a + current_t - destination, axis=1)
        cutoff = float(np.quantile(residual, tr))
        keep = np.where(residual <= cutoff)[0]
        if len(keep) < int(min_matches):
            diag["n_matches_after_trim"] = int(len(keep))
            diag["trim_kept_fraction"] = float(len(keep) / max(n, 1))
            diag["fallback_reason"] = "insufficient_matches_after_trim"
            diag["median_residual_after"] = diag["median_residual_before"]
            diag["mean_residual_after"] = diag["mean_residual_before"]
            return initial_a, initial_t, diag
        w_keep = None if weights is None else np.asarray(weights, float).reshape(-1)[keep]
        (current_a, current_t), hdiag = fit_affine_weighted_huber(
            source[keep],
            destination[keep],
            w_keep,
            ridge=ridge,
            huber_delta=huber_delta,
        )
    residual_after = np.linalg.norm(source[keep] @ current_a + current_t - destination[keep], axis=1)
    diag.update({
        "n_matches_after_trim": int(len(keep)),
        "trim_kept_fraction": float(len(keep) / max(n, 1)),
        "median_residual_after": float(np.median(residual_after)) if len(residual_after) else np.nan,
        "mean_residual_after": float(np.mean(residual_after)) if len(residual_after) else np.nan,
        "huber_delta": hdiag.get("huber_delta", np.nan),
        "fallback_reason": "",
    })
    return current_a, current_t, diag


def _expression_guided_assignment(
    current,
    target,
    moving_expr,
    reference_expr,
    moving_clusters,
    reference_clusters,
    moving_soft_clusters,
    reference_soft_clusters,
    expression_guidance_mode,
    expression_confidence_mode,
    cluster_guidance_mode,
    expression_lambda_g,
    expression_lambda_f,
    alpha_max,
    tau_margin,
    expr_temperature,
    cluster_cross_weight,
    cluster_soft_min,
    spatial_knn,
    use_local_structure=False,
    local_k=8,
    local_structure_weight=1.0,
    use_edge_consistency=False,
    edge_weight=1.0,
    edge_distance_quantile=0.75,
    edge_near_threshold=0.15,
):
    k = min(max(int(spatial_knn), 1), len(target))
    dist, cand = cKDTree(target).query(current, k=k)
    if k == 1:
        dist = dist[:, None]
        cand = cand[:, None]
    expr_sim = None
    if expression_guidance_mode != "none" and moving_expr is not None and reference_expr is not None:
        expr_sim = candidate_cosine(moving_expr, reference_expr, cand)
    cweight = cluster_candidate_weight(
        moving_clusters,
        reference_clusters,
        cand,
        moving_soft_clusters,
        reference_soft_clusters,
        cluster_guidance_mode,
        cluster_cross_weight,
        cluster_soft_min,
    )
    cfg = ExpressionGuidanceConfig(
        expression_guidance_mode=expression_guidance_mode,
        expression_confidence_mode=expression_confidence_mode,
        cluster_guidance_mode=cluster_guidance_mode,
        lambda_g=expression_lambda_g,
        lambda_f=expression_lambda_f,
        alpha_max=alpha_max,
        tau_margin=tau_margin,
        expr_temperature=expr_temperature,
        cluster_cross_weight=cluster_cross_weight,
        cluster_soft_min=cluster_soft_min,
    )
    score, confidence, stats = score_spatial_expression_candidates(dist, expr_sim, cweight, cfg)
    if use_local_structure:
        local_score, local_stats = local_structure_candidate_score(current, target, cand, local_k)
        score = score * np.power(local_score, float(local_structure_weight))
        stats.update(local_stats)
        stats["use_local_structure"] = True
        stats["local_k"] = int(local_k)
        stats["local_structure_weight"] = float(local_structure_weight)
    else:
        stats["use_local_structure"] = False
    edge_weights = np.ones_like(score)
    if use_edge_consistency:
        edge_score, edge_weights, edge_stats = edge_candidate_score(
            current,
            target,
            cand,
            edge_weight=edge_weight,
            edge_distance_quantile=edge_distance_quantile,
            edge_near_threshold=edge_near_threshold,
        )
        score = score * edge_score
        stats.update(edge_stats)
        stats["use_edge_consistency"] = True
    else:
        stats["use_edge_consistency"] = False
    pick = np.argmax(score, axis=1)
    j = cand[np.arange(len(current)), pick]
    picked_dist = dist[np.arange(len(current)), pick]
    picked_expr = np.full(len(current), np.nan) if expr_sim is None else expr_sim[np.arange(len(current)), pick]
    picked_cluster = cweight[np.arange(len(current)), pick]
    picked_weight = edge_weights[np.arange(len(current)), pick]
    return j, picked_dist, picked_expr, picked_cluster, confidence, stats, int(dist.size), picked_weight


def boundary_points(x):
    x = np.asarray(x, float)
    if len(x) < 3:
        return x
    try:
        return x[ConvexHull(x).vertices]
    except Exception:
        return x


def normalized_boundary_distance(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.zeros(0, dtype=float)
    b = boundary_points(x)
    if len(b) == 0:
        return np.zeros(len(x), dtype=float)
    d = cKDTree(b).query(x)[0]
    span = np.ptp(x, axis=0)
    denom = max(float(np.linalg.norm(span)), 1e-9)
    return np.nan_to_num(d / denom, nan=0.0, posinf=0.0, neginf=0.0)


def edge_candidate_score(
    current,
    target,
    cand,
    edge_weight=1.0,
    edge_distance_quantile=0.75,
    edge_near_threshold=0.15,
):
    dm = normalized_boundary_distance(current)
    dr = normalized_boundary_distance(target)
    cand = np.asarray(cand, int)
    diff = np.abs(dm[:, None] - dr[cand])
    finite = diff[np.isfinite(diff)]
    if finite.size == 0:
        return np.ones_like(diff), np.ones_like(diff), {
            "use_edge_consistency": True,
            "edge_fallback_reason": "no_valid_edge_points",
            "sigma_edge": np.nan,
            "mean_s_edge": np.nan,
            "median_s_edge": np.nan,
            "n_edge_pairs": 0,
            "n_matches_total": int(diff.size),
            "n_matches_edge_weighted": 0,
        }
    q = float(np.clip(edge_distance_quantile, 0.1, 0.95))
    sigma = float(np.quantile(finite, q))
    positive = finite[finite > 0]
    if sigma <= 1e-9:
        sigma = float(np.median(positive)) if positive.size else 1.0
    sigma = max(sigma, 1e-6)
    score = np.exp(-(diff * diff) / (sigma * sigma))
    near = (dm[:, None] <= edge_near_threshold) & (dr[cand] <= edge_near_threshold)
    solver_weight = 1.0 + float(edge_weight) * near.astype(float)
    score = np.nan_to_num(score, nan=1.0, posinf=1.0, neginf=0.0)
    solver_weight = np.nan_to_num(solver_weight, nan=1.0, posinf=1.0, neginf=1.0)
    return score, solver_weight, {
        "sigma_edge": sigma,
        "edge_weight": float(edge_weight),
        "edge_distance_quantile": float(edge_distance_quantile),
        "edge_near_threshold": float(edge_near_threshold),
        "mean_s_edge": float(np.mean(score)),
        "median_s_edge": float(np.median(score)),
        "n_edge_pairs": int(np.sum(near)),
        "n_matches_total": int(diff.size),
        "n_matches_edge_weighted": int(np.sum(near)),
    }


def local_structure_features(x, k=8):
    """Per-point local spatial structure: kNN mean/std distance and density.

    Features are robustly standardized inside each slice before candidate
    comparison.  This keeps the term scale-adaptive and avoids a fixed
    scene-specific distance threshold.
    """
    x = np.asarray(x, float)
    n = len(x)
    if n <= 2:
        return np.zeros((n, 3), dtype=float)
    kk = min(max(int(k) + 1, 2), n)
    d = cKDTree(x).query(x, k=kk)[0][:, 1:]
    mean = np.mean(d, axis=1)
    std = np.std(d, axis=1)
    density = 1.0 / np.maximum(mean, 1e-9)
    h = np.column_stack([mean, std, density])
    med = np.nanmedian(h, axis=0)
    mad = np.nanmedian(np.abs(h - med), axis=0)
    scale = np.maximum(1.4826 * mad, 1e-9)
    return np.nan_to_num((h - med) / scale, nan=0.0, posinf=0.0, neginf=0.0)


def local_structure_candidate_score(current, target, cand, local_k=8):
    hm = local_structure_features(current, local_k)
    hr = local_structure_features(target, local_k)
    diff = hm[:, None, :] - hr[np.asarray(cand, int)]
    dist2 = np.sum(diff * diff, axis=2)
    d = np.sqrt(np.maximum(dist2, 0.0))
    finite = d[np.isfinite(d)]
    if finite.size:
        sigma = float(np.quantile(finite, 0.75))
        if sigma <= 1e-12:
            sigma = float(np.median(finite[finite > 0])) if np.any(finite > 0) else 1.0
    else:
        sigma = 1.0
    score = np.exp(-dist2 / max(sigma * sigma, 1e-12))
    score = np.nan_to_num(score, nan=1.0, posinf=1.0, neginf=0.0)
    return score, {
        "sigma_local": sigma,
        "mean_s_local": float(np.mean(score)),
        "median_s_local": float(np.median(score)),
    }


def _legacy_local_assignment(
    current,
    target,
    spatial_knn=8,
    use_local_structure=False,
    local_k=8,
    local_structure_weight=1.0,
    use_edge_consistency=False,
    edge_weight=1.0,
    edge_distance_quantile=0.75,
    edge_near_threshold=0.15,
):
    k = min(max(int(spatial_knn), 1), len(target))
    dist, cand = cKDTree(target).query(current, k=k)
    if k == 1:
        dist = dist[:, None]
        cand = cand[:, None]
    sigma = max(float(np.nanmedian(dist[np.isfinite(dist)])), 1e-9)
    geo = np.exp(-((dist / sigma) ** 2))
    score = geo.copy()
    local_stats = {"use_local_structure": False}
    if use_local_structure:
        local_score, local_stats = local_structure_candidate_score(current, target, cand, local_k)
        score = score * np.power(local_score, float(local_structure_weight))
        local_stats["use_local_structure"] = True
    edge_weights = np.ones_like(score)
    if use_edge_consistency:
        edge_score, edge_weights, edge_stats = edge_candidate_score(
            current,
            target,
            cand,
            edge_weight=edge_weight,
            edge_distance_quantile=edge_distance_quantile,
            edge_near_threshold=edge_near_threshold,
        )
        score = score * edge_score
        local_stats.update(edge_stats)
        local_stats["use_edge_consistency"] = True
    else:
        local_stats["use_edge_consistency"] = False
    pick = np.argmax(score, axis=1)
    j = cand[np.arange(len(current)), pick]
    picked_dist = dist[np.arange(len(current)), pick]
    picked_weight = edge_weights[np.arange(len(current)), pick]
    stats = {
        "confidence_mean": 0.0,
        "expression_guidance_used": False,
        "use_local_structure": bool(use_local_structure),
        "local_k": int(local_k),
        "local_structure_weight": float(local_structure_weight),
        **local_stats,
    }
    return j, picked_dist, stats, int(dist.size), picked_weight


def affine_refine(
    rigid_source,
    rigid_target,
    rounds=12,
    trim=0.70,
    max_pairs=6000,
    seed=2026,
    moving_expr=None,
    reference_expr=None,
    moving_clusters=None,
    reference_clusters=None,
    moving_soft_clusters=None,
    reference_soft_clusters=None,
    expression_guidance_mode="legacy",
    expression_confidence_mode="combined",
    cluster_guidance_mode="none",
    expression_lambda_g=0.70,
    expression_lambda_f=0.30,
    alpha_max=0.30,
    tau_margin=0.10,
    expr_temperature=0.10,
    cluster_cross_weight=0.20,
    cluster_soft_min=0.20,
    spatial_knn=8,
    use_local_structure=False,
    local_k=8,
    local_structure_weight=1.0,
    use_edge_consistency=False,
    edge_sigma="auto",
    edge_weight=1.0,
    edge_distance_quantile=0.75,
    edge_near_threshold=0.15,
    use_trimmed_huber=False,
    trim_ratio=0.8,
    max_trim_iter=1,
    min_matches=30,
    huber_delta="auto",
    residual_metric="euclidean",
    return_diagnostics=False,
):
    rng=np.random.RandomState(seed); base=np.asarray(rigid_source,float); current=base.copy()
    candidates=[(np.eye(2),np.zeros(2),base.copy(),"identity")]
    diagnostics={
        "expression_guidance_mode": expression_guidance_mode,
        "expression_confidence_mode": expression_confidence_mode,
        "cluster_guidance_mode": cluster_guidance_mode,
        "spatial_knn": int(spatial_knn),
        "use_local_structure": bool(use_local_structure),
        "local_k": int(local_k),
        "local_structure_weight": float(local_structure_weight),
        "use_edge_consistency": bool(use_edge_consistency),
        "edge_sigma": str(edge_sigma),
        "edge_weight": float(edge_weight),
        "edge_distance_quantile": float(edge_distance_quantile),
        "edge_near_threshold": float(edge_near_threshold),
        "use_trimmed_huber": bool(use_trimmed_huber),
        "trim_ratio": float(trim_ratio),
        "max_trim_iter": int(max_trim_iter),
        "min_matches": int(min_matches),
        "huber_delta": str(huber_delta),
        "residual_metric": str(residual_metric),
        "iterations": [],
        "confidence": {},
        "legacy_path": expression_guidance_mode == "legacy",
    }
    # Moment-matching initialization recovers global anisotropic scale/shear
    # even when nearest-neighbour ICP alone starts outside its capture range.
    def sqrtm_spd(c, inverse=False):
        w,v=np.linalg.eigh(c+1e-8*np.eye(2));w=np.maximum(w,1e-8)
        return (v*(1/np.sqrt(w) if inverse else np.sqrt(w)))@v.T
    cs=np.cov(base,rowvar=False);ct=np.cov(rigid_target,rowvar=False)
    a0=sqrtm_spd(cs,True)@sqrtm_spd(ct,False);t0=rigid_target.mean(0)-base.mean(0)@a0
    sv0=np.linalg.svd(a0,compute_uv=False)
    if np.linalg.det(a0)>0 and sv0.min()>=.45 and sv0.max()<=2.20 and sv0.max()/sv0.min()<=4:
        current=base@a0+t0;candidates.append((a0,t0,current.copy(),"moment_init"))
    for iteration in range(rounds):
        pair_weight=np.ones(len(current),dtype=float)
        if expression_guidance_mode == "legacy":
            if use_local_structure or use_edge_consistency:
                j,dist,stats,candidate_pairs,pair_weight=_legacy_local_assignment(
                    current,rigid_target,spatial_knn,use_local_structure,local_k,local_structure_weight,
                    use_edge_consistency,edge_weight,edge_distance_quantile,edge_near_threshold)
            else:
                tree_t=cKDTree(rigid_target);dist,j=tree_t.query(current)
                stats={"confidence_mean":0.0,"expression_guidance_used":False,"use_local_structure":False,"use_edge_consistency":False};candidate_pairs=len(current)
            expr_picked=np.full(len(current),np.nan);cluster_picked=np.ones(len(current));conf=np.zeros(len(current))
        else:
            j,dist,expr_picked,cluster_picked,conf,stats,candidate_pairs,pair_weight=_expression_guided_assignment(
                current,rigid_target,moving_expr,reference_expr,moving_clusters,reference_clusters,
                moving_soft_clusters,reference_soft_clusters,expression_guidance_mode,
                expression_confidence_mode,cluster_guidance_mode,expression_lambda_g,
                expression_lambda_f,alpha_max,tau_margin,expr_temperature,
                cluster_cross_weight,cluster_soft_min,spatial_knn,
                use_local_structure,local_k,local_structure_weight,
                use_edge_consistency,edge_weight,edge_distance_quantile,edge_near_threshold)
        back=cKDTree(current).query(rigid_target)[1]
        ids=np.arange(len(current));mask=back[j]==ids
        idx=np.where(mask)[0]
        if len(idx)<100: idx=np.arange(len(current))
        cutoff=np.quantile(dist[idx],trim);idx=idx[dist[idx]<=cutoff]
        if len(idx)>max_pairs: idx=np.sort(rng.choice(idx,max_pairs,replace=False))
        if len(idx)<6: break
        iter_diag = {
            "iteration": iteration + 1,
            "candidate_pairs": int(candidate_pairs),
            "mnn_or_fallback_points": int(np.sum(mask)) if len(mask) else 0,
            "trimmed_pairs": int(len(idx)),
            "confidence_mean": float(np.nanmean(conf)) if len(conf) else 0.0,
            "confidence_std": float(np.nanstd(conf)) if len(conf) else 0.0,
            "picked_expression_similarity_mean": (
                float(np.nanmean(expr_picked[idx]))
                if len(idx) and np.isfinite(expr_picked[idx]).any()
                else np.nan
            ),
            "picked_cluster_weight_mean": float(np.nanmean(cluster_picked[idx])) if len(idx) else np.nan,
            "edge_solver_weight_mean": float(np.nanmean(pair_weight[idx])) if len(idx) else np.nan,
            **{k: v for k, v in stats.items() if isinstance(v, (int, float, bool))},
        }
        solver_weights = pair_weight[idx] if use_edge_consistency else None
        if use_trimmed_huber:
            a,t,trim_diag=fit_affine_trimmed_huber(
                base[idx],rigid_target[j[idx]],solver_weights,
                trim_ratio=trim_ratio,max_trim_iter=max_trim_iter,
                min_matches=min_matches,huber_delta=huber_delta,
                residual_metric=residual_metric)
            iter_diag.update({k: v for k, v in trim_diag.items() if isinstance(v, (int, float, bool, str))})
        else:
            a,t=fit_affine_weighted(base[idx],rigid_target[j[idx]],solver_weights)
            iter_diag.update({
                "use_trimmed_huber": False,
                "n_matches_before_trim": int(len(idx)),
                "n_matches_after_trim": int(len(idx)),
                "trim_kept_fraction": 1.0,
                "fallback_reason": "",
            })
        diagnostics["iterations"].append(iter_diag)
        singular=np.linalg.svd(a,compute_uv=False);det=float(np.linalg.det(a))
        if det<=0 or singular.min()<0.45 or singular.max()>2.20 or singular.max()/singular.min()>4.0: break
        current=base@a+t;candidates.append((a,t,current.copy(),f"iter_{iteration+1}"))
    scored=[]
    for a,t,xy,name in candidates:
        m=geometry(xy,rigid_target);score=m["symmetric_chamfer"]+0.35*m["boundary_error"]
        scored.append((score,a,t,xy,name,m))
    scored.sort(key=lambda x:x[0]);best=scored[0];raw=geometry(base,rigid_target)
    accepted=(best[4]!="identity" and best[5]["symmetric_chamfer"]<=raw["symmetric_chamfer"]*1.01 and best[5]["boundary_error"]<=raw["boundary_error"]*1.05)
    if not accepted: best=next(x for x in scored if x[4]=="identity")
    if diagnostics["iterations"]:
        vals=[x.get("confidence_mean",0.0) for x in diagnostics["iterations"]]
        diagnostics["confidence"]["mean_over_iterations"]=float(np.mean(vals))
        diagnostics["confidence"]["last_iteration_mean"]=float(vals[-1])
        diagnostics["confidence"]["n_iterations"]=len(vals)
    diagnostics.update({
        "selected_candidate": best[4],
        "accepted": bool(accepted),
        "n_candidates": int(len(candidates)),
        "before_geometry": raw,
        "after_geometry": best[5],
    })
    if return_diagnostics:
        return best[3],best[1],best[2],best[4],best[5],raw,accepted,len(candidates),diagnostics
    return best[3],best[1],best[2],best[4],best[5],raw,accepted,len(candidates)


def limits(*arrays):
    z=np.vstack(arrays);lo=z.min(0);hi=z.max(0);pad=np.maximum((hi-lo)*.04,1.0);return lo-pad,hi+pad


def plot_triptych(path, raw_s, raw_t, rigid_s, rigid_t, affine_s, affine_t, title, contour=False):
    lo,hi=limits(raw_s,raw_t,rigid_s,rigid_t,affine_s,affine_t)
    fig,axes=plt.subplots(1,3,figsize=(15,5),dpi=300)
    panels=[("Before Alignment",raw_s,raw_t),("Spateo Rigid",rigid_s,rigid_t),("Spateo Rigid + Affine",affine_s,affine_t)]
    for ax,(name,s,t) in zip(axes,panels):
        if contour:
            for xy,color,label in ((s,"#2166ac","Moving"),(t,"#d73027","Reference")):
                try:q=xy[ConvexHull(xy).vertices];q=np.vstack([q,q[0]]);ax.plot(q[:,0],q[:,1],color=color,lw=1.2,label=label)
                except Exception:ax.scatter(xy[:,0],xy[:,1],s=.2,c=color,alpha=.4,label=label)
        else:
            ax.scatter(t[:,0],t[:,1],s=.35,c="#d73027",alpha=.28,lw=0,label="Reference")
            ax.scatter(s[:,0],s[:,1],s=.35,c="#2166ac",alpha=.28,lw=0,label="Moving")
        ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]));ax.set_aspect("equal",adjustable="box");ax.set_title(name);ax.axis("off")
    axes[0].legend(loc="best",markerscale=5,frameon=False);fig.suptitle(title);fig.tight_layout();fig.savefig(path,bbox_inches="tight");plt.close(fig)


def run_single(args):
    import spateo as st
    moving_path,target_path=Path(args.input_h5ad),Path(args.target_h5ad);name=moving_path.stem.replace("basic_usage_demo_1_","")
    out=Path(args.output_dir)/name;out.mkdir(parents=True,exist_ok=True);start=time.time()
    moving,target,n_common=prepare(moving_path,target_path);raw_s=moving.obsm["spatial"].copy();raw_t=target.obsm["spatial"].copy()
    cache=out/"coordinates_all_stages.npz"
    if args.resume_coordinates and cache.exists():
        z=np.load(cache);raw_s=z["moving_before"];raw_t=z["reference_before"];rigid_s=z["moving_rigid"];rigid_t=z["reference_rigid"];affine_s=z["moving_rigid_affine"];affine_t=z["reference_rigid_affine"]
        a,t=fit_affine(rigid_s,affine_s,ridge=0);candidate="recovered_from_saved_coordinates";before=geometry(rigid_s,rigid_t);after=geometry(affine_s,affine_t);accepted=not np.allclose(rigid_s,affine_s);n_candidates=np.nan
    else:
        # Reuse the preprocessing-compatible PCA embeddings already stored in both files.
        if "X_pca" not in moving.obsm or "X_pca" not in target.obsm: raise KeyError("X_pca missing")
        models,_=st.align.morpho_align(models=[moving,target],spatial_key="spatial",key_added="spateo_rigid",
            mode="SN-S",device=args.device,rep_layer="X_pca",rep_field="obsm",dissimilarity="cos",
            max_iter=args.rigid_iter,verbose=False)
        key="spateo_rigid_rigid" if "spateo_rigid_rigid" in models[0].obsm else "spateo_rigid"
        rigid_s=np.asarray(models[0].obsm[key])[:,:2];rigid_t=np.asarray(models[1].obsm[key])[:,:2]
        affine_s,a,t,candidate,after,before,accepted,n_candidates=affine_refine(rigid_s,rigid_t,seed=args.seed)
        affine_t=rigid_t.copy()
        np.savez_compressed(cache,moving_before=raw_s,reference_before=raw_t,
            moving_rigid=rigid_s,reference_rigid=rigid_t,moving_rigid_affine=affine_s,reference_rigid_affine=affine_t)
    runtime=time.time()-start
    for obj,xy,p in ((moving,rigid_s,out/"moving_spateo_rigid.h5ad"),(moving,affine_s,out/"moving_spateo_rigid_affine.h5ad"),(target,rigid_t,out/"reference_aligned_frame.h5ad")):
        q=clean_for_h5ad(obj);q.obsm["aligned_spatial"]=xy;q.write_h5ad(p)
    transform={"matrix_A":a.tolist(),"translation_t":t.tolist(),"determinant":float(np.linalg.det(a)),
        "singular_values":np.linalg.svd(a,compute_uv=False).tolist(),"selected_candidate":candidate,
        "accepted":bool(accepted),"candidate_count":n_candidates,"source_coordinate_convention":"x_prime = x @ A + t"}
    (out/"affine_transform.json").write_text(json.dumps(transform,indent=2))
    plot_triptych(out/"alignment_triptych.png",raw_s,raw_t,rigid_s,rigid_t,affine_s,affine_t,name)
    plot_triptych(out/"boundary_triptych.png",raw_s,raw_t,rigid_s,rigid_t,affine_s,affine_t,name+" boundaries",True)
    rows=[]
    for stage,s,tg in (("before",raw_s,raw_t),("spateo_rigid",rigid_s,rigid_t),("spateo_rigid_affine",affine_s,affine_t)):
        rows.append(dict(dataset=name,stage=stage,**geometry(s,tg),runtime_sec=runtime,n_moving=len(s),n_reference=len(tg),n_common_genes=n_common,status="SUCCESS"))
    pd.DataFrame(rows).to_csv(out/"geometry_metrics.csv",index=False)
    result=dict(dataset=name,status="SUCCESS",runtime_sec=runtime,input_h5ad=str(moving_path),output_dir=str(out),affine_accepted=accepted,**transform)
    (out/"run_status.json").write_text(json.dumps(result,indent=2));print(json.dumps(result));return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--input-dir",default=str(ROOT/"aug_h5ad"));p.add_argument("--input-h5ad");p.add_argument("--target-h5ad",default=str(ROOT/"basic_usage_demo_2.h5ad"));p.add_argument("--output-dir",default=str(DEFAULT_OUT));p.add_argument("--device",default="cuda:0");p.add_argument("--devices",default="cuda:0,cuda:1,cuda:2,cuda:3");p.add_argument("--workers",type=int,default=4);p.add_argument("--rigid-iter",type=int,default=200);p.add_argument("--seed",type=int,default=2026);p.add_argument("--resume-coordinates",action="store_true");a=p.parse_args()
    if a.input_h5ad: return run_single(a)
    files=sorted(Path(a.input_dir).glob("*.h5ad"));devices=a.devices.split(",");out=Path(a.output_dir);(out/"logs").mkdir(parents=True,exist_ok=True)
    def launch(i,f):
        device=devices[i%len(devices)];log=out/"logs"/(f.stem+".log");cmd=[sys.executable,__file__,"--input-h5ad",str(f),"--target-h5ad",a.target_h5ad,"--output-dir",str(out),"--device",device,"--rigid-iter",str(a.rigid_iter),"--seed",str(a.seed+i)]
        if a.resume_coordinates:cmd.append("--resume-coordinates")
        env={**os.environ,"PYTHONPATH":"/ldap_shared/home/t_jyy/STAlign_project/spateo-release","MPLCONFIGDIR":f"/tmp/mpl_spateo_affine_{i}"};r=subprocess.run(cmd,text=True,capture_output=True,env=env);log.write_text(" ".join(cmd)+"\nSTDOUT\n"+r.stdout+"\nSTDERR\n"+r.stderr)
        try:return json.loads((out/f.stem.replace("basic_usage_demo_1_","")/"run_status.json").read_text())
        except Exception:return dict(dataset=f.stem,status="FAILED",error=f"exit {r.returncode}",log=str(log))
    rows=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        fs=[ex.submit(launch,i,f) for i,f in enumerate(files)]
        for f in as_completed(fs):rows.append(f.result());pd.DataFrame(rows).to_csv(out/"summary_partial.csv",index=False)
    pd.DataFrame(rows).sort_values("dataset").to_csv(out/"summary.csv",index=False);print(out)


if __name__=="__main__":main()
