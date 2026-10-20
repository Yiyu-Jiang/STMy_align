#!/usr/bin/env python3
"""Adaptive expression guidance utilities for GDEBA affine matching.

The functions here are deliberately independent from the affine optimizer so
that the fixed and adaptive scoring rules can be unit-tested without running
Spateo.  The confidence value q_i is a local candidate-set discriminability
score, not a calibrated probability of a correct correspondence.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


EPS = 1e-12


@dataclass
class ExpressionGuidanceConfig:
    expression_guidance_mode: str = "fixed"  # fixed | adaptive | none
    expression_confidence_mode: str = "combined"  # margin | entropy | combined
    cluster_guidance_mode: str = "hard"  # hard | none | soft
    lambda_g: float = 0.70
    lambda_f: float = 0.30
    alpha_max: float = 0.30
    tau_margin: float = 0.10
    expr_temperature: float = 0.10
    cluster_cross_weight: float = 0.20
    cluster_soft_min: float = 0.20


def l2_normalize(x: np.ndarray, eps: float = EPS) -> np.ndarray:
    """Row-wise L2 normalization with finite-value protection."""
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def candidate_cosine(moving_expr: np.ndarray, reference_expr: np.ndarray, candidate_idx: np.ndarray) -> np.ndarray:
    """Cosine similarities for each moving row and its spatial candidates."""
    zm = l2_normalize(moving_expr)
    zr = l2_normalize(reference_expr)
    cand = np.asarray(candidate_idx, dtype=int)
    return np.einsum("ij,ikj->ik", zm, zr[cand])


def stable_softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.nan_to_num(x, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    m = np.nanmax(x, axis=axis, keepdims=True)
    m[~np.isfinite(m)] = 0.0
    y = np.exp(np.clip(x - m, -80.0, 80.0))
    s = np.sum(y, axis=axis, keepdims=True)
    return y / np.maximum(s, EPS)


def expression_confidence(sim: np.ndarray, mode: str = "combined", tau_margin: float = 0.10,
                          temperature: float = 0.10) -> tuple[np.ndarray, dict]:
    """Compute q_i in [0, 1] from candidate expression similarities.

    Invalid rows, single-candidate rows and rows with no discriminability safely
    return confidence zero.
    """
    e = np.asarray(sim, dtype=float)
    e = np.where(np.isfinite(e), e, np.nan)
    n, k = e.shape if e.ndim == 2 else (len(e), 1)
    if k <= 1:
        q0 = np.zeros(n, dtype=float)
        return q0, {
            "confidence_margin_mean": 0.0,
            "confidence_entropy_mean": 0.0,
            "confidence_mean": 0.0,
            "effective_candidate_min": int(k),
            "effective_candidate_median": float(k),
        }

    valid = np.isfinite(e)
    valid_counts = valid.sum(axis=1)
    filled = np.where(valid, e, -np.inf)
    part = np.sort(filled, axis=1)[:, ::-1]
    top1 = part[:, 0]
    top2 = part[:, 1]
    margin = np.zeros(n, dtype=float)
    ok_margin = valid_counts >= 2
    margin[ok_margin] = top1[ok_margin] - top2[ok_margin]
    q_margin = np.clip(margin / max(float(tau_margin), EPS), 0.0, 1.0)
    q_margin[~np.isfinite(q_margin)] = 0.0

    temp = max(float(temperature), EPS)
    prob = stable_softmax(np.where(valid, e / temp, -np.inf), axis=1)
    entropy = -np.sum(np.where(prob > 0, prob * np.log(prob + EPS), 0.0), axis=1)
    denom = np.log(np.maximum(valid_counts, 2))
    q_entropy = np.where(valid_counts >= 2, 1.0 - entropy / np.maximum(denom, EPS), 0.0)
    q_entropy = np.clip(q_entropy, 0.0, 1.0)
    q_entropy[~np.isfinite(q_entropy)] = 0.0

    if mode == "margin":
        q = q_margin
    elif mode == "entropy":
        q = q_entropy
    elif mode == "combined":
        q = q_margin * q_entropy
    else:
        raise ValueError(f"unknown expression_confidence_mode={mode!r}")
    q = np.clip(np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)
    stats = {
        "confidence_margin_mean": float(np.mean(q_margin)),
        "confidence_entropy_mean": float(np.mean(q_entropy)),
        "confidence_mean": float(np.mean(q)),
        "confidence_std": float(np.std(q)),
        "confidence_p10": float(np.quantile(q, 0.10)),
        "confidence_p50": float(np.quantile(q, 0.50)),
        "confidence_p90": float(np.quantile(q, 0.90)),
        "high_confidence_ratio": float(np.mean(q >= 0.5)),
        "low_confidence_ratio": float(np.mean(q <= 0.1)),
        "effective_candidate_min": int(valid_counts.min()) if len(valid_counts) else 0,
        "effective_candidate_median": float(np.median(valid_counts)) if len(valid_counts) else 0.0,
    }
    return q, stats


def cluster_candidate_weight(moving_clusters=None, reference_clusters=None, candidate_idx=None,
                             moving_soft=None, reference_soft=None,
                             mode: str = "hard", cross_weight: float = 0.20,
                             soft_min: float = 0.20) -> np.ndarray:
    """Return c(i,j) for each spatial candidate."""
    cand = np.asarray(candidate_idx, dtype=int)
    if mode == "none" or cand.size == 0:
        return np.ones(cand.shape, dtype=float)
    if mode == "soft":
        if moving_soft is None or reference_soft is None:
            return np.ones(cand.shape, dtype=float)
        pm = np.asarray(moving_soft, dtype=float)
        pr = np.asarray(reference_soft, dtype=float)
        dot = np.einsum("ij,ikj->ik", pm, pr[cand])
        return float(soft_min) + (1.0 - float(soft_min)) * np.clip(dot, 0.0, 1.0)
    if mode == "hard":
        if moving_clusters is None or reference_clusters is None:
            return np.ones(cand.shape, dtype=float)
        cm = np.asarray(moving_clusters)
        cr = np.asarray(reference_clusters)
        return np.where(cm[:, None] == cr[cand], 1.0, float(cross_weight))
    raise ValueError(f"unknown cluster_guidance_mode={mode!r}")


def score_spatial_expression_candidates(dist: np.ndarray, expr_sim: np.ndarray | None,
                                        cluster_weight: np.ndarray | None,
                                        cfg: ExpressionGuidanceConfig) -> tuple[np.ndarray, np.ndarray, dict]:
    """Compute fixed/adaptive candidate scores within an unchanged spatial KNN set."""
    d = np.asarray(dist, dtype=float)
    sigma = max(float(np.nanmedian(d[np.isfinite(d)])) if np.isfinite(d).any() else 1.0, EPS)
    s_geo = np.exp(-d * d / (2.0 * sigma * sigma))
    c = np.ones_like(s_geo) if cluster_weight is None else np.asarray(cluster_weight, dtype=float)
    e = np.zeros_like(s_geo) if expr_sim is None else np.nan_to_num(np.asarray(expr_sim, dtype=float), nan=0.0)
    e_pos = np.maximum(e, 0.0)
    mode = cfg.expression_guidance_mode
    if mode == "none":
        q = np.zeros(d.shape[0], dtype=float)
        score = s_geo * c
        stats = {"confidence_mean": 0.0, "expression_guidance_used": False}
    elif mode == "fixed":
        q = np.ones(d.shape[0], dtype=float)
        score = s_geo * (cfg.lambda_g + (1.0 - cfg.lambda_g) *
                         (cfg.lambda_f + (1.0 - cfg.lambda_f) * e_pos)) * c
        stats = {"confidence_mean": 1.0, "expression_guidance_used": True}
    elif mode == "adaptive":
        q, stats = expression_confidence(e, cfg.expression_confidence_mode, cfg.tau_margin, cfg.expr_temperature)
        alpha = np.clip(cfg.alpha_max * q, 0.0, 1.0)[:, None]
        g_expr = 1.0 - alpha * (1.0 - e_pos)
        score = s_geo * (cfg.lambda_g + (1.0 - cfg.lambda_g) * g_expr) * c
        stats["expression_guidance_used"] = True
        stats["alpha_mean"] = float(np.mean(alpha))
    else:
        raise ValueError(f"unknown expression_guidance_mode={mode!r}")
    score = np.nan_to_num(score, nan=-np.inf, posinf=np.inf, neginf=-np.inf)
    return score, q, stats


def soft_cluster_assignments(z: np.ndarray, centers: np.ndarray, temperature: float = 0.25) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    centers = np.asarray(centers, dtype=float)
    d2 = ((z[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return stable_softmax(-d2 / max(float(temperature), EPS), axis=1)
