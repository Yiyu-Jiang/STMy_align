#!/usr/bin/env python3
"""Shared, configuration-driven utilities for the strict GDEBA module ablation."""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
from scipy import sparse

from evaluate_real_gdeba_pair import LABELS, metrics
from gdeba_dual_gate import dual_gate, final_nonrigid_gate, load_config, topology_metrics
from run_expression_guided_affine import joint_cluster, normalize_log_hvg_pca
from run_gdeba_affine import optimize_mode


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def gdeba_args(enforce_optimizer_safety: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        spatial_knn=20, interior_distance_percentile=75,
        boundary_distance_percentile=95, expression_weight_floor=.3,
        geometry_base_weight=.7, cross_cluster_weight=.2, boundary_weight=3,
        max_interior_matches=3600, max_boundary_matches=400,
        max_affine_iterations=4, convergence_tol=1e-4,
        affine_regularization_weight=1e-2, min_singular_value=.5,
        max_singular_value=2, min_correspondences=50,
        enforce_optimizer_safety=enforce_optimizer_safety,
    )


def load_pair(moving_path: Path, reference_path: Path):
    moving = ad.read_h5ad(moving_path); reference = ad.read_h5ad(reference_path)
    mx=sparse.csr_matrix(moving.X); rx=sparse.csr_matrix(reference.X)
    mp=np.asarray(moving.obsm['X_pca']); rp=np.asarray(reference.obsm['X_pca'])
    key=next((k for k in LABELS if k in moving.obs and k in reference.obs),None)
    ml=np.asarray(moving.obs[key]) if key else None; rl=np.asarray(reference.obs[key]) if key else None
    variance=np.asarray(sparse.vstack([mx,rx]).multiply(sparse.vstack([mx,rx])).mean(0)).ravel()-np.asarray(sparse.vstack([mx,rx]).mean(0)).ravel()**2
    markers=np.argsort(variance)[-10:]
    return moving,reference,mx,rx,mp,rp,ml,rl,key,markers


def build_candidates(moving_path: Path, reference_path: Path, rigid_npz: Path, seed: int):
    seed_all(seed); started=time.time()
    moving,reference,mx,rx,mp,rp,ml,rl,key,markers=load_pair(moving_path,reference_path)
    z=np.load(rigid_npz); raw=z['moving_input']; target=z['reference']; rigid=z['moving_rigid']
    # Recompute embeddings identically to the frozen full implementation.
    zm,zr,features,npc=normalize_log_hvg_pca(moving,reference,2000,30,seed)
    cm,cr,nclusters,_=joint_cluster(zm,zr,12,seed)
    modes={
        'no_expression':'geometry_boundary',
        'no_boundary':'expression_soft',
        'full':'expression_boundary',
    }
    candidates={}; diagnostics={}
    for name,mode in modes.items():
        t0=time.time()
        coords,A,t,A0,init_t,matches,history=optimize_mode(
            rigid,target,zm,zr,cm,cr,mode,gdeba_args(),seed)
        optimization_runtime=time.time()-t0
        tg=time.time(); gate=dual_gate(rigid,coords,target,mx,rx,zm,zr,A,t,load_config(),'dual'); gate_runtime=time.time()-tg
        candidates[name]=coords
        diagnostics[name]={
            'mode':mode,'A':A.tolist(),'t':t.tolist(),'A0':A0.tolist(),
            'history':history,'optimization_runtime_sec':optimization_runtime,
            'gate_runtime_sec':gate_runtime,'gate':gate,
        }
    meta={'seed':seed,'n_common_genes':moving.n_vars,'n_expression_features':len(features),
          'n_pcs':npc,'n_clusters':nclusters,'total_runtime_sec':time.time()-started,
          'annotation_used_for_gate':False,'ground_truth_correspondence_used':False,
          'candidates':diagnostics}
    return dict(raw=raw,target=target,rigid=rigid,zm=zm,zr=zr,mx=mx,rx=rx,mp=mp,rp=rp,
                ml=ml,rl=rl,key=key,markers=markers,candidates=candidates,meta=meta)


def initial_decision(config: dict, candidate_diag: dict) -> tuple[bool, dict]:
    gate=candidate_diag['gate']; physical=gate['affine_safe'] if config['use_affine_physical_safety'] else True
    geometry=gate['geometry_gate_pass'] if config['use_geometry_gate'] else True
    biology=gate['biology_gate_pass'] if config['use_biology_gate'] else True
    accepted=bool(physical and geometry and biology)
    return accepted,{'physical_gate_pass':bool(physical),'geometry_gate_pass':bool(geometry),
                      'biology_gate_pass':bool(biology),'raw_affine_safe':bool(gate['affine_safe']),
                      'raw_geometry_gate_pass':bool(gate['geometry_gate_pass']),
                      'raw_biology_gate_pass':bool(gate['biology_gate_pass']),
                      'reject_reason':gate['reject_reason'] if not accepted else []}


def candidate_name(config: dict) -> str:
    if not config['use_expression_guidance']: return 'no_expression'
    if not config['use_boundary_constraint']: return 'no_boundary'
    return 'full'


def array_hash(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x).view(np.uint8)).hexdigest()


def evaluate_variant(name, config, bundle, candidate, initial_accepted, decision,
                     original_nr, nr_raw, runtimes):
    raw,target,rigid=bundle['raw'],bundle['target'],bundle['rigid']
    if name=='A10_spateo_rigid':
        final=rigid; final_source='spateo_rigid'; final_diag={}; nr_input=rigid
    elif name=='A0_original_spateo_nr':
        final=original_nr; final_source='original_spateo_nr'; final_diag={}; nr_input=rigid
    else:
        initialization=candidate if initial_accepted else rigid
        nr_input=initialization
        if not config['use_nonrigid']:
            final=initialization; final_source='gdeba' if initial_accepted else 'rigid_fallback'; final_diag={}
        elif not initial_accepted:
            final=original_nr; final_source='original_spateo_nr_reused'; final_diag={'accepted':True,'equivalent_reuse':True}
        else:
            full_gate=final_nonrigid_gate(initialization,nr_raw,target,bundle['mx'],bundle['rx'],bundle['mp'],bundle['rp'])
            consistency=bool(full_gate['geometry_gate_pass'] and full_gate['biology_gate_pass'])
            topology=bool(full_gate['topology_gate_pass'])
            if not config['use_final_nr_gate'] and not config['use_topology_gate']: accept_nr=True
            elif not config['use_final_nr_gate']: accept_nr=topology
            elif not config['use_topology_gate']: accept_nr=consistency
            else: accept_nr=consistency and topology
            final=nr_raw if accept_nr else initialization
            final_source='nonrigid_output' if accept_nr else 'accepted_initialization'
            final_diag={**full_gate,'accepted':bool(accept_nr),'consistency_gate_pass':consistency,
                        'topology_gate_pass':topology}
    q=metrics(name,final,target,bundle['mx'],bundle['rx'],bundle['mp'],bundle['rp'],
              bundle['ml'],bundle['rl'],bundle['markers'])
    try: top=topology_metrics(nr_input,final)
    except Exception: top={k:np.nan for k in ('fold_ratio','jacobian_low_ratio','jacobian_high_ratio','jacobian_min','jacobian_p01','jacobian_p05','jacobian_median','jacobian_p95','jacobian_p99','jacobian_max')}
    q.update({
        'method':name,'GDEBA_candidate_generated':bool(config['use_gdeba']),
        'GDEBA_accepted':bool(initial_accepted) if config['use_gdeba'] else False,
        **decision,
        'final_nr_gate_pass':final_diag.get('consistency_gate_pass',final_diag.get('accepted',np.nan)),
        'topology_gate_pass':final_diag.get('topology_gate_pass',np.nan),
        'fold_ratio':top.get('fold_ratio'),'jacobian_lt_0_5_ratio':top.get('jacobian_low_ratio'),
        'jacobian_gt_2_ratio':top.get('jacobian_high_ratio'),
        **{k:top.get(k) for k in ('jacobian_min','jacobian_p01','jacobian_p05','jacobian_median','jacobian_p95','jacobian_p99','jacobian_max')},
        **runtimes,'runtime_total':sum(float(v) for k,v in runtimes.items() if k.endswith('_runtime')),
        'final_source':final_source,'Status':'SUCCESS','failure_reason':'',
    })
    return q,final,final_diag


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value,indent=2,default=lambda x: float(x) if np.isscalar(x) else str(x)))
