#!/usr/bin/env python3
"""Independently evaluate before/rigid/rigid+affine coordinates.

All stages use the same reference-to-moving nearest-neighbour assignments and
the same expression/label matrices.  No metric is copied from a method output.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import ConvexHull, Delaunay, cKDTree
from sklearn.metrics import balanced_accuracy_score, f1_score


ROOT = Path(__file__).resolve().parent


def boundary(x: np.ndarray) -> np.ndarray:
    try:
        return x[ConvexHull(x).vertices]
    except Exception:
        return x


def edge_iou(source: np.ndarray, target: np.ndarray, bins: int = 96) -> float:
    z = np.vstack((source, target)); lo = z.min(0)
    span = np.maximum(z.max(0) - lo, 1e-9)
    def occupied(x):
        return {tuple(v) for v in np.clip(((x-lo)/span*(bins-1)).astype(int), 0, bins-1)}
    a, b = occupied(source), occupied(target)
    return len(a & b) / max(1, len(a | b))


def row_pearson(x, y, chunk: int = 1024) -> np.ndarray:
    """Row-wise Pearson correlation, evaluated sparsely and in chunks."""
    x = sparse.csr_matrix(x); y = sparse.csr_matrix(y)
    n, p = x.shape; out = np.full(n, np.nan)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n); a=x[start:stop]; b=y[start:stop]
        sa=np.asarray(a.sum(1)).ravel(); sb=np.asarray(b.sum(1)).ravel()
        saa=np.asarray(a.multiply(a).sum(1)).ravel()
        sbb=np.asarray(b.multiply(b).sum(1)).ravel()
        sab=np.asarray(a.multiply(b).sum(1)).ravel()
        num=sab-sa*sb/p
        den=np.sqrt(np.maximum(saa-sa*sa/p, 0)*np.maximum(sbb-sb*sb/p, 0))
        ok=den>1e-12; out[start:stop][ok]=num[ok]/den[ok]
    return out


def knn_preservation(raw: np.ndarray, aligned: np.ndarray, k: int = 10) -> float:
    k=min(k, len(raw)-1)
    if k < 1: return np.nan
    a=cKDTree(raw).query(raw,k=k+1)[1][:,1:]
    b=cKDTree(aligned).query(aligned,k=k+1)[1][:,1:]
    return float(np.mean([len(set(x).intersection(y))/k for x,y in zip(a,b)]))


def fold_rate(raw: np.ndarray, aligned: np.ndarray) -> float:
    try:
        tri=Delaunay(raw).simplices
        def signed(x):
            p=x[tri]; return np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0])
        a,b=signed(raw),signed(aligned); valid=np.abs(a)>1e-12
        return float(np.mean(a[valid]*b[valid] < 0)) if valid.any() else np.nan
    except Exception:
        return np.nan


def area_change(raw: np.ndarray, aligned: np.ndarray) -> float:
    try:
        return float(ConvexHull(aligned).volume/ConvexHull(raw).volume-1)
    except Exception:
        return np.nan


def evaluate(stage, source, target, raw_source, moving_x, target_x, moving_labels, target_labels):
    started=time.time()
    tree_s=cKDTree(source); dt, idx=tree_s.query(target)
    ds=cKDTree(target).query(source)[0]
    # MAE is symmetric L1 nearest-neighbour error; Mean NN is directed
    # reference-to-moving Euclidean NN distance, so these are not duplicates.
    l1_t=np.abs(target-source[idx]).sum(1)
    idx_back=cKDTree(target).query(source)[1]
    l1_s=np.abs(source-target[idx_back]).sum(1)
    corr=row_pearson(target_x, moving_x[idx])
    overlap=dt <= np.quantile(dt, .75)
    bs,bt=boundary(source),boundary(target)
    bd=np.r_[cKDTree(bt).query(bs)[0],cKDTree(bs).query(bt)[0]]
    row=dict(stage=stage, PCC=float(np.nanmean(corr)), Local_PCC=float(np.nanmean(corr[overlap])),
             MAE=float((l1_s.mean()+l1_t.mean())/2), Mean_NN_Distance=float(dt.mean()),
             symmetric_chamfer=float((ds.mean()+dt.mean())/2), Boundary_Error=float(bd.mean()),
             Edge_IoU=float(edge_iou(source,target)), hausdorff95=float(max(np.quantile(ds,.95),np.quantile(dt,.95))),
             kNN_Preservation=knn_preservation(raw_source,source), Fold_Rate=fold_rate(raw_source,source),
             Area_Change=area_change(raw_source,source), n_moving=len(source), n_reference=len(target),
             matched_reference_n=len(idx), overlap_reference_n=int(overlap.sum()), metric_runtime_sec=time.time()-started)
    if moving_labels is None or target_labels is None:
        row.update(CI=np.nan, Local_CI=np.nan, macro_F1=np.nan, balanced_accuracy=np.nan)
    else:
        pred=moving_labels[idx].astype(str); true=target_labels.astype(str)
        row.update(CI=float(np.mean(pred==true)),Local_CI=float(np.mean(pred[overlap]==true[overlap])),
                   macro_F1=float(f1_score(true,pred,average="macro",zero_division=0)),
                   balanced_accuracy=float(balanced_accuracy_score(true,pred)))
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--input-dir',default=str(ROOT/'aug_h5ad'))
    p.add_argument('--target-h5ad',default=str(ROOT/'basic_usage_demo_2.h5ad'))
    p.add_argument('--results-dir',default=str(ROOT/'results_spateo_affine'))
    p.add_argument('--label-key',default='cellbin_SpatialDomain')
    args=p.parse_args(); result=Path(args.results_dir)
    target0=ad.read_h5ad(args.target_h5ad)
    all_rows=[]
    for moving_path in sorted(Path(args.input_dir).glob('*.h5ad')):
        name=moving_path.stem.replace('basic_usage_demo_1_',''); out=result/name
        cache=out/'coordinates_all_stages.npz'
        if not cache.exists():
            all_rows.append(dict(dataset=name,stage='FAILED',Status='FAILED',failure_reason='coordinates_all_stages.npz missing')); continue
        moving0=ad.read_h5ad(moving_path)
        common=np.intersect1d(moving0.var_names[~moving0.var_names.duplicated()],target0.var_names[~target0.var_names.duplicated()])
        moving=moving0[:,common]; target=target0[:,common]
        mx=sparse.csr_matrix(moving.X); tx=sparse.csr_matrix(target.X)
        ml=np.asarray(moving.obs[args.label_key]) if args.label_key in moving.obs and args.label_key in target.obs else None
        tl=np.asarray(target.obs[args.label_key]) if ml is not None else None
        z=np.load(cache)
        stages=[('before',z['moving_before'],z['reference_before']),('spateo_rigid',z['moving_rigid'],z['reference_rigid']),('spateo_rigid_affine',z['moving_rigid_affine'],z['reference_rigid_affine'])]
        raw=stages[0][1]; rows=[]
        for stage,s,t in stages:
            row=evaluate(stage,s,t,raw,mx,tx,ml,tl)
            row.update(dataset=name,n_common_genes=len(common),label_key=args.label_key if ml is not None else '',Status='SUCCESS',failure_reason='')
            rows.append(row); all_rows.append(row)
        pd.DataFrame(rows).to_csv(out/'all_stage_metrics.csv',index=False)
    long=pd.DataFrame(all_rows)
    long.to_csv(result/'spateo_affine_all_stage_metrics.csv',index=False)
    metrics=['PCC','CI','Local_PCC','Local_CI','MAE','Mean_NN_Distance','symmetric_chamfer','Boundary_Error','Edge_IoU','hausdorff95','kNN_Preservation','Fold_Rate','Area_Change']
    comparisons=[]
    for dataset,g in long[long.Status=='SUCCESS'].groupby('dataset'):
        r=g.set_index('stage'); row={'dataset':dataset}
        for m in metrics:
            rv=float(r.loc['spateo_rigid',m]); av=float(r.loc['spateo_rigid_affine',m])
            row[f'rigid_{m}']=rv; row[f'affine_{m}']=av; row[f'delta_affine_minus_rigid_{m}']=av-rv
        accepted=json.loads((result/dataset/'affine_transform.json').read_text()).get('accepted',False)
        row['affine_accepted']=accepted; comparisons.append(row)
    comp=pd.DataFrame(comparisons).sort_values('dataset')
    comp.to_csv(result/'spateo_rigid_vs_affine_metrics.csv',index=False)
    report=['# Spateo Rigid vs Rigid + Affine metric report','',
            'All metrics were recomputed from each method\'s saved coordinates; no metric or coordinate was copied.',
            'PCC is mean row-wise Pearson correlation over all unique shared genes after reference-to-moving nearest-neighbour matching.',
            'CI uses the same matches and `cellbin_SpatialDomain`. MAE is symmetric L1 NN error; Mean NN Distance is directed reference-to-moving Euclidean distance.',
            'Local PCC/CI use the closest 75% reference matches. Positive deltas favor Affine for PCC/CI/Edge IoU; negative deltas favor Affine for error metrics.','',
            f'Rows evaluated: {len(long)}; datasets: {long.dataset.nunique()}.']
    (result/'metrics_calculation_report.md').write_text('\n'.join(report)+'\n')
    print(comp.to_string(index=False)); print(result/'spateo_rigid_vs_affine_metrics.csv')


if __name__=='__main__': main()
