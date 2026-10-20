#!/usr/bin/env python3
"""Geometry–biology dual gates for GDEBA and its non-rigid refinement."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull,Delaunay,cKDTree
from compute_spateo_affine_metrics import edge_iou,row_pearson

ROOT=Path(__file__).resolve().parent
LOWER=('Mean_NN','Chamfer','Boundary_Error','Boundary_Chamfer','Hausdorff')
HIGHER=('Edge_IoU',)

def load_config(path=None):return json.loads(Path(path or ROOT/'isbi_gate_config.json').read_text())
def _boundary(x):
 try:return x[ConvexHull(x).vertices]
 except Exception:return x
def geometry_metrics(x,target):
 dt=cKDTree(x).query(target)[0];ds=cKDTree(target).query(x)[0];bx=_boundary(x);bt=_boundary(target);a=cKDTree(bt).query(bx)[0];b=cKDTree(bx).query(bt)[0]
 return {'Mean_NN':float(dt.mean()),'Chamfer':float((dt.mean()+ds.mean())/2),'Edge_IoU':float(edge_iou(x,target)),'Boundary_Error':float(np.r_[a,b].mean()),'Boundary_Chamfer':float(np.mean(a*a)+np.mean(b*b)),'Hausdorff':float(max(a.max(),b.max()))}
def biology_metrics(x,target,mx,rx,mpca,rpca):
 idx=cKDTree(x).query(target)[1];corr=row_pearson(rx,mx[idx]);a=rpca;b=mpca[idx];cos=np.sum(a*b,1)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-12)
 return {'PCC':float(np.nanmean(corr)),'Expression_Cosine':float(np.nanmean(cos)),'CI':None,'CI_gate_eligible':False,'CI_exclusion_reason':'project CI is annotation-derived and evaluation-only'}
def relative_gains(base,candidate):
 eps=1e-12;z={}
 for k in LOWER:z[k]=(base[k]-candidate[k])/(abs(base[k])+eps)
 for k in HIGHER:z[k]=(candidate[k]-base[k])/(abs(base[k])+eps)
 return z
def physical_safety(A,t,cfg):
 A=np.asarray(A);t=np.asarray(t);reasons=[]
 if not np.isfinite(A).all() or not np.isfinite(t).all():reasons.append('non_finite_affine')
 try:
  sv=np.linalg.svd(A,compute_uv=False);det=float(np.linalg.det(A));cond=float(sv.max()/max(sv.min(),1e-12));shear=float(np.dot(A[:,0],A[:,1])/(np.linalg.norm(A[:,0])*np.linalg.norm(A[:,1])));area=abs(det)
 except Exception:sv=np.array([np.nan,np.nan]);det=cond=shear=area=np.nan;reasons.append('affine_decomposition_failed')
 if not det>0:reasons.append('mirror_or_nonpositive_determinant')
 if not sv.min()>=cfg['min_singular_value']:reasons.append('extreme_compression')
 if not sv.max()<=cfg['max_singular_value']:reasons.append('extreme_stretch')
 if not cond<=cfg['max_condition_number']:reasons.append('condition_number')
 if not cfg['min_area_ratio']<=area<=cfg['max_area_ratio']:reasons.append('area_ratio')
 if not abs(shear)<=cfg['max_abs_shear_cosine']:reasons.append('extreme_shear')
 return not reasons,{'det':det,'singular_values':sv.tolist(),'condition_number':cond,'area_ratio':area,'shear_cosine':shear,'failure_reasons':reasons}
def dual_gate(base_coords,candidate_coords,target,mx,rx,mpca,rpca,A,t,config=None,mode='dual'):
 cfg=config or load_config();bg=geometry_metrics(base_coords,target);cg=geometry_metrics(candidate_coords,target);gain=relative_gains(bg,cg);score=sum(cfg['geometry_metric_weights'][k]*gain[k] for k in cfg['geometry_metric_weights']);geo_reasons=[]
 if score<=cfg['geometry_gain_threshold']:geo_reasons.append('insufficient_geometry_benefit')
 if gain['Mean_NN'] < -cfg['max_allowed_mean_nn_drop']:geo_reasons.append('mean_nn_degradation')
 if gain['Edge_IoU'] < -cfg['max_allowed_edge_iou_drop']:geo_reasons.append('edge_iou_degradation')
 if gain['Boundary_Error'] < -cfg['max_allowed_boundary_error_drop']:geo_reasons.append('boundary_error_degradation')
 geo_pass=not geo_reasons;bb=biology_metrics(base_coords,target,mx,rx,mpca,rpca);cb=biology_metrics(candidate_coords,target,mx,rx,mpca,rpca);changes={'PCC':cb['PCC']-bb['PCC'],'Expression_Cosine':cb['Expression_Cosine']-bb['Expression_Cosine'],'CI':None};valid_rel=[];bio_reasons=[]
 if changes['PCC'] < -cfg['pcc_tolerance']:bio_reasons.append('pcc_degradation')
 if changes['Expression_Cosine'] < -cfg['expr_cosine_tolerance']:bio_reasons.append('expression_cosine_degradation')
 for k in ('PCC','Expression_Cosine'):
  if np.isfinite(bb[k]) and np.isfinite(cb[k]):valid_rel.append((cb[k]-bb[k])/(abs(bb[k])+1e-12))
 bio_score=float(np.mean(valid_rel)) if valid_rel else np.nan
 if valid_rel and bio_score<cfg['biology_score_min']:bio_reasons.append('aggregate_biology_degradation')
 bio_pass=not bio_reasons;safe,safe_diag=physical_safety(A,t,cfg)
 accepted={'no_gate':safe,'geometry_only':safe and geo_pass,'dual':safe and geo_pass and bio_pass}[mode];reasons=safe_diag['failure_reasons']+geo_reasons+([] if mode!='dual' else bio_reasons)
 return {'gate_mode':mode,'accepted':bool(accepted),'accept_reason':'all_enabled_gates_passed' if accepted else '','reject_reason':reasons,'affine_safe':safe,'geometry_gate_pass':geo_pass,'biology_gate_pass':bio_pass,'rigid_geometry':bg,'candidate_geometry':cg,'geometry_relative_gains':gain,'geometry_score':score,'rigid_biology':bb,'candidate_biology':cb,'biology_changes':changes,'biology_preservation_score':bio_score,'affine_diagnostics':safe_diag,'gate_threshold_source':cfg['gate_threshold_source'],'annotation_used_for_gate':False}
def topology_metrics(base,out):
 tri=Delaunay(base).simplices;a=base[tri];b=out[tri];cross=lambda p:np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]);ca=cross(a);cb=cross(b);valid=np.abs(ca)>1e-12;j=cb[valid]/ca[valid]
 d=out-base;nn=cKDTree(base).query(base,k=min(9,len(base)))[1][:,1:];smooth=float(np.mean(np.linalg.norm(d[:,None]-d[nn],axis=2)/np.maximum(np.linalg.norm(base[:,None]-base[nn],axis=2),1e-12)))
 return {'mean_displacement':float(np.linalg.norm(d,axis=1).mean()),'smoothness':smooth,'jacobian_min':float(j.min()),'jacobian_p01':float(np.quantile(j,.01)),'jacobian_p05':float(np.quantile(j,.05)),'jacobian_median':float(np.median(j)),'jacobian_mean':float(j.mean()),'jacobian_p95':float(np.quantile(j,.95)),'jacobian_p99':float(np.quantile(j,.99)),'jacobian_max':float(j.max()),'fold_ratio':float(np.mean(j<=0)),'jacobian_low_ratio':float(np.mean(j<.5)),'jacobian_high_ratio':float(np.mean(j>2))}
def final_nonrigid_gate(base,candidate,target,mx,rx,mpca,rpca,config=None):
 cfg=dict(config or load_config());cfg['geometry_gain_threshold']=cfg['final_geometry_gain_threshold'];identity=np.eye(2);z=dual_gate(base,candidate,target,mx,rx,mpca,rpca,identity,np.zeros(2),cfg,'dual');top=topology_metrics(base,candidate);top_reasons=[]
 if top['fold_ratio']>cfg['max_fold_ratio']:top_reasons.append('fold_ratio')
 if top['jacobian_low_ratio']>cfg['max_jacobian_low_ratio']:top_reasons.append('excess_local_compression')
 if top['jacobian_high_ratio']>cfg['max_jacobian_high_ratio']:top_reasons.append('excess_local_expansion')
 geo=relative_gains(z['rigid_geometry'],z['candidate_geometry']);subset=np.mean([geo[k] for k in ('Mean_NN','Chamfer','Edge_IoU','Boundary_Error')]);geometry_pass=z['geometry_gate_pass'] and subset>=cfg['final_geometry_gain_threshold'];accepted=geometry_pass and z['biology_gate_pass'] and not top_reasons
 z.update(accepted=bool(accepted),geometry_gate_pass=bool(geometry_pass),topology_gate_pass=not top_reasons,topology=top,reject_reason=list(z['reject_reason'])+top_reasons,final_stage='nonrigid_output' if accepted else 'accepted_initialization');return z
