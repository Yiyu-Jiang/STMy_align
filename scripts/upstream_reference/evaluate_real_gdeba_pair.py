#!/usr/bin/env python3
"""Independent, correspondence-free evaluation for one real slice pair."""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import anndata as ad
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from scipy import sparse
from scipy.spatial import cKDTree
from compute_spateo_affine_metrics import boundary,edge_iou,row_pearson
from run_gdeba_affine import boundary_stats,boundary_ids
from gdeba_dual_gate import final_nonrigid_gate

LABELS=('cellbin_SpatialDomain','cell_type','celltype','Cell_class','annotation','cluster','domain','spatial_domain','region','class','subclass','layer','layer_guess_reordered','layer_guess')
def cosine_rows(a,b):
    num=(a*b).sum(1);den=np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1);return num/np.maximum(den,1e-12)
def sparse_var(x):
    x=sparse.csr_matrix(x);mu=np.asarray(x.mean(0)).ravel();return np.asarray(x.multiply(x).mean(0)).ravel()-mu*mu
def metrics(name,x,target,mx,rx,mpca,rpca,ml,rl,markers):
    it=cKDTree(x).query(target)[1];dt=cKDTree(x).query(target)[0];ds=cKDTree(target).query(x)[0];corr=row_pearson(rx,mx[it]);cos=cosine_rows(rpca,mpca[it]);bs=boundary_stats(x,target);bx=x[boundary_ids(x)];bt=target[boundary_ids(target)];bd=np.r_[cKDTree(bt).query(bx)[0],cKDTree(bx).query(bt)[0]]
    marker=[]
    for j in markers:
        a=np.asarray(rx[:,j].toarray()).ravel();b=np.asarray(mx[it,j].toarray()).ravel();marker.append(np.corrcoef(a,b)[0,1] if np.std(a)>0 and np.std(b)>0 else np.nan)
    return {'method':name,'PCC':float(np.nanmean(corr)),'CI':float(np.mean(ml[it].astype(str)==rl.astype(str))) if ml is not None else np.nan,'Pairwise_MAE':np.nan,'Mean_NN':float(dt.mean()),'Chamfer':float((dt.mean()+ds.mean())/2),'Edge_IoU':edge_iou(x,target),'Boundary_Error':float(bd.mean()),'Boundary_Chamfer':bs['boundary_chamfer'],'Hausdorff':bs['boundary_hausdorff'],'Expression_NN_Cosine_Mean':float(np.nanmean(cos)),'Expression_NN_Cosine_Median':float(np.nanmedian(cos)),'Label_Consistency':float(np.mean(ml[it].astype(str)==rl.astype(str))) if ml is not None else np.nan,'Marker_Spatial_PCC_Mean':float(np.nanmean(marker)),'Marker_Spatial_PCC_Median':float(np.nanmedian(marker)),'Status':'SUCCESS'}
def panels(path,items,target):
    z=np.vstack([target,*[v for _,v in items]]);lo=z.min(0);hi=z.max(0);pad=np.maximum((hi-lo)*.04,1);fig,axs=plt.subplots(1,len(items),figsize=(4*len(items),4),dpi=300)
    for ax,(title,x) in zip(axs,items):ax.scatter(target[:,0],target[:,1],s=.25,c='#d73027',alpha=.25,lw=0);ax.scatter(x[:,0],x[:,1],s=.25,c='#2166ac',alpha=.25,lw=0);ax.set_xlim(lo[0]-pad[0],hi[0]+pad[0]);ax.set_ylim(lo[1]-pad[1],hi[1]+pad[1]);ax.set_aspect('equal');ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)
def boundaries(path,items,target):
    fig,ax=plt.subplots(figsize=(7,6),dpi=300)
    for label,x,c in [('Reference',target,'#d73027'),*[(n,v,c) for (n,v),c in zip(items,['#777777','#e6ab02','#2166ac','#1b9e77'])]]:
        ids=boundary_ids(x);q=np.vstack([x[ids],x[ids[0]]]);ax.plot(q[:,0],q[:,1],c=c,lw=1,label=label)
    ax.set_aspect('equal');ax.axis('off');ax.legend(frameon=False);fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)
def correspondence(path,x,target):
    idx=cKDTree(target).query(x)[1];conf=cKDTree(target).query(x)[0];ids=np.argsort(conf)[:min(300,len(x))];fig,ax=plt.subplots(figsize=(7,6),dpi=300);ax.scatter(target[:,0],target[:,1],s=.2,c='#d73027',alpha=.2);ax.scatter(x[:,0],x[:,1],s=.2,c='#2166ac',alpha=.2)
    for i in ids:ax.plot([x[i,0],target[idx[i],0]],[x[i,1],target[idx[i],1]],c='#555555',alpha=.12,lw=.3)
    ax.set_aspect('equal');ax.axis('off');fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)
def marker_plot(path,coords,mx,genes,markers):
    use=list(markers[-min(5,len(markers)):]);fig,axs=plt.subplots(len(use),len(coords),figsize=(4*len(coords),3.5*len(use)),dpi=300,squeeze=False)
    for i,j in enumerate(use):
        val=np.asarray(mx[:,j].toarray()).ravel();lo,hi=np.nanquantile(val,[.01,.99])
        for k,(title,x) in enumerate(coords):
            ax=axs[i,k];ax.scatter(x[:,0],x[:,1],c=val,s=.35,cmap='viridis',vmin=lo,vmax=hi,lw=0);ax.set_aspect('equal');ax.axis('off');ax.set_title(f'{genes[j]} | {title}')
    fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)
def main():
    p=argparse.ArgumentParser();p.add_argument('--moving',required=True);p.add_argument('--reference',required=True);p.add_argument('--result-dir',required=True);p.add_argument('--dataset',required=True);p.add_argument('--moving-slice',required=True);p.add_argument('--reference-slice',required=True);a=p.parse_args();out=Path(a.result_dir);out.mkdir(parents=True,exist_ok=True)
    m=ad.read_h5ad(a.moving);r=ad.read_h5ad(a.reference);mx=sparse.csr_matrix(m.X);rx=sparse.csr_matrix(r.X);mp=np.asarray(m.obsm['X_pca']);rp=np.asarray(r.obsm['X_pca']);key=next((k for k in LABELS if k in m.obs and k in r.obs),None);ml=np.asarray(m.obs[key]) if key else None;rl=np.asarray(r.obs[key]) if key else None
    b=np.load(out/'spateo_original.npz');z=np.load(out/'gdeba_initialization.npz');e=np.load(out/'gdeba_nonrigid.npz');raw=z['raw_coords'];target=z['reference_coords'];rigid=z['A_rigid'];gdeba_raw=z['C_gdeba_raw'];gdeba=z['C_gdeba_final'];spateo=b['moving_nonrigid'];enh_raw=e['moving_nonrigid'];markers=np.argsort(sparse_var(sparse.vstack([mx,rx]))) [-10:]
    bm=json.loads((out/'spateo_original.json').read_text());rm=json.loads((out/'rigid.json').read_text());gm=json.loads((out/'gdeba_initialization.json').read_text());em=json.loads((out/'gdeba_nonrigid.json').read_text());accepted=gm['GDEBA']['accepted']
    # When GDEBA is rejected, Method D is mathematically the same pipeline as
    # Method C (Rigid -> Original Non-rigid).  The worker memoizes that exact
    # execution and records the reuse explicitly.  Do not apply a D-only final
    # gate here, otherwise identical inputs/seeds/parameters can be reported as
    # different methods solely because their post-processing differs.
    shared_rejected_execution=(not accepted) and bool(em.get('shared_execution_due_to_identical_pipeline',False))
    if shared_rejected_execution:
        final_gate={'accepted':True,'final_stage':'spateo_original_nonrigid','shared_execution_due_to_identical_pipeline':True,'reason':'GDEBA rejected: Method C and D are the identical deterministic Original Non-rigid execution'}
        enh=enh_raw
    else:
        final_gate=final_nonrigid_gate(gdeba,enh_raw,target,mx,rx,mp,rp);enh=enh_raw if final_gate['accepted'] else gdeba.copy()
    rows=[]
    for name,x,runtime in [('rigid',rigid,rm['runtime_sec']),('gdeba_raw',gdeba_raw,gm['runtime_sec']),('gdeba',gdeba,gm['runtime_sec']),('spateo_original_nonrigid',spateo,bm['runtime_sec']),('gdeba_nonrigid_raw',enh_raw,rm['runtime_sec']+gm['runtime_sec']+em['runtime_sec']),('gdeba_nonrigid',enh,rm['runtime_sec']+gm['runtime_sec']+em['runtime_sec'])]:
        q=metrics(name,x,target,mx,rx,mp,rp,ml,rl,markers);q.update(dataset_group=a.dataset,moving_slice=a.moving_slice,reference_slice=a.reference_slice,GDEBA_Accepted=accepted,Final_NR_Accepted=final_gate['accepted'],Fallback_stage='none' if accepted else 'rigid',Final_stage=final_gate['final_stage'],Runtime=runtime,Rigid_runtime=rm['runtime_sec'] if name.startswith('gdeba_nonrigid') else np.nan,GDEBA_runtime=gm['runtime_sec'] if name.startswith('gdeba_nonrigid') else np.nan,Nonrigid_runtime=em['runtime_sec'] if name.startswith('gdeba_nonrigid') else np.nan,label_key=key or '',ground_truth_spot_correspondence_used=False);rows.append(q)
    pd.DataFrame(rows).to_csv(out/'pair_metrics.csv',index=False);np.savez_compressed(out/'coordinates.npz',raw_moving_coords=raw,reference_coords=target,rigid_coords=rigid,gdeba_raw_coords=gdeba_raw,gdeba_final_coords=gdeba,spateo_original_nonrigid_coords=spateo,gdeba_nonrigid_raw_coords=enh_raw,gdeba_nonrigid_coords=enh)
    items=[('Before Alignment',raw),('Spateo Rigid',rigid),('GDEBA',gdeba),('Spateo Original',spateo),('GDEBA + Non-rigid',enh)];panels(out/'comparison_5panel.png',items,target);panels(out/'isbi_method_comparison.png',items,target);boundaries(out/'boundary_comparison.png',[('Before',raw),('Spateo Final',spateo),('GDEBA',gdeba),('GDEBA + NR',enh)],target);correspondence(out/'spateo_correspondence.png',spateo,target);correspondence(out/'gdeba_nonrigid_correspondence.png',enh,target);marker_plot(out/'biology_preservation_comparison.png',[('Rigid',rigid),('GDEBA',gdeba),('Original NR',spateo),('GDEBA+NR',enh)],mx,m.var_names,markers)
    (out/'parameters.json').write_text(json.dumps({'dataset':a.dataset,'moving_slice':a.moving_slice,'reference_slice':a.reference_slice,'label_key':key,'marker_genes':[str(m.var_names[i]) for i in markers],'GDEBA_Accepted':accepted,'Fallback_stage':'none' if accepted else 'rigid','GDEBA_gate':gm['GDEBA'].get('dual_gate',{}),'gate_ablation':gm['GDEBA'].get('gate_ablation',{}),'Final_NR_Accepted':final_gate['accepted'],'final_nonrigid_gate':final_gate,'Ground-truth spot correspondence used':False,'Annotation used for gate':False},indent=2))
if __name__=='__main__':main()
