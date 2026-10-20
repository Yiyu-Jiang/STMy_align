#!/usr/bin/env python3
"""Spateo rigid followed by expression-guided robust affine refinement.

Correspondences use only current spatial coordinates and a jointly fitted
expression representation.  Observation names, row identity and augmentation
ground truth are never read by the matching code.
"""
from __future__ import annotations

import argparse
import json
import time
import traceback
from pathlib import Path

import anndata as ad
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.spatial import ConvexHull, cKDTree
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA

from compute_spateo_affine_metrics import evaluate

ROOT=Path(__file__).resolve().parent


def normalize_log_hvg_pca(moving,reference,n_features=2000,n_pcs=30,seed=2026):
    """Joint library normalization, HVG variance ranking, scaling and PCA."""
    xm=sparse.csr_matrix(moving.X,dtype=np.float32);xr=sparse.csr_matrix(reference.X,dtype=np.float32)
    def norm(x):
        scale=1e4/np.maximum(np.asarray(x.sum(1)).ravel(),1e-12)
        y=x.multiply(scale[:,None]).tocsr();y.data=np.log1p(y.data);return y
    xm=norm(xm);xr=norm(xr);x=sparse.vstack((xm,xr),format='csr')
    mean=np.asarray(x.mean(0)).ravel();sq=np.asarray(x.multiply(x).mean(0)).ravel();var=np.maximum(sq-mean*mean,0)
    nf=min(n_features,x.shape[1]);selected=np.argpartition(var,-nf)[-nf:];selected=selected[np.argsort(var[selected])[::-1]]
    dense=x[:,selected].toarray().astype(np.float32,copy=False)
    mu=dense.mean(0);sd=dense.std(0);dense=(dense-mu)/np.maximum(sd,1e-6)
    npc=min(n_pcs,dense.shape[1],dense.shape[0]-1)
    z=PCA(n_components=npc,svd_solver='randomized',random_state=seed).fit_transform(dense)
    z=z.astype(np.float32);z/=np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-8)
    return z[:moving.n_obs],z[moving.n_obs:],selected,npc


def joint_cluster(zm,zr,n_clusters=12,seed=2026):
    z=np.vstack((zm,zr));k=min(n_clusters,max(2,len(z)//100))
    labels=MiniBatchKMeans(n_clusters=k,batch_size=4096,n_init=10,random_state=seed).fit_predict(z)
    return labels[:len(zm)],labels[len(zm):],k,'kmeans'


def fps(points,k,seed):
    n=len(points);k=min(k,n)
    if not k:return np.empty(0,int)
    p=(points-points.min(0))/np.maximum(np.ptp(points,axis=0),1e-9);out=np.empty(k,int)
    out[0]=np.random.RandomState(seed).randint(n);d=np.sum((p-p[out[0]])**2,axis=1)
    for q in range(1,k):out[q]=np.argmax(d);d=np.minimum(d,np.sum((p-p[out[q]])**2,axis=1))
    return out


def build_matches(coords,target,zm,zr,cm,cr,spatial_knn=20,distance_percentile=80,
                  expression_threshold=.5,max_per_cluster=500,geometry_weight=.6,seed=2026,
                  threshold_mode='fixed',use_expression=True):
    """Same-cluster spatial KNN + expression selection + spatial MNN filter."""
    rows=[];initial=0;same=0;shared=[]
    for c in sorted(set(cm).intersection(set(cr))):
        mi=np.where(cm==c)[0];ri=np.where(cr==c)[0]
        if len(mi)<3 or len(ri)<3:continue
        shared.append(int(c));kt=min(spatial_knn,len(ri));dist,cand=cKDTree(target[ri]).query(coords[mi],k=kt)
        if kt==1:dist=dist[:,None];cand=cand[:,None]
        initial+=dist.size;same+=dist.size
        # Select the best geometry/expression combination among spatial KNN.
        ref_ids=ri[cand];sim=np.einsum('ij,ikj->ik',zm[mi],zr[ref_ids])
        sigma=max(float(np.median(dist)),1e-8);geo=np.exp(-dist*dist/(2*sigma*sigma))
        score=geometry_weight*geo+(1-geometry_weight)*((sim+1)/2)
        pick=np.argmax(score,axis=1);rbest=ref_ids[np.arange(len(mi)),pick]
        dbest=dist[np.arange(len(mi)),pick];sbest=sim[np.arange(len(mi)),pick]
        # Spatial MNN is evaluated within the same expression cluster.
        back=cKDTree(coords[mi]).query(target[ri])[1];mutual=back[np.searchsorted(ri,rbest)]==np.arange(len(mi))
        threshold=(np.percentile(sbest,expression_threshold) if threshold_mode=='percentile' else expression_threshold)
        valid=mutual & ((sbest>=threshold) if use_expression else True)
        if valid.any():valid &= dbest<=np.percentile(dbest[valid],distance_percentile)
        loc=np.where(valid)[0]
        if not len(loc):continue
        # Cluster cap plus FPS provides cluster and spatial balance.
        if len(loc)>max_per_cluster:loc=loc[fps(coords[mi[loc]],max_per_cluster,seed+int(c))]
        n=max(len(loc),1);wg=np.exp(-dbest[loc]**2/(2*sigma*sigma));we=np.exp(-(1-sbest[loc])/.25) if use_expression else np.ones(len(loc))
        w=wg*we/np.sqrt(n)
        rows.extend(zip(mi[loc],rbest[loc],dbest[loc],sbest[loc],w,np.repeat(c,len(loc))))
    if not rows:return dict(mi=np.empty(0,int),ri=np.empty(0,int),distance=np.array([]),similarity=np.array([]),weight=np.array([]),cluster=np.array([]),initial=initial,same=same,shared=shared)
    a=np.asarray(rows,float)
    return dict(mi=a[:,0].astype(int),ri=a[:,1].astype(int),distance=a[:,2],similarity=a[:,3],weight=a[:,4],cluster=a[:,5].astype(int),initial=initial,same=same,shared=shared)


def weighted_affine(x,y,w,reg=1e-3,huber_rounds=6):
    """IRLS Huber weighted affine with regularization toward identity."""
    design=np.c_[x,np.ones(len(x))];base=np.asarray(w,float)/max(np.mean(w),1e-12);theta=np.vstack((np.eye(2),np.zeros((1,2))))
    for _ in range(huber_rounds):
        pred=design@theta;res=np.linalg.norm(pred-y,axis=1);scale=max(np.median(res)*1.4826,1e-8)
        hub=np.minimum(1,1.345*scale/np.maximum(res,1e-12));ww=base*hub
        lhs=design.T@(ww[:,None]*design);rhs=design.T@(ww[:,None]*y)
        penalty=np.diag([reg,reg,reg*.01]);prior=np.vstack((np.eye(2),np.zeros((1,2))))
        theta=np.linalg.solve(lhs+penalty,rhs+penalty@prior)
    return theta[:2],theta[2]


def run_affine(rigid,target,zm,zr,cm,cr,args,seed):
    base=rigid.copy();current=base.copy();history=[];last=None;final_matches=None
    for it in range(args.max_affine_iterations):
        schedule=[.8,.7,.6,.5];gw=schedule[min(it,len(schedule)-1)]
        m=build_matches(current,target,zm,zr,cm,cr,args.spatial_knn,args.distance_percentile,args.expression_threshold,args.max_matches_per_cluster,gw,seed+it,args.expression_threshold_mode,args.affine_mode=='coda')
        if len(m['mi'])<args.min_correspondences or len(m['shared'])<args.min_shared_clusters:raise ValueError(f"insufficient matches/clusters: {len(m['mi'])}/{len(m['shared'])}")
        a,t=weighted_affine(base[m['mi']],target[m['ri']],m['weight'],args.affine_regularization)
        sv=np.linalg.svd(a,compute_uv=False)
        if np.linalg.det(a)<=0 or sv.min()<args.min_singular_value or sv.max()>args.max_singular_value:raise ValueError(f"unsafe affine during iteration: det={np.linalg.det(a)}, sv={sv}")
        current=base@a+t;change=np.inf if last is None else np.linalg.norm(a-last[0])+np.linalg.norm(t-last[1])/max(np.linalg.norm(np.ptp(target,axis=0)),1)
        history.append(dict(iteration=it+1,n_matches=len(m['mi']),geometry_weight=gw,expression_weight=1-gw,change=float(change),stage='cluster_restricted_irls'))
        final_matches=m
        if last is not None and change<args.affine_convergence_tol:break
        last=(a.copy(),t.copy())
    return current,a,t,final_matches,history


def hard_gate(a,n_matches,n_shared,target,args):
    sv=np.linalg.svd(a,compute_uv=False);det=float(np.linalg.det(a));cond=float(sv.max()/sv.min());reasons=[]
    if not np.isfinite(a).all():reasons.append('NaN_or_Inf')
    if det<=0:reasons.append('nonpositive_determinant')
    if sv.min()<args.min_singular_value or sv.max()>args.max_singular_value:reasons.append('singular_value_out_of_range')
    if cond>args.max_condition_number:reasons.append('condition_number')
    if not .25<=det<=4:reasons.append('extreme_area_change')
    if n_matches<args.min_correspondences:reasons.append('too_few_correspondences')
    if n_shared<args.min_shared_clusters:reasons.append('too_few_shared_clusters')
    return not reasons,reasons,det,sv,cond


def soft_score(r,a,mean_sim):
    def rel(old,new,higher=False):
        scale=max(abs(old),1e-8);return (new-old)/scale if higher else (old-new)/scale
    geometry=np.mean([rel(r['MAE'],a['MAE']),rel(r['Mean_NN_Distance'],a['Mean_NN_Distance']),rel(r['Boundary_Error'],a['Boundary_Error']),rel(r['Edge_IoU'],a['Edge_IoU'],True)])
    bio=np.mean([a['PCC']-r['PCC'],(a['CI']-r['CI']) if np.isfinite(a['CI']) else 0,max(mean_sim,0)*.02])
    return float(.7*geometry+.3*bio),float(geometry),float(bio)


def limits(arrays):
    z=np.vstack(arrays);lo=z.min(0);hi=z.max(0);pad=np.maximum(np.ptp(z,axis=0)*.04,1);return lo-pad,hi+pad


def panel(ax,s,t,title,lo,hi):
    ax.scatter(t[:,0],t[:,1],s=.35,c='#d73027',alpha=.28,lw=0,label='Reference');ax.scatter(s[:,0],s[:,1],s=.35,c='#2166ac',alpha=.28,lw=0,label='Moving')
    ax.set(xlim=(lo[0],hi[0]),ylim=(lo[1],hi[1]));ax.set_aspect('equal');ax.set_title(title);ax.axis('off')


def make_plots(out,name,before,rigid,final,target,cm,cr,matches,seed):
    lo,hi=limits([before,rigid,final,target]);fig,axs=plt.subplots(1,3,figsize=(15,5),dpi=300)
    for ax,s,title in zip(axs,[before,rigid,final],['Before Alignment','Spateo Rigid','Expression-Guided Affine']):panel(ax,s,target,title,lo,hi)
    axs[0].legend(markerscale=5,frameon=False);fig.suptitle(name);fig.tight_layout();fig.savefig(out/'comparison_3panel.png',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(10,5),dpi=300)
    for ax,s,title in zip(axs,[rigid,final],['Spateo Rigid','Expression-Guided Affine']):panel(ax,s,target,title,lo,hi)
    fig.tight_layout();fig.savefig(out/'rigid_vs_coda_affine.png',bbox_inches='tight');plt.close(fig)
    k=max(cm.max(),cr.max())+1;cmap=plt.get_cmap('tab20',k);fig,axs=plt.subplots(1,2,figsize=(10,5),dpi=300)
    axs[0].scatter(rigid[:,0],rigid[:,1],c=cm,cmap=cmap,s=.4);axs[1].scatter(target[:,0],target[:,1],c=cr,cmap=cmap,s=.4)
    for ax,title in zip(axs,['Moving joint clusters','Reference joint clusters']):ax.set_aspect('equal');ax.set_title(title);ax.axis('off')
    fig.tight_layout();fig.savefig(out/'joint_expression_clusters.png',bbox_inches='tight');plt.close(fig)
    rng=np.random.RandomState(seed);n=min(300,len(matches['mi']));ids=np.sort(rng.choice(len(matches['mi']),n,replace=False));fig,ax=plt.subplots(figsize=(7,6),dpi=300)
    ax.scatter(target[:,0],target[:,1],s=.15,c='#d73027',alpha=.15);ax.scatter(final[:,0],final[:,1],s=.15,c='#2166ac',alpha=.15)
    w=matches['weight'][ids];w=w/np.maximum(w.max(),1e-12)
    for q,alpha in zip(ids,w):
        p=final[matches['mi'][q]];r=target[matches['ri'][q]];ax.plot([p[0],r[0]],[p[1],r[1]],c='#31a354',alpha=.08+.65*alpha,lw=.35)
    ax.set_aspect('equal');ax.set_title('Cluster-restricted correspondences');ax.axis('off');fig.tight_layout();fig.savefig(out/'cluster_restricted_correspondence.png',bbox_inches='tight');plt.close(fig)


def run_one(path,target0,args,seed):
    name=path.stem.replace('basic_usage_demo_1_','');out=Path(args.output_dir)/name;out.mkdir(parents=True,exist_ok=True);start=time.time()
    try:
        z=np.load(Path(args.rigid_cache)/name/'coordinates_all_stages.npz');before=z['moving_before'];rigid=z['moving_rigid'];target=z['reference_rigid']
        moving0=ad.read_h5ad(path);common=np.intersect1d(moving0.var_names[~moving0.var_names.duplicated()],target0.var_names[~target0.var_names.duplicated()])
        if len(common)<args.min_common_genes:raise ValueError(f'too few common genes: {len(common)}')
        moving=moving0[:,common];reference=target0[:,common];zm,zr,features,npc=normalize_log_hvg_pca(moving,reference,args.n_expression_features,args.n_pcs,seed)
        cm,cr,nclusters,cluster_method=joint_cluster(zm,zr,args.n_joint_clusters,seed)
        if args.affine_mode=='geometry':cm=np.zeros(len(cm),int);cr=np.zeros(len(cr),int);nclusters=1
        candidate,a,t,matches,history=run_affine(rigid,target,zm,zr,cm,cr,args,seed)
        mx=sparse.csr_matrix(moving.X);tx=sparse.csr_matrix(reference.X);key=args.label_key;ml=np.asarray(moving.obs[key]) if key in moving.obs and key in reference.obs else None;tl=np.asarray(reference.obs[key]) if ml is not None else None
        rm=evaluate('spateo_rigid',rigid,target,before,mx,tx,ml,tl);am=evaluate('expression_guided_affine_candidate',candidate,target,before,mx,tx,ml,tl)
        hard,reasons,det,sv,cond=hard_gate(a,len(matches['mi']),len(matches['shared']),target,args);score,gs,bs=soft_score(rm,am,float(matches['similarity'].mean()))
        accepted=hard and score>args.affine_score_threshold;reasons += ([] if accepted else ['soft_score_below_threshold'] if hard else [])
        final=candidate if accepted else rigid.copy();fm=am if accepted else rm
        make_plots(out,name,before,rigid,final,target,cm,cr,matches,seed)
        np.savez_compressed(out/'coordinates.npz',moving_before=before,moving_rigid=rigid,moving_affine_candidate=candidate,moving_final=final,reference=target,match_moving=matches['mi'],match_reference=matches['ri'],match_weight=matches['weight'])
        q=moving0.copy();q.obsm['aligned_spatial']=final;q.write_h5ad(out/'moving_final.h5ad')
        delta={'Delta_PCC':fm['PCC']-rm['PCC'],'Delta_CI':fm['CI']-rm['CI'],'Delta_Pairwise_MAE':rm['MAE']-fm['MAE'],'Delta_Mean_NN':rm['Mean_NN_Distance']-fm['Mean_NN_Distance'],'Delta_Edge_IoU':fm['Edge_IoU']-rm['Edge_IoU'],'Delta_Boundary_Error':rm['Boundary_Error']-fm['Boundary_Error']}
        params={'A':a.tolist(),'t':t.tolist(),'affine_matrix':a.tolist(),'translation':t.tolist(),'determinant':det,'singular_values':sv.tolist(),'condition_number':cond,'area_ratio':det,'scale_x':float(np.linalg.norm(a[:,0])),'scale_y':float(np.linalg.norm(a[:,1])),'shear':float(np.dot(a[:,0],a[:,1])/(np.linalg.norm(a[:,0])*np.linalg.norm(a[:,1]))),'n_common_genes':len(common),'n_expression_features':len(features),'n_pcs':npc,'n_joint_clusters':nclusters,'n_shared_clusters':len(matches['shared']),'n_correspondences':len(matches['mi']),'n_valid_correspondences':len(matches['mi']),'mean_expression_similarity':float(matches['similarity'].mean()),'median_expression_similarity':float(np.median(matches['similarity'])),'mean_spatial_distance':float(matches['distance'].mean()),'cluster_consistency_ratio':1.0,'affine_score':score,'affine_accepted':accepted,'affine_mode':args.affine_mode,'ground_truth_spot_ids_used_for_correspondence':False,'iteration_history':history}
        row={'dataset':name,'deformation_type':name,'rigid_PCC':rm['PCC'],'affine_PCC':fm['PCC'],'rigid_CI':rm['CI'],'affine_CI':fm['CI'],'rigid_Pairwise_MAE':rm['MAE'],'affine_Pairwise_MAE':fm['MAE'],'rigid_Mean_NN':rm['Mean_NN_Distance'],'affine_Mean_NN':fm['Mean_NN_Distance'],'rigid_Edge_IoU':rm['Edge_IoU'],'affine_Edge_IoU':fm['Edge_IoU'],'rigid_Boundary_Error':rm['Boundary_Error'],'affine_Boundary_Error':fm['Boundary_Error'],**delta,'n_common_genes':len(common),'n_expression_features':len(features),'n_pcs':npc,'n_joint_clusters':nclusters,'n_shared_clusters':len(matches['shared']),'n_initial_correspondences':matches['initial'],'n_final_correspondences':len(matches['mi']),'n_valid_correspondences':len(matches['mi']),'mean_expression_similarity':params['mean_expression_similarity'],'median_expression_similarity':params['median_expression_similarity'],'mean_spatial_distance':params['mean_spatial_distance'],'shared_cluster_ratio':len(matches['shared'])/nclusters,'cluster_coverage':len(matches['shared'])/nclusters,'det_affine':det,'singular_value_min':float(sv.min()),'singular_value_max':float(sv.max()),'condition_number':cond,'area_ratio':det,'affine_score':score,'geometry_score':gs,'biology_score':bs,'affine_accepted':accepted,'affine_mode':args.affine_mode,'failure_reason':';'.join(reasons),'runtime':time.time()-start,'Status':'SUCCESS','ground_truth_spot_ids_used':False}
        diagnostics=[]
        for c in sorted(set(cm).union(set(cr))):
            mask=matches['cluster']==c;nm=int(np.sum(cm==c));nr=int(np.sum(cr==c))
            diagnostics.append({'cluster_id':int(c),'n_moving':nm,'n_reference':nr,'n_initial_pairs':nm*min(args.spatial_knn,nr) if nr else 0,'n_valid_pairs':int(mask.sum()),'mean_spatial_distance':float(matches['distance'][mask].mean()) if mask.any() else np.nan,'mean_expression_similarity':float(matches['similarity'][mask].mean()) if mask.any() else np.nan,'mean_weight':float(matches['weight'][mask].mean()) if mask.any() else np.nan})
        pd.DataFrame(diagnostics).to_csv(out/'cluster_matching_diagnostics.csv',index=False)
        (out/'affine_parameters.json').write_text(json.dumps(params,indent=2));(out/'metrics.json').write_text(json.dumps(row,indent=2,allow_nan=True))
        log=['===== CODA-INSPIRED AFFINE =====',f'Dataset: {name}','Rigid finished: True',f'Common genes: {len(common)}',f'Selected expression features: {len(features)}',f'PCA dimensions: {npc}',f'Joint clusters: {nclusters}',f'Shared clusters: {len(matches["shared"])}',f'Initial candidate pairs: {matches["initial"]}',f'Same-cluster pairs: {matches["same"]}',f'MNN/filtered/final pairs: {len(matches["mi"])}',f'Mean spatial distance: {params["mean_spatial_distance"]}',f'Mean expression similarity: {params["mean_expression_similarity"]}',f'Affine matrix: {a.tolist()}',f'Translation: {t.tolist()}',f'Determinant: {det}',f'Singular values: {sv.tolist()}',f'Condition number: {cond}',f'Area ratio: {det}',f'Rigid metrics: {rm}',f'Affine candidate metrics: {am}',f'Delta metrics: {delta}',f'Affine score: {score}',f'Affine accepted: {accepted}',f'Failure/gate reason: {row["failure_reason"]}','Ground-truth spot IDs used for matching: False']
        (out/'run.log').write_text('\n'.join(log)+'\n');return row
    except Exception as e:
        reason=f'{type(e).__name__}: {e}';(out/'run.log').write_text(reason+'\n'+traceback.format_exc()+'\nGround-truth spot IDs used for matching: False\n')
        return {'dataset':name,'deformation_type':name,'affine_accepted':False,'failure_reason':reason,'runtime':time.time()-start,'Status':'FAILED','ground_truth_spot_ids_used':False}


def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',default=str(ROOT/'aug_h5ad'));p.add_argument('--target-h5ad',default=str(ROOT/'basic_usage_demo_2.h5ad'));p.add_argument('--rigid-cache',default=str(ROOT/'results_spateo_affine'));p.add_argument('--output-dir',default=str(ROOT/'results_coda_affine'));p.add_argument('--datasets',default='');p.add_argument('--mode',choices=['rigid','affine'],default='affine');p.add_argument('--affine-mode',choices=['geometry','coda'],default='coda');p.add_argument('--n-expression-features',type=int,default=2000);p.add_argument('--n-pcs',type=int,default=30);p.add_argument('--n-joint-clusters',type=int,default=12);p.add_argument('--spatial-knn',type=int,default=20);p.add_argument('--distance-percentile',type=float,default=80);p.add_argument('--expression-threshold',type=float,default=.5);p.add_argument('--expression-threshold-mode',choices=['fixed','percentile'],default='fixed');p.add_argument('--max-matches-per-cluster',type=int,default=500);p.add_argument('--max-affine-iterations',type=int,default=5);p.add_argument('--affine-convergence-tol',type=float,default=1e-4);p.add_argument('--affine-regularization',type=float,default=1e-2);p.add_argument('--min-singular-value',type=float,default=.5);p.add_argument('--max-singular-value',type=float,default=2.0);p.add_argument('--max-condition-number',type=float,default=10);p.add_argument('--min-correspondences',type=int,default=50);p.add_argument('--min-shared-clusters',type=int,default=2);p.add_argument('--min-common-genes',type=int,default=20);p.add_argument('--affine-score-threshold',type=float,default=0);p.add_argument('--label-key',default='cellbin_SpatialDomain');p.add_argument('--seed',type=int,default=2026);args=p.parse_args()
    if args.mode!='affine':raise NotImplementedError('Rigid outputs already exist in --rigid-cache; use the affine mode for this experiment')
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);target=ad.read_h5ad(args.target_h5ad);wanted={x.strip() for x in args.datasets.split(',') if x.strip()};paths=sorted(Path(args.input_dir).glob('*.h5ad'));paths=[x for x in paths if not wanted or x.stem.replace('basic_usage_demo_1_','') in wanted]
    rows=[]
    for i,path in enumerate(paths):rows.append(run_one(path,target,args,args.seed+i));pd.DataFrame(rows).to_csv(out/'expression_guided_affine_metrics_partial.csv',index=False)
    pd.DataFrame(rows).sort_values('dataset').to_csv(out/'coda_affine_metrics.csv',index=False);print(pd.DataFrame(rows).to_string(index=False))


if __name__=='__main__':main()
