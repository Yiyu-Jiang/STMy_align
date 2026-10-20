#!/usr/bin/env python3
"""Geometry-Dominant Expression-Guided Boundary-Aware affine (GDEBA).

Rigid coordinates are loaded from the saved Spateo run.  No non-rigid module
and no ground-truth spot identity is used.
"""
from __future__ import annotations
import argparse,json,time,traceback
from pathlib import Path
import anndata as ad
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np,pandas as pd
from scipy import sparse
from scipy.spatial import ConvexHull,cKDTree

from compute_spateo_affine_metrics import evaluate
from run_expression_guided_affine import normalize_log_hvg_pca,joint_cluster,weighted_affine,fps

ROOT=Path(__file__).resolve().parent

def boundary_ids(x):
    try:return ConvexHull(x).vertices
    except Exception:return np.arange(len(x))

def geometry_initialization(source,target,min_sv=.5,max_sv=2):
    def sqrt_spd(c,inverse=False):
        w,v=np.linalg.eigh(c+1e-8*np.eye(2));w=np.maximum(w,1e-8)
        return (v*(1/np.sqrt(w) if inverse else np.sqrt(w)))@v.T
    a=sqrt_spd(np.cov(source,rowvar=False),True)@sqrt_spd(np.cov(target,rowvar=False),False)
    u,s,vt=np.linalg.svd(a);a=u@np.diag(np.clip(s,min_sv,max_sv))@vt
    t=target.mean(0)-source.mean(0)@a
    return a,t

def match_global(current,target,zm,zr,cm,cr,mode,args,seed):
    k=min(args.spatial_knn,len(target));dist,j=cKDTree(target).query(current,k=k)
    if k==1:dist=dist[:,None];j=j[:,None]
    sim=np.einsum('ij,ikj->ik',zm,zr[j]);sigma=max(np.median(dist),1e-8);geo=np.exp(-dist*dist/(2*sigma*sigma))
    same=cm[:,None]==cr[j]
    if mode=='cluster_hard':score=.7*geo+.3*((sim+1)/2);score[~same]=-np.inf
    elif mode in ('expression_soft','expression_boundary'):
        expr=args.expression_weight_floor+(1-args.expression_weight_floor)*np.maximum(sim,0)
        cluster=np.where(same,1,args.cross_cluster_weight);score=geo*(args.geometry_base_weight+(1-args.geometry_base_weight)*expr)*cluster
    else:score=geo
    pick=np.argmax(score,axis=1);ri=j[np.arange(len(current)),pick];dd=dist[np.arange(len(current)),pick];ss=sim[np.arange(len(current)),pick];same1=same[np.arange(len(current)),pick]
    back=cKDTree(current).query(target)[1];keep=back[ri]==np.arange(len(current))
    if mode=='cluster_hard':keep &= same1
    if keep.any():keep &= dd<=np.percentile(dd[keep],args.interior_distance_percentile)
    mi=np.where(keep)[0];ri=ri[keep];dd=dd[keep];ss=ss[keep];same1=same1[keep]
    wg=np.exp(-dd*dd/(2*sigma*sigma));expr=args.expression_weight_floor+(1-args.expression_weight_floor)*np.maximum(ss,0)
    if mode in ('geometry_only','geometry_boundary'):
        w=wg
    else:
        wc=np.where(same1,1,args.cross_cluster_weight)
        w=wg*(args.geometry_base_weight+(1-args.geometry_base_weight)*expr)*wc
    if len(mi)>args.max_interior_matches:
        take=fps(current[mi],args.max_interior_matches,seed);mi,ri,dd,ss,w=mi[take],ri[take],dd[take],ss[take],w[take]
    return dict(mi=mi,ri=ri,distance=dd,similarity=ss,weight=w,is_boundary=np.zeros(len(mi),bool),same_cluster=same1,initial=int(dist.size))

def boundary_matches(current,target,zm,zr,args,seed,use_expression=True):
    sm=boundary_ids(current);sr=boundary_ids(target);d,j=cKDTree(target[sr]).query(current[sm]);back=cKDTree(current[sm]).query(target[sr])[1]
    keep=back[j]==np.arange(len(sm))
    if keep.any() and args.boundary_distance_percentile<100:keep &= d<=np.percentile(d[keep],args.boundary_distance_percentile)
    mi=sm[keep];ri=sr[j[keep]];dd=d[keep];ss=np.sum(zm[mi]*zr[ri],axis=1)
    if len(mi)>args.max_boundary_matches:
        take=fps(current[mi],args.max_boundary_matches,seed);mi,ri,dd,ss=mi[take],ri[take],dd[take],ss[take]
    sigma=max(np.median(dd),1e-8);w=args.boundary_weight*np.exp(-dd*dd/(2*sigma*sigma))
    if use_expression:
        w*=args.geometry_base_weight+(1-args.geometry_base_weight)*(args.expression_weight_floor+(1-args.expression_weight_floor)*np.maximum(ss,0))
    return dict(mi=mi,ri=ri,distance=dd,similarity=ss,weight=w,is_boundary=np.ones(len(mi),bool),same_cluster=np.ones(len(mi),bool),initial=len(sm)*len(sr))

def _stratified_limit(current,mi,clusters,max_keep,seed):
    """Limit matches while preserving cluster and coarse spatial coverage."""
    if len(mi)<=max_keep:return np.arange(len(mi))
    rng=np.random.RandomState(seed)
    pts=current[mi]
    lo=pts.min(0);span=np.maximum(np.ptp(pts,axis=0),1e-9)
    grid=np.clip(((pts-lo)/span*5).astype(int),0,4)
    strata=[(int(c),int(g[0]),int(g[1])) for c,g in zip(clusters,grid)]
    groups={}
    for i,s in enumerate(strata):groups.setdefault(s,[]).append(i)
    keys=list(groups)
    rng.shuffle(keys)
    chosen=[]
    # First pass: one representative from as many strata as possible.
    for key in keys:
        if len(chosen)>=max_keep:break
        chosen.append(rng.choice(groups[key]))
    if len(chosen)<max_keep:
        remaining=np.setdiff1d(np.arange(len(mi)),np.asarray(chosen,dtype=int),assume_unique=False)
        if len(remaining):
            # Farthest-point sampling over the remaining matches for spatial spread.
            need=min(max_keep-len(chosen),len(remaining))
            take=fps(current[mi[remaining]],need,seed+17)
            chosen.extend(remaining[take].tolist())
    return np.asarray(chosen[:max_keep],dtype=int)

def match_global_prefiltered(current,target,zm,zr,cm,cr,mode,args,seed):
    """Lightweight pre-filtered interior matches for faster affine solving.

    The original match_global remains available.  This version reduces the KNN
    candidate pool before expensive scoring/IRLS using spatial KNN, expression
    top-ratio filtering, mutual NN filtering, and stratified/FPS sampling.
    """
    log={'fallback':False,'fallback_reason':''}
    k=min(int(getattr(args,'prefilter_knn',8)),len(target))
    dist,j=cKDTree(target).query(current,k=k)
    if k==1:dist=dist[:,None];j=j[:,None]
    n,k2=dist.shape
    mi_all=np.repeat(np.arange(n),k2)
    ri_all=j.reshape(-1)
    dd_all=dist.reshape(-1)
    sim_all=np.einsum('ij,ij->i',zm[mi_all],zr[ri_all])
    same_all=cm[mi_all]==cr[ri_all]
    log['raw_candidate_pairs']=int(len(mi_all))

    finite=np.isfinite(dd_all)&np.isfinite(sim_all)
    if finite.any():
        dist_thr=np.percentile(dd_all[finite],float(getattr(args,'prefilter_distance_percentile',90)))
        keep=finite&(dd_all<=dist_thr)
    else:
        keep=finite
    log['after_spatial_distance_filter']=int(keep.sum())

    use_expr=mode in ('expression_soft','expression_boundary','cluster_hard') and zm is not None and zr is not None
    if use_expr and keep.any():
        top_ratio=float(getattr(args,'prefilter_top_ratio',0.4))
        top_ratio=min(max(top_ratio,0.01),1.0)
        thr=np.quantile(sim_all[keep],1-top_ratio)
        keep &= sim_all>=thr
        log['expression_filter_used']=True
        log['expression_similarity_threshold']=float(thr)
    else:
        log['expression_filter_used']=False
        log['expression_similarity_threshold']=np.nan
    log['after_expression_similarity_filter']=int(keep.sum())

    if mode=='cluster_hard':keep &= same_all
    if not keep.any():
        log.update(fallback=True,fallback_reason='empty_after_expression_or_cluster_filter')
        keep=finite

    # Pick the best surviving pair for each moving point.
    idx=np.where(keep)[0]
    sigma=max(np.median(dd_all[idx]) if len(idx) else np.median(dd_all[finite]),1e-8)
    geo=np.exp(-dd_all[idx]*dd_all[idx]/(2*sigma*sigma))
    if mode in ('expression_soft','expression_boundary','cluster_hard'):
        expr=getattr(args,'expression_weight_floor',.3)+(1-getattr(args,'expression_weight_floor',.3))*np.maximum(sim_all[idx],0)
        cluster=np.where(same_all[idx],1,getattr(args,'cross_cluster_weight',.2))
        score=geo*(getattr(args,'geometry_base_weight',.7)+(1-getattr(args,'geometry_base_weight',.7))*expr)*cluster
    else:
        score=geo
    order=np.lexsort((-score,mi_all[idx]))
    sorted_idx=idx[order]
    _,first=np.unique(mi_all[sorted_idx],return_index=True)
    chosen=sorted_idx[first]
    mi=mi_all[chosen];ri=ri_all[chosen];dd=dd_all[chosen];ss=sim_all[chosen];same1=same_all[chosen]
    log['after_best_per_moving']=int(len(mi))

    if bool(getattr(args,'enable_mutual_nn',True)) and len(mi):
        back=cKDTree(current).query(target)[1]
        mkeep=back[ri]==mi
        if mkeep.any():
            mi,ri,dd,ss,same1=mi[mkeep],ri[mkeep],dd[mkeep],ss[mkeep],same1[mkeep]
            log['after_mutual_nn_filter']=int(len(mi))
        else:
            log.update(fallback=True,fallback_reason=(log['fallback_reason']+';empty_after_mutual_nn').strip(';'))
            log['after_mutual_nn_filter']=0
    else:
        log['after_mutual_nn_filter']=int(len(mi))

    if len(mi) and getattr(args,'interior_distance_percentile',100)<100:
        q=np.percentile(dd,float(args.interior_distance_percentile))
        dkeep=dd<=q
        mi,ri,dd,ss,same1=mi[dkeep],ri[dkeep],dd[dkeep],ss[dkeep],same1[dkeep]
    log['after_interior_distance_percentile']=int(len(mi))

    if len(mi)<int(getattr(args,'min_matches_for_affine',getattr(args,'min_correspondences',50))):
        # Controlled fallback to the original robust matcher when pre-filtering
        # becomes too strict.  This preserves correctness and logs the event.
        orig=match_global(current,target,zm,zr,cm,cr,mode,args,seed)
        orig['prefilter_log']={**log,'fallback':True,'fallback_reason':(log['fallback_reason']+';too_few_prefilter_matches').strip(';'),'used_original_matcher':True,'final_interior_matches':int(len(orig['mi']))}
        return orig

    wg=np.exp(-dd*dd/(2*sigma*sigma))
    expr=getattr(args,'expression_weight_floor',.3)+(1-getattr(args,'expression_weight_floor',.3))*np.maximum(ss,0)
    if mode in ('geometry_only','geometry_boundary'):
        w=wg
    else:
        wc=np.where(same1,1,getattr(args,'cross_cluster_weight',.2))
        w=wg*(getattr(args,'geometry_base_weight',.7)+(1-getattr(args,'geometry_base_weight',.7))*expr)*wc

    max_keep=int(getattr(args,'max_interior_matches',2500))
    if len(mi)>max_keep:
        if bool(getattr(args,'enable_stratified_sampling',True)):
            take=_stratified_limit(current,mi,cm[mi],max_keep,seed)
        else:
            take=fps(current[mi],max_keep,seed)
        mi,ri,dd,ss,w,same1=mi[take],ri[take],dd[take],ss[take],w[take],same1[take]
    log['sampled_interior_matches']=int(len(mi))
    log['used_original_matcher']=False
    log['final_interior_matches']=int(len(mi))
    return dict(mi=mi,ri=ri,distance=dd,similarity=ss,weight=w,is_boundary=np.zeros(len(mi),bool),same_cluster=same1,initial=int(dist.size),prefilter_log=log)

def boundary_matches_prefiltered(current,target,zm,zr,args,seed,use_expression=True):
    sm=boundary_ids(current);sr=boundary_ids(target)
    log={'raw_boundary_candidate_pairs':int(len(sm)*len(sr)),'fallback':False,'fallback_reason':''}
    if not len(sm) or not len(sr):
        log.update(fallback=True,fallback_reason='empty_boundary_ids')
        return boundary_matches(current,target,zm,zr,args,seed,use_expression)
    d,j=cKDTree(target[sr]).query(current[sm])
    back=cKDTree(current[sm]).query(target[sr])[1]
    keep=back[j]==np.arange(len(sm))
    log['after_boundary_mutual_nn']=int(keep.sum())
    if keep.any() and args.boundary_distance_percentile<100:
        keep &= d<=np.percentile(d[keep],args.boundary_distance_percentile)
    log['after_boundary_distance_filter']=int(keep.sum())
    if not keep.any():
        log.update(fallback=True,fallback_reason='empty_boundary_after_filter')
        keep=np.ones(len(sm),dtype=bool)
    mi=sm[keep];ri=sr[j[keep]];dd=d[keep];ss=np.sum(zm[mi]*zr[ri],axis=1)
    maxb=int(getattr(args,'max_boundary_matches',800))
    if len(mi)>maxb:
        take=fps(current[mi],maxb,seed)
        mi,ri,dd,ss=mi[take],ri[take],dd[take],ss[take]
    sigma=max(np.median(dd),1e-8)
    w=args.boundary_weight*np.exp(-dd*dd/(2*sigma*sigma))
    if use_expression:
        w*=args.geometry_base_weight+(1-args.geometry_base_weight)*(args.expression_weight_floor+(1-args.expression_weight_floor)*np.maximum(ss,0))
    log['sampled_boundary_matches']=int(len(mi))
    out=dict(mi=mi,ri=ri,distance=dd,similarity=ss,weight=w,is_boundary=np.ones(len(mi),bool),same_cluster=np.ones(len(mi),bool),initial=len(sm)*len(sr))
    out['prefilter_log']=log
    return out

def merge(a,b=None):
    if b is None:return a
    out={}
    for k in a:
        if k=='initial':out[k]=a[k]+b[k]
        elif k=='prefilter_log':out[k]={'interior':a.get(k,{}),'boundary':b.get(k,{})}
        else:out[k]=np.concatenate((a[k],b[k]))
    return out

def solve_prior(x,y,w,a0,t0,reg=.01,rounds=6):
    d=np.c_[x,np.ones(len(x))];prior=np.vstack((a0,t0));theta=prior.copy();base=w/max(np.mean(w),1e-12)
    for _ in range(rounds):
        res=np.linalg.norm(d@theta-y,axis=1);scale=max(np.median(res)*1.4826,1e-8);ww=base*np.minimum(1,1.345*scale/np.maximum(res,1e-12))
        pen=np.diag([reg,reg,reg*.01]);theta=np.linalg.solve(d.T@(ww[:,None]*d)+pen,d.T@(ww[:,None]*y)+pen@prior)
    return theta[:2],theta[2]

def optimize_mode(base,target,zm,zr,cm,cr,mode,args,seed):
    if mode in ('geometry_only','geometry_boundary','expression_soft','expression_boundary'):
        a0,t0=geometry_initialization(base,target,args.min_singular_value,args.max_singular_value);current=base@a0+t0
    else:a0=np.eye(2);t0=np.zeros(2);current=base.copy()
    history=[];matches=None
    for it in range(args.max_affine_iterations):
        matches=match_global(current,target,zm,zr,cm,cr,mode,args,seed+it)
        if mode in ('expression_boundary','geometry_boundary'):
            matches=merge(matches,boundary_matches(current,target,zm,zr,args,seed+100+it,use_expression=mode=='expression_boundary'))
        if len(matches['mi'])<args.min_correspondences:raise ValueError(f'too few correspondences: {len(matches["mi"])}')
        a,t=solve_prior(base[matches['mi']],target[matches['ri']],matches['weight'],a0,t0,args.affine_regularization_weight)
        sv=np.linalg.svd(a,compute_uv=False)
        if getattr(args,'enforce_optimizer_safety',True) and (np.linalg.det(a)<=0 or sv.min()<args.min_singular_value or sv.max()>args.max_singular_value):raise ValueError(f'unsafe affine det={np.linalg.det(a)} sv={sv}')
        new=base@a+t;change=np.mean(np.linalg.norm(new-current,axis=1));current=new
        history.append({'iteration':it+1,'n_correspondences':len(matches['mi']),'n_boundary':int(matches['is_boundary'].sum()),'mean_update':float(change)})
        if change<args.convergence_tol:break
    return current,a,t,a0,t0,matches,history

def optimize_mode_prefiltered(base,target,zm,zr,cm,cr,mode,args,seed):
    """Optimized GDEBA affine path with lightweight pre-filtering.

    API-compatible with optimize_mode; each history item contains detailed
    candidate-count diagnostics for runtime and quality-control reports.
    """
    if mode in ('geometry_only','geometry_boundary','expression_soft','expression_boundary'):
        a0,t0=geometry_initialization(base,target,args.min_singular_value,args.max_singular_value);current=base@a0+t0
    else:a0=np.eye(2);t0=np.zeros(2);current=base.copy()
    history=[];matches=None
    for it in range(args.max_affine_iterations):
        iter_t=time.time()
        interior=match_global_prefiltered(current,target,zm,zr,cm,cr,mode,args,seed+it)
        matches=interior
        boundary=None
        if mode in ('expression_boundary','geometry_boundary'):
            boundary=boundary_matches_prefiltered(current,target,zm,zr,args,seed+100+it,use_expression=mode=='expression_boundary')
            matches=merge(matches,boundary)
        if len(matches['mi'])<args.min_correspondences:raise ValueError(f'too few correspondences: {len(matches["mi"])}')
        solve_t=time.time()
        a,t=solve_prior(base[matches['mi']],target[matches['ri']],matches['weight'],a0,t0,args.affine_regularization_weight)
        solve_runtime=time.time()-solve_t
        sv=np.linalg.svd(a,compute_uv=False)
        if getattr(args,'enforce_optimizer_safety',True) and (np.linalg.det(a)<=0 or sv.min()<args.min_singular_value or sv.max()>args.max_singular_value):raise ValueError(f'unsafe affine det={np.linalg.det(a)} sv={sv}')
        new=base@a+t;change=np.mean(np.linalg.norm(new-current,axis=1));current=new
        ilog=interior.get('prefilter_log',{})
        blog={} if boundary is None else boundary.get('prefilter_log',{})
        history.append({'iteration':it+1,'n_correspondences':len(matches['mi']),'n_boundary':int(matches['is_boundary'].sum()),'mean_update':float(change),'iteration_runtime_sec':time.time()-iter_t,'solver_runtime_sec':solve_runtime,**{f'interior_{k}':v for k,v in ilog.items()},**{f'boundary_{k}':v for k,v in blog.items()}})
        if change<args.convergence_tol:break
    return current,a,t,a0,t0,matches,history

def boundary_stats(source,target):
    bs=source[boundary_ids(source)];bt=target[boundary_ids(target)];ds=cKDTree(bt).query(bs)[0];dt=cKDTree(bs).query(bt)[0]
    return {'boundary_chamfer':float(np.mean(ds*ds)+np.mean(dt*dt)),'boundary_hausdorff':float(max(ds.max(),dt.max()))}

def scale_diag(before,after,target,a0,a):
    def ratios(x):return np.ptp(target,axis=0)/np.maximum(np.ptp(x,axis=0),1e-9)
    rb=ratios(before);ra=ratios(after);eb=np.sort(np.linalg.eigvalsh(np.cov(target,rowvar=False))/np.maximum(np.linalg.eigvalsh(np.cov(before,rowvar=False)),1e-12));ea=np.sort(np.linalg.eigvalsh(np.cov(target,rowvar=False))/np.maximum(np.linalg.eigvalsh(np.cov(after,rowvar=False)),1e-12))
    s0=np.linalg.svd(a0,compute_uv=False);sf=np.linalg.svd(a,compute_uv=False)
    return dict(scale_x_initial=float(s0[0]),scale_y_initial=float(s0[1]),scale_x_final=float(sf[0]),scale_y_final=float(sf[1]),bbox_width_ratio_before=float(rb[0]),bbox_height_ratio_before=float(rb[1]),bbox_width_ratio_after=float(ra[0]),bbox_height_ratio_after=float(ra[1]),cov_eigenvalue_ratio_before=eb.tolist(),cov_eigenvalue_ratio_after=ea.tolist())

def metric_row(dataset,method,coords,target,raw,mx,tx,ml,tl,a,t,a0,matches,runtime,accepted):
    m=evaluate(method,coords,target,raw,mx,tx,ml,tl);sv=np.linalg.svd(a,compute_uv=False);b=boundary_stats(coords,target)
    return {'dataset':dataset,'method':method,'PCC':m['PCC'],'CI':m['CI'],'Pairwise_MAE':m['MAE'],'Mean_NN':m['Mean_NN_Distance'],'Edge_IoU':m['Edge_IoU'],'Boundary_Error':m['Boundary_Error'],**b,'scale_x':float(sv[0]),'scale_y':float(sv[1]),'shear':float(np.dot(a[:,0],a[:,1])/(np.linalg.norm(a[:,0])*np.linalg.norm(a[:,1]))),'det':float(np.linalg.det(a)),'condition_number':float(sv.max()/sv.min()),'n_correspondences':0 if matches is None else len(matches['mi']),'n_boundary_correspondences':0 if matches is None else int(matches['is_boundary'].sum()),'mean_expression_similarity':np.nan if matches is None or not len(matches['mi']) else float(matches['similarity'].mean()),'affine_accepted':accepted,'runtime':runtime,'Status':'SUCCESS'}

def plot_ablation(path,name,coords,target):
    z=np.vstack([target,*coords.values()]);lo=z.min(0);hi=z.max(0);pad=np.maximum(np.ptp(z,axis=0)*.04,1);fig,axs=plt.subplots(1,len(coords),figsize=(5*len(coords),5),dpi=300)
    for ax,(title,s) in zip(axs,coords.items()):
        ax.scatter(target[:,0],target[:,1],s=.35,c='#d73027',alpha=.28,lw=0);ax.scatter(s[:,0],s[:,1],s=.35,c='#2166ac',alpha=.28,lw=0);ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]));ax.set_aspect('equal');ax.set_title(title);ax.axis('off')
    fig.suptitle(name);fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)

def plot_correspondence(path,current,target,matches,title,seed):
    rng=np.random.RandomState(seed);n=min(300,len(matches['mi']));ids=np.sort(rng.choice(len(matches['mi']),n,replace=False));fig,ax=plt.subplots(figsize=(7,6),dpi=300)
    ax.scatter(target[:,0],target[:,1],s=.18,c='#d73027',alpha=.16);ax.scatter(current[:,0],current[:,1],s=.18,c='#2166ac',alpha=.16)
    w=matches['weight'][ids];w=w/max(w.max(),1e-12)
    for q,alpha in zip(ids,w):
        a=current[matches['mi'][q]];b=target[matches['ri'][q]];ax.plot([a[0],b[0]],[a[1],b[1]],c='#31a354' if not matches['is_boundary'][q] else '#756bb1',lw=.35,alpha=.08+.65*alpha)
    ax.set_aspect('equal');ax.set_title(title);ax.axis('off');fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)

def plot_boundaries(path,rigid,final,target):
    fig,ax=plt.subplots(figsize=(7,6),dpi=300)
    for x,c,label in [(target,'#d73027','Reference boundary'),(rigid,'#636363','Rigid boundary'),(final,'#2166ac','Final GDEBA boundary')]:
        ids=boundary_ids(x);q=np.vstack((x[ids],x[ids[0]]));ax.plot(q[:,0],q[:,1],c=c,lw=1.4,label=label)
    ax.set_aspect('equal');ax.axis('off');ax.legend(frameon=False);fig.tight_layout();fig.savefig(path,bbox_inches='tight');plt.close(fig)

def run_dataset(path,target0,args):
    name=path.stem.replace('basic_usage_demo_1_','');out=Path(args.output_dir)/name;out.mkdir(parents=True,exist_ok=True);started=time.time();rows=[]
    try:
        z=np.load(Path(args.rigid_cache)/name/'coordinates_all_stages.npz');raw=z['moving_before'];rigid=z['moving_rigid'];target=z['reference_rigid']
        mov0=ad.read_h5ad(path);common=np.intersect1d(mov0.var_names[~mov0.var_names.duplicated()],target0.var_names[~target0.var_names.duplicated()]);mov=mov0[:,common];ref=target0[:,common]
        zm,zr,features,npc=normalize_log_hvg_pca(mov,ref,args.n_expression_features,args.n_pcs,args.seed);cm,cr,nclusters,_=joint_cluster(zm,zr,args.n_joint_clusters,args.seed)
        mx=sparse.csr_matrix(mov.X);tx=sparse.csr_matrix(ref.X);key=args.label_key;ml=np.asarray(mov.obs[key]) if key in mov.obs and key in ref.obs else None;tl=np.asarray(ref.obs[key]) if ml is not None else None
        identity=np.eye(2);zero=np.zeros(2);rows.append(metric_row(name,'rigid',rigid,target,raw,mx,tx,ml,tl,identity,zero,identity,None,0,True));coordinates={'Rigid':rigid};details={}
        modes=args.modes.split(',')
        for mode in modes:
            st=time.time();coords,a,t,a0,t0,matches,history=optimize_mode(rigid,target,zm,zr,cm,cr,mode,args,args.seed);hard=np.linalg.det(a)>0 and np.linalg.cond(a)<=args.max_condition_number
            r=metric_row(name,mode,coords,target,raw,mx,tx,ml,tl,a,t,a0,matches,time.time()-st,hard);base=rows[0];geometry=np.mean([(base['Pairwise_MAE']-r['Pairwise_MAE'])/base['Pairwise_MAE'],(base['Mean_NN']-r['Mean_NN'])/base['Mean_NN'],(base['Boundary_Error']-r['Boundary_Error'])/base['Boundary_Error'],(r['Edge_IoU']-base['Edge_IoU'])/base['Edge_IoU'],(base['boundary_chamfer']-r['boundary_chamfer'])/base['boundary_chamfer']]);biology=np.mean([r['PCC']-base['PCC'],r['CI']-base['CI']]);r['affine_score']=.75*geometry+.25*biology;r['affine_accepted']=hard and r['affine_score']>0
            if not r['affine_accepted']:coords=rigid.copy();r=metric_row(name,mode,coords,target,raw,mx,tx,ml,tl,a,t,a0,matches,time.time()-st,False)|{'affine_score':r['affine_score']}
            rows.append(r);coordinates[mode]=coords;details[mode]={'A':a.tolist(),'t':t.tolist(),'A0':a0.tolist(),'history':history,**scale_diag(rigid,coords,target,a0,a),**boundary_stats(rigid,target),**{f'after_{k}':v for k,v in boundary_stats(coords,target).items()},'ground_truth_spot_ids_used':False}
            corr_name={'geometry_only':'geometry_correspondence.png','cluster_hard':'cluster_hard_correspondence.png','expression_soft':'expression_soft_correspondence.png','expression_boundary':'boundary_correspondence.png'}[mode]
            plot_correspondence(out/corr_name,coords,target,matches,mode.replace('_',' ').title()+' correspondence',args.seed)
        pd.DataFrame(rows).to_csv(out/'affine_ablation_metrics.csv',index=False);(out/'affine_parameters.json').write_text(json.dumps(details,indent=2));np.savez_compressed(out/'coordinates.npz',reference=target,**{k.replace(' ','_').lower():v for k,v in coordinates.items()})
        plot_ablation(out/(f'{name}_affine_ablation.png'),name,coordinates,target)
        # Required three-stage final plot uses expression_boundary as formal result.
        plot_ablation(out/'comparison_3panel.png',name,{'Before Alignment':raw,'Spateo Rigid':rigid,'GDEBA':coordinates.get('expression_boundary',rigid)},target)
        plot_boundaries(out/'boundary_overlay.png',rigid,coordinates.get('expression_boundary',rigid),target)
        log=['===== AFFINE DEBUG =====',f'Dataset: {name}','Rigid finished',f'Modes: {modes}',f'Common genes: {len(common)}',f'PCA: {npc}',f'Clusters: {nclusters}',pd.DataFrame(rows).to_string(index=False),'Ground-truth spot IDs used for matching: False'];(out/'run.log').write_text('\n'.join(log)+'\n')
        return rows
    except Exception as e:
        (out/'run.log').write_text(f'{type(e).__name__}: {e}\n{traceback.format_exc()}\nGround-truth spot IDs used for matching: False\n');return [{'dataset':name,'method':'FAILED','Status':'FAILED','failure_reason':str(e)}]

def main():
    p=argparse.ArgumentParser();p.add_argument('--input-dir',default=str(ROOT/'aug_h5ad'));p.add_argument('--target-h5ad',default=str(ROOT/'basic_usage_demo_2.h5ad'));p.add_argument('--rigid-cache',default=str(ROOT/'results_spateo_affine'));p.add_argument('--output-dir',default=str(ROOT/'results_gdeba_affine'));p.add_argument('--datasets',default='compression');p.add_argument('--modes',default='geometry_only,cluster_hard,expression_soft,expression_boundary');p.add_argument('--n-expression-features',type=int,default=2000);p.add_argument('--n-pcs',type=int,default=30);p.add_argument('--n-joint-clusters',type=int,default=12);p.add_argument('--spatial-knn',type=int,default=20);p.add_argument('--interior-distance-percentile',type=float,default=75);p.add_argument('--boundary-distance-percentile',type=float,default=95);p.add_argument('--expression-weight-floor',type=float,default=.3);p.add_argument('--geometry-base-weight',type=float,default=.7);p.add_argument('--cross-cluster-weight',type=float,default=.2);p.add_argument('--boundary-weight',type=float,default=3);p.add_argument('--max-interior-matches',type=int,default=3600);p.add_argument('--max-boundary-matches',type=int,default=400);p.add_argument('--max-affine-iterations',type=int,default=4);p.add_argument('--convergence-tol',type=float,default=1e-4);p.add_argument('--affine-regularization-weight',type=float,default=1e-2);p.add_argument('--min-singular-value',type=float,default=.5);p.add_argument('--max-singular-value',type=float,default=2);p.add_argument('--max-condition-number',type=float,default=10);p.add_argument('--min-correspondences',type=int,default=50);p.add_argument('--label-key',default='cellbin_SpatialDomain');p.add_argument('--seed',type=int,default=2026);args=p.parse_args()
    target=ad.read_h5ad(args.target_h5ad);wanted={x.strip() for x in args.datasets.split(',') if x.strip()};paths=[x for x in sorted(Path(args.input_dir).glob('*.h5ad')) if not wanted or x.stem.replace('basic_usage_demo_1_','') in wanted];allrows=[]
    for path in paths:allrows.extend(run_dataset(path,target,args))
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True);pd.DataFrame(allrows).to_csv(out/'affine_ablation_metrics.csv',index=False);print(pd.DataFrame(allrows).to_string(index=False))
if __name__=='__main__':main()
