#!/usr/bin/env python3
"""Isolated official-Spateo non-rigid worker for the A/B/E/F experiment."""
from __future__ import annotations
import argparse,json,time,traceback,os,random
from pathlib import Path
import anndata as ad
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--moving',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True);p.add_argument('--device',required=True);p.add_argument('--kind',choices=['rigid','original','residual'],required=True);p.add_argument('--input-npz');p.add_argument('--input-key');p.add_argument('--reference-npz');p.add_argument('--reference-key');p.add_argument('--max-iter',type=int,default=200);p.add_argument('--K',type=int,default=15);p.add_argument('--beta',type=float,default=.01);p.add_argument('--lambda-vf',type=float,default=100);p.add_argument('--partial-robust-level',type=float,default=10);p.add_argument('--batch-size',type=int,default=1024);p.add_argument('--seed',type=int,default=42);a=p.parse_args()
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);started=time.time();status={}
    try:
        os.environ['PYTHONHASHSEED']=str(a.seed);random.seed(a.seed);np.random.seed(a.seed)
        import torch;torch.manual_seed(a.seed)
        if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
        import spateo as st
        moving=ad.read_h5ad(a.moving);reference=ad.read_h5ad(a.reference)
        key='six_method_input';moving.obsm[key]=np.asarray(moving.obsm['spatial'])[:,:2].copy();reference.obsm[key]=np.asarray(reference.obsm['spatial'])[:,:2].copy()
        if a.input_npz:moving.obsm[key]=np.load(a.input_npz)[a.input_key]
        if a.reference_npz:reference.obsm[key]=np.load(a.reference_npz)[a.reference_key]
        kwargs=dict(models=[reference,moving],spatial_key=key,key_added='six_spateo',mode='SN-S' if a.kind=='rigid' else 'SN-N',rep_layer='X_pca',rep_field='obsm',dissimilarity='cos',max_iter=a.max_iter,device=a.device,verbose=False,K=a.K,beta=a.beta,lambdaVF=a.lambda_vf,partial_robust_level=a.partial_robust_level,SVI_mode=True,batch_size=a.batch_size,pre_compute_dist=False,iter_key_added=None,vecfld_key_added=None)
        internal_rigid=True
        if a.kind=='residual':
            kwargs.update(nn_init=False,init_transform=False,nonrigid_start_iter=-1,update_R=False)
            # update_R=False freezes rotation, but official _update_rigid still
            # updates translation at every iteration.
            internal_rigid=True
        models,_=st.align.morpho_align(**kwargs);aligned=models[1]
        rigid=np.asarray(aligned.obsm['six_spateo_rigid'])[:,:2];nonrigid=np.asarray(aligned.obsm['six_spateo_nonrigid'])[:,:2]
        np.savez_compressed(out,moving_input=moving.obsm[key],reference=reference.obsm[key],moving_rigid=rigid,moving_nonrigid=nonrigid)
        status={'status':'SUCCESS','kind':a.kind,'device':a.device,'physical_cuda_visible_devices':__import__('os').environ.get('CUDA_VISIBLE_DEVICES'),'runtime_sec':time.time()-started,'seed':a.seed,'rng':{'PYTHONHASHSEED':str(a.seed),'python_random':a.seed,'numpy':a.seed,'torch':a.seed,'torch_cuda':a.seed},'nonrigid_internal_rigid_update':internal_rigid,'input_stage':a.input_key or 'raw','parameters':{'K':a.K,'beta':a.beta,'lambdaVF':a.lambda_vf,'max_iter':a.max_iter,'partial_robust_level':a.partial_robust_level,'batch_size':a.batch_size,'SVI_mode':True,'pre_compute_dist':False,'nn_init':a.kind in ('rigid','original'),'init_transform':a.kind in ('rigid','original'),'nonrigid_start_iter':80 if a.kind in ('rigid','original') else -1,'update_R':a.kind in ('rigid','original')},'ground_truth_correspondence_used':False}
    except Exception as e:
        status={'status':'FAILED','kind':a.kind,'device':a.device,'runtime_sec':time.time()-started,'error':f'{type(e).__name__}: {e}','traceback':traceback.format_exc(),'ground_truth_correspondence_used':False}
    out.with_suffix('.json').write_text(json.dumps(status,indent=2));print(json.dumps(status));raise SystemExit(0 if status['status']=='SUCCESS' else 1)
if __name__=='__main__':main()
