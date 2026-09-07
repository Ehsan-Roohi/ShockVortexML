"""Train-only rear-front repair with multiscale continuity supervision.

Only shock_decoder and shock_head can change.  Targets are entropy-free
compression/jump/NMS proposals on exposed development trajectories.  They are
weak supervision, never independent ground truth.
"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import torch
import torch.nn.functional as F
from jcp2026.common import ROOT, read_json, write_json, sha256
from jcp2026.diagnostics import build_inputs
from jcp2026.infer import predict_tiled, clean
from jcp2026.models import make_model
from ml.flow_aligned import FreestreamReference
from show_front_vs_response_20260906 import narrow_front

CONFIG=ROOT/'configs/rear_shock_repair_20260907_v1.json'
PREFIXES=('shock_decoder.','shock_head.')

def reference(case):
    m=2.7 if case=='ellipse_m2p7' else 2.2
    return FreestreamReference(1.,1/1.4,m,0.,1.4,1.)

def fields(path):
    with np.load(path) as z:return {k:z[k] for k in ('rho','pressure','u','v','x','y','geometry','observation_mask')}

def shock_logits(model,x):
    with torch.no_grad():
        a=model.encoder_1(x);b=model.encoder_2(model.pool(a));c=model.encoder_3(model.pool(b));latent=model.bottleneck(model.pool(c))
    return model.shock_head(model.shock_decoder(latent,c,b,a))[:,0]

def masked_bce(logits,target,valid,cap):
    positive=(target*valid).sum(); negative=((1-target)*valid).sum(); weight=(negative/positive.clamp_min(1)).clamp(1,cap)
    loss=F.binary_cross_entropy_with_logits(logits,target,reduction='none',pos_weight=weight)
    return (loss*valid).sum()/valid.sum().clamp_min(1)

def multiscale(logits,target,valid,cap):
    answer=logits.new_zeros(())
    for scale in (2,4):
        q=F.max_pool2d(logits[:,None],scale)[:,0]
        y=F.max_pool2d(target[:,None],scale)[:,0]
        v=-F.max_pool2d(-valid[:,None],scale)[:,0]
        answer+=masked_bce(q,y,v,cap)
    return answer/2

def prepare(inputs,predictions,out,cfg):
    bank=out/'teacher';bank.mkdir(parents=True);rows=[]
    for case_dir in sorted(p for p in inputs.iterdir() if p.is_dir()):
        selected=sorted(case_dir.glob('*.npz'))[::cfg['frame_stride']]
        for src in selected:
            f=fields(src); ref=reference(case_dir.name); domain=f['observation_mask'].astype(bool)&~f['geometry'].astype(bool)
            target,_=narrow_front(f,domain,ref.to_dict());target=ndi.binary_dilation(target,iterations=cfg['teacher_dilation_cells'])&domain
            halo=ndi.binary_dilation(target,iterations=cfg['teacher_ignore_cells'])&~target
            valid=domain&~halo
            with np.load(predictions/case_dir.name/src.name) as z:wake=z['wake_shear'].astype(bool)
            path=bank/(case_dir.name+'_'+src.name);np.savez_compressed(path,target=target,valid=valid,wake=wake)
            rows.append(dict(case=case_dir.name,input=str(src),teacher=str(path),input_sha256=sha256(src),teacher_sha256=sha256(path),positive_pixels=int(target.sum())))
    write_json(bank/'INDEX.json',dict(frames=rows,human_ground_truth=False,independent_accuracy=False,teacher='entropy-free compression/jump/NMS'))
    return rows

def patch(record,size,rng):
    f=fields(Path(record['input']));ref=reference(record['case']);x,_=build_inputs(f,ref)
    with np.load(record['teacher']) as z:y=z['target'];v=z['valid'];wake=z['wake']
    roll=rng.random();focus=y if roll<.65 else (wake&~y if roll<.85 else v)
    pts=np.argwhere(focus)
    if not len(pts):pts=np.argwhere(v)
    j,k=pts[int(rng.integers(len(pts)))];h,w=y.shape
    top=int(np.clip(j-rng.integers(size//4,3*size//4),0,h-size));left=int(np.clip(k-rng.integers(size//4,3*size//4),0,w-size));sl=np.s_[top:top+size,left:left+size]
    return x[:,sl[0],sl[1]].copy(),y[sl].astype(np.float32),v[sl].astype(np.float32),dict(case=record['case'],file=Path(record['input']).name,top=top,left=left)

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=read_json(CONFIG);t=cfg['training']
    if a.output.exists():raise FileExistsError('Use a new output directory')
    a.output.mkdir(parents=True);rows=prepare(a.inputs,a.predictions,a.output,cfg)
    cp_path=ROOT/cfg['parent_checkpoint'];assert sha256(cp_path)==cfg['parent_sha256'];cp=torch.load(cp_path,map_location='cpu',weights_only=True)
    model=make_model(cp['spec'],cp['base_channels']);model.load_state_dict(cp['model_state_dict']);parent=make_model(cp['spec'],cp['base_channels']);parent.load_state_dict(cp['model_state_dict'])
    for name,q in model.named_parameters():q.requires_grad_(name.startswith(PREFIXES))
    for q in parent.parameters():q.requires_grad_(False)
    model.eval();parent.eval();torch.set_num_threads(t['threads']);torch.manual_seed(cfg['seed']);torch.use_deterministic_algorithms(True);rng=np.random.default_rng(cfg['seed'])
    params=[q for q in model.parameters() if q.requires_grad];opt=torch.optim.AdamW(params,lr=t['learning_rate'],weight_decay=t['weight_decay']);sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,t['steps'],eta_min=t['minimum_learning_rate']);history=[];schedule=[];started=time.perf_counter()
    for step in range(t['steps']):
        samples=[patch(rows[int(rng.integers(len(rows)))],t['patch_size'],rng) for _ in range(t['batch_size'])];x,y,v=[torch.from_numpy(np.stack([q[i] for q in samples])) for i in range(3)]
        opt.zero_grad(set_to_none=True);logits=shock_logits(model,x);primary=masked_bce(logits,y,v,t['positive_weight_cap']);coarse=multiscale(logits,y,v,t['positive_weight_cap'])
        with torch.no_grad():base=shock_logits(parent,x)
        preserve=((logits-base).square()*(v*(1-y))).sum()/(v*(1-y)).sum().clamp_min(1);loss=primary+t['multiscale_weight']*coarse+t['parent_preservation_weight']*preserve
        loss.backward();torch.nn.utils.clip_grad_norm_(params,5.);opt.step();sched.step();schedule.extend(q[3] for q in samples)
        if (step+1)%50==0:
            row=dict(step=step+1,loss=float(loss.detach()),primary=float(primary.detach()),multiscale=float(coarse.detach()),preservation=float(preserve.detach()),seconds=time.perf_counter()-started);history.append(row);write_json(a.output/'PROGRESS.json',dict(status='TRAINING',history=history));print(row,flush=True)
    changed=[n for n,q in model.state_dict().items() if not torch.equal(q,cp['model_state_dict'][n])]
    assert changed and all(n.startswith(PREFIXES) for n in changed)
    new={**cp,'model_state_dict':model.state_dict(),'schema_version':'rear_shock_repair_20260907_v1','parent_checkpoint_sha256':cfg['parent_sha256'],'config_sha256':sha256(CONFIG),'claim_boundary':cfg['claim_boundary']};torch.save(new,a.output/'checkpoint.pt')
    write_json(a.output/'TRAINING_REPORT.json',dict(status='PASS_DEVELOPMENT_ONLY',checkpoint_sha256=sha256(a.output/'checkpoint.pt'),changed_state_names=changed,all_other_state_exact=True,teacher_frames=len(rows),history=history,schedule=schedule,seconds=time.perf_counter()-started,claim_boundary=cfg['claim_boundary']))

if __name__=='__main__':main()
