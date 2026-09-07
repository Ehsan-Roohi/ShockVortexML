"""Replay a repaired shock branch while preserving frozen core/wake outputs."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import torch
from jcp2026.diagnostics import build_inputs
from jcp2026.infer import predict_tiled, clean
from jcp2026.models import make_model
from ml.flow_aligned import FreestreamReference

def ref(case):
    m=2.7 if case=='ellipse_m2p7' else 2.2
    return FreestreamReference(1.,1/1.4,m,0.,1.4,1.)

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError('Use a new output directory')
    a.output.mkdir(parents=True);cp=torch.load(a.checkpoint,map_location='cpu',weights_only=True);model=make_model(cp['spec'],cp['base_channels']);model.load_state_dict(cp['model_state_dict']);model.eval();torch.set_num_threads(4);rows=[]
    for case_dir in sorted(q for q in a.inputs.iterdir() if q.is_dir()):
        target=a.output/case_dir.name;target.mkdir()
        for src in sorted(case_dir.glob('*.npz')):
            with np.load(src) as z:f={k:z[k] for k in ('rho','pressure','u','v','x','y','geometry','observation_mask')}
            domain=f['observation_mask'].astype(bool)&~f['geometry'].astype(bool);x,_=build_inputs(f,ref(case_dir.name));prob=predict_tiled(model,torch,x,tile_size=384,overlap=64,batch_size=4)[0];shock=clean((prob>=.97)&domain,9)
            with np.load(a.baseline/case_dir.name/src.name) as z:arrays={k:z[k] for k in z.files}
            old=arrays['shock'].astype(bool);arrays['p_shock']=prob;arrays['shock']=shock;arrays['background_other']=domain&~(shock|arrays['vortex_core']|arrays['wake_shear']);np.savez_compressed(target/src.name,**arrays)
            rows.append(dict(case=case_dir.name,frame=src.name,old_shock_pixels=int(old.sum()),new_shock_pixels=int(shock.sum())))
    (a.output/'SUMMARY.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
