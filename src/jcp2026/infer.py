"""Portable ML-only prediction from primitive CFD arrays; no label dependency."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from scipy import ndimage
import torch
from jcp2026.common import CONTRACT, read_json, write_json, sha256
from jcp2026.diagnostics import build_inputs
from jcp2026.models import make_model
from ml.flow_aligned import FreestreamReference


def predict_tiled(model, torch, inputs, *, tile_size, overlap, batch_size):
    if not 0 <= overlap < tile_size or batch_size < 1:
        raise ValueError("Invalid tile/overlap/batch setting")
    _,height,width=inputs.shape
    padded=np.pad(inputs,((0,0),(0,max(0,tile_size-height)),(0,max(0,tile_size-width))),mode="edge")
    def starts(n):
        return sorted(set(range(0,n-tile_size+1,tile_size-overlap))|{n-tile_size})
    positions=[(y,x) for y in starts(padded.shape[1]) for x in starts(padded.shape[2])]
    window1d=np.maximum(np.hanning(tile_size).astype(np.float32),.05)
    window=window1d[:,None]*window1d[None,:]
    result=np.zeros((6,*padded.shape[1:]),np.float32);weights=np.zeros(padded.shape[1:],np.float32)
    with torch.inference_mode():
        for start in range(0,len(positions),batch_size):
            batch=positions[start:start+batch_size]
            x=np.stack([padded[:,r:r+tile_size,c:c+tile_size] for r,c in batch])
            prediction=torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(x)))).cpu().numpy()
            for p,(r,c) in zip(prediction,batch):
                result[:,r:r+tile_size,c:c+tile_size]+=p*window
                weights[r:r+tile_size,c:c+tile_size]+=window
    result=result[:,:height,:width]/weights[None,:height,:width]
    if not np.isfinite(result).all(): raise FloatingPointError("Nonfinite network probabilities")
    return result


def clean(mask,minimum):
    labels,n=ndimage.label(mask,structure=np.ones((3,3)))
    counts=np.bincount(labels.ravel(),minlength=n+1);keep=counts>=minimum;keep[0]=False
    return keep[labels]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input",type=Path,required=True);p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--checkpoint",type=Path,required=True);p.add_argument("--thresholds",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--threads",type=int,default=8)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError("Prediction exists; use a new output name")
    checkpoint=torch.load(args.checkpoint,map_location="cpu",weights_only=True)
    if checkpoint["diagnostic_contract"]!=CONTRACT: raise ValueError("Checkpoint input contract mismatch")
    frozen=read_json(args.thresholds)
    if frozen["checkpoint_sha256"]!=sha256(args.checkpoint): raise ValueError("Thresholds belong to another checkpoint")
    reference=FreestreamReference(**read_json(args.reference));reference.validate()
    with np.load(args.input,allow_pickle=False) as z:
        fields={k:z[k] for k in ("rho","pressure","u","v","x","y")}
        geometry=z["geometry"].astype(bool) if "geometry" in z else np.zeros_like(fields["rho"],bool)
        observation=z["observation_mask"].astype(bool) if "observation_mask" in z else ~geometry
    domain=observation&~geometry
    if not domain.any(): raise ValueError("No observed fluid")
    for k in ("rho","pressure","u","v"):
        if not np.isfinite(fields[k][domain]).all(): raise ValueError("Nonfinite observed fluid")
    if np.any(fields["rho"][domain]<=0) or np.any(fields["pressure"][domain]<=0): raise ValueError("Nonpositive observed fluid")
    inputs,_=build_inputs(fields,reference)
    torch.set_num_threads(args.threads)
    model=make_model(checkpoint["spec"],checkpoint["base_channels"]);model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
    probability=predict_tiled(model,torch,inputs[:checkpoint["input_channels"]],tile_size=384,overlap=64,batch_size=4)
    masks=[clean((probability[h]>=frozen["thresholds"][name])&domain,9) for h,name in enumerate(("shock","vortex_core"))]
    instances,count=ndimage.label(masks[1],structure=np.ones((3,3)))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,p_shock=probability[0],p_vortex_core=probability[1],p_background_head=probability[2],shock=masks[0],vortex_core=masks[1],background_other=domain&~(masks[0]|masks[1]),vortex_instances=instances,geometry=geometry,observation_mask=observation,x=fields["x"],y=fields["y"])
    write_json(args.output.with_suffix(".json"),{"branch":"ML_ONLY","diagnostic_contract":CONTRACT,"input_sha256":sha256(args.input),"reference":reference.to_dict(),"checkpoint_sha256":sha256(args.checkpoint),"threshold_file_sha256":sha256(args.thresholds),"output_sha256":sha256(args.output),"vortex_candidate_instances":int(count),"instance_scope":"frame-local connected-component IDs, not physical truth or temporal tracks","probability_note":"Sigmoid scores; no independent probability calibration claim","generalization":"No guarantee on arbitrary geometry, solver, resolution or regime"})


if __name__=="__main__":main()
