"""Frozen-checkpoint clean/adverse Mach-3 alpha-40 regression, not test accuracy."""
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import torch
from scipy import ndimage
from jcp2026.common import ROOT, CONTRACT, read_json, write_json, sha256
from jcp2026.diagnostics import build_inputs
from jcp2026.models import make_model
from jcp2026.evaluate import binary_masks, hybrid_masks, figure
from ml.flow_aligned import FreestreamReference
from jcp2026.infer import predict_tiled


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--config",type=Path,default=ROOT/"configs/jcp_v4_campaign.json")
    parser.add_argument("--variant",required=True);parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args();cfg=read_json(args.config);ev=cfg["evaluation"]
    run=ROOT/cfg["results"]/args.variant/str(args.seed);out=run/"regression"
    if (out/"report.json").exists(): raise FileExistsError("Completed regression is immutable")
    checkpoint=torch.load(run/"checkpoint.pt",map_location="cpu",weights_only=True)
    if checkpoint["diagnostic_contract"]!=CONTRACT or checkpoint["config_sha256"]!=sha256(args.config):
        raise ValueError("Checkpoint contract mismatch")
    thresholds_doc=read_json(run/"evaluation/thresholds.json")
    if thresholds_doc["checkpoint_sha256"]!=sha256(run/"checkpoint.pt"): raise ValueError("Threshold mismatch")
    thresholds=[thresholds_doc["thresholds"][h] for h in ("shock","vortex_core")]
    model=make_model(checkpoint["spec"],checkpoint["base_channels"]);model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
    torch.set_num_threads(cfg["training"]["torch_num_threads"])
    source=ROOT/"data/processed/mfc_a40_stage2"
    with np.load(source/"grid_and_geometry.npz",allow_pickle=False) as z:
        x,y=z["x"],z["y"]
    ref=FreestreamReference(1,1/1.4,3*math.cos(math.radians(40)),3*math.sin(math.radians(40)))
    rows=[]
    for step,role,noise in [(2700,"clean_core_a40",False),(5400,"adverse_development_a40",False),(16200,"late_wake_a40",False),(2700,"synthetic_1pct_velocity_noise_a40",True)]:
        path=source/"frames"/f"step_{step:05d}.npz"
        with np.load(path,allow_pickle=False) as z:
            raw=z["input_fields"].astype(np.float64)
            fields=dict(rho=np.exp(raw[0]),pressure=np.exp(raw[1]),u=raw[2],v=raw[3],x=x,y=y)
            geometry=z["geometry_mask"].astype(bool)
            targets=np.zeros((6,*geometry.shape),np.uint8)
            targets[0],targets[1]=z["shock_mask"],z["vortex_mask"]
        if noise:
            rng=np.random.default_rng(20260904)
            for key in ("u","v"):
                fields[key]=fields[key]+np.where(geometry,0,rng.normal(0,.01*ref.speed_inf,geometry.shape))
        inputs,_=build_inputs(fields,ref)
        loaded=dict(inputs=inputs,targets=targets,geometry=geometry,observation_mask=~geometry,wall_band=ndimage.binary_dilation(geometry,iterations=2)&~geometry,x=x,y=y)
        probability=predict_tiled(model,torch,inputs[:checkpoint["input_channels"]],tile_size=ev["tile_size"],overlap=ev["tile_overlap"],batch_size=4)
        if not np.isfinite(probability).all(): raise ValueError("Nonfinite regression probability")
        learned=binary_masks(probability,thresholds,~geometry,ev["minimum_component_pixels"])
        hybrid=hybrid_masks(probability,learned,loaded,ev["hybrid"],ev["minimum_component_pixels"])
        item={"dataset_id":f"core_{step:05d}_{role}","role":role,"checkpoint_sha256":sha256(run/"checkpoint.pt"),"config_sha256":sha256(args.config),"input_sha256":sha256(path),"noise_seed":20260904 if noise else None,"noise_std_velocity":.03 if noise else 0,"thresholds":thresholds_doc["thresholds"],"learned_pixels":[int(m.sum()) for m in learned],"hybrid_pixels":[int(m.sum()) for m in hybrid],"metrics":{"finite_probability":True,"geometry_overlap":int(np.count_nonzero(learned & geometry))},"accuracy":None,"reference_note":"Old weak proposals, not independent truth; synthetic noisy reference is inherited from clean fields."}
        out.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(out/(role+".npz"),probabilities=probability,ml_shock=learned[0],ml_vortex_core=learned[1],hybrid_shock=hybrid[0],hybrid_vortex_core=hybrid[1],geometry=geometry,x=x,y=y)
        figure(out/"figures"/role,loaded,probability,targets[:2].astype(bool),learned,hybrid,item)
        rows.append(item);print("regression "+role,flush=True)
    write_json(out/"report.json",{"status":"PASS","scope":"numerical/visual regression; not detector accuracy","frames":rows,"independent_accuracy":False})


if __name__=="__main__":main()
