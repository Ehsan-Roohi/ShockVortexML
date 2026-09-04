"""Validation-only operating points and separate development ablations."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import torch

from jcp2026.common import ROOT, CONTRACT, read_json, write_json, sha256
from jcp2026.dataset import Frames
from jcp2026.models import make_model
from jcp2026.infer import predict_tiled
from evaluate_geometry_robust_v2 import clean_components, counts, metrics
from evaluate_stage5_joint import object_match_counts, symmetric_mask_distance


HEADS=("shock","vortex_core")


def binary_masks(probability, thresholds, domain, minimum):
    return np.stack([clean_components((probability[h]>=thresholds[h])&domain,minimum) for h in range(2)])


def hybrid_masks(probability, high, loaded, policy, minimum):
    """Fixed, instantaneous support; no temporal claim or overlap veto."""
    domain=loaded["observation_mask"] & ~loaded["geometry"]
    support=ndimage.binary_dilation(loaded["targets"][0].astype(bool),iterations=policy["shock_dilation_cells"])
    shock=high[0]|((probability[0]>=policy["shock_seed_floor"])&support&domain)
    inputs=loaded["inputs"]
    rotation=(np.abs(inputs[4])>=np.tanh(0.5/8))&(inputs[5]>=np.tanh(0.02/24))&(inputs[6]>=np.tanh(0.2/8))
    pool=(probability[1]>=policy["vortex_seed_floor"])&domain
    labels,n=ndimage.label(pool,structure=np.ones((3,3)))
    sizes=np.bincount(labels.ravel(),minlength=n+1)
    rotational=np.bincount(labels.ravel(),weights=rotation.ravel(),minlength=n+1)
    walls=np.bincount(labels.ravel(),weights=loaded["wall_band"].ravel(),minlength=n+1)
    keep=(sizes>=minimum)&(rotational/np.maximum(sizes,1)>=policy["rotation_support_fraction"])&(walls/np.maximum(sizes,1)<=policy["maximum_wall_fraction"])
    keep[0]=False
    vortex=high[1]|keep[labels]
    return np.stack([clean_components(shock,minimum),clean_components(vortex,minimum)])


def figure(path, loaded, probability, physics, learned, hybrid, metadata):
    x,y=loaded["x"],loaded["y"]; extent=(x[0],x[-1],y[0],y[-1])
    fig,axes=plt.subplots(1,5,figsize=(19,4.5),layout="constrained")
    fields=[(probability[0],"P(shock)","magma"),(probability[1],"P(vortex core)","magma")]
    for ax,(field,title,cmap) in zip(axes,fields):
        ax.imshow(field,origin="lower",extent=extent,cmap=cmap,vmin=0,vmax=1,aspect="equal");ax.set_title(title)
    for ax,masks,title in zip(axes[2:],(physics,learned,hybrid),("Physics weak proposal","ML only","ML-primary hybrid")):
        ax.imshow(loaded["inputs"][1],origin="lower",extent=extent,cmap="Greys",aspect="equal",vmin=-1,vmax=1)
        for mask,color in zip(masks,("#d55e00","#cc79a7")):
            if mask.any() and (~mask).any():
                ax.contour(x,y,mask.astype(float),levels=[.5],colors=[color],linewidths=.65)
        ax.set_title(title)
    for ax in axes:
        if loaded["geometry"].any():
            ax.contourf(x,y,loaded["geometry"].astype(float),levels=[.5,1.5],colors=["black"])
        ax.set_aspect("equal",adjustable="box");ax.set_xlabel("x / L");ax.set_ylabel("y / L")
    fig.suptitle(f"{metadata['dataset_id']} — full stored field; weak-reference development audit",fontsize=10)
    path.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(path.with_suffix(".png"),dpi=240);fig.savefig(path.with_suffix(".pdf"));plt.close(fig)
    write_json(path.with_suffix(".json"),metadata)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",type=Path,default=ROOT/"configs/jcp_v4_campaign.json")
    parser.add_argument("--variant",required=True);parser.add_argument("--seed",type=int,required=True)
    args=parser.parse_args();cfg=read_json(args.config);ev=cfg["evaluation"]
    run=ROOT/cfg["results"]/args.variant/str(args.seed);out=run/"evaluation"
    if (out/"report.json").exists():
        raise FileExistsError("Completed evaluation is immutable")
    out.mkdir(parents=True,exist_ok=True)
    checkpoint=torch.load(run/"checkpoint.pt",map_location="cpu",weights_only=True)
    if checkpoint["diagnostic_contract"]!=CONTRACT or checkpoint["config_sha256"]!=sha256(args.config):
        raise ValueError("Checkpoint input/config contract mismatch")
    torch.set_num_threads(cfg["training"]["torch_num_threads"])
    model=make_model(checkpoint["spec"],checkpoint["base_channels"]);model.load_state_dict(checkpoint["model_state_dict"]);model.eval()
    index=ROOT/cfg["output"]/"dataset_index.json"; grid=ev["thresholds"]; minimum=ev["minimum_component_pixels"]
    checkpoint_hash=sha256(run/"checkpoint.pt")
    threshold_path=out/"thresholds.json"
    prediction_metadata=[]

    def predict(ds,i):
        loaded=ds.load(i);r=loaded["record"]; dest=out/"probabilities"/(Path(r["cache_file"]).stem+".npz")
        if dest.exists():
            with np.load(dest,allow_pickle=False) as z:
                if str(z["checkpoint_sha256"])!=checkpoint_hash or str(z["cache_sha256"])!=r["cache_sha256"]:
                    raise ValueError("Stale saved prediction")
                probability=z["probabilities"].astype(np.float32)
                elapsed=float(z["inference_seconds"])
        else:
            if sha256(ROOT/r["cache_file"])!=r["cache_sha256"]:
                raise ValueError("Cache changed after freeze")
            n=checkpoint["input_channels"];started=time.perf_counter()
            probability=predict_tiled(model,torch,loaded["inputs"][:n],tile_size=ev["tile_size"],overlap=ev["tile_overlap"],batch_size=4)
            elapsed=time.perf_counter()-started
            dest.parent.mkdir(parents=True,exist_ok=True)
            # Float32 ensures saved/reloaded thresholds agree exactly.
            np.savez_compressed(dest,probabilities=probability,checkpoint_sha256=checkpoint_hash,cache_sha256=r["cache_sha256"],inference_seconds=elapsed)
        return loaded,probability,elapsed,dest

    validation=Frames(index,"validation",cache_size=1)
    if threshold_path.exists():
        frozen=read_json(threshold_path)
        if frozen["checkpoint_sha256"]!=checkpoint_hash:
            raise ValueError("Threshold/checkpoint mismatch")
    else:
        aggregate=defaultdict(lambda:np.zeros((2,len(grid),3),dtype=np.int64))
        for i,r in enumerate(validation.frames):
            loaded,prob,_,_=predict(validation,i); domain=loaded["observation_mask"]&~loaded["geometry"]
            for h in range(2):
                valid=loaded["valid"][h].astype(bool)&domain
                for ti,threshold in enumerate(grid):
                    mask=clean_components((prob[h]>=threshold)&domain,minimum)
                    aggregate[r["family"]][h,ti]+=counts(mask,loaded["targets"][h],valid)
            print(f"threshold calibration {args.variant} {i+1}/{len(validation)}",flush=True)
        scores=np.mean(np.stack([np.asarray([[metrics(c)["weak_dice"] for c in head] for head in value]) for value in aggregate.values()]),axis=0)
        thresholds=[float(grid[int(np.argmax(scores[h]))]) for h in range(2)]
        frozen={"checkpoint_sha256":checkpoint_hash,"dataset_index_sha256":sha256(index),"config_sha256":sha256(args.config),"thresholds":dict(zip(HEADS,thresholds)),"selection":"validation-only family-macro weak Dice","grid":grid,"family_macro_scores":scores.tolist(),"independent_calibration":False}
        write_json(threshold_path,frozen)
    thresholds=[frozen["thresholds"][h] for h in HEADS]
    accumulator=defaultdict(lambda:np.zeros((2,3),dtype=np.int64));objects=defaultdict(lambda:np.zeros(3,dtype=np.int64));rows=[]
    for split in ("validation","test"):
        ds=Frames(index,split,cache_size=1)
        last_by_case={r["case"]:max(z["time"] for z in ds.frames if z["case"]==r["case"]) for r in ds.frames}
        for i,r in enumerate(ds.frames):
            loaded,prob,seconds,prob_path=predict(ds,i);domain=loaded["observation_mask"]&~loaded["geometry"]
            learned=binary_masks(prob,thresholds,domain,minimum)
            equal=binary_masks(prob,[ev["equal_threshold_audit"]]*2,domain,minimum)
            physics=loaded["targets"][:2].astype(bool)&domain
            hybrid=hybrid_masks(prob,learned,loaded,ev["hybrid"],minimum)
            if np.any(learned & ~hybrid):
                raise AssertionError("Hybrid unexpectedly vetoed a learned pixel")
            frame_metrics={}
            for branch,masks in [("ml_only",learned),("hybrid",hybrid),("equal_threshold_ml",equal)]:
                key=(split,r["family"],branch); frame_metrics[branch]={}
                for h,name in enumerate(HEADS):
                    valid=loaded["valid"][h].astype(bool)&domain
                    c=counts(masks[h],physics[h],valid);accumulator[key][h]+=c
                    frame_metrics[branch][name]=metrics(c)
                valid=loaded["valid"][1].astype(bool)&domain
                obj=np.asarray(object_match_counts(masks[1]&valid,physics[1]&valid,minimum_pixels=minimum,minimum_iou=.1))
                objects[key]+=obj
                edge=lambda a:a & ~ndimage.binary_erosion(a)
                frame_metrics[branch]["shock_boundary_distance"]=symmetric_mask_distance(edge(masks[0])&loaded["valid"][0].astype(bool),edge(physics[0])&loaded["valid"][0].astype(bool),spacing=(float(np.mean(np.diff(loaded["y"]))),float(np.mean(np.diff(loaded["x"])))))
            for branch,masks in [("ml_only",learned),("physics_only",physics),("hybrid",hybrid)]:
                path=out/branch/(Path(r["cache_file"]).stem+".npz");path.parent.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(path,shock=masks[0],vortex_core=masks[1],background_other=domain&~np.any(masks,axis=0),geometry=loaded["geometry"],x=loaded["x"],y=loaded["y"])
            item={"dataset_id":r["dataset_id"],"case":r["case"],"family":r["family"],"time":r["time"],"split":split,"inference_seconds":seconds,"metrics":frame_metrics,"geometry_raw_positive_counts":[int(np.count_nonzero((prob[h]>=thresholds[h])&loaded["geometry"])) for h in range(2)],"checkpoint_sha256":checkpoint_hash,"config_sha256":sha256(args.config),"input_sha256":r["cache_sha256"],"probability_file":str(prob_path.relative_to(ROOT)),"thresholds":frozen["thresholds"]}
            rows.append(item)
            if r["time"]==last_by_case[r["case"]]:
                figure(out/"figures"/(split+"_"+r["case"]),loaded,prob,physics,learned,hybrid,item)
            print(f"evaluation {args.variant} {split} {i+1}/{len(ds)}",flush=True)
    summary=[]
    for (split,family,branch),value in accumulator.items():
        obj=objects[(split,family,branch)];denom=2*obj[0]+obj[1]+obj[2]
        summary.append({"split":split,"family":family,"branch":branch,"heads":{name:metrics(value[h]) for h,name in enumerate(HEADS)},"vortex_object_counts":obj.tolist(),"vortex_object_f1":float(2*obj[0]/denom) if denom else None})
    write_json(out/"report.json",{"status":"PASS","claim_boundary":cfg["claim_boundary"],"independent_accuracy":False,"physics_self_score_omitted":True,"hybrid_type":"instantaneous component support; temporal training ablation is distinct","threshold_freeze":frozen,"summary":summary,"frames":rows})


if __name__=="__main__":
    main()
