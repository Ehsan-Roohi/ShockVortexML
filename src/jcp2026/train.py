"""Matched-update, from-scratch training; no test access or legacy weights."""
from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
import numpy as np
import torch

from jcp2026.common import ROOT, CONTRACT, read_json, write_json, sha256, process_memory
from jcp2026.models import make_model
from train_stage5_joint import masked_bce_dice_loss, gradients_or_zeros, project_conflicting_gradients


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,default=ROOT/"configs/jcp_v4_campaign.json")
    p.add_argument("--variant",required=True)
    p.add_argument("--seed",type=int,required=True)
    args=p.parse_args()
    cfg=read_json(args.config); t=cfg["training"]; spec=cfg["variants"][args.variant]
    if args.seed not in cfg["seeds"]:
        raise ValueError("Seed is not preregistered")
    out=ROOT/cfg["results"]/args.variant/str(args.seed)
    out.mkdir(parents=True,exist_ok=True)
    if (out/"checkpoint.pt").exists():
        raise FileExistsError("Checkpoint already exists; refusing overwrite")
    bank=ROOT/cfg["output"]/"patches"; manifest=read_json(bank/"manifest.json")
    if manifest["config_sha256"]!=sha256(args.config):
        raise ValueError("Config changed after patch freeze")
    target_key="temporal_targets" if spec["temporal_targets"] else "targets"
    valid_key="temporal_valid" if spec["temporal_targets"] else "valid"
    arrays={k:np.load(bank/(k+".npy"),mmap_mode="r") for k in ("inputs",target_key,valid_key)}
    for k in arrays:
        if sha256(bank/(k+".npy"))!=manifest["hashes"][k]:
            raise ValueError("Patch bank hash mismatch")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.set_num_threads(t["torch_num_threads"]); torch.use_deterministic_algorithms(True)
    model=make_model(spec,t["base_channels"])
    optimizer=torch.optim.AdamW(model.parameters(),lr=t["learning_rate"],weight_decay=t["weight_decay"])
    nsteps=t["epochs"]*t["steps_per_epoch"]
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,nsteps,eta_min=t["minimum_learning_rate"])
    shared=model.shared_encoder_parameters(); history=[]; started=time.perf_counter()
    print({"variant":args.variant,"seed":args.seed,"parameters":sum(p.numel() for p in model.parameters()),"steps":nsteps},flush=True)
    for epoch in range(t["epochs"]):
        totals=[]; cosines=[]; conflicts=0; epoch_start=time.perf_counter(); model.train()
        for step in range(t["steps_per_epoch"]):
            first=(epoch*t["steps_per_epoch"]+step)*t["batch_size"]; sl=slice(first,first+t["batch_size"])
            x=torch.from_numpy(np.array(arrays["inputs"][sl,:7 if spec["diagnostics"] else 4],copy=True))
            y=torch.from_numpy(np.array(arrays[target_key][sl],dtype=np.float32)); v=torch.from_numpy(np.array(arrays[valid_key][sl],dtype=np.float32))
            optimizer.zero_grad(set_to_none=True); logits=model(x)
            losses=[masked_bce_dice_loss(logits[:,h],y[:,h],v[:,h],t["positive_weight_cap"][h])[0]*t["head_loss_weights"][h] for h in range(6)]
            shock=losses[0]+losses[4]; vortex=losses[1]+losses[5]; aux=losses[2]+losses[3]; total=shock+vortex+aux
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite loss")
            if spec["pcgrad"] and step%t["pcgrad_interval"]==0:
                gs=gradients_or_zeros(shock,shared,retain_graph=True); gv=gradients_or_zeros(vortex,shared,retain_graph=True); ga=gradients_or_zeros(aux,shared,retain_graph=True)
                gs,gv,cos,conflict=project_conflicting_gradients(gs,gv); cosines.append(cos); conflicts+=int(conflict)
                total.backward()
                for param,a,b,c in zip(shared,gs,gv,ga):
                    param.grad=(a+b+c).detach()
            else:
                total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step();scheduler.step();totals.append(float(total.detach()))
        row={"epoch":epoch+1,"loss":float(np.mean(totals)),"seconds":time.perf_counter()-epoch_start,"gradient_cosine":float(np.mean(cosines)) if cosines else None,"conflicts":conflicts,"gradient_audits":len(cosines)}
        history.append(row); print(row,flush=True)
        write_json(out/"progress.json",{"status":"TRAINING","variant":args.variant,"seed":args.seed,"history":history})
    checkpoint={"schema_version":"jcp_v4","diagnostic_contract":CONTRACT,"variant":args.variant,"seed":args.seed,"spec":spec,"base_channels":t["base_channels"],"model_state_dict":model.state_dict(),"config_sha256":sha256(args.config),"patch_manifest_sha256":sha256(bank/"manifest.json"),"input_channels":7 if spec["diagnostics"] else 4,"independent_accuracy":False}
    torch.save(checkpoint,out/"checkpoint.pt")
    report={"status":"PASS","scientific_status":"weak_pretraining_not_independent_validation","variant":args.variant,"seed":args.seed,"spec":spec,"seconds":time.perf_counter()-started,"parameters":sum(p.numel() for p in model.parameters()),"torch_version":str(torch.__version__),"device":"cpu","config_sha256":sha256(args.config),"dataset_index_sha256":manifest["dataset_index_sha256"],"patch_manifest_sha256":sha256(bank/"manifest.json"),"checkpoint_sha256":sha256(out/"checkpoint.pt"),"training_source_sha256":sha256(Path(__file__)),"history":history}
    report["process_memory"]=process_memory()
    write_json(out/"training_report.json",report)
    write_json(out/"progress.json",{"status":"TRAINED","checkpoint_sha256":report["checkpoint_sha256"]})


if __name__=="__main__":
    main()
