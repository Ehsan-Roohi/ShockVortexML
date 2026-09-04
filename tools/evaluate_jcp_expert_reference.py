"""Fixed-output comparison against a completed, hashed expert reference."""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from PIL import Image
from scipy import ndimage
from jcp2026.common import ROOT, read_json, write_json, sha256
from evaluate_geometry_robust_v2 import counts
from evaluate_stage5_joint import object_match_counts, symmetric_mask_distance


def main():
    p=argparse.ArgumentParser();p.add_argument("--reference",type=Path,required=True);p.add_argument("--run",type=Path,required=True)
    args=p.parse_args();ref=read_json(args.reference);out=args.run/"expert_evaluation.json"
    if out.exists(): raise FileExistsError("Expert comparison exists; no silent overwrite")
    if ref["status"]!="EXPERT_LABELS_FROZEN": raise ValueError("Actual frozen expert review required")
    cfg=read_json(ROOT/"configs/jcp_v4_campaign.json")
    index=read_json(ROOT/cfg["output"]/"dataset_index.json")
    index_rows={r["dataset_id"]:r for r in index["records"]}
    dev=read_json(args.run/"evaluation/report.json")
    if dev["status"]!="PASS": raise ValueError("Incomplete fixed-output evaluation")
    checkpoint_hash=sha256(args.run/"checkpoint.pt")
    if dev["threshold_freeze"]["checkpoint_sha256"]!=checkpoint_hash: raise ValueError("Checkpoint changed")
    total=defaultdict(lambda:np.zeros(3,np.int64));rows=[]
    for item in ref["rows"]:
        if item["split"]=="train": continue
        record=index_rows[item["dataset_id"]]
        if record["split"]!=item["split"]: raise ValueError("Split changed")
        field=Path(item["source_frame"])
        if sha256(field)!=item["source_sha256"]: raise ValueError("Expert source fields changed")
        with np.load(field,allow_pickle=False) as z:
            domain=z["observation_mask"].astype(bool)&~z["geometry"].astype(bool)
            spacing=(float(np.mean(np.diff(z["y"]))),float(np.mean(np.diff(z["x"]))))
        labels={}
        for layer,entry in item["masks"].items():
            path=Path(entry["path"])
            if sha256(path)!=entry["sha256"]: raise ValueError("Expert mask changed after freeze")
            with Image.open(path) as image: labels[layer]=np.asarray(image.convert("L"))>127
        for branch in ("ml_only","physics_only","hybrid"):
            path=args.run/"evaluation"/branch/(Path(record["cache_file"]).stem+".npz")
            with np.load(path,allow_pickle=False) as z:
                predicted={h:z[h].astype(bool) for h in ("shock","vortex_core")}
            for head in ("shock","vortex_core"):
                decision_head="vortex" if head=="vortex_core" else head
                if item["decisions"][decision_head]=="not_applicable": continue
                valid=domain&~labels[decision_head+"_ignore"]
                c=counts(predicted[head],labels[head],valid)
                key=(item["split"],item["family"],branch,head);total[key]+=c
                extra={}
                if head=="vortex_core":
                    extra["connected_component_matches_fp_fn"]=list(object_match_counts(predicted[head]&valid,labels[head]&valid,minimum_pixels=9,minimum_iou=.1))
                else:
                    edge=lambda a:a&~ndimage.binary_erosion(a)
                    extra["envelope_boundary_distance"]=symmetric_mask_distance(edge(predicted[head])&valid,edge(labels[head])&valid,spacing=spacing)
                rows.append({"dataset_id":item["dataset_id"],"branch":branch,"head":head,"tp_fp_fn":c.tolist(),"prediction_sha256":sha256(path),**extra})
    summary=[]
    for key,c in total.items():
        tp,fp,fn=map(int,c);denom=2*tp+fp+fn
        summary.append(dict(zip(("split","family","branch","head"),key),tp=tp,fp=fp,fn=fn,dice=2*tp/denom if denom else None,precision=tp/(tp+fp) if tp+fp else None,recall=tp/(tp+fn) if tp+fn else None))
    write_json(out,{"status":"PASS","reference_sha256":sha256(args.reference),"checkpoint_sha256":checkpoint_hash,"thresholds":dev["threshold_freeze"],"summary":summary,"rows":rows,"train_excluded":True,"untouched_final_test":False,"claim":"Fixed-output agreement with human development reference; not untouched-case generalization or CFD validation. Component IDs are algorithmic, not human trajectory IDs."})


if __name__=="__main__":main()
