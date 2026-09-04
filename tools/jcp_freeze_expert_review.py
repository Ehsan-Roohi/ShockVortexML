"""Freeze actual completed expert labels, never blank seeds or AI substitutes."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jcp2026.common import ROOT, read_json, write_json, sha256
from jcp2026.review import REVIEW_ROOT, check, LAYERS
from annotation_server import AnnotationProject


def main():
    p=argparse.ArgumentParser();p.add_argument("--attestation",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists(): raise FileExistsError("Frozen reference exists")
    att=read_json(args.attestation)
    for key in ("actual_human_review_completed","predictions_hidden_during_labeling","ambiguity_recorded","prior_exposure_disclosed"):
        if att.get(key) is not True: raise ValueError("Expert attestation incomplete: "+key)
    if not att.get("reviewer","").strip() or not att.get("signed_date","").strip(): raise ValueError("Reviewer/date missing")
    packs=read_json(REVIEW_ROOT/"packs.json")["packs"]
    rows=[]
    for pack in packs:
        if check(pack["name"]): raise ValueError("Unfinished or invalid expert review")
        project=AnnotationProject(ROOT/pack["manifest"],ROOT/pack["working_output"])
        for step in project.steps:
            frame=project.frames[step];row=project.rows[step]
            if row["reviewer"].strip()!=att["reviewer"].strip(): raise ValueError("Reviewer differs from signed attestation")
            masks={}
            for layer in LAYERS:
                path=project.output_mask_path(step,layer)
                if not path.exists(): raise ValueError("Reviewed mask was not explicitly saved")
                masks[layer]={"path":str(path),"sha256":sha256(path)}
            rows.append({"dataset_id":frame["dataset_id"],"case":frame["case"],"family":frame["family"],"split":frame["original_split"],"leakage_group_id":frame["leakage_group_id"],"source_frame":frame["source_frame"],"source_sha256":frame["source_sha256"],"reviewer":row["reviewer"],"review_notes":row["review_notes"],"decisions":{h:row[h+"_decision"] for h in project.review_heads},"masks":masks})
    write_json(args.output,{"status":"EXPERT_LABELS_FROZEN","attestation_sha256":sha256(args.attestation),"attestation":att,"rows":rows,"untouched_case_test":False,"scope":"Human-reviewed development reference. Train rows excluded from held-case metrics. Not proof of an unbiased final benchmark."})
    print(f"Frozen {len(rows)} actual expert reviews")


if __name__=="__main__":main()
