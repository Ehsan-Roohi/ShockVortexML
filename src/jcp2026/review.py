"""Prediction-blind expert review pack; never manufacture completed reviews."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from jcp2026.common import ROOT, read_json, write_json, sha256

LAYERS = ("shock", "shock_centerline", "vortex_core", "shock_ignore", "vortex_ignore")
REVIEW_ROOT = ROOT / "data/processed/jcp_v4/expert_review"


def prepare(config: Path):
    cfg = read_json(config)
    index_path = ROOT / cfg["output"] / "dataset_index.json"
    index = read_json(index_path)
    groups = defaultdict(list)
    for r in index["records"]:
        groups[r["case"]].append(r)
    selected = []
    for case, rows in sorted(groups.items()):
        rows.sort(key=lambda r: r["time"])
        # Selection uses time only, never labels, losses or network confidence.
        selected.extend(rows[i] for i in np.unique(np.linspace(0,len(rows)-1,6).round().astype(int)))
    by_shape = defaultdict(list)
    for step, record in enumerate(selected):
        with np.load(ROOT / record["cache_file"], allow_pickle=False) as z:
            by_shape[tuple(z["geometry"].shape)].append((step, record))
    packs = []
    for shape, pairs in sorted(by_shape.items()):
        name = "x".join(map(str, shape))
        folder = REVIEW_ROOT / name
        manifest_path = folder / "annotation_manifest.json"
        if manifest_path.exists():
            raise FileExistsError("Review seed pack already exists; refusing overwrite")
        frames = []
        for step, r in pairs:
            source = ROOT / r["cache_file"]
            if sha256(source) != r["cache_sha256"]:
                raise ValueError("Changed input cache")
            target = folder / "fields" / f"frame_{step:04d}.npz"
            target.parent.mkdir(parents=True, exist_ok=True)
            # No targets, valid-label masks, ML predictions or teacher proposals.
            with np.load(source, allow_pickle=False) as z:
                arrays = {k:z[k] for k in ("rho","pressure","u","v","x","y","geometry","observation_mask","diagnostics")}
                np.savez_compressed(target, **arrays)
            masks = {}
            for layer in LAYERS:
                mask_path = folder / "blank_masks" / f"frame_{step:04d}_{layer}.png"
                mask_path.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(np.zeros(shape,dtype=np.uint8)).save(mask_path)
                masks[layer] = str(mask_path)
            frames.append({"step":step,"source_step":r["step"],"time":r["time"],"dataset_id":r["dataset_id"],"case":r["case"],"family":r["family"],"original_split":r["split"],"leakage_group_id":r["leakage_group_id"],"source_frame":str(target),"source_sha256":sha256(target),"mask_files":masks,"review_status":"unreviewed","reviewer":"","review_notes":"","evaluation_role":"previously_inspected_development; train rows excluded from held-case scores"})
        manifest = {"schema_version":"jcp_v4_blind_review","case_group_id":"multiple_explicit_frame_groups","overlap_policy":"shock and vortex may overlap; review each independently","editable_layers":list(LAYERS),"review_heads":["shock","vortex"],"selected_steps":[f["step"] for f in frames],"frames":frames,"diagnostic_stats":{k:{"std":v} for k,v in zip(("grad_log_rho","grad_log_pressure","compression","vorticity","q_deviatoric","lambda_ci"),(3,3,5,10,25,5))},"display_scaling_only":True,"selection_rule":"six uniformly spaced record times per case; no output-informed selection","named_reviewer_designate":"Ehsan Roohi Golkhatmi (user volunteered; no reviews completed)","independent_accuracy_enabled":False,"untouched_case_test":False,"dataset_index_sha256":sha256(index_path)}
        write_json(manifest_path,manifest)
        packs.append({"name":name,"frames":len(frames),"manifest":str(manifest_path.relative_to(ROOT)),"working_output":str((folder/"reviewed_working").relative_to(ROOT)),"manifest_sha256":sha256(manifest_path)})
    write_json(REVIEW_ROOT/"packs.json",{"status":"AWAITING_EXPERT","reviewer_designate":"Ehsan Roohi Golkhatmi","number_of_frames":len(selected),"packs":packs,"no_completed_human_labels":True})
    print(f"Prepared {len(selected)} blank, prediction-blind frames in {len(packs)} raster-shape packs",flush=True)


def check(pack: str):
    from annotation_server import AnnotationProject
    folder = REVIEW_ROOT / pack
    project = AnnotationProject(folder/"annotation_manifest.json",folder/"reviewed_working")
    issues = []
    checked = []
    for step in project.steps:
        frame = project.frames[step]; row = project.rows[step]
        if sha256(Path(frame["source_frame"])) != frame["source_sha256"]:
            issues.append(f"{step}: changed source fields")
        for key in ("reviewer","review_notes"):
            if not row.get(key,"").strip(): issues.append(f"{step}: missing {key}")
        if row["review_status"] != "reviewed": issues.append(f"{step}: not reviewed")
        for head in project.review_heads:
            if row[f"{head}_decision"] not in {"accepted","corrected","rejected","not_applicable"}:
                issues.append(f"{step}: {head} decision missing")
        geometry = project.load_geometry(step)
        with np.load(frame["source_frame"],allow_pickle=False) as z:
            outside = ~z["observation_mask"].astype(bool) | geometry
        masks = {k:project.load_mask(step,k) for k in LAYERS}
        for k in ("shock","vortex_core","shock_centerline"):
            if np.any(masks[k] & outside): issues.append(f"{step}: {k} outside observed fluid")
        if np.any(masks["shock_centerline"] & ~masks["shock"]):
            issues.append(f"{step}: shock centerline lies outside annotated shock envelope")
        checked.append({"step":step,"dataset_id":frame["dataset_id"],"reviewer":row.get("reviewer"),"status":row["review_status"]})
    report = {"status":"NEEDS_REVIEW" if issues else "STRUCTURAL_QA_PASS","issues":issues,"frames":checked,"independent_accuracy_enabled":False,"note":"Structural QA does not certify reviewer independence, label correctness or unseen-case status. Expert attestation and model/data freeze are separate."}
    write_json(folder/"review_qa.json",report)
    print(f"{pack}: {report['status']}, {len(issues)} outstanding checks",flush=True)
    return bool(issues)


def main():
    p=argparse.ArgumentParser();p.add_argument("action",choices=["prepare","check"])
    p.add_argument("--config",type=Path,default=ROOT/"configs/jcp_v4_campaign.json");p.add_argument("--pack")
    args=p.parse_args()
    if args.action=="prepare": prepare(args.config)
    elif args.pack: raise SystemExit(int(check(args.pack)))
    else: p.error("--pack is required for check")


if __name__=="__main__": main()
