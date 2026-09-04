"""Stage approved code, first-model artifacts and QC-passed sample fields."""
from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from jcp2026.common import ROOT,read_json,write_json,sha256
from jcp2026.diagnostics import gradient_diagnostics
from ml.geometry_robust_dataset import _reference
from build_jcp_release import local_closure

DEST=ROOT/"output/public_release/ShockVortexML"


def main():
    if not (DEST/".git").is_dir():raise ValueError("Expected separate, already cloned release repository")
    selected=local_closure((ROOT/"src/jcp2026").glob("*.py"))
    selected.update((ROOT/"tests").glob("test_jcp*.py"))
    selected.update((ROOT/"docs").glob("JCP_*.md"))
    selected.update((ROOT/"configs").glob("jcp*.json"))
    selected.update((ROOT/"src/annotation_ui").glob("*"))
    selected.update(ROOT/p for p in ("requirements-jcp-v4.txt","tools/jcp_freeze_expert_review.py","tools/evaluate_jcp_expert_reference.py","tools/summarize_jcp_campaign.py","tools/build_jcp_release.py","tools/build_jcp_review_context.py","tools/prepare_jcp_public_assets.py"))
    for source in sorted(selected):
        if not source.is_file():continue
        target=DEST/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists() and sha256(target)!=sha256(source):raise ValueError("Refusing to overwrite a different staged file: "+str(target))
        if not target.exists():shutil.copy2(source,target)
    cfg=read_json(ROOT/"configs/jcp_v4_campaign.json")
    index=read_json(ROOT/cfg["output"]/"dataset_index.json")
    rows=[]
    for case,name in [("re1e4_f180","airfoil_m3_a40_re10000_t6"),("baseline_f90_cfl0p20","cylinder_euler_m2p7_t8")]:
        record=max((r for r in index["records"] if r["case"]==case and r["split"]=="train"),key=lambda r:r["time"])
        source=ROOT/record["cache_file"]
        if sha256(source)!=record["cache_sha256"]:raise ValueError("Input cache failed integrity check")
        with np.load(source,allow_pickle=False) as z:
            arrays={k:z[k] for k in ("rho","pressure","u","v","x","y","geometry","observation_mask")}
        shape=arrays["rho"].shape;domain=arrays["observation_mask"].astype(bool)&~arrays["geometry"].astype(bool)
        if shape!=(len(arrays["y"]),len(arrays["x"])) or not domain.any():raise ValueError("Invalid grid/domain")
        if np.any(arrays["geometry"].astype(bool)&arrays["observation_mask"].astype(bool)):raise ValueError("Overlapping solid/fluid masks")
        stats={}
        for key in ("rho","pressure","u","v"):
            field=arrays[key]
            if field.shape!=shape or not np.isfinite(field).all():raise ValueError("Nonfinite or inconsistent sample array")
            stats[key]={"observed_min":float(field[domain].min()),"observed_max":float(field[domain].max())}
        if np.any(arrays["rho"][domain]<=0) or np.any(arrays["pressure"][domain]<=0):raise ValueError("Nonpositive observed fluid")
        d=gradient_diagnostics(arrays["u"],arrays["v"],arrays["x"],arrays["y"])
        residual=float(np.max(np.abs(d["q_legacy"]-d["q_deviatoric"]+d["divergence"]**2/4))/max(1.,float(np.max(np.abs(d["q_legacy"])))))
        if residual>1e-10:raise ValueError("Differential identity failed")
        target=DEST/"sample_data"/(name+".npz");target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():raise FileExistsError("Sample data already staged")
        np.savez_compressed(target,**arrays)
        with np.load(target,allow_pickle=False) as z:
            if set(z.files)!=set(arrays):raise ValueError("Unexpected publication payload")
            for key in arrays:np.testing.assert_array_equal(arrays[key],z[key])
        ref=DEST/"sample_data"/(name+"_reference.json");write_json(ref,asdict(_reference(record)))
        rows.append({"name":name,"dataset_id":record["dataset_id"],"case":case,"time":record["time"],"development_split":"train; previously inspected, not a held-out benchmark","path":target.relative_to(DEST).as_posix(),"sha256":sha256(target),"size_bytes":target.stat().st_size,"reference_path":ref.relative_to(DEST).as_posix(),"reference_sha256":sha256(ref),"upstream_harmonized_cache_sha256":record["cache_sha256"],"primitive_source_sha256":record["primitive_source_sha256"],"shape":list(shape),"geometry_pixels":int(arrays["geometry"].sum()),"observed_fluid_pixels":int(domain.sum()),"primitive_stats":stats,"q_identity_relative_residual":residual,"qc":"PASS finite/positive-fluid/grid/integrity/round-trip checks","cfd_convergence_certified":False,"shock_vortex_truth_labels_included":False,"failed_viscous_cylinder":False})
    write_json(DEST/"sample_data/DATA_QC.json",{"status":"NUMERICAL_DATA_QC_PASS_NOT_CFD_VALIDATION","samples":rows,"scope":"Author-approved illustrative data for ML pipeline reproduction. No human labels or physical shedding claims. All source CFD preserved.","exclusions":["failed viscous-cylinder states","private papers and email","unreviewed masks presented as truth","bulk raw archives"]})
    run=ROOT/cfg["results"]/"joint_qdev/20260904"
    report=read_json(run/"training_report.json")
    if report["status"]!="PASS" or report["checkpoint_sha256"]!=sha256(run/"checkpoint.pt"):raise ValueError("Invalid model")
    modeldir=DEST/"models/jcp_v4_joint_qdev_seed20260904";modeldir.mkdir(parents=True,exist_ok=True)
    for source,name in [(run/"checkpoint.pt","checkpoint.pt"),(run/"training_report.json","training_report.json"),(run/"evaluation/thresholds.json","thresholds.json")]:
        target=modeldir/name
        if target.exists():raise FileExistsError(target)
        shutil.copy2(source,target)
    if (run/"evaluation/report.json").exists():
        evaluated=read_json(run/"evaluation/report.json")
        write_json(modeldir/"development_metrics.json",{k:evaluated[k] for k in ("status","claim_boundary","independent_accuracy","physics_self_score_omitted","hybrid_type","threshold_freeze","summary")})
    manifest=[]
    for path in sorted(DEST.rglob("*")):
        if not path.is_file() or ".git" in path.relative_to(DEST).parts:continue
        if path.name=="PUBLIC_ASSET_MANIFEST.json":continue
        manifest.append({"path":path.relative_to(DEST).as_posix(),"size_bytes":path.stat().st_size,"sha256":sha256(path)})
    write_json(DEST/"PUBLIC_ASSET_MANIFEST.json",{"status":"STAGED_NOT_YET_PUSHED","repository":"https://github.com/Ehsan-Roohi/ShockVortexML","development_model_count":1,"planned_campaign_models":15,"sample_fields":2,"human_ground_truth":False,"cfd_validation":False,"files":manifest})
    print(f"Staged {len(manifest)} files; two numerical-QC-passed sample fields; one development checkpoint",flush=True)


if __name__=="__main__":main()
