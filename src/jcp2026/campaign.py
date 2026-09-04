"""Sequential, resumable campaign; no silent failed-model exclusion."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
from jcp2026.common import ROOT, read_json, write_json, sha256


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",type=Path,default=ROOT/"configs/jcp_v4_campaign.json")
    p.add_argument("--train-only",action="store_true")
    args=p.parse_args();cfg=read_json(args.config);out=ROOT/cfg["results"];out.mkdir(parents=True,exist_ok=True)
    steps=[]
    lock=out/"campaign_source_freeze.json"
    sources=list((ROOT/"src/jcp2026").glob("*.py"))+[ROOT/p for p in ["src/ml/stage5_model.py","src/ml/geometry_robust_dataset.py","src/ml/flow_aligned.py","src/train_stage5_joint.py","src/evaluate_stage5_joint.py","src/evaluate_geometry_robust_v2.py","src/analyze_mfc_euler_cylinder.py"]]
    fingerprints={str(p.relative_to(ROOT)):sha256(p) for p in sorted(sources)}
    if lock.exists():
        old=read_json(lock)
        if old["config_sha256"]!=sha256(args.config) or old["sources"]!=fingerprints:
            raise ValueError("Campaign source/config changed after freeze; use a new version")
    else:
        write_json(lock,{"utc":datetime.now(timezone.utc).isoformat(),"config_sha256":sha256(args.config),"sources":fingerprints})
    for seed in cfg["seeds"]:
        for variant in cfg["variants"]:
            run=out/variant/str(seed)
            for action,filename in [("train","training_report.json"),("evaluate","evaluation/report.json"),("regression","regression/report.json")]:
                if args.train_only and action!="train": continue
                report=run/filename
                if report.exists():
                    if read_json(report)["status"]!="PASS": raise ValueError("Non-passing existing report")
                    if action=="train" and read_json(report)["checkpoint_sha256"]!=sha256(run/"checkpoint.pt"):
                        raise ValueError("Changed checkpoint")
                    steps.append({"variant":variant,"seed":seed,"action":action,"status":"VERIFIED_EXISTING"});continue
                run.mkdir(parents=True,exist_ok=True)
                status={"status":"RUNNING","current":{"variant":variant,"seed":seed,"action":action},"completed":steps,"independent_accuracy":False}
                write_json(out/"campaign_progress.json",status)
                print(f"START {action} {variant} seed={seed}",flush=True)
                logfile=run/(action+".log")
                with logfile.open("a",encoding="utf-8") as log:
                    process=subprocess.run([sys.executable,"-m","jcp2026."+action,"--config",str(args.config),"--variant",variant,"--seed",str(seed)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                if process.returncode:
                    status.update(status="FAILED",returncode=process.returncode,log=str(logfile.relative_to(ROOT)))
                    write_json(out/"campaign_progress.json",status)
                    raise SystemExit(process.returncode)
                if not report.exists(): raise RuntimeError("Process finished without completion report")
                steps.append({"variant":variant,"seed":seed,"action":action,"status":"PASS"})
                print(f"PASS {action} {variant} seed={seed}",flush=True)
    write_json(out/"campaign_progress.json",{"status":"TRAINING_COMPLETE_EVALUATION_PENDING" if args.train_only else "DEVELOPMENT_CAMPAIGN_COMPLETE","completed":steps,"independent_accuracy":False,"final_test":"pending untouched case and expert labels"})


if __name__=="__main__":main()
