"""Evidence-only campaign status and tables; missing seeds remain explicit."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from jcp2026.common import ROOT,read_json,write_json,sha256


def mean_std(values):
    values=[v for v in values if v is not None]
    return {"n":len(values),"mean":float(np.mean(values)) if values else None,"sample_std":float(np.std(values,ddof=1)) if len(values)>1 else None}


def main():
    cfg=read_json(ROOT/"configs/jcp_v4_campaign.json");out=ROOT/cfg["results"]
    rows=[];grouped=defaultdict(list)
    for variant in cfg["variants"]:
        for seed in cfg["seeds"]:
            run=out/variant/str(seed);record={"variant":variant,"seed":seed,"training":"pending","evaluation":"pending","regression":"pending"}
            if (run/"training_report.json").exists():
                trained=read_json(run/"training_report.json")
                if trained["checkpoint_sha256"]!=sha256(run/"checkpoint.pt"):raise ValueError("Checkpoint changed")
                record.update(training=trained["status"],parameters=trained["parameters"],training_seconds=trained["seconds"],peak_resident_bytes=trained.get("process_memory",{}).get("peak_resident_bytes"))
            if (run/"evaluation/report.json").exists():
                evaluated=read_json(run/"evaluation/report.json");record["evaluation"]=evaluated["status"]
                for item in evaluated["summary"]:
                    grouped[(variant,item["split"],item["family"],item["branch"])].append(dict(seed=seed,**item))
            if (run/"regression/report.json").exists():record["regression"]=read_json(run/"regression/report.json")["status"]
            rows.append(record)
    aggregate=[]
    for key,values in sorted(grouped.items()):
        record=dict(zip(("variant","split","family","branch"),key))
        record["seeds"]=[v["seed"] for v in values]
        for head in ("shock","vortex_core"):
            record[head]={name:mean_std([v["heads"][head][name] for v in values]) for name in ("weak_dice","weak_precision","weak_recall")}
        record["vortex_object_f1"]=mean_std([v["vortex_object_f1"] for v in values]);aggregate.append(record)
    doc={"status":"DEVELOPMENT_ONLY","planned_runs":15,"training_completed":sum(r["training"]=="PASS" for r in rows),"evaluation_completed":sum(r["evaluation"]=="PASS" for r in rows),"regression_completed":sum(r["regression"]=="PASS" for r in rows),"runs":rows,"aggregate":aggregate,"uncertainty_scope":"Sample standard deviation over initialization seeds only, not a confidence interval for case generalization.","physics_self_scores":"intentionally omitted pending expert reference","independent_accuracy":False,"untouched_case":"pending"}
    write_json(out/"campaign_summary.json",doc)
    lines=["# JCP v4 campaign status", "", f"Completed training: {doc['training_completed']}/15; evaluation: {doc['evaluation_completed']}/15; regression: {doc['regression_completed']}/15.","", "All current metrics are agreement with weak development proposals, not independently validated accuracy.","", "| Variant | Seed | Training | Evaluation | Regression |","|---|---:|---|---|---|"]
    lines.extend(f"| {r['variant']} | {r['seed']} | {r['training']} | {r['evaluation']} | {r['regression']} |" for r in rows)
    lines.extend(["","Physics-only products are saved separately, but comparing them with themselves would not establish accuracy. Expert review and an untouched case remain outstanding."])
    (out/"CAMPAIGN_STATUS.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(lines[2],flush=True)


if __name__=="__main__":main()
