"""Build a non-public, allowlisted and verified release candidate ZIP."""
from __future__ import annotations
import ast
from datetime import datetime,timezone
from pathlib import Path
import sys
import zipfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from jcp2026.common import ROOT, read_json, write_json, sha256


def local_closure(initial):
    selected=set();pending=list(initial)
    while pending:
        path=pending.pop()
        if path in selected:continue
        selected.add(path)
        if path.suffix!=".py":continue
        tree=ast.parse(path.read_text(encoding="utf-8-sig"))
        for node in ast.walk(tree):
            names=[a.name for a in node.names] if isinstance(node,ast.Import) else ([node.module] if isinstance(node,ast.ImportFrom) and node.module and not node.level else [])
            for name in names:
                relative=Path(*name.split("."))
                for candidate in (ROOT/"src"/relative.with_suffix(".py"),ROOT/"src"/relative/"__init__.py"):
                    if candidate.is_file():pending.append(candidate)
                parts=relative.parts
                for n in range(1,len(parts)):
                    init=ROOT/"src"/Path(*parts[:n])/"__init__.py"
                    if init.is_file():pending.append(init)
    return selected


def main():
    cfg=read_json(ROOT/"configs/jcp_v4_campaign.json")
    selected=local_closure((ROOT/"src/jcp2026").glob("*.py"))
    selected.update((ROOT/"tests").glob("test_jcp*.py"))
    selected.update((ROOT/"docs").glob("JCP_*.md"))
    selected.update((ROOT/"configs").glob("jcp*.json"))
    selected.update(ROOT/p for p in ("requirements-jcp-v4.txt","tools/build_jcp_release.py","tools/jcp_freeze_expert_review.py","tools/evaluate_jcp_expert_reference.py"))
    selected.update((ROOT/"src/annotation_ui").glob("*"))
    selected={p for p in selected if p.is_file()}
    completed=[];missing=[]
    for variant in cfg["variants"]:
        for seed in cfg["seeds"]:
            run=ROOT/cfg["results"]/variant/str(seed)
            report=run/"training_report.json"
            if not report.exists(): missing.append({"variant":variant,"seed":seed,"training":"not complete"});continue
            trained=read_json(report)
            if trained["status"]!="PASS" or trained["checkpoint_sha256"]!=sha256(run/"checkpoint.pt"):
                raise ValueError("Invalid completed checkpoint")
            selected.update((report,run/"checkpoint.pt"))
            for folder in (run/"evaluation",run/"regression"):
                if (folder/"report.json").exists():
                    selected.update(folder.rglob("*.json"));selected.update(folder.rglob("*.png"));selected.update(folder.rglob("*.pdf"))
            completed.append({"variant":variant,"seed":seed,"evaluation_complete":(run/"evaluation/report.json").exists(),"regression_complete":(run/"regression/report.json").exists()})
    for name in ("campaign_source_freeze.json","campaign_progress.json"):
        path=ROOT/cfg["results"]/name
        if path.exists():selected.add(path)
    metadata={"status":"LOCAL_RELEASE_CANDIDATE_NOT_PUBLISHED","created_utc":datetime.now(timezone.utc).isoformat(),"target":"Journal of Computational Physics","config_sha256":sha256(ROOT/"configs/jcp_v4_campaign.json"),"completed_runs":completed,"missing_runs":missing,"independent_reference":"pending actual expert review","untouched_case":"not available","publication_gates":["confirm repository and source/model/data licensing","complete campaign and scientific review"],"data_policy":"No raw or harmonized CFD data are redistributed here. Full training requires the registered data assets; this is not a self-contained full-data replication package.","files":[]}
    for path in sorted(selected):
        metadata["files"].append({"path":path.relative_to(ROOT).as_posix(),"size_bytes":path.stat().st_size,"sha256":sha256(path)})
    destination=ROOT/"output/packages"/("JCP_v4_candidate_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+".zip")
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    import json
    with zipfile.ZipFile(destination,"x",compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for path in sorted(selected):archive.write(path,path.relative_to(ROOT).as_posix())
        archive.writestr("RELEASE_MANIFEST.json",json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    with zipfile.ZipFile(destination) as archive:
        bad=archive.testzip()
        if bad:raise ValueError("ZIP CRC failed: "+bad)
        import hashlib
        for item in metadata["files"]:
            if hashlib.sha256(archive.read(item["path"])).hexdigest()!=item["sha256"]:raise ValueError("ZIP hash mismatch")
    write_json(destination.with_suffix(".json"),{"archive":str(destination.relative_to(ROOT)),"sha256":sha256(destination),"files":len(selected),"completed_training_runs":len(completed),"missing_training_runs":len(missing),"publicly_published":False})
    print(str(destination),flush=True)


if __name__=="__main__":main()
