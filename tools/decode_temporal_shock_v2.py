"""Create development V2 predictions from frozen neural probability sequences."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from jcp2026.temporal_shock import decode_temporal_hysteresis


def main():
    p=argparse.ArgumentParser(); p.add_argument('--inputs',type=Path,required=True); p.add_argument('--predictions',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Use a new output directory')
    a.output.mkdir(parents=True); summary=[]
    for case_dir in sorted(p for p in a.inputs.iterdir() if p.is_dir()):
        sources=sorted(case_dir.glob('*.npz')); probabilities=[]; domains=[]; geometries=[]; old=[]
        for src in sources:
            with np.load(src) as f:
                geom=f['geometry'].astype(bool); domain=f['observation_mask'].astype(bool)&~geom
            with np.load(a.predictions/case_dir.name/src.name) as z:
                probabilities.append(z['p_shock']); old.append({k:z[k] for k in z.files})
            domains.append(domain); geometries.append(geom)
        decoded=decode_temporal_hysteresis(probabilities,domains,geometries)
        target=a.output/case_dir.name; target.mkdir()
        for src,arrays,new,domain in zip(sources,old,decoded,domains):
            previous=arrays['shock'].copy()
            arrays['shock']=new; arrays['background_other']=domain&~(new|arrays['vortex_core']|arrays['wake_shear'])
            np.savez_compressed(target/src.name,**arrays)
            summary.append(dict(case=case_dir.name,frame=src.name,old_shock_pixels=int(previous.sum()),v2_shock_pixels=int(new.sum())))
    (a.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__': main()
