"""Topology-limited hysteresis for short gaps in the learned shock front.

The high-confidence neural mask remains the seed.  A pixel can be added only
when both the learned shock probability and an entropy-free density-gradient
support test pass.  Vortex and wake/shear arrays are copied bit-for-bit.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi


def robust01(a: np.ndarray, domain: np.ndarray) -> np.ndarray:
    q0, q1 = np.quantile(a[domain], (0.70, 0.995))
    return np.clip((a-q0)/max(q1-q0, 1e-12), 0.0, 1.0)


def main() -> None:
    p=argparse.ArgumentParser()
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--high',type=float,default=.97)
    p.add_argument('--low',type=float,default=.32)
    p.add_argument('--gradient',type=float,default=.18)
    p.add_argument('--iterations',type=int,default=10)
    a=p.parse_args()
    if a.output.exists(): raise FileExistsError('Use a new output directory')
    a.output.mkdir(parents=True); rows=[]
    structure=np.ones((3,3),bool)
    for case_dir in sorted(q for q in a.predictions.iterdir() if q.is_dir()):
        out=a.output/case_dir.name; out.mkdir()
        for pred_path in sorted(case_dir.glob('*.npz')):
            with np.load(pred_path) as z: arrays={k:z[k] for k in z.files}
            with np.load(a.inputs/case_dir.name/pred_path.name) as z:
                rho=z['rho'].astype(float); geom=z['geometry'].astype(bool); obs=z['observation_mask'].astype(bool)
            domain=obs & ~geom
            gy,gx=np.gradient(np.log(np.maximum(rho,1e-12)))
            grad=robust01(np.hypot(gx,gy),domain)
            prob=arrays['p_shock'].astype(float)
            seed=(prob>=a.high)&domain
            support=(prob>=a.low)&(grad>=a.gradient)&domain
            grown=ndi.binary_propagation(seed,structure=structure,mask=support,iterations=a.iterations)
            # Close only sub-grid interruptions already enclosed by supported front pixels.
            closed=ndi.binary_closing(grown,structure=np.ones((3,3),bool),iterations=1)&support
            shock=seed|grown|closed
            old=arrays['shock'].astype(bool)
            vortex=arrays['vortex_core'].copy(); wake=arrays['wake_shear'].copy()
            arrays['shock']=shock
            arrays['background_other']=domain&~(shock|vortex.astype(bool)|wake.astype(bool))
            np.savez_compressed(out/pred_path.name,**arrays)
            rows.append(dict(case=case_dir.name,frame=pred_path.name,seed_pixels=int(seed.sum()),added_pixels=int((shock&~seed).sum()),vortex_preserved=bool(np.array_equal(vortex,arrays['vortex_core'])),wake_preserved=bool(np.array_equal(wake,arrays['wake_shear']))))
    (a.output/'SUMMARY.json').write_text(json.dumps(rows,indent=2)+'\n')
    print(f"status=PASS frames={len(rows)} added={sum(r['added_pixels'] for r in rows)}")


if __name__=='__main__': main()
