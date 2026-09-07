"""Read all registered Silo states and export qualified, derived NPZ fields."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import h5py
import numpy as np

CASES = ("ellipse_m2p7", "circle_m2p2", "ellipse_m2p2")

def decode(v):
    if isinstance(v, np.ndarray) and v.shape == (): v = v.item()
    if isinstance(v, (bytes, np.bytes_)): return bytes(v).decode().rstrip("\0")
    return str(v)

def payload(h, name, shape):
    ref = decode(h[name].attrs["silo"]["value0"]).lstrip("/")
    return np.asarray(h[ref][...], dtype=np.float64).reshape(shape, order="C")

def read_silo(path):
    with h5py.File(path, "r") as h:
        mesh = h["rectilinear_grid"].attrs["silo"]
        xn = np.asarray(h[decode(mesh["coord0"]).lstrip("/")][...])
        yn = np.asarray(h[decode(mesh["coord1"]).lstrip("/")][...])
        x, y = .5*(xn[:-1]+xn[1:]), .5*(yn[:-1]+yn[1:])
        shape = (len(y), len(x))
        return x, y, {n: payload(h, n, shape) for n in ("rho","pres","vel1","vel2","ib_markers")}

def analytic_body(x, y, lx, ly):
    yy, xx = np.meshgrid(y, x, indexing="ij")
    return (xx/(.5*lx))**2 + (yy/(.5*ly))**2 <= 1

def iou(a, b):
    return np.count_nonzero(a & b) / max(1, np.count_nonzero(a | b))

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run",type=Path,required=True); p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); a.output.mkdir(parents=True,exist_ok=False); report=[]
    for case in CASES:
        src=a.run/"production"/case; meta=json.loads((src/"METADATA.json").read_text()); states=json.loads((src/"FILES_COMPLETE.json").read_text())["states"]
        if len(states)!=81: raise RuntimeError(f"{case}: expected 81 states")
        out=a.output/case; out.mkdir()
        for state in states:
            t=float(state["time"]); step=int(state["step"]); x,y,f=read_silo(src/"silo_hdf5"/"p0"/f"{step}.silo")
            expected=analytic_body(x,y,*meta["full_axis_lengths"])
            choices=(f["ib_markers"]>.5,f["ib_markers"]<.5); geom=max(choices,key=lambda q:iou(q,expected)); giou=float(iou(geom,expected)); fluid=~geom
            good=all(np.isfinite(f[k]).all() for k in ("rho","pres","vel1","vel2")) and np.all(f["rho"][fluid]>0) and np.all(f["pres"][fluid]>0)
            if giou<.97 or not good: raise RuntimeError(f"qualification failed {case} t={t}")
            np.savez_compressed(out/f"t{t:04.1f}.npz",rho=f["rho"].astype("f4"),pressure=f["pres"].astype("f4"),u=f["vel1"].astype("f4"),v=f["vel2"].astype("f4"),x=x,y=y,geometry=geom,observation_mask=np.ones_like(geom,bool))
            report.append(dict(case=case,time=t,step=step,geometry_iou=giou))
    (a.output/"QUALIFICATION.json").write_text(json.dumps(dict(status="PASS",frames=len(report),records=report),indent=2)+"\n")
    print(f"status=PASS frames={len(report)}")
if __name__=="__main__": main()
