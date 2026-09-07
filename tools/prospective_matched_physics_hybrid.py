"""Matched physics-only and frozen-rule hybrid audit for prospective fields.

This script does not tune thresholds and does not create independent truth.
"""
from pathlib import Path
import argparse, json
import numpy as np
from scipy import ndimage as ndi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from jcp2026.diagnostics import build_inputs
from ml.flow_aligned import FreestreamReference
from physics_proposals import derivatives, shock_proposal, vortex_proposal

POLICY = dict(shock_seed_floor=.75, vortex_seed_floor=.75,
              rotation_support_fraction=.5, maximum_wall_fraction=.25,
              shock_dilation_cells=6, minimum_component_pixels=9)


def clean(mask, minimum=9):
    labels, count = ndi.label(mask, np.ones((3, 3)))
    keep = np.bincount(labels.ravel(), minlength=count + 1) >= minimum
    keep[0] = False
    return keep[labels]


def reference(mach):
    return FreestreamReference(1., 1./1.4, mach, 0., 1.4, 1., "registered_case_metadata")


def dice(a, b):
    return float(2 * np.count_nonzero(a & b) / max(np.count_nonzero(a) + np.count_nonzero(b), 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, required=True)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    records = []
    palette = dict(shock="#e67e22", vortex="#9b2fae")
    for case_dir in sorted(p for p in args.inputs.iterdir() if p.is_dir()):
        frames = sorted(case_dir.glob("*.npz"), key=lambda p: float(p.stem[1:]))
        fig, axes = plt.subplots(len(frames), 4, figsize=(18, 11), constrained_layout=True)
        mach = 2.7 if case_dir.name.endswith("m2p7") else 2.2
        ref = reference(mach)
        for row, source in enumerate(frames):
            with np.load(source, allow_pickle=False) as z:
                f = {k: z[k] for k in ("rho", "pressure", "u", "v", "x", "y", "geometry", "observation_mask")}
            with np.load(args.predictions / case_dir.name / source.name, allow_pickle=False) as z:
                ml = np.stack([z["shock"].astype(bool), z["vortex_core"].astype(bool)])
                prob = np.stack([z["p_shock"], z["p_vortex_core"]])
            geom = f["geometry"].astype(bool); domain = f["observation_mask"].astype(bool) & ~geom
            dx, dy = float(np.median(np.diff(f["x"]))), float(np.median(np.diff(f["y"])))
            flow = derivatives(f["rho"], f["pressure"], f["u"], f["v"], dx=dx, dy=dy)
            guard = ndi.binary_dilation(geom, iterations=5)
            shock = shock_proposal(f["rho"], f["pressure"], flow, guard, dx=dx, dy=dy,
                                   jump_offset_pixels=2., leading_edge_anchor_distance=.5)
            vortex = vortex_proposal(flow, geom, shock, f["x"], f["y"], dx=dx, dy=dy)
            physics = np.stack([shock.band & domain, vortex.core & domain])
            inputs, _ = build_inputs(f, ref)
            support = ndi.binary_dilation(physics[0], iterations=POLICY["shock_dilation_cells"])
            hshock = ml[0] | ((prob[0] >= POLICY["shock_seed_floor"]) & support & domain)
            rotation = ((np.abs(inputs[4]) >= np.tanh(.5/8)) &
                        (inputs[5] >= np.tanh(.02/24)) & (inputs[6] >= np.tanh(.2/8)))
            wall = domain & (ndi.distance_transform_edt(~geom, sampling=(dy, dx)) <= .05)
            pool = (prob[1] >= POLICY["vortex_seed_floor"]) & domain
            labels, count = ndi.label(pool, np.ones((3, 3)))
            sizes = np.bincount(labels.ravel(), minlength=count + 1)
            rot = np.bincount(labels.ravel(), weights=rotation.ravel(), minlength=count + 1)
            walls = np.bincount(labels.ravel(), weights=wall.ravel(), minlength=count + 1)
            keep = ((sizes >= POLICY["minimum_component_pixels"]) &
                    (rot / np.maximum(sizes, 1) >= POLICY["rotation_support_fraction"]) &
                    (walls / np.maximum(sizes, 1) <= POLICY["maximum_wall_fraction"]))
            keep[0] = False
            hvortex = ml[1] | keep[labels]
            hybrid = np.stack([clean(hshock), clean(hvortex)])
            outdir = args.output / "arrays" / case_dir.name; outdir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(outdir / source.name, physics_shock=physics[0], physics_vortex=physics[1],
                                hybrid_shock=hybrid[0], hybrid_vortex=hybrid[1])
            records.append(dict(case=case_dir.name, time=float(source.stem[1:]),
                ml_shock_pixels=int(ml[0].sum()), physics_shock_pixels=int(physics[0].sum()), hybrid_shock_pixels=int(hybrid[0].sum()),
                ml_vortex_pixels=int(ml[1].sum()), physics_vortex_pixels=int(physics[1].sum()), hybrid_vortex_pixels=int(hybrid[1].sum()),
                ml_physics_shock_dice=dice(ml[0], physics[0]), ml_physics_vortex_dice=dice(ml[1], physics[1])))
            gy, gx = np.gradient(f["rho"], f["y"], f["x"]); schl = np.hypot(gx, gy)
            schl /= max(np.percentile(schl[domain], 99.5), 1e-12)
            for col, (masks, title) in enumerate(((None, "Density schlieren"), (ml, "ML-only"), (physics, "Physics-only"), (hybrid, "Frozen hybrid"))):
                ax=axes[row,col]; ax.imshow(np.clip(schl,0,1), origin="lower", extent=(f["x"][0],f["x"][-1],f["y"][0],f["y"][-1]), cmap="gray_r", vmin=0, vmax=1, aspect="equal")
                ax.contour(f["x"],f["y"],geom,levels=[.5],colors="black",linewidths=.8)
                if masks is not None:
                    for mask,color in zip(masks,(palette["shock"],palette["vortex"])):
                        if mask.any(): ax.contour(f["x"],f["y"],mask,levels=[.5],colors=color,linewidths=1.)
                ax.set(xlim=(-2,5),ylim=(-2.5,2.5),xlabel="x/L",ylabel="y/L",title=f"t={source.stem[1:]} | {title}")
        fig.suptitle(f"Matched untouched comparison: {case_dir.name} | fixed rules",fontsize=16)
        fig.savefig(args.output/f"{case_dir.name}_matched.png",dpi=220); plt.close(fig)
    (args.output/"SUMMARY.json").write_text(json.dumps({"policy":POLICY,"status":"descriptive_not_independent_accuracy","records":records},indent=2))


if __name__ == "__main__": main()
