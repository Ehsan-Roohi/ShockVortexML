"""Create fixed-scale, full-field review sheets for prospective ML-only runs."""
from pathlib import Path
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def schlieren(rho, x, y, fluid):
    gy, gx = np.gradient(rho, y, x)
    value = np.hypot(gx, gy)
    scale = np.percentile(value[fluid], 99.5)
    return np.clip(value / max(scale, 1e-12), 0, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    summary = []
    colors = {"shock": "#e67e22", "vortex_core": "#9b2fae", "wake_shear": "#159f82"}
    for case_dir in sorted(p for p in args.inputs.iterdir() if p.is_dir()):
        frames = sorted(case_dir.glob("*.npz"), key=lambda p: float(p.stem[1:]))
        fig, axes = plt.subplots(len(frames), 4, figsize=(18, 11), constrained_layout=True)
        for row, source in enumerate(frames):
            pred_path = args.predictions / case_dir.name / source.name
            with np.load(source, allow_pickle=False) as z:
                rho, x, y = z["rho"], z["x"], z["y"]
                geom = z["geometry"].astype(bool)
                obs = z["observation_mask"].astype(bool)
            with np.load(pred_path, allow_pickle=False) as p:
                probs = {name: p["p_" + name] for name in colors}
                masks = {name: p[name].astype(bool) for name in colors}
            fluid = obs & ~geom
            image = schlieren(rho, x, y, fluid)
            extent = (x[0], x[-1], y[0], y[-1])
            axes[row, 0].imshow(image, origin="lower", extent=extent, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
            axes[row, 0].contour(x, y, geom, levels=[.5], colors="black", linewidths=.8)
            axes[row, 0].set_title(f"t={source.stem[1:]} | density schlieren")
            for col, name in enumerate(colors, start=1):
                axes[row, col].imshow(probs[name], origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1, aspect="equal")
                axes[row, col].contour(x, y, geom, levels=[.5], colors="white", linewidths=.7)
                if masks[name].any():
                    axes[row, col].contour(x, y, masks[name], levels=[.5], colors=colors[name], linewidths=1.1)
                axes[row, col].set_title(f"P({name}) | fixed mask")
                summary.append({"case": case_dir.name, "time": float(source.stem[1:]), "class": name,
                                "positive_pixels": int(masks[name].sum()),
                                "maximum_probability": float(np.max(probs[name][fluid]))})
            for ax in axes[row]:
                ax.set_xlim(-2, 5); ax.set_ylim(-2.5, 2.5); ax.set_xlabel("x/L"); ax.set_ylabel("y/L")
        fig.suptitle(f"Untouched prospective transfer: {case_dir.name} | ML-only, frozen thresholds", fontsize=16)
        fig.savefig(args.output / f"{case_dir.name}_review.png", dpi=220)
        plt.close(fig)
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
