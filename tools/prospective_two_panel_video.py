"""Render fixed-scale physics-left / learned-structures-right figures and MP4 frames."""
from __future__ import annotations
import argparse, json, subprocess
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS={"shock":"#e67e22","vortex_core":"#9b2fae","wake_shear":"#159f82"}
def schlieren(rho,x,y,fluid,scale):
    gy,gx=np.gradient(rho,y,x); q=np.hypot(gx,gy); return np.clip(q/max(scale,1e-12),0,1)
def main():
    p=argparse.ArgumentParser(); p.add_argument("--inputs",type=Path,required=True); p.add_argument("--predictions",type=Path,required=True); p.add_argument("--output",type=Path,required=True); p.add_argument("--resume",action="store_true"); a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=a.resume); summary=[]
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        ffmpeg = "ffmpeg"
    for case_dir in sorted(p for p in a.inputs.iterdir() if p.is_dir()):
        frames=sorted(case_dir.glob("*.npz")); movie=a.output/case_dir.name; movie.mkdir(exist_ok=a.resume); scales=[]
        for src in frames:
            with np.load(src) as z:
                fluid=~z["geometry"].astype(bool); gy,gx=np.gradient(z["rho"],z["y"],z["x"]); scales.append(np.percentile(np.hypot(gx,gy)[fluid],99.5))
        scale=float(np.median(scales))
        for i,src in enumerate(frames):
            with np.load(src) as z: rho,x,y,geom=z["rho"],z["x"],z["y"],z["geometry"].astype(bool)
            with np.load(a.predictions/case_dir.name/src.name) as z: masks={k:z[k].astype(bool) for k in COLORS}
            image=schlieren(rho,x,y,~geom,scale); extent=(x[0],x[-1],y[0],y[-1]); fig,axs=plt.subplots(1,2,figsize=(12.8,5.8),constrained_layout=True)
            for ax in axs:
                ax.imshow(image,origin="lower",extent=extent,cmap="gray_r",vmin=0,vmax=1,aspect="equal"); ax.contour(x,y,geom,levels=[.5],colors="black",linewidths=1); ax.set(xlim=(-2,5),ylim=(-2.5,2.5),xlabel="x/L",ylabel="y/L")
            axs[0].set_title("Physical field: density schlieren")
            axs[1].set_title("Frozen neural structure segmentation")
            for name,color in COLORS.items():
                if masks[name].any(): axs[1].contour(x,y,masks[name],levels=[.5],colors=color,linewidths=1.4)
            t=float(src.stem[1:]); fig.suptitle(f"{case_dir.name} | t={t:.1f} L/a_inf | orange: shock, magenta: vortex core, green: wake/shear")
            target=movie/f"frame_{i:04d}.png"
            if not (a.resume and target.exists()): fig.savefig(target,dpi=140)
            plt.close(fig)
            summary.append(dict(case=case_dir.name,time=t,frame=target.name,**{k:int(v.sum()) for k,v in masks.items()}))
        subprocess.run([ffmpeg,"-y","-framerate","10","-i",str(movie/"frame_%04d.png"),"-c:v","libx264","-pix_fmt","yuv420p",str(a.output/f"{case_dir.name}_physics_vs_ml.mp4")],check=True)
        for pick in (0,len(frames)//2,len(frames)-1):
            (a.output/f"{case_dir.name}_t{pick:02d}.png").write_bytes((movie/f"frame_{pick:04d}.png").read_bytes())
    (a.output/"SUMMARY.json").write_text(json.dumps(summary,indent=2)+"\n")
if __name__=="__main__": main()
