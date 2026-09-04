"""Raw-field context for the human reviewer; no model or weak-label access."""
from __future__ import annotations
import html
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from jcp2026.common import ROOT,read_json,write_json,sha256


def main():
    root=ROOT/"data/processed/jcp_v4/expert_review"
    packs=read_json(root/"packs.json")["packs"]
    for pack in packs:
        manifest=read_json(ROOT/pack["manifest"]);folder=root/pack["name"]/"raw_context"
        folder.mkdir(parents=True,exist_ok=True);links=[]
        for record in manifest["frames"]:
            name=f"frame_{record['step']:04d}";path=folder/(name+".png")
            if path.exists():raise FileExistsError("Context already exists")
            source=Path(record["source_frame"])
            if sha256(source)!=record["source_sha256"]:raise ValueError("Changed reviewer field")
            with np.load(source,allow_pickle=False) as z:
                data={k:z[k] for k in z.files}
            x,y=data["x"],data["y"];extent=(x[0],x[-1],y[0],y[-1]);masked=~data["observation_mask"].astype(bool)|data["geometry"].astype(bool)
            fig,axes=plt.subplots(1,4,figsize=(18,5),layout="constrained")
            fields=[data["rho"],data["pressure"],np.hypot(data["u"],data["v"]),data["diagnostics"][3]]
            for ax,values,title,cmap in zip(axes,fields,("Density","Pressure","Speed and streamlines","Signed vorticity"),("Greys","Greys","magma","PuOr")):
                plotted=ax.imshow(np.ma.masked_where(masked,values),origin="lower",extent=extent,aspect="equal",cmap=cmap)
                fig.colorbar(plotted,ax=ax,shrink=.72)
                if data["geometry"].any():ax.contourf(x,y,data["geometry"].astype(float),levels=[.5,1.5],colors=["black"])
                ax.set_title(title);ax.set_aspect("equal",adjustable="box");ax.set_xlabel("x / L");ax.set_ylabel("y / L")
            axes[2].streamplot(np.linspace(float(x[0]),float(x[-1]),len(x),dtype=np.float64),np.linspace(float(y[0]),float(y[-1]),len(y),dtype=np.float64),np.ma.masked_where(masked,data["u"]),np.ma.masked_where(masked,data["v"]),density=1.4,color="white",linewidth=.35,arrowsize=.5)
            fig.suptitle(f"{record['dataset_id']} | t={record['time']:.5g} | raw fields; NO predictions",fontsize=10)
            fig.savefig(path,dpi=180);plt.close(fig)
            write_json(folder/(name+".json"),{"source_sha256":record["source_sha256"],"generator_sha256":sha256(Path(__file__)),"dataset_id":record["dataset_id"],"checkpoint":None,"model_or_weak_labels_used":False,"display":"Full stored domain; scalar ranges are display-only, not segmentation thresholds."})
            links.append(f'<li><a href="{name}.png">{html.escape(record["dataset_id"])}</a> — t={record["time"]:.5g}</li>')
            print(pack["name"]+" "+name,flush=True)
        (folder/"index.html").write_text('<!doctype html><meta charset="utf-8"><title>Prediction-blind CFD review</title><h1>Raw-field context — no network predictions</h1><p>Use with the blank annotation UI. Color ranges are display-only.</p><ul>'+"\n".join(links)+"</ul>",encoding="utf-8")


if __name__=="__main__":main()
