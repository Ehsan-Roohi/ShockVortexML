from pathlib import Path
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(os.environ.get("ALPHA20_DATA_DIR", Path.cwd())).resolve()
FILES = sorted(ROOT.glob("restart_circ_*.csv"))
FILES = [p for p in FILES if 2986 <= int(re.search(r"(\d{5})", p.name).group(1)) <= 3185]
FILES = [p for p in FILES if int(re.search(r"(\d{5})", p.name).group(1)) % 2 == 0]
OUT = ROOT / "frames_4k"
OUT.mkdir(exist_ok=True)
GAMMA = 1.4
DT = 4.79998e-6

def load(path):
    a = np.loadtxt(path, delimiter=",", skiprows=1, usecols=(1,2,3,4,5,6,7))
    a = a.reshape(611, 1060, 7)
    x,y,rho,mx,my,E,rhok = [a[...,i] for i in range(7)]
    u, v = mx/rho, my/rho
    p = (GAMMA-1.0)*(E - 0.5*rho*(u*u+v*v) - rhok)
    mach = np.sqrt(u*u+v*v)/np.sqrt(GAMMA*p/rho)
    re_, rc = np.gradient(rho)
    xe, xc = np.gradient(x); ye, yc = np.gradient(y)
    det = xc*ye-xe*yc
    det[np.abs(det) < 1e-20] = np.nan
    rx = (rc*ye-re_*yc)/det
    ry = (re_*xc-rc*xe)/det
    grad = np.sqrt(rx*rx+ry*ry)
    return x,y,mach,grad

# Fixed movie-wide normalization from representative frames.
samples=[]
for p in (FILES[0], FILES[len(FILES)//2], FILES[-1]):
    *_,g=load(p); samples.append(g[np.isfinite(g)])
gref=np.concatenate([s[::50] for s in samples])
glo,ghi=np.percentile(np.log1p(gref),[2,99.75])

for n,path in enumerate(FILES):
    it=int(re.search(r"(\d{5})",path.name).group(1))
    x,y,mach,grad=load(path)
    schl=np.clip((np.log1p(grad)-glo)/(ghi-glo),0,1)
    fig,axs=plt.subplots(2,1,figsize=(16,9),dpi=240,gridspec_kw={"hspace":0.012})
    fig.patch.set_facecolor("black")
    for ax in axs:
        ax.set_facecolor("black"); ax.set_xlim(-0.45,2.75); ax.set_ylim(-0.50,0.50)
        ax.set_aspect("auto"); ax.axis("off")
    axs[0].pcolormesh(x,y,schl,shading="gouraud",cmap="gray_r",vmin=0,vmax=1,rasterized=True)
    pm=axs[1].pcolormesh(x,y,mach,shading="gouraud",cmap="turbo",vmin=0,vmax=3.3,rasterized=True)
    axs[1].contour(x,y,mach,levels=[1.0],colors="white",linewidths=0.55,alpha=0.92)
    box=dict(facecolor="black",alpha=.62,edgecolor="none",pad=3)
    axs[0].text(.012,.955,"Density-gradient schlieren",transform=axs[0].transAxes,color="white",fontsize=14,va="top",bbox=box)
    axs[1].text(.012,.955,"Mach number  |  white: M = 1",transform=axs[1].transAxes,color="white",fontsize=14,va="top",bbox=box)
    axs[0].text(.988,.955,f"M=3  |  alpha=20 deg  |  SST-URANS  |  t={(it-2986)*DT:.6f}",transform=axs[0].transAxes,color="white",fontsize=12,ha="right",va="top",bbox=box)
    cax=fig.add_axes([.944,.055,.012,.405]); cb=fig.colorbar(pm,cax=cax); cb.ax.tick_params(colors="white",labelsize=8,length=2); cb.outline.set_edgecolor("white")
    fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    fig.savefig(OUT/f"frame_{n:04d}.png",facecolor=fig.get_facecolor(),pad_inches=0)
    plt.close(fig)
print(f"RENDERED={len(FILES)}")
