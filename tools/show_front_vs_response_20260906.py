"""ML response region vs physics-supported narrow front; separate hybrid display."""
from pathlib import Path
import numpy as np
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from jcp2026.common import ROOT,read_json,write_json,sha256
from build_paper_videos_20260905 import load_fields
from plot_jcp_schlieren_panels import schlieren
from audit_movie_waves_20260906 import sample

OUT=ROOT/'results/front_vs_response_display_20260906'
SRC=ROOT/'results/wake_v3_f270_two_panel_20260906'
POLICY=dict(smoothing_cells=1.,wall_clearance_cells=4,compression_L_U_min=.5,grad_logp_L_min=1.,jump_logp_min=.055,jump_logrho_min=.025,normal_velocity_drop_U_min=.01,nms_pitch=1.,jump_pitch=3.,minimum_component_pixels=5,neural_mask_unchanged=True,output='Separate hybrid diagnostic, not a new neural prediction or independent truth')

def narrow_front(f,ml,reference):
    domain=f['observation_mask']&~f['geometry'];safe=ndi.binary_erosion(domain,iterations=POLICY['wall_clearance_cells'])
    w=ndi.gaussian_filter(domain.astype(float),1.)
    def smooth(a):return ndi.gaussian_filter(np.where(domain,a,0.),1.)/np.maximum(w,1e-12)
    p,rho=[np.maximum(smooth(f[k]),1e-12) for k in ('pressure','rho')];u,v=[smooth(f[k]) for k in ('u','v')]
    x,y=f['x'],f['y'];X,Y=np.meshgrid(x,y);L=reference['reference_length'];U=reference['speed_inf']
    py,px=np.gradient(np.log(p),y,x);g=np.hypot(px,py);nx=px/np.maximum(g,1e-12);ny=py/np.maximum(g,1e-12)
    dx=np.gradient(x)[None,:];dy=np.gradient(y)[:,None]
    h=1/np.sqrt((nx/dx)**2+(ny/dy)**2+1e-20)
    h=np.minimum(h,np.maximum(dx,dy))
    def at(a,n):return sample(a,f,X+n*h*nx,Y+n*h*ny)
    # NMS uses physical normals sampled on actual nonuniform coordinates.
    ridge=(g>=at(g,1))&(g>=at(g,-1))
    uy,ux=np.gradient(u,y,x);vy,vx=np.gradient(v,y,x);compression=-(ux+vy)*L/U
    jp=at(np.log(p),3)-at(np.log(p),-3);jr=at(np.log(rho),3)-at(np.log(rho),-3)
    du=((at(u,3)-at(u,-3))*nx+(at(v,3)-at(v,-3))*ny)/U
    candidate=ml&safe&ridge&(compression>=.5)&(g*L>=1.)&(jp>=.055)&(jr>=.025)&(du<=-.01)
    # Component grouping permits one-cell adjacency but adds no new line pixels.
    lab,n=ndi.label(ndi.binary_dilation(candidate),np.ones((3,3)))
    counts=np.bincount(lab[candidate],minlength=n+1);keep=counts>=5;keep[0]=False
    front=candidate&keep[lab]
    assert not (front&~ml).any()
    return front,dict(front_pixels=int(front.sum()),ml_response_pixels=int(ml.sum()),candidate_pixels=int(candidate.sum()))

def controls():
    x=np.linspace(-1,1,121);y=np.linspace(-1,1,121);X,Y=np.meshgrid(x,y);one=np.ones(X.shape)
    f=dict(x=x,y=y,geometry=np.zeros(X.shape,bool),observation_mask=np.ones(X.shape,bool),rho=one,pressure=one,u=3*one,v=0*one)
    ref=dict(reference_length=1.,speed_inf=3.);mask=np.ones(X.shape,bool)
    q=.5*(1+np.tanh(X/.04));states=dict(uniform=f,shear={**f,'u':3+Y},clean_compression={**f,'rho':1+.7*q,'pressure':1+q,'u':3-1.1*q},expansion={**f,'rho':1-.3*q,'pressure':1-.4*q,'u':3+.6*q})
    rng=np.random.default_rng(20260906);states['noisy_compression']={k:(v*(1+.01*rng.standard_normal(v.shape)) if k in ('rho','pressure','u','v') else v) for k,v in states['clean_compression'].items()}
    result={k:narrow_front(z,mask,ref)[1] for k,z in states.items()}
    assert all(result[k]['front_pixels']==0 for k in ('uniform','shear','expansion'))
    assert all(result[k]['front_pixels']>0 for k in ('clean_compression','noisy_compression'))
    return dict(results=result,scope='Engineering sign/noise checks, not physical validation')

def paint(path,r,f,m,exp,front,zoom=False):
    x,y=f['x'],f['y'];domain=f['observation_mask']&~f['geometry'];bg,_=schlieren(f['rho'],x,y,domain,r['reference'])
    fig,axes=plt.subplots(1,2,figsize=(16,9),dpi=160)
    fig.subplots_adjust(left=.045,right=.99,bottom=.2,top=.85,wspace=.12)
    for j,ax in enumerate(axes):
        ax.pcolormesh(x,y,bg,cmap='gray',vmin=0,vmax=1,shading='nearest')
        for z,c,a in [(exp,'#277bb9',.15),(m[2],'#159b80',.13)]+([(m[0],'#e2a54b',.3)] if j else []):
            ax.pcolormesh(x,y,np.ma.masked_where(~z,np.ones(domain.shape)),cmap=ListedColormap([c]),alpha=a,shading='nearest')
        for z,c in zip(m[1:],['#97269a','#159b80']):
            if z.any():ax.contour(x,y,z,[.5],colors=[c],linewidths=.9)
        if j:ax.pcolormesh(x,y,np.ma.masked_where(~front,np.ones(domain.shape)),cmap=ListedColormap(['#ca4300']),shading='nearest')
        else:ax.contour(x,y,m[0],[.5],colors=['#d55e00'],linewidths=.9)
        ax.pcolormesh(x,y,np.ma.masked_where(~f['geometry'],np.ones(domain.shape)),cmap=ListedColormap(['black']),shading='nearest')
        ax.set(aspect='equal',xlim=((.1,1.5) if zoom else (x[0],x[-1])),ylim=((-0.1,.95) if zoom else (y[0],y[-1])),xlabel='x / L',ylabel='y / L')
        ax.set_title(['Previous: boundary of ML shock mask','Separated: narrow front + ML response region'][j],fontsize=15)
    fig.suptitle(f'f270 | stored t = {r["time"]:.2f} | same frozen neural outputs',fontsize=18,y=.95)
    handles=[Line2D([],[],color='#ca4300',label='Narrow front (hybrid diagnostic)',lw=2),Patch(color='#e2a54b',alpha=.3,label='Original ML shock-response region'),Line2D([],[],color='#97269a',label='Core candidates'),Line2D([],[],color='#159b80',label='Wake/shear'),Patch(color='#277bb9',alpha=.2,label='Expansion (physics)')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.09),ncol=3,frameon=False,fontsize=11)
    fig.text(.5,.045,'The shaded response is NOT a validated shock-influence region. Narrow lines require ML support plus physical ridge/jump cues.',ha='center',fontsize=10)
    fig.text(.5,.02,'No retraining, gap filling, manual curve drawing or changes to vortex/wake/expansion outputs.',ha='center',fontsize=10)
    fig.savefig(path,dpi=160);plt.close(fig)

def main():
    OUT.mkdir(exist_ok=True);check=controls();rows=[]
    for i in (60,90,120):
        original=read_json(SRC/f'frame_{i:04d}.json');r=original['record'];f,digest=load_fields(r);assert digest==original['source_sha256']
        pred=SRC/f'prediction_{i:04d}.npz';assert sha256(pred)==original['prediction_sha256']
        with np.load(pred) as z:m=[z[k] for k in ('shock','vortex_core','wake_shear')]
        ep=ROOT/f'results/f270_expansion_video_20260906/expansion_{i:04d}.npz'
        with np.load(ep) as z:exp=z['expansion_physics_only']
        front,metrics=narrow_front(f,m[0],r['reference'])
        out=OUT/f'frame_{i:04d}.npz';np.savez_compressed(out,hybrid_front=front,original_shock_response=m[0],vortex_core=m[1],wake_shear=m[2],expansion_physics_only=exp)
        paint(OUT/f'full_{i:04d}.png',r,f,m,exp,front)
        if i==120:paint(OUT/'detail_0120.png',r,f,m,exp,front,True)
        rows.append(dict(record=r,source_sha256=digest,original_prediction_sha256=sha256(pred),expansion_sha256=sha256(ep),metrics=metrics,output_sha256=sha256(out),figure_sha256=sha256(OUT/f'full_{i:04d}.png'),unchanged_neural_outputs=True))
        print(i,metrics,flush=True)
    write_json(OUT/'RESULTS.json',dict(rows=rows,policy=POLICY,controls=check,script_sha256=sha256(Path(__file__)),model_changed=False,independent_accuracy=False))

if __name__=='__main__':main()
