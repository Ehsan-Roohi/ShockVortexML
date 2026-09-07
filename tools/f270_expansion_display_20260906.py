"""Native-coordinate physics-only expansion diagnostic; never a neural output."""
import numpy as np
from scipy import ndimage as ndi

def diagnose(f,reference,cfg):
    domain=f['observation_mask']&~f['geometry']
    sigma=cfg['smoothing_cells'];w=ndi.gaussian_filter(domain.astype(float),sigma)
    def smooth(a):return ndi.gaussian_filter(np.where(domain,a,0.),sigma)/np.maximum(w,1e-12)
    u,v=[smooth(f[k]) for k in ('u','v')]
    rho=np.maximum(smooth(f['rho']),1e-12);p=np.maximum(smooth(f['pressure']),1e-12)
    speed=np.hypot(u,v);ex=u/np.maximum(speed,1e-12);ey=v/np.maximum(speed,1e-12)
    L=reference['reference_length'];U=reference['speed_inf']
    def gradient(a):return np.gradient(a,f['y'],f['x'],edge_order=2)
    def along(a):ay,ax=gradient(a);return (ex*ax+ey*ay)*L
    uy,ux=gradient(u);vy,vx=gradient(v)
    mach=speed/np.sqrt(reference['gamma']*p/rho)
    a=dict(mach=mach,divergence=(ux+vy)*L/U,pressure_drop=-along(np.log(p)),density_drop=-along(np.log(rho)),speed_gain=along(speed)/U,mach_gain=along(mach),vorticity=(vx-uy)*L/U)
    safe=ndi.binary_erosion(domain,iterations=cfg['wall_clearance_cells'])
    assert all(np.isfinite(z[safe]).all() for z in a.values())
    mask=safe.copy()
    for k,t in [('mach','mach_min'),('divergence','divergence_L_U_min'),('pressure_drop','pressure_drop_L_min'),('density_drop','density_drop_L_min'),('speed_gain','speed_gain_L_U_min'),('mach_gain','mach_gain_L_min')]:mask&=a[k]>=cfg[t]
    labels,count=ndi.label(mask,np.ones((3,3)));keep=np.bincount(labels.ravel())>=cfg['minimum_component_pixels'];keep[0]=False
    a['expansion_physics_only']=keep[labels];a['safe_domain']=safe
    return a

def controls(cfg):
    x=np.linspace(0,1,81);y=np.linspace(-.5,.5,81);X,Y=np.meshgrid(x,y)
    base=dict(x=x,y=y,geometry=np.zeros(X.shape,bool),observation_mask=np.ones(X.shape,bool))
    ref=dict(reference_length=1.,speed_inf=3.,gamma=1.4)
    def field(u,p,r):return dict(base,u=u,v=np.zeros(X.shape),pressure=p,rho=r)
    one=np.ones(X.shape)
    states={'uniform':field(3*one,one,one),'shear':field(3+Y,one,one),'compression':field(3-.6*X,np.exp(.5*X),np.exp(.3*X)),'expansion_sign_control':field(3+.6*X,np.exp(-.5*X),np.exp(-.3*X))}
    counts={k:int(diagnose(f,ref,cfg)['expansion_physics_only'].sum()) for k,f in states.items()}
    assert counts['uniform']==counts['shear']==counts['compression']==0 and counts['expansion_sign_control']>0
    return dict(counts=counts,scope='Engineering sign tests, not exact CFD solutions or scientific validation')
