"""Independent learned wake/shear head. Existing shock/core branches are untouched.

The teacher is TRAIN-only weak supervision. Inference sees canonical CFD inputs,
not teacher masks, body coordinates, manually selected wake corridors or labels.
"""
from collections import OrderedDict
import numpy as np
from scipy import ndimage as ndi
import torch
from torch import nn
from jcp2026.common import ROOT, read_json, sha256
from jcp2026.dataset import Frames
from jcp2026.diagnostics import gradient_diagnostics
from jcp2026.shock_repair_noise import noisy_patch
from ml.flow_aligned import FreestreamReference


class WakeNet(nn.Module):
    """Finite-receptive-field FCN; no batch/spatial-statistic normalization."""
    def __init__(self,width=16):
        super().__init__()
        self.net=nn.Sequential(nn.Conv2d(7,width,3,padding=1),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=2,dilation=2),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=4,dilation=4),nn.SiLU(),
            nn.Conv2d(width,width,3,padding=8,dilation=8),nn.SiLU(),
            nn.Conv2d(width,1,1))

    def forward(self,x):return self.net(x)


def clean(mask,minimum=9):
    lab,count=ndi.label(mask,np.ones((3,3)))
    keep=np.bincount(lab.ravel(),minlength=count+1)>=minimum;keep[0]=False
    return keep[lab]


def downstream_of_body(fields,reference):
    """TRAIN wake semantics only, rotation covariant; never used by predict."""
    body=fields['geometry'].astype(bool)
    if not body.any():return np.ones(body.shape,bool)
    X,Y=np.meshgrid(fields['x'],fields['y'])
    # Cell-area weighting also supports stretched observation grids.
    area=np.gradient(fields['y'])[:,None]*np.gradient(fields['x'])[None,:]
    weight=area*body;cx=float((weight*X).sum()/weight.sum());cy=float((weight*Y).sum()/weight.sum())
    return (X-cx)*reference.u_inf_x+(Y-cy)*reference.u_inf_y>=0


def weak_wake(fields,reference,seeds,policy):
    """Localized shear grown from rotating TRAIN seeds, not an expansion label."""
    domain=fields['observation_mask']&~fields['geometry']
    support=ndi.binary_erosion(domain,iterations=policy['wall_clearance_cells'])
    def smooth(z,sigma):
        w=ndi.gaussian_filter(domain.astype(float),sigma)
        return ndi.gaussian_filter(np.where(domain,z,0.),sigma)/np.maximum(w,1e-10)
    u,v=[smooth(fields[k],policy['velocity_smoothing_cells']) for k in ('u','v')]
    d=gradient_diagnostics(u,v,fields['x'],fields['y'])
    factor=reference.reference_length/reference.speed_inf
    omega=d['vorticity']*factor;comp=np.maximum(-d['divergence']*factor,0)
    dx,dy=[float(np.median(np.diff(fields[k]))) for k in ('x','y')]
    sig=lambda length:(length*reference.reference_length/dy,length*reference.reference_length/dx)
    enstrophy=smooth(omega**2,sig(policy['rms_radius_L']))
    compression=smooth(comp**2,sig(policy['rms_radius_L']))
    ratio=enstrophy/(enstrophy+compression+1e-8)
    parallel=(u*reference.u_inf_x+v*reference.u_inf_y)/reference.speed_inf**2
    deficit=smooth(parallel,sig(policy['local_mean_radius_L']))-parallel
    pool=support&(enstrophy>=policy['minimum_rms_vorticity']**2)
    pool&=ratio>=policy['minimum_rotation_to_compression_fraction']
    pool&=deficit>=policy['minimum_local_parallel_deficit']
    if policy.get('downstream_centroid_only',False):
        pool&=downstream_of_body(fields,reference)
    # Seeds do not override compression/shear compatibility or wall clearance.
    seed=seeds.astype(bool)&pool
    target=clean(ndi.binary_propagation(seed,mask=pool,structure=np.ones((3,3))),policy['minimum_component_pixels'])
    edge=ndi.binary_dilation(target,iterations=policy['ignore_boundary_cells'])&~target
    valid=domain&~edge&support
    valid[fields['geometry']]=True
    hard=valid&~target&((ratio<.3)|(enstrophy>.04))
    return dict(target=target,valid=valid,hard_negative=hard,pool=pool,
                rms_vorticity=np.sqrt(enstrophy),compression_fraction=1-ratio,local_deficit=deficit)


class WakeFrames(Frames):
    def __init__(self,cfg,prepared=True):
        super().__init__(ROOT/cfg['index'],'train',cache_size=4)
        self.cfg=cfg;native=read_json(ROOT/cfg['native_index'])
        self.native={r['dataset_id']:r for r in native['records'] if r['split']=='train'}
        assert set(self.native)=={r['dataset_id'] for r in self.frames}
        self.native_root=(ROOT/cfg['native_index']).parent
        self.seed_root=(ROOT/cfg['core_seed_index']).parent
        self.seeds=read_json(ROOT/cfg['core_seed_index'])['frames']
        assert set(self.seeds)==set(self.native)
        self.bank=ROOT/cfg['results']/'teacher_train';self._primitive_cache=OrderedDict()
        self.bank_index=read_json(self.bank/'INDEX.json')['frames'] if prepared else None
        self.source_hashes={}

    def primitive(self,ident):
        if ident not in self._primitive_cache:
            r=self.native[ident];path=self.native_root/r['primitive_file']
            assert sha256(path)==r['primitive_sha256'];self.source_hashes[path.relative_to(ROOT).as_posix()]=r['primitive_sha256']
            with np.load(path) as z:f={k:z[k] for k in ('rho','pressure','u','v','x','y','geometry','observation_mask')}
            self._primitive_cache[ident]=f
            while len(self._primitive_cache)>4:self._primitive_cache.popitem(last=False)
        return self._primitive_cache[ident]

    def seed(self,ident):
        row=self.seeds[ident];path=self.seed_root/row['file'];assert sha256(path)==row['sha256']
        with np.load(path) as z:return z['positive']

    def load(self,i):
        f=super().load(i)
        if self.bank_index is not None and 'wake_target' not in f:
            row=self.bank_index[f['record']['dataset_id']];path=self.bank/row['file'];assert sha256(path)==row['sha256']
            with np.load(path) as z:
                for k in ('target','valid','hard_negative'):f['wake_'+k]=z[k]
        return f

    def sample(self,rng):
        t=self.cfg['training'];size=t['patch_size']
        family=rng.choice(self.family_names);group=rng.choice(self.family_groups[family]);i=int(rng.choice(self.grouped_indices[group]))
        f=self.load(i);ident=f['record']['dataset_id'];roll=rng.random()
        pos=f['wake_target'];neg=f['wake_hard_negative']
        focus=pos if roll<t['positive_sampling_probability'] else (neg if roll<t['positive_sampling_probability']+t['hard_negative_sampling_probability'] else f['observation_mask'])
        coords=np.argwhere(focus)
        if not len(coords):coords=np.argwhere(f['observation_mask'])
        j,k=coords[int(rng.integers(len(coords)))];height,width=pos.shape
        top=int(np.clip(j-rng.integers(size//4,3*size//4),0,height-size));left=int(np.clip(k-rng.integers(size//4,3*size//4),0,width-size))
        x=f['inputs'][:,top:top+size,left:left+size].copy()
        fraction=float(rng.choice(t['noise_fractions'])) if rng.random()<t['noise_probability'] else 0.
        seed=int(rng.integers(2**32-1))
        if fraction:
            ref=FreestreamReference(**self.native[ident]['reference'])
            primitive=self.primitive(ident)
            x=noisy_patch(primitive,primitive['geometry'],primitive['observation_mask'],ref,top,left,size,fraction,seed)
        target=pos[top:top+size,left:left+size].astype(np.float32)
        valid=f['wake_valid'][top:top+size,left:left+size].astype(np.float32)
        turns=int(rng.integers(4))
        x=np.rot90(x,turns,axes=(-2,-1)).copy();target=np.rot90(target,turns).copy();valid=np.rot90(valid,turns).copy()
        return x,target,valid,dict(dataset_id=ident,split='train',top=top,left=left,quarter_turns=turns,noise_fraction=fraction,noise_seed=seed)


def predict(model,inputs,domain,threshold=.5,minimum=9):
    with torch.inference_mode():
        p=torch.sigmoid(model(torch.from_numpy(np.ascontiguousarray(inputs))[None]))[0,0].numpy()
    if not np.isfinite(p).all():raise FloatingPointError('Nonfinite wake output')
    return p,clean((p>=threshold)&domain,minimum)
