#!/usr/bin/env python3
"""Fixed prospective Euler cases for MFC 0c9a1d43. No ML labels generated."""
import argparse
import json
import math

CASES = {'ellipse_m2p7': (2.7, 1., .6), 'circle_m2p2': (2.2, 1., 1.), 'ellipse_m2p2': (2.2, 1., .6)}
COMMIT = '0c9a1d434410175ac483b8d71646455444e3b7eb'


def build(case_id, smoke=False):
    mach, lx, ly = CASES[case_id]
    # Full identical extent for all cases. L=major body length=1, a_inf=1.
    nx, ny = (220,200) if smoke else (990,900)
    save_dt, end = (.025,.05) if smoke else (.1,8.)
    dtmax = .2*min(11/nx,10/ny)/(mach+1)
    every = math.ceil(save_dt/dtmax)
    dt = save_dt/every  # exact save lattice, never exceed requested CFL
    c = {
        'run_time_info':'T','x_domain%beg':-5.,'x_domain%end':6.,
        'y_domain%beg':-5.,'y_domain%end':5.,'m':nx-1,'n':ny-1,'p':0,
        'dt':dt,'t_step_start':0,'t_step_stop':round(end/save_dt)*every,'t_step_save':every,
        'num_patches':1,'model_eqns':2,'num_fluids':1,'alt_soundspeed':'F',
        'mpp_lim':'F','mixture_err':'T','time_stepper':3,'weno_order':5,'weno_eps':1.e-16,
        'weno_Re_flux':'F','weno_avg':'T','avg_state':2,'mapped_weno':'F','null_weights':'F',
        'mp_weno':'F','riemann_solver':2,'wave_speeds':1,'viscous':'F','fd_order':4,
        'bc_x%beg':-11,'bc_x%end':-12,'bc_y%beg':-6,'bc_y%end':-6,
        'ib':'T','num_ibs':1,'ib_neighborhood_radius':4,
        'patch_ib(1)%geometry':2 if lx == ly else 6,
        'patch_ib(1)%x_centroid':0.,'patch_ib(1)%y_centroid':0.,'patch_ib(1)%slip':'T',
        'format':1,'precision':2,'prim_vars_wrt':'T','ib_state_wrt':'T','rho_wrt':'T',
        'pres_wrt':'T','vel_wrt(1)':'T','vel_wrt(2)':'T','omega_wrt(3)':'T',
        'schlieren_wrt':'T','schlieren_alpha(1)':.5,'parallel_io':'T',
        'fluid_pp(1)%gamma':2.5,'fluid_pp(1)%pi_inf':0.,
        'patch_icpp(1)%geometry':3,'patch_icpp(1)%x_centroid':.5,'patch_icpp(1)%y_centroid':0.,
        'patch_icpp(1)%length_x':11.,'patch_icpp(1)%length_y':10.,
        'patch_icpp(1)%vel(1)':mach,'patch_icpp(1)%vel(2)':0.,
        'patch_icpp(1)%pres':1/1.4,'patch_icpp(1)%alpha_rho(1)':1.,'patch_icpp(1)%alpha(1)':1.
    }
    if lx == ly: c['patch_ib(1)%radius'] = .5
    else: c.update({'patch_ib(1)%length_x':lx,'patch_ib(1)%length_y':ly})
    meta = dict(case_id=case_id, mach=mach, full_axis_lengths=[lx,ly], cells=[nx,ny],
        reference=dict(rho_inf=1.,pressure_inf=1/1.4,u_inf_x=mach,u_inf_y=0.,gamma=1.4,reference_length=1.,source='registered MFC upstream state'),
        code_time_units='L/a_inf', end_code_time=end,end_convective_time=mach*end,
        dt=dt, save_dt=save_dt, expected_states=round(end/save_dt)+1, mfc_commit=COMMIT,
        smoke=smoke, role='engineering_smoke_only' if smoke else 'prospective_transfer',
        physical_shedding_validated=False)
    return c,meta


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--mfc',type=json.loads,default={})
    p.add_argument('--case-id',choices=CASES,required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--metadata',action='store_true')
    a = p.parse_args()
    c,m = build(a.case_id,a.smoke)
    print(json.dumps(m if a.metadata else c))
