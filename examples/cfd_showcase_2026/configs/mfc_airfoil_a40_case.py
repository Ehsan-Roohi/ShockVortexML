#!/usr/bin/env python3
"""Post-process and restart the MFC A40 viscous/no-model screen safely.

The restart clock is expressed in physical nondimensional time.  When the
time step is reduced, checkpoint names are re-indexed by the launcher so that
``step * dt`` remains the physical time shown in diagnostics and movies.
"""

import argparse
import json
import math


parser = argparse.ArgumentParser()
parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
parser.add_argument("--mode", choices=("initial", "restart"), default="initial")
parser.add_argument("--grid", choices=("f180", "f270", "f405"), default="f270")
parser.add_argument("--start-time", type=float, default=0.0)
parser.add_argument("--final-time", type=float, default=0.4)
parser.add_argument("--save-dt", type=float, default=0.05)
parser.add_argument("--dt-factor", type=int, default=1)
parser.add_argument("--format", choices=("silo", "binary"), default="binary")
args = parser.parse_args()

if args.dt_factor < 1:
    parser.error("--dt-factor must be a positive integer")
if args.start_time < 0.0 or args.final_time < args.start_time:
    parser.error("require 0 <= --start-time <= --final-time")
interval = args.final_time - args.start_time
if args.save_dt <= 0.0 or (interval > 0.0 and args.save_dt > interval):
    parser.error("--save-dt must be positive and fit inside a nonzero interval")
if args.mode == "initial" and (args.start_time != 0.0 or args.dt_factor != 1):
    parser.error("initial mode requires --start-time 0 and --dt-factor 1")
if args.mode == "restart" and args.start_time <= 0.0:
    parser.error("restart mode requires a positive --start-time")

gamma = 1.4
mach = 3.0
alpha_deg = 40.0
re_chord = 1.0e6
rho_inf = 1.0
p_inf = 1.0 / gamma
a_inf = math.sqrt(gamma * p_inf / rho_inf)
speed_inf = mach * a_inf
alpha = math.radians(alpha_deg)
u_inf = speed_inf * math.cos(alpha)
v_inf = speed_inf * math.sin(alpha)

grids = {
    "f180": {"m": 1979, "n": 1799, "x": (-5.0, 6.0), "y": (-5.0, 5.0)},
    "f270": {"m": 2969, "n": 2699, "x": (-5.0, 6.0), "y": (-5.0, 5.0)},
    "f405": {"m": 4454, "n": 4049, "x": (-5.0, 6.0), "y": (-5.0, 5.0)},
}
grid = grids[args.grid]
x_beg, x_end = grid["x"]
y_beg, y_end = grid["y"]
dx = (x_end - x_beg) / (grid["m"] + 1)
dy = (y_end - y_beg) / (grid["n"] + 1)
base_dt = 0.20 * min(dx, dy) / (speed_inf + a_inf)
dt = base_dt / args.dt_factor


def exact_step(value: float, label: str) -> int:
    step = round(value / dt)
    if not math.isclose(step * dt, value, rel_tol=0.0, abs_tol=1.0e-12):
        parser.error(f"{label}={value} is not an integer multiple of dt={dt}")
    return step


start_step = exact_step(args.start_time, "--start-time")
stop_step = exact_step(args.final_time, "--final-time")
save_every = exact_step(args.save_dt, "--save-dt")
if (stop_step - start_step) % save_every:
    parser.error("the requested interval must contain an integer number of saves")

inverse_mu = re_chord / speed_inf


def boundary_pair(normal_velocity: float) -> tuple[int, int]:
    if abs(normal_velocity) <= 1.0e-12 * a_inf:
        return -6, -6
    if normal_velocity >= a_inf:
        return -11, -12
    if normal_velocity <= -a_inf:
        return -12, -11
    if normal_velocity >= 0.0:
        return -7, -8
    return -8, -7


bc_x_beg, bc_x_end = boundary_pair(u_inf)
bc_y_beg, bc_y_end = boundary_pair(v_inf)

case = {
    "run_time_info": "T",
    "m": grid["m"],
    "n": grid["n"],
    "p": 0,
    # MFC 0c9a1d4's stage-local validator requires explicit bounds even
    # when old_grid=T.  The pre-processor subsequently replaces them with
    # the checkpoint grid, so these exact original bounds are safe.
    "x_domain%beg": x_beg,
    "x_domain%end": x_end,
    "y_domain%beg": y_beg,
    "y_domain%end": y_end,
    "dt": dt,
    "t_step_start": start_step,
    "t_step_stop": stop_step,
    "t_step_save": save_every,
    "num_patches": 0 if args.mode == "restart" else 1,
    "model_eqns": 2,
    "alt_soundspeed": "F",
    "num_fluids": 1,
    "mpp_lim": "F",
    "mixture_err": "T",
    "time_stepper": 3,
    "weno_order": 5,
    "weno_eps": 1.0e-16,
    "weno_Re_flux": "T",
    "weno_avg": "T",
    "avg_state": 2,
    "mapped_weno": "F",
    "null_weights": "F",
    "mp_weno": "F",
    "riemann_solver": 2,
    "wave_speeds": 1,
    "viscous": "T",
    "fd_order": 4,
    "bc_x%beg": bc_x_beg,
    "bc_x%end": bc_x_end,
    "bc_y%beg": bc_y_beg,
    "bc_y%end": bc_y_end,
    "ib": "T",
    "num_ibs": 1,
    "ib_neighborhood_radius": 4,
    "num_stl_models": 1,
    "patch_ib(1)%geometry": 5,
    "patch_ib(1)%model_id": 1,
    "stl_models(1)%model_filepath": "Diamond_Airfoil_2D_MFC.stl",
    "stl_models(1)%model_threshold": 0.90,
    "patch_ib(1)%slip": "F",
    "format": 2 if args.format == "binary" else 1,
    "precision": 2,
    "prim_vars_wrt": "T",
    "ib_state_wrt": "T",
    "rho_wrt": "T",
    "pres_wrt": "T",
    "vel_wrt(1)": "T",
    "vel_wrt(2)": "T",
    "omega_wrt(3)": "T",
    "schlieren_wrt": "T",
    "schlieren_alpha(1)": 0.5,
    "schlieren_alpha(2)": 0.5,
    "parallel_io": "T",
    "fluid_pp(1)%gamma": 1.0 / (gamma - 1.0),
    "fluid_pp(1)%pi_inf": 0.0,
    "fluid_pp(1)%Re(1)": inverse_mu,
}

if args.mode == "initial":
    case.update(
        {
            "x_domain%beg": x_beg,
            "x_domain%end": x_end,
            "y_domain%beg": y_beg,
            "y_domain%end": y_end,
            "patch_icpp(1)%geometry": 3,
            "patch_icpp(1)%x_centroid": 0.5 * (x_beg + x_end),
            "patch_icpp(1)%y_centroid": 0.5 * (y_beg + y_end),
            "patch_icpp(1)%length_x": x_end - x_beg,
            "patch_icpp(1)%length_y": y_end - y_beg,
            "patch_icpp(1)%vel(1)": u_inf,
            "patch_icpp(1)%vel(2)": v_inf,
            "patch_icpp(1)%pres": p_inf,
            "patch_icpp(1)%alpha_rho(1)": rho_inf,
            "patch_icpp(1)%alpha(1)": 1.0,
        }
    )
else:
    # MFC's documented restart form: retain only m/n/p, read both the old
    # grid and conservative state, and add no new initial-condition patches.
    case.update({"old_ic": "T", "old_grid": "T", "t_step_old": 0})


def add_subsonic_boundary_targets(axis: str, beginning: int, end: int) -> None:
    if beginning == -7 or end == -7:
        case[f"bc_{axis}%grcbc_in"] = "T"
        case[f"bc_{axis}%vel_in(1)"] = u_inf
        case[f"bc_{axis}%vel_in(2)"] = v_inf
        case[f"bc_{axis}%pres_in"] = p_inf
        case[f"bc_{axis}%alpha_rho_in(1)"] = rho_inf
        case[f"bc_{axis}%alpha_in(1)"] = 1.0
    if beginning == -8 or end == -8:
        case[f"bc_{axis}%grcbc_out"] = "T"
        case[f"bc_{axis}%pres_out"] = p_inf


add_subsonic_boundary_targets("x", bc_x_beg, bc_x_end)
add_subsonic_boundary_targets("y", bc_y_beg, bc_y_end)
print(json.dumps(case))
