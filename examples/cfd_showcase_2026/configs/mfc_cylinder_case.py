#!/usr/bin/env python3
"""MFC 2-D Euler flow over a circular cylinder.

The case is a shock-only validation geometry.  It deliberately uses a slip
immersed boundary and no viscosity; consequently no Reynolds number is
defined and wake vorticity must not be interpreted as viscous vortex shedding.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass


GAMMA = 1.4
RHO_INF = 1.0
P_INF = 1.0 / GAMMA
CYLINDER_DIAMETER = 1.0
CYLINDER_RADIUS = 0.5 * CYLINDER_DIAMETER
CFL_COEFFICIENT = 0.20


@dataclass(frozen=True)
class Grid:
    m: int
    n: int
    x_beg: float
    x_end: float
    y_beg: float
    y_end: float
    cells_per_diameter: int


# MFC stores the maximum zero-based index, so the number of cells is m+1 by
# n+1.  The scientific grids share the domain used by the existing Mach-3 MFC
# airfoil sequence, which makes later ML raster comparisons unambiguous.
GRIDS = {
    "smoke": Grid(119, 99, -2.0, 4.0, -2.5, 2.5, 20),
    "f90": Grid(989, 899, -5.0, 6.0, -5.0, 5.0, 90),
    "f180": Grid(1979, 1799, -5.0, 6.0, -5.0, 5.0, 180),
    "f270": Grid(2969, 2699, -5.0, 6.0, -5.0, 5.0, 270),
}


def build_case(
    mach: float,
    grid_name: str,
    final_time: float,
    save_dt: float,
    output_format: str = "silo",
) -> tuple[dict[str, object], dict[str, float | int | str]]:
    """Return an MFC case dictionary and deterministic run metadata."""

    if not math.isfinite(mach) or mach <= 1.0:
        raise ValueError("--mach must be finite and supersonic")
    if grid_name not in GRIDS:
        raise ValueError(f"unknown grid {grid_name!r}")
    if not math.isfinite(final_time) or final_time <= 0.0:
        raise ValueError("--final-time must be finite and positive")
    if not math.isfinite(save_dt) or save_dt <= 0.0 or save_dt > final_time:
        raise ValueError("--save-dt must be positive and no larger than --final-time")
    if output_format not in {"silo", "binary"}:
        raise ValueError("output_format must be 'silo' or 'binary'")

    grid = GRIDS[grid_name]
    dx = (grid.x_end - grid.x_beg) / (grid.m + 1)
    dy = (grid.y_end - grid.y_beg) / (grid.n + 1)
    a_inf = math.sqrt(GAMMA * P_INF / RHO_INF)
    speed_inf = mach * a_inf
    dt = CFL_COEFFICIENT * min(dx, dy) / (speed_inf + a_inf)

    # Keep every requested output interval equal and make the final state a
    # saved state.  The actual times differ from the request by less than one
    # explicit time step, and are recorded in the metadata.
    save_every = max(1, round(save_dt / dt))
    save_count = max(1, round(final_time / (save_every * dt)))
    stop_step = save_count * save_every
    actual_save_dt = save_every * dt
    actual_final_time = stop_step * dt

    case: dict[str, object] = {
        "run_time_info": "T",
        "x_domain%beg": grid.x_beg,
        "x_domain%end": grid.x_end,
        "y_domain%beg": grid.y_beg,
        "y_domain%end": grid.y_end,
        "m": grid.m,
        "n": grid.n,
        "p": 0,
        "dt": dt,
        "t_step_start": 0,
        "t_step_stop": stop_step,
        "t_step_save": save_every,
        "num_patches": 1,
        "model_eqns": 2,
        "alt_soundspeed": "F",
        "num_fluids": 1,
        "mpp_lim": "F",
        "mixture_err": "T",
        "time_stepper": 3,
        "weno_order": 5,
        "weno_eps": 1.0e-16,
        "weno_Re_flux": "F",
        "weno_avg": "T",
        "avg_state": 2,
        "mapped_weno": "F",
        "null_weights": "F",
        "mp_weno": "F",
        "riemann_solver": 2,
        "wave_speeds": 1,
        "viscous": "F",
        "fd_order": 4,
        "bc_x%beg": -11,
        "bc_x%end": -12,
        "bc_y%beg": -6,
        "bc_y%end": -6,
        "ib": "T",
        "num_ibs": 1,
        "ib_neighborhood_radius": 4,
        "patch_ib(1)%geometry": 2,
        "patch_ib(1)%x_centroid": 0.0,
        "patch_ib(1)%y_centroid": 0.0,
        "patch_ib(1)%radius": CYLINDER_RADIUS,
        "patch_ib(1)%slip": "T",
        "format": 1 if output_format == "silo" else 2,
        "precision": 2,
        "prim_vars_wrt": "T",
        "ib_state_wrt": "T",
        "rho_wrt": "T",
        "pres_wrt": "T",
        "vel_wrt(1)": "T",
        "vel_wrt(2)": "T",
        # Vorticity is retained only as a false-positive audit channel.  This
        # Euler case provides no viscous vortex-shedding ground truth.
        "omega_wrt(3)": "T",
        "schlieren_wrt": "T",
        "schlieren_alpha(1)": 0.5,
        "schlieren_alpha(2)": 0.5,
        "parallel_io": "T",
        "patch_icpp(1)%geometry": 3,
        "patch_icpp(1)%x_centroid": 0.5 * (grid.x_beg + grid.x_end),
        "patch_icpp(1)%y_centroid": 0.5 * (grid.y_beg + grid.y_end),
        "patch_icpp(1)%length_x": grid.x_end - grid.x_beg,
        "patch_icpp(1)%length_y": grid.y_end - grid.y_beg,
        "patch_icpp(1)%vel(1)": speed_inf,
        "patch_icpp(1)%vel(2)": 0.0,
        "patch_icpp(1)%pres": P_INF,
        "patch_icpp(1)%alpha_rho(1)": RHO_INF,
        "patch_icpp(1)%alpha(1)": 1.0,
        "fluid_pp(1)%gamma": 1.0 / (GAMMA - 1.0),
        "fluid_pp(1)%pi_inf": 0.0,
    }
    metadata: dict[str, float | int | str] = {
        "mach": mach,
        "gamma": GAMMA,
        "grid": grid_name,
        "cells_x": grid.m + 1,
        "cells_y": grid.n + 1,
        "cells_per_diameter": grid.cells_per_diameter,
        "dx_over_d": dx / CYLINDER_DIAMETER,
        "dy_over_d": dy / CYLINDER_DIAMETER,
        "dt": dt,
        "requested_final_time": final_time,
        "actual_final_time": actual_final_time,
        "requested_save_dt": save_dt,
        "actual_save_dt": actual_save_dt,
        "saved_states_including_initial": save_count + 1,
        "output_format": output_format,
        "physics": "two-dimensional inviscid Euler; continuum; slip cylinder",
        "label_scope": "shock and background/other only; vortex is not ground truth",
    }
    return case, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mfc", type=json.loads, default="{}", metavar="DICT")
    parser.add_argument("--mach", type=float, default=2.7)
    parser.add_argument("--grid", choices=tuple(GRIDS), default="f90")
    parser.add_argument("--final-time", type=float, default=3.0)
    parser.add_argument("--save-dt", type=float, default=0.1)
    parser.add_argument("--format", choices=("silo", "binary"), default="silo")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        case, _ = build_case(
            args.mach, args.grid, args.final_time, args.save_dt, args.format
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(case))


if __name__ == "__main__":
    main()
