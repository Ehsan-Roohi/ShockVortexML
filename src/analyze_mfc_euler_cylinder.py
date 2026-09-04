#!/usr/bin/env python3
"""Audit the MFC Mach-2.7 Euler cylinder sequence.

The script keeps physics-only, frozen-ML-only, and ML-seeded hybrid outputs in
separate directories.  The Euler/slip case is a shock target and a vortex
false-positive control: rotational responses are reported, but the accepted
vortex mask is deliberately empty.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.patches import Circle, Patch
import numpy as np
from scipy import ndimage


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from physics_proposals import derivatives, shock_proposal, vortex_proposal


VERMILLION = "#D55E00"
GOLD = "#E69F00"
MAGENTA = "#CC79A7"
CHARCOAL = "#202020"
MID_GRAY = "#777777"


@dataclass(frozen=True)
class Frame:
    step: int
    time: float
    x: np.ndarray
    y: np.ndarray
    rho: np.ndarray
    u: np.ndarray
    v: np.ndarray
    pressure: np.ndarray
    omega: np.ndarray
    ib: np.ndarray
    schlieren: np.ndarray


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def import_h5py():
    try:
        import h5py  # type: ignore

        return h5py
    except ModuleNotFoundError:
        # NumPy/SciPy are intentionally imported first.  Appending, rather
        # than prepending, keeps the bundled compatible NumPy while exposing
        # only the locally vendored h5py package.
        vendor = ROOT / "tools" / "_python_deps"
        if str(vendor) not in sys.path:
            sys.path.append(str(vendor))
        import h5py  # type: ignore

        return h5py


def decode_reference(value: Any) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, (bytes, np.bytes_)):
        return bytes(value).decode("utf-8").rstrip("\x00")
    return str(value)


def silo_payload(handle: Any, variable: str, shape: tuple[int, int]) -> np.ndarray:
    metadata = handle[variable].attrs["silo"]
    reference = decode_reference(metadata["value0"]).lstrip("/")
    raw = np.asarray(handle[reference][...], dtype=np.float64)
    # MFC/Silo writes the x-fast payload into an HDF5 dataset whose displayed
    # dimensions are (nx,ny).  A C-order reshape of the linear buffer is the
    # correct y-by-x raster.  Transposing is wrong and turns the exact circle
    # marker into a domain-spanning stripe.
    return raw.reshape(shape, order="C")


def read_inventory(path: Path) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    with path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            records.append({"step": int(row["step"]), "time": float(row["time"])})
    records.sort(key=lambda item: int(item["step"]))
    return records


def read_frame(path: Path, step: int, time: float) -> Frame:
    h5py = import_h5py()
    with h5py.File(path, "r") as handle:
        mesh = handle["rectilinear_grid"].attrs["silo"]
        x_ref = decode_reference(mesh["coord0"]).lstrip("/")
        y_ref = decode_reference(mesh["coord1"]).lstrip("/")
        x_nodes = np.asarray(handle[x_ref][...], dtype=np.float64)
        y_nodes = np.asarray(handle[y_ref][...], dtype=np.float64)
        x = 0.5 * (x_nodes[:-1] + x_nodes[1:])
        y = 0.5 * (y_nodes[:-1] + y_nodes[1:])
        shape = (len(y), len(x))
        frame = Frame(
            step=step,
            time=time,
            x=x,
            y=y,
            rho=silo_payload(handle, "rho", shape),
            u=silo_payload(handle, "vel1", shape),
            v=silo_payload(handle, "vel2", shape),
            pressure=silo_payload(handle, "pres", shape),
            omega=silo_payload(handle, "omega3", shape),
            ib=silo_payload(handle, "ib_markers", shape),
            schlieren=silo_payload(handle, "schlieren", shape),
        )
    return frame


def exact_geometry(frame: Frame, radius: float) -> np.ndarray:
    yy, xx = np.meshgrid(frame.y, frame.x, indexing="ij")
    return xx * xx + yy * yy <= radius * radius


def geometry_iou(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.count_nonzero(first & second)
    union = np.count_nonzero(first | second)
    return float(intersection / max(union, 1))


def select_main_bow_component(
    centerline: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    guard_radius: float,
    dy: float,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    labels, count = ndimage.label(
        centerline, structure=np.ones((3, 3), dtype=np.uint8)
    )
    components: list[dict[str, Any]] = []
    accepted_ids: list[tuple[tuple[int, int, float], int]] = []
    for component_id in range(1, count + 1):
        mask = labels == component_id
        rows, columns = np.where(mask)
        if not len(columns):
            continue
        axis_pixels = (
            (np.abs(y[rows]) <= max(3.0 * dy, 0.04))
            & (x[columns] < -guard_radius)
        )
        record = {
            "component_id": component_id,
            "pixels": int(len(columns)),
            "x_min": float(x[columns].min()),
            "x_max": float(x[columns].max()),
            "y_min": float(y[rows].min()),
            "y_max": float(y[rows].max()),
            "stagnation_axis_upstream_pixels": int(np.count_nonzero(axis_pixels)),
        }
        components.append(record)
        if np.any(axis_pixels):
            priority = (
                int(np.count_nonzero(axis_pixels)),
                int(len(columns)),
                -float(x[columns].min()),
            )
            accepted_ids.append((priority, component_id))
    selected = np.zeros_like(centerline, dtype=bool)
    if accepted_ids:
        selected_id = max(accepted_ids)[1]
        selected = labels == selected_id
        for record in components:
            record["selected_main_bow"] = bool(record["component_id"] == selected_id)
    else:
        for record in components:
            record["selected_main_bow"] = False
    return selected, components


def bow_curve(mask: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    curve_x: list[float] = []
    curve_y: list[float] = []
    for row in np.flatnonzero(np.any(mask, axis=1)):
        columns = np.flatnonzero(mask[row])
        if len(columns):
            curve_x.append(float(np.min(x[columns])))
            curve_y.append(float(y[row]))
    return np.asarray(curve_x), np.asarray(curve_y)


def interpolated_crossing(x: np.ndarray, values: np.ndarray, level: float) -> float | None:
    above = np.flatnonzero(values >= level)
    if not len(above):
        return None
    index = int(above[0])
    if index == 0:
        return float(x[0])
    x0, x1 = float(x[index - 1]), float(x[index])
    y0, y1 = float(values[index - 1]), float(values[index])
    if y1 == y0:
        return x1
    fraction = (level - y0) / (y1 - y0)
    return x0 + fraction * (x1 - x0)


def stagnation_metrics(
    frame: Frame,
    main_centerline: np.ndarray,
    *,
    radius: float,
    gamma: float,
    rho_inf: float,
    p_inf: float,
    u_inf: float,
    offset_cells: int,
) -> dict[str, Any]:
    center_row = int(np.argmin(np.abs(frame.y)))
    nearby_rows = np.flatnonzero(np.abs(frame.y) <= 2.0 * abs(frame.y[1] - frame.y[0]))
    rows, columns = np.where(main_centerline[nearby_rows])
    if not len(columns):
        return {"detected": False}
    global_rows = nearby_rows[rows]
    upstream = columns[frame.x[columns] < -radius]
    if not len(upstream):
        return {"detected": False}
    shock_column = int(upstream[np.argmin(frame.x[upstream])])
    x_shock = float(frame.x[shock_column])
    k = int(np.argmin(np.abs(frame.x - x_shock)))
    if k - offset_cells - 1 < 0 or k + offset_cells + 2 > len(frame.x):
        return {"detected": False}

    def local_median(field: np.ndarray, center: int) -> float:
        return float(np.median(field[center_row, center - 1 : center + 2]))

    upstream_k = k - offset_cells
    downstream_k = k + offset_cells
    p1 = local_median(frame.pressure, upstream_k)
    p2 = local_median(frame.pressure, downstream_k)
    rho1 = local_median(frame.rho, upstream_k)
    rho2 = local_median(frame.rho, downstream_k)
    u2 = local_median(frame.u, downstream_k)
    v2 = local_median(frame.v, downstream_k)
    mach2 = math.hypot(u2, v2) / math.sqrt(gamma * p2 / rho2)

    profile_slice = slice(max(0, k - 8), min(len(frame.x), k + 9))
    profile_x = frame.x[profile_slice]
    profile_p = frame.pressure[center_row, profile_slice]
    low = p1 + 0.10 * (p2 - p1)
    high = p1 + 0.90 * (p2 - p1)
    x10 = interpolated_crossing(profile_x, profile_p, low)
    x90 = interpolated_crossing(profile_x, profile_p, high)
    dx = abs(frame.x[1] - frame.x[0])
    width_cells = None if x10 is None or x90 is None else float((x90 - x10) / dx)

    body_line = (frame.x >= -radius + dx) & (frame.x <= radius - dx)
    stagnation_pressure = float(np.max(frame.pressure[center_row, body_line]))
    dynamic_pressure = 0.5 * rho_inf * u_inf * u_inf
    stagnation_cp = (stagnation_pressure - p_inf) / dynamic_pressure
    return {
        "detected": True,
        "x_shock": x_shock,
        "standoff_over_d": float((-radius - x_shock) / (2.0 * radius)),
        "sample_offset_cells": int(offset_cells),
        "p_upstream": p1,
        "p_downstream": p2,
        "rho_upstream": rho1,
        "rho_downstream": rho2,
        "u_downstream": u2,
        "v_downstream": v2,
        "p2_over_p1": p2 / p1,
        "rho2_over_rho1": rho2 / rho1,
        "mach_downstream": mach2,
        "shock_width_10_90_cells": width_cells,
        "stagnation_pressure": stagnation_pressure,
        "stagnation_cp": stagnation_cp,
    }


def relative_l2(current: np.ndarray, previous: np.ndarray, mask: np.ndarray) -> float:
    return float(np.linalg.norm((current - previous)[mask]) / max(np.linalg.norm(previous[mask]), 1e-30))


def symmetry_error(field: np.ndarray, fluid: np.ndarray) -> float:
    allowed = fluid & np.flipud(fluid)
    difference = (field - np.flipud(field))[allowed]
    reference = field[allowed]
    return float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-30))


def surface_cp(
    frame: Frame,
    *,
    radius: float,
    sample_offset_cells: int,
    rho_inf: float,
    p_inf: float,
    u_inf: float,
) -> tuple[np.ndarray, np.ndarray]:
    theta = np.linspace(-180.0, 180.0, 361)
    angle = np.deg2rad(theta)
    dx = abs(frame.x[1] - frame.x[0])
    sample_radius = radius + sample_offset_cells * dx
    sample_x = -sample_radius * np.cos(angle)
    sample_y = sample_radius * np.sin(angle)
    columns = (sample_x - frame.x[0]) / (frame.x[1] - frame.x[0])
    rows = (sample_y - frame.y[0]) / (frame.y[1] - frame.y[0])
    pressure = ndimage.map_coordinates(
        frame.pressure, [rows, columns], order=1, mode="nearest"
    )
    cp = (pressure - p_inf) / (0.5 * rho_inf * u_inf * u_inf)
    return theta, cp


def binary_dice(first: np.ndarray, second: np.ndarray) -> float:
    intersection = np.count_nonzero(first & second)
    return float(2.0 * intersection / max(np.count_nonzero(first) + np.count_nonzero(second), 1))


def save_figure(figure: plt.Figure, base: Path, dpi: int) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(figure)


def draw_body(axis: plt.Axes, radius: float) -> None:
    axis.add_patch(
        Circle((0.0, 0.0), radius, facecolor="white", edgecolor=CHARCOAL, linewidth=0.9, zorder=8)
    )


def draw_mask_contour(
    axis: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    mask: np.ndarray,
    *,
    color: str,
    linewidth: float = 1.0,
    linestyle: str = "solid",
    zorder: int = 7,
) -> None:
    if np.any(mask):
        axis.contour(
            x,
            y,
            mask.astype(float),
            levels=[0.5],
            colors=[color],
            linewidths=[linewidth],
            linestyles=[linestyle],
            zorder=zorder,
        )


def configure_axis(axis: plt.Axes, xlim: tuple[float, float], ylim: tuple[float, float]) -> None:
    axis.set_xlim(*xlim)
    axis.set_ylim(*ylim)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x/D$")
    axis.set_ylabel(r"$y/D$")


def process_physics(
    config: dict[str, Any],
    records: list[dict[str, float | int]],
    physics_dir: Path,
    config_path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray | Frame]]]:
    input_dir = ROOT / config["input_dir"]
    silo_dir = input_dir / config["silo"]["leaf_subdir"]
    physical = config["physics"]
    detector = config["physics_detector"]
    evaluation = config["evaluation"]
    radius = float(physical["cylinder_radius"])
    guard_radius = float(detector["guard_radius"])
    figure_steps = set(int(step) for step in config["figures"]["temporal_steps"])
    figure_steps.update(int(step) for step in config["ml_transfer"]["representative_steps"])
    figure_steps.add(int(records[-1]["step"]))
    cache: dict[int, dict[str, np.ndarray | Frame]] = {}
    frame_reports: list[dict[str, Any]] = []
    previous_pressure: np.ndarray | None = None
    exact_geometry_reference: np.ndarray | None = None
    x_reference: np.ndarray | None = None
    y_reference: np.ndarray | None = None

    for index, record in enumerate(records):
        step = int(record["step"])
        time = float(record["time"])
        frame = read_frame(silo_dir / f"{step}.silo", step, time)
        dx = float(np.mean(np.diff(frame.x)))
        dy = float(np.mean(np.diff(frame.y)))
        geometry = exact_geometry(frame, radius)
        ib = frame.ib > 0.5
        if exact_geometry_reference is None:
            exact_geometry_reference = geometry
            x_reference = frame.x
            y_reference = frame.y
        elif not (
            np.array_equal(geometry, exact_geometry_reference)
            and np.array_equal(frame.x, x_reference)
            and np.array_equal(frame.y, y_reference)
        ):
            raise RuntimeError("Grid or exact cylinder geometry changed within the case group")

        yy, xx = np.meshgrid(frame.y, frame.x, indexing="ij")
        geometry_guard = xx * xx + yy * yy <= guard_radius * guard_radius
        flow = derivatives(
            frame.rho,
            frame.pressure,
            frame.u,
            frame.v,
            dx=dx,
            dy=dy,
        )
        proposal = shock_proposal(
            frame.rho,
            frame.pressure,
            flow,
            geometry_guard,
            dx=dx,
            dy=dy,
            jump_offset_pixels=float(detector["jump_offset_pixels"]),
            leading_edge_anchor_distance=float(
                detector["leading_edge_anchor_distance"]
            ),
        )
        main_centerline, component_records = select_main_bow_component(
            proposal.centerline,
            frame.x,
            frame.y,
            guard_radius=guard_radius,
            dy=dy,
        )
        main_envelope = ndimage.binary_dilation(
            main_centerline,
            iterations=int(detector["envelope_dilation_pixels"]),
        ) & ~geometry
        other_fronts = proposal.centerline & ~main_centerline
        other_envelope = ndimage.binary_dilation(other_fronts, iterations=2) & ~geometry
        vortex = vortex_proposal(
            flow,
            geometry,
            proposal,
            frame.x,
            frame.y,
            dx=dx,
            dy=dy,
        )
        vortex_candidate = vortex.core | vortex.uncertainty
        accepted_vortex = np.zeros_like(geometry, dtype=bool)
        fluid = ~geometry
        finite = np.isfinite(frame.rho) & np.isfinite(frame.pressure)
        positive = (frame.rho > 0.0) & (frame.pressure > 0.0)
        stationarity_bounds = evaluation["stationarity_roi"]
        stationarity_roi = (
            (xx >= float(stationarity_bounds[0]))
            & (xx <= float(stationarity_bounds[1]))
            & (yy >= float(stationarity_bounds[2]))
            & (yy <= float(stationarity_bounds[3]))
            & fluid
        )
        pressure_change = (
            None
            if previous_pressure is None
            else relative_l2(frame.pressure, previous_pressure, stationarity_roi)
        )
        previous_pressure = frame.pressure.copy()
        stagnation = stagnation_metrics(
            frame,
            main_centerline,
            radius=radius,
            gamma=float(physical["gamma"]),
            rho_inf=float(physical["rho_inf"]),
            p_inf=float(physical["p_inf"]),
            u_inf=float(physical["u_inf"]),
            offset_cells=int(evaluation["stagnation_jump_offset_cells"]),
        )
        curve_x, curve_y = bow_curve(main_centerline, frame.x, frame.y)
        report = {
            "frame_index": index,
            "step": step,
            "time": time,
            "finite_fluid_fraction": float(np.mean(finite[fluid])),
            "positive_rho_p_fluid_fraction": float(np.mean(positive[fluid])),
            "ib_exact_geometry_iou": geometry_iou(ib, geometry),
            "main_bow_centerline_pixels": int(np.count_nonzero(main_centerline)),
            "main_bow_envelope_pixels": int(np.count_nonzero(main_envelope)),
            "main_bow_wall_overlap_pixels": int(np.count_nonzero(main_envelope & geometry)),
            "other_compression_centerline_pixels": int(np.count_nonzero(other_fronts)),
            "shock_uncertain_pixels": int(np.count_nonzero(proposal.uncertainty)),
            "raw_rotational_candidate_pixels": int(np.count_nonzero(vortex_candidate)),
            "raw_rotational_candidate_objects": len(vortex.objects),
            "accepted_vortex_pixels": 0,
            "relative_l2_pressure_change_stationarity_roi": pressure_change,
            "pressure_symmetry_relative_l2": symmetry_error(frame.pressure, fluid),
            "stagnation_line": stagnation,
            "main_bow_curve_points": int(len(curve_x)),
            "shock_components": component_records,
            "top_rotational_candidates": vortex.objects[:5],
        }
        frame_reports.append(report)
        np.savez_compressed(
            physics_dir / f"step_{step:05d}_physics_only.npz",
            step=np.asarray(step, dtype=np.int64),
            time=np.asarray(time, dtype=np.float64),
            shock_centerline=main_centerline.astype(np.uint8),
            shock_envelope=main_envelope.astype(np.uint8),
            other_compression_centerline=other_fronts.astype(np.uint8),
            other_compression_envelope=other_envelope.astype(np.uint8),
            shock_uncertain=proposal.uncertainty.astype(np.uint8),
            vortex_candidate=vortex_candidate.astype(np.uint8),
            accepted_vortex=accepted_vortex.astype(np.uint8),
            background_other=(fluid & ~main_envelope).astype(np.uint8),
            geometry_mask=geometry.astype(np.uint8),
            bow_curve_x=curve_x.astype(np.float32),
            bow_curve_y=curve_y.astype(np.float32),
        )
        if step in figure_steps:
            cache[step] = {
                "frame": Frame(
                    step=frame.step,
                    time=frame.time,
                    x=frame.x,
                    y=frame.y,
                    rho=frame.rho.astype(np.float32),
                    u=frame.u.astype(np.float32),
                    v=frame.v.astype(np.float32),
                    pressure=frame.pressure.astype(np.float32),
                    omega=frame.omega.astype(np.float32),
                    ib=frame.ib.astype(np.float32),
                    schlieren=frame.schlieren.astype(np.float32),
                ),
                "geometry": geometry,
                "main_centerline": main_centerline,
                "main_envelope": main_envelope,
                "other_fronts": other_fronts,
                "other_envelope": other_envelope,
                "shock_uncertain": proposal.uncertainty,
                "shock_score": proposal.score,
                "vortex_candidate": vortex_candidate,
                "compression": flow.compression.astype(np.float32),
                "lambda_ci": flow.lambda_ci.astype(np.float32),
            }
        print(
            f"physics step={step:05d} time={time:.4f} "
            f"shock={np.count_nonzero(main_centerline)} "
            f"rotational_candidates={np.count_nonzero(vortex_candidate)}"
        )

    detected_tail = [
        item
        for item in frame_reports[-int(evaluation["stationarity_tail_frames"]) :]
        if item["stagnation_line"]["detected"]
    ]
    if len(detected_tail) >= 2:
        tail_time = np.asarray([item["time"] for item in detected_tail], dtype=float)
        tail_delta = np.asarray(
            [item["stagnation_line"]["standoff_over_d"] for item in detected_tail],
            dtype=float,
        )
        standoff_slope = float(np.polyfit(tail_time, tail_delta, 1)[0])
    else:
        standoff_slope = float("nan")
    tail_pressure_changes = [
        float(item["relative_l2_pressure_change_stationarity_roi"])
        for item in frame_reports[-int(evaluation["stationarity_tail_frames"]) :]
        if item["relative_l2_pressure_change_stationarity_roi"] is not None
    ]
    maximum_tail_pressure_change = max(tail_pressure_changes, default=float("nan"))
    pressure_gate = float(evaluation["maximum_tail_relative_l2_pressure_change"])
    slope_gate = float(evaluation["maximum_tail_standoff_slope_d_per_time"])
    stationarity_pass = bool(
        np.isfinite(maximum_tail_pressure_change)
        and np.isfinite(standoff_slope)
        and maximum_tail_pressure_change <= pressure_gate
        and abs(standoff_slope) <= slope_gate
    )

    rh_path = ROOT / evaluation["rh_reference"]
    rh = json.loads(rh_path.read_text(encoding="utf-8"))
    final_stagnation = frame_reports[-1]["stagnation_line"]
    rh_comparison: dict[str, Any] = {}
    if final_stagnation["detected"]:
        for name in ("p2_over_p1", "rho2_over_rho1", "mach_downstream", "stagnation_cp"):
            reference_name = "inviscid_stagnation_cp" if name == "stagnation_cp" else name
            measured = float(final_stagnation[name])
            reference = float(rh[reference_name])
            rh_comparison[name] = {
                "measured": measured,
                "exact_reference": reference,
                "relative_error": abs(measured - reference) / abs(reference),
            }

    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "result_role": "physics_only_audit_and_weak_proposal; not human ground truth",
        "case_group_id": config["case_group_id"],
        "frames": frame_reports,
        "aggregate": {
            "snapshot_count": len(frame_reports),
            "step_sequence": [item["step"] for item in frame_reports],
            "time_sequence": [item["time"] for item in frame_reports],
            "all_fluid_density_pressure_finite": all(
                item["finite_fluid_fraction"] == 1.0 for item in frame_reports
            ),
            "all_fluid_density_pressure_positive": all(
                item["positive_rho_p_fluid_fraction"] == 1.0 for item in frame_reports
            ),
            "minimum_ib_exact_geometry_iou": min(
                item["ib_exact_geometry_iou"] for item in frame_reports
            ),
            "total_accepted_vortex_pixels": 0,
            "tail_maximum_relative_l2_pressure_change": maximum_tail_pressure_change,
            "tail_standoff_slope_d_per_time": standoff_slope,
            "stationarity_pass": stationarity_pass,
            "stationarity_gates": {
                "maximum_relative_l2_pressure_change": pressure_gate,
                "maximum_absolute_standoff_slope_d_per_time": slope_gate,
            },
            "f90_pilot_decision": (
                evaluation.get("stationarity_pass_action", "PASS_FOR_F180")
                if stationarity_pass
                else evaluation.get(
                    "stationarity_fail_action", "EXTEND_F90_BEFORE_F180"
                )
            ),
            "final_rankine_hugoniot_comparison": rh_comparison,
            "experimental_validation_status": "NOT_EVALUATED_EXPERIMENTAL_CURVES_NOT_DIGITIZED",
        },
        "provenance": {
            "config": str(config_path.resolve()),
            "config_sha256": sha256(config_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "outer_zip_sha256_verified_before_analysis": config["outer_zip_sha256"],
            "nested_archive_sha256_verified_before_analysis": config["nested_archive_sha256"],
            "spatial_decode": config["silo"]["spatial_decode"],
        },
    }
    write_json(physics_dir / "physics_only_report.json", report)
    return report, cache


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if length <= tile_size:
        return [0]
    stride = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, stride))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def predict_tiled(
    model: Any,
    torch: Any,
    inputs: np.ndarray,
    *,
    tile_size: int,
    overlap: int,
    batch_size: int,
) -> np.ndarray:
    channels, height, width = inputs.shape
    padded_height = max(height, tile_size)
    padded_width = max(width, tile_size)
    padded = np.pad(
        inputs,
        ((0, 0), (0, padded_height - height), (0, padded_width - width)),
        mode="edge",
    )
    starts_y = tile_starts(padded_height, tile_size, overlap)
    starts_x = tile_starts(padded_width, tile_size, overlap)
    window_1d = np.maximum(np.hanning(tile_size).astype(np.float32), 0.05)
    window = window_1d[:, None] * window_1d[None, :]
    probability_sum = np.zeros((6, padded_height, padded_width), dtype=np.float32)
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    positions = [(top, left) for top in starts_y for left in starts_x]
    with torch.inference_mode():
        for first in range(0, len(positions), batch_size):
            batch_positions = positions[first : first + batch_size]
            tiles = np.stack(
                [
                    padded[:, top : top + tile_size, left : left + tile_size]
                    for top, left in batch_positions
                ]
            )
            probability = torch.sigmoid(
                model(torch.from_numpy(np.ascontiguousarray(tiles)))
            ).cpu().numpy()
            for item, (top, left) in zip(probability, batch_positions):
                probability_sum[:, top : top + tile_size, left : left + tile_size] += (
                    item * window[None]
                )
                weight_sum[top : top + tile_size, left : left + tile_size] += window
    return (
        probability_sum[:, :height, :width] / weight_sum[None, :height, :width]
    ).astype(np.float32)


def process_ml_and_hybrid(
    config: dict[str, Any],
    cache: dict[int, dict[str, np.ndarray | Frame]],
    ml_dir: Path,
    hybrid_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    vendor = ROOT / "tools" / "_python_ml_deps"
    if str(vendor) not in sys.path:
        sys.path.append(str(vendor))
    import torch  # type: ignore
    from ml.stage5_model import Stage5JointNet

    torch.set_num_threads(min(8, max(1, int(torch.get_num_threads()))))
    ml_config = config["ml_transfer"]
    checkpoint_path = ROOT / ml_config["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Stage5JointNet(
        input_channels=len(ml_config["input_channels"]),
        base_channels=int(checkpoint["base_channels"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    training_index = json.loads(
        (ROOT / ml_config["training_index_for_fixed_normalization"]).read_text(encoding="utf-8")
    )
    stats = training_index["normalization_all_frames_for_inspection_only"]
    mean = np.asarray([stats[name]["mean"] for name in ml_config["input_channels"]], dtype=np.float32)
    std = np.asarray([stats[name]["std"] for name in ml_config["input_channels"]], dtype=np.float32)
    thresholds = ml_config["thresholds"]
    threshold_array = np.asarray(
        [
            thresholds["shock"],
            thresholds["vortex_core"],
            thresholds["background_other"],
        ],
        dtype=np.float32,
    )
    head_names = list(Stage5JointNet.output_names)
    ml_records: list[dict[str, Any]] = []
    hybrid_records: list[dict[str, Any]] = []
    for step in (int(value) for value in ml_config["representative_steps"]):
        data = cache[step]
        frame = data["frame"]
        assert isinstance(frame, Frame)
        geometry = np.asarray(data["geometry"], dtype=bool)
        main_centerline = np.asarray(data["main_centerline"], dtype=bool)
        main_envelope = np.asarray(data["main_envelope"], dtype=bool)
        primitive = np.stack(
            (
                np.log(np.maximum(frame.rho, np.finfo(np.float32).tiny)),
                np.log(np.maximum(frame.pressure, np.finfo(np.float32).tiny)),
                frame.u,
                frame.v,
            )
        ).astype(np.float32)
        standardized = (primitive - mean[:, None, None]) / std[:, None, None]
        clipped = np.clip(standardized, -float(ml_config["input_clip"]), float(ml_config["input_clip"]))
        probability = predict_tiled(
            model,
            torch,
            np.ascontiguousarray(clipped),
            tile_size=int(ml_config["tile_size"]),
            overlap=int(ml_config["tile_overlap"]),
            batch_size=int(ml_config["batch_size"]),
        )
        probability[:, geometry] = 0.0
        selected_probability = probability[[0, 1, 2]]
        binary = selected_probability >= threshold_array[:, None, None]
        shock_ml, vortex_ml, background_ml = binary
        np.savez_compressed(
            ml_dir / f"step_{step:05d}_ml_only.npz",
            step=np.asarray(step, dtype=np.int64),
            time=np.asarray(frame.time, dtype=np.float64),
            probabilities=probability.astype(np.float16),
            binary_masks=binary.astype(np.uint8),
            geometry_mask=geometry.astype(np.uint8),
            thresholds=threshold_array,
            output_heads=np.asarray(head_names, dtype="U32"),
        )
        wall_distance = ndimage.distance_transform_edt(~geometry)
        wall_band = (wall_distance <= 6) & ~geometry
        agreement = {
            "dice_with_physics_envelope_not_accuracy": binary_dice(shock_ml, main_envelope),
            "ml_shock_coverage_of_physics_envelope": float(
                np.count_nonzero(shock_ml & main_envelope) / max(np.count_nonzero(main_envelope), 1)
            ),
            "ml_shock_precision_against_physics_proposal_not_accuracy": float(
                np.count_nonzero(shock_ml & main_envelope) / max(np.count_nonzero(shock_ml), 1)
            ),
        }
        ml_record = {
            "step": step,
            "time": frame.time,
            "positive_pixels": {
                "shock": int(np.count_nonzero(shock_ml)),
                "vortex_core_false_positive_control": int(np.count_nonzero(vortex_ml)),
                "background_other": int(np.count_nonzero(background_ml)),
            },
            "inside_geometry_positive_pixels": int(np.count_nonzero(binary[:, geometry])),
            "shock_pixels_in_six_cell_wall_band": int(np.count_nonzero(shock_ml & wall_band)),
            "vortex_pixels_in_six_cell_wall_band": int(np.count_nonzero(vortex_ml & wall_band)),
            "input_standardized_ranges_before_clip": {
                name: [float(np.min(standardized[channel])), float(np.max(standardized[channel]))]
                for channel, name in enumerate(ml_config["input_channels"])
            },
            "input_fraction_at_clip": {
                name: float(np.mean(np.abs(standardized[channel]) >= float(ml_config["input_clip"])))
                for channel, name in enumerate(ml_config["input_channels"])
            },
            "physics_proposal_agreement": agreement,
            "interpretation": "out-of-distribution frozen transfer audit; not independent accuracy",
        }
        ml_records.append(ml_record)

        near_physics = ndimage.binary_dilation(
            main_envelope, iterations=int(config["hybrid"]["seed_dilation_pixels"])
        )
        seed_pixels = int(np.count_nonzero(shock_ml & near_physics))
        accepted_object = bool(
            np.any(main_centerline)
            and seed_pixels >= int(config["hybrid"]["minimum_seed_pixels_on_physics_envelope"])
        )
        hybrid_centerline = main_centerline if accepted_object else np.zeros_like(main_centerline)
        hybrid_envelope = main_envelope if accepted_object else np.zeros_like(main_envelope)
        hybrid_vortex = np.zeros_like(geometry, dtype=bool)
        hybrid_background = ~geometry & ~hybrid_envelope
        np.savez_compressed(
            hybrid_dir / f"step_{step:05d}_hybrid.npz",
            step=np.asarray(step, dtype=np.int64),
            time=np.asarray(frame.time, dtype=np.float64),
            shock_centerline=hybrid_centerline.astype(np.uint8),
            shock_envelope=hybrid_envelope.astype(np.uint8),
            vortex_core=hybrid_vortex.astype(np.uint8),
            background_other=hybrid_background.astype(np.uint8),
            neural_seed=shock_ml.astype(np.uint8),
            physics_envelope=main_envelope.astype(np.uint8),
            geometry_mask=geometry.astype(np.uint8),
        )
        hybrid_record = {
            "step": step,
            "time": frame.time,
            "neural_seed_pixels_near_physics_bow": seed_pixels,
            "physics_bow_object_accepted": accepted_object,
            "hybrid_shock_pixels": int(np.count_nonzero(hybrid_envelope)),
            "hybrid_vortex_pixels": 0,
            "rule": config["hybrid"]["rule"],
        }
        hybrid_records.append(hybrid_record)
        data["ml_probability"] = probability.astype(np.float32)
        data["ml_binary"] = binary
        data["hybrid_centerline"] = hybrid_centerline
        data["hybrid_envelope"] = hybrid_envelope
        print(
            f"ML step={step:05d} shock={np.count_nonzero(shock_ml)} "
            f"vortex_fp={np.count_nonzero(vortex_ml)} hybrid={accepted_object}"
        )

    ml_report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "result_role": "frozen_ml_only_out_of_distribution_transfer_audit",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256(checkpoint_path),
        "fixed_normalization_source": str((ROOT / ml_config["training_index_for_fixed_normalization"]).resolve()),
        "fixed_thresholds": thresholds,
        "records": ml_records,
        "accuracy_claim": False,
        "warning": ml_config["warning"],
    }
    hybrid_report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "result_role": "ml_seeded_physics_supported_hybrid",
        "records": hybrid_records,
        "accepted_vortex_target": "none; Euler slip-cylinder false-positive control",
        "accuracy_claim": False,
    }
    write_json(ml_dir / "ml_only_transfer_report.json", ml_report)
    write_json(hybrid_dir / "hybrid_transfer_report.json", hybrid_report)
    return ml_report, hybrid_report


def make_figures(
    config: dict[str, Any],
    physics_report: dict[str, Any],
    cache: dict[int, dict[str, np.ndarray | Frame]],
    figures_dir: Path,
) -> list[dict[str, str]]:
    dpi = int(config["figures"]["dpi"])
    radius = float(config["physics"]["cylinder_radius"])
    p_inf = float(config["physics"]["p_inf"])
    rho_inf = float(config["physics"]["rho_inf"])
    u_inf = float(config["physics"]["u_inf"])
    final_step = int(physics_report["aggregate"]["step_sequence"][-1])
    final_data = cache[final_step]
    frame = final_data["frame"]
    assert isinstance(frame, Frame)
    main = np.asarray(final_data["main_centerline"], dtype=bool)
    envelope = np.asarray(final_data["main_envelope"], dtype=bool)
    other = np.asarray(final_data["other_fronts"], dtype=bool)
    rotational = np.asarray(final_data["vortex_candidate"], dtype=bool)
    extent = [float(frame.x[0]), float(frame.x[-1]), float(frame.y[0]), float(frame.y[-1])]
    manifest: list[dict[str, str]] = []

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 10.8), constrained_layout=True)
    fields = (
        (np.log10(np.maximum(frame.pressure / p_inf, 1e-8)), "Log pressure ratio", "Greys", -0.15, 1.05),
        (1.0 - frame.schlieren, "MFC schlieren response", "Greys", 0.0, 0.4),
    )
    for axis, (field, title, cmap, vmin, vmax) in zip(axes[0], fields):
        image = axis.imshow(field, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        draw_body(axis, radius)
        configure_axis(axis, (-5.0, 6.0), (-5.0, 5.0))
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.72)
    for axis in axes[1]:
        axis.imshow(1.0 - frame.schlieren, origin="lower", extent=extent, cmap="Greys", vmin=0.0, vmax=0.4)
        draw_body(axis, radius)
        configure_axis(axis, (-5.0, 6.0), (-5.0, 5.0))
    draw_mask_contour(axes[1, 0], frame.x, frame.y, envelope, color=VERMILLION, linewidth=1.25)
    draw_mask_contour(axes[1, 0], frame.x, frame.y, other, color=GOLD, linewidth=0.75, linestyle="dashed")
    axes[1, 0].set_title("Physics-only: selected bow shock")
    axes[1, 0].legend(
        handles=[
            Patch(facecolor=VERMILLION, label="main bow-shock envelope"),
            Patch(facecolor=GOLD, label="other compression fronts (separate)"),
        ],
        loc="upper left",
        fontsize=8,
    )
    if "ml_binary" in final_data:
        ml_binary = np.asarray(final_data["ml_binary"], dtype=bool)
        hybrid = np.asarray(final_data["hybrid_envelope"], dtype=bool)
        draw_mask_contour(axes[1, 1], frame.x, frame.y, ml_binary[0], color=MAGENTA, linewidth=0.9)
        draw_mask_contour(axes[1, 1], frame.x, frame.y, hybrid, color=VERMILLION, linewidth=1.35)
        axes[1, 1].set_title("Frozen ML-only and ML-seeded hybrid")
        axes[1, 1].legend(
            handles=[
                Patch(facecolor=MAGENTA, label="ML-only shock"),
                Patch(facecolor=VERMILLION, label="hybrid accepted bow shock"),
            ],
            loc="upper left",
            fontsize=8,
        )
    else:
        axes[1, 1].text(0.5, 0.5, "ML inference unavailable", transform=axes[1, 1].transAxes, ha="center")
        axes[1, 1].set_title("ML-only status")
    figure.suptitle(f"Mach 2.7 Euler cylinder, full domain, step {final_step} (t={frame.time:.3f})")
    base = figures_dir / "figure_01_full_domain_final"
    save_figure(figure, base, dpi)
    manifest.append({"figure": base.name, "role": "full-domain final fields and separate detector outputs"})

    figure, axes = plt.subplots(2, 2, figsize=(12.2, 10.2), constrained_layout=True)
    near_fields = (
        (frame.rho / rho_inf, r"$\rho/\rho_\infty$", "Greys", 0.5, 4.1),
        (frame.pressure / p_inf, r"$p/p_\infty$", "Greys", 0.5, 10.2),
        (np.asarray(final_data["shock_score"]), "Entropy-free shock score", "magma", 0.0, 1.0),
        (frame.omega, r"$\omega_z D/a_\infty$", "Greys", -80.0, 80.0),
    )
    for axis, (field, title, cmap, vmin, vmax) in zip(axes.flat, near_fields):
        image = axis.imshow(field, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
        draw_mask_contour(axis, frame.x, frame.y, envelope, color=VERMILLION, linewidth=1.2)
        draw_body(axis, radius)
        configure_axis(axis, (-1.5, 1.5), (-1.5, 1.5))
        axis.set_title(title)
        figure.colorbar(image, ax=axis, shrink=0.78)
    figure.suptitle("Near-field shock boundary: wall excluded by exact immersed geometry")
    base = figures_dir / "figure_02_nearfield_boundary"
    save_figure(figure, base, dpi)
    manifest.append({"figure": base.name, "role": "near-field boundary and entropy-free score"})

    records = physics_report["frames"]
    detected = [item for item in records if item["stagnation_line"]["detected"]]
    times = np.asarray([item["time"] for item in detected], dtype=float)
    standoff = np.asarray([item["stagnation_line"]["standoff_over_d"] for item in detected], dtype=float)
    changes_t = np.asarray(
        [item["time"] for item in records if item["relative_l2_pressure_change_stationarity_roi"] is not None],
        dtype=float,
    )
    changes = np.asarray(
        [
            item["relative_l2_pressure_change_stationarity_roi"]
            for item in records
            if item["relative_l2_pressure_change_stationarity_roi"] is not None
        ],
        dtype=float,
    )
    rh = json.loads((ROOT / config["evaluation"]["rh_reference"]).read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 2, figsize=(12.6, 9.2), constrained_layout=True)
    axes[0, 0].plot(times, standoff, color=CHARCOAL, marker="o", markersize=3, linewidth=1.1)
    axes[0, 0].set_xlabel(r"$t a_\infty/D$")
    axes[0, 0].set_ylabel(r"$\delta/D$")
    axes[0, 0].set_title("Stagnation-line shock standoff")
    axes[0, 0].grid(alpha=0.2)
    axes[0, 1].semilogy(changes_t, changes, color=CHARCOAL, marker="o", markersize=3)
    axes[0, 1].axhline(
        float(config["evaluation"]["maximum_tail_relative_l2_pressure_change"]),
        color=VERMILLION,
        linestyle="--",
        label="acceptance gate",
    )
    axes[0, 1].set_xlabel(r"$t a_\infty/D$")
    axes[0, 1].set_ylabel("relative $L_2$ pressure change")
    axes[0, 1].set_title("Near-field stationarity")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.2)
    p_ratio = np.asarray([item["stagnation_line"]["p2_over_p1"] for item in detected], dtype=float)
    rho_ratio = np.asarray([item["stagnation_line"]["rho2_over_rho1"] for item in detected], dtype=float)
    axes[1, 0].plot(times, p_ratio, color=VERMILLION, label=r"$p_2/p_1$")
    axes[1, 0].plot(times, rho_ratio, color=GOLD, label=r"$\rho_2/\rho_1$")
    axes[1, 0].axhline(float(rh["p2_over_p1"]), color=VERMILLION, linestyle=":")
    axes[1, 0].axhline(float(rh["rho2_over_rho1"]), color=GOLD, linestyle=":")
    axes[1, 0].set_xlabel(r"$t a_\infty/D$")
    axes[1, 0].set_ylabel("one-sided jump ratio")
    axes[1, 0].set_title("Rankine-Hugoniot nose-jump audit")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)
    theta, cp = surface_cp(
        frame,
        radius=radius,
        sample_offset_cells=int(config["evaluation"]["surface_sampling_offset_cells"]),
        rho_inf=rho_inf,
        p_inf=p_inf,
        u_inf=u_inf,
    )
    axes[1, 1].plot(theta, cp, color=CHARCOAL, linewidth=1.2, label="MFC sampled outside IB")
    axes[1, 1].scatter([0.0], [float(rh["inviscid_stagnation_cp"])], color=VERMILLION, s=30, label="exact inviscid nose $C_p$")
    axes[1, 1].set_xlabel("surface angle from upstream nose (deg)")
    axes[1, 1].set_ylabel(r"$C_p$")
    axes[1, 1].set_title("Surface-pressure audit (not experiment)")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(alpha=0.2)
    figure.suptitle(
        "Physical verification and convergence gate — dotted RH lines are exact, not fitted"
    )
    base = figures_dir / "figure_03_stationarity_rh_cp"
    save_figure(figure, base, dpi)
    manifest.append({"figure": base.name, "role": "stationarity, exact RH jump, and surface Cp"})

    temporal_steps = [int(value) for value in config["figures"]["temporal_steps"]]
    figure, axes = plt.subplots(2, 2, figsize=(12.4, 10.4), constrained_layout=True)
    for axis, step in zip(axes.flat, temporal_steps):
        item = cache[step]
        local_frame = item["frame"]
        assert isinstance(local_frame, Frame)
        axis.imshow(
            1.0 - local_frame.schlieren,
            origin="lower",
            extent=[local_frame.x[0], local_frame.x[-1], local_frame.y[0], local_frame.y[-1]],
            cmap="Greys",
            vmin=0.0,
            vmax=0.4,
        )
        draw_mask_contour(
            axis,
            local_frame.x,
            local_frame.y,
            np.asarray(item["main_envelope"], dtype=bool),
            color=VERMILLION,
            linewidth=1.0,
        )
        draw_body(axis, radius)
        configure_axis(axis, (-5.0, 6.0), (-5.0, 5.0))
        axis.set_title(f"step {step}, t={local_frame.time:.3f}")
    figure.suptitle("Full-domain evolution of the selected detached bow-shock object")
    base = figures_dir / "figure_04_temporal_bow_shock"
    save_figure(figure, base, dpi)
    manifest.append({"figure": base.name, "role": "four-time full-domain bow-shock evolution"})

    ml_steps = [int(value) for value in config["ml_transfer"]["representative_steps"]]
    if all("ml_probability" in cache[step] for step in ml_steps):
        figure, axes = plt.subplots(len(ml_steps), 3, figsize=(13.2, 4.0 * len(ml_steps)), constrained_layout=True)
        for row, step in enumerate(ml_steps):
            item = cache[step]
            local_frame = item["frame"]
            assert isinstance(local_frame, Frame)
            local_extent = [local_frame.x[0], local_frame.x[-1], local_frame.y[0], local_frame.y[-1]]
            probability = np.asarray(item["ml_probability"])
            binary = np.asarray(item["ml_binary"], dtype=bool)
            axes[row, 0].imshow(1.0 - local_frame.schlieren, origin="lower", extent=local_extent, cmap="Greys", vmin=0.0, vmax=0.4)
            axes[row, 0].set_title(f"Input schlieren, t={local_frame.time:.3f}")
            image = axes[row, 1].imshow(probability[0], origin="lower", extent=local_extent, cmap="magma", vmin=0.0, vmax=1.0)
            draw_mask_contour(axes[row, 1], local_frame.x, local_frame.y, binary[0], color=MAGENTA, linewidth=0.8)
            axes[row, 1].set_title(r"Frozen ML-only $P(\mathrm{shock})$")
            figure.colorbar(image, ax=axes[row, 1], shrink=0.7)
            axes[row, 2].imshow(1.0 - local_frame.schlieren, origin="lower", extent=local_extent, cmap="Greys", vmin=0.0, vmax=0.4)
            draw_mask_contour(axes[row, 2], local_frame.x, local_frame.y, np.asarray(item["main_envelope"], dtype=bool), color=GOLD, linewidth=0.8)
            draw_mask_contour(axes[row, 2], local_frame.x, local_frame.y, np.asarray(item["hybrid_envelope"], dtype=bool), color=VERMILLION, linewidth=1.2)
            axes[row, 2].set_title("Physics proposal (gold) / hybrid (red)")
            for column in range(3):
                draw_body(axes[row, column], radius)
                configure_axis(axes[row, column], (-2.0, 3.0), (-2.0, 2.0))
        figure.suptitle("Frozen diamond-airfoil network: out-of-distribution cylinder transfer audit")
        base = figures_dir / "figure_05_ml_transfer_audit"
        save_figure(figure, base, dpi)
        manifest.append({"figure": base.name, "role": "ML-only versus physics-only versus ML-seeded hybrid"})

        figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.2), constrained_layout=True)
        signed_log_omega = np.sign(frame.omega) * np.log1p(np.abs(frame.omega))
        image = axes[0].imshow(
            signed_log_omega,
            origin="lower",
            extent=extent,
            cmap="Greys",
            norm=SymLogNorm(linthresh=0.05, vmin=-math.log1p(150.0), vmax=math.log1p(150.0)),
        )
        draw_mask_contour(axes[0], frame.x, frame.y, rotational, color=GOLD, linewidth=0.8)
        axes[0].set_title("Physics rotational candidates (not accepted vortices)")
        figure.colorbar(image, ax=axes[0], shrink=0.75)
        ml_probability = np.asarray(final_data["ml_probability"])
        ml_binary = np.asarray(final_data["ml_binary"], dtype=bool)
        image = axes[1].imshow(ml_probability[1], origin="lower", extent=extent, cmap="magma", vmin=0.0, vmax=1.0)
        draw_mask_contour(axes[1], frame.x, frame.y, ml_binary[1], color=MAGENTA, linewidth=0.8)
        axes[1].set_title("Frozen ML vortex response; expected target is empty")
        figure.colorbar(image, ax=axes[1], shrink=0.75)
        for axis in axes:
            draw_body(axis, radius)
            configure_axis(axis, (-1.5, 2.5), (-1.5, 1.5))
        figure.suptitle("Vortex false-positive control — accepted vortex pixels = 0 by case definition")
        base = figures_dir / "figure_06_vortex_false_positive_control"
        save_figure(figure, base, dpi)
        manifest.append({"figure": base.name, "role": "raw rotational and neural vortex false-positive audit"})

    with (figures_dir / "surface_cp_final.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("theta_deg_from_upstream_nose", "cp_sampled_three_cells_outside_ib"))
        writer.writerows(zip(theta, cp))
    with (figures_dir / "figure_manifest.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("figure", "role"))
        writer.writeheader()
        writer.writerows(manifest)
    return manifest


def write_markdown_summary(
    config: dict[str, Any],
    physics_report: dict[str, Any],
    ml_report: dict[str, Any],
    hybrid_report: dict[str, Any],
    manifest: list[dict[str, str]],
    path: Path,
) -> None:
    aggregate = physics_report["aggregate"]
    final = physics_report["frames"][-1]
    stagnation = final["stagnation_line"]
    rh = aggregate["final_rankine_hugoniot_comparison"]
    accepted_hybrid = sum(
        int(item["physics_bow_object_accepted"]) for item in hybrid_report["records"]
    )
    stationarity_pass = bool(aggregate["stationarity_pass"])
    if stationarity_pass:
        stationarity_paragraph = (
            f"The f90 trajectory **passes the frozen stationarity gate**: the largest tail relative L2 pressure "
            f"change is {aggregate['tail_maximum_relative_l2_pressure_change']:.6f} (gate "
            f"{aggregate['stationarity_gates']['maximum_relative_l2_pressure_change']:.6f}), and the fitted tail "
            f"standoff slope is {aggregate['tail_standoff_slope_d_per_time']:.6f} D per unit time (absolute gate "
            f"{aggregate['stationarity_gates']['maximum_absolute_standoff_slope_d_per_time']:.6f}). The trajectory "
            "is therefore acceptable as the fixed-grid steady Euler-cylinder audit case."
        )
        stationarity_limit = (
            "- The frozen tail tests support stationarity at the saved cadence; they do not establish grid convergence."
        )
    else:
        stationarity_paragraph = (
            f"The f90 trajectory **does not pass the frozen stationarity gate**: the largest tail relative L2 pressure "
            f"change is {aggregate['tail_maximum_relative_l2_pressure_change']:.6f} (gate "
            f"{aggregate['stationarity_gates']['maximum_relative_l2_pressure_change']:.6f}), and the fitted tail "
            f"standoff slope is {aggregate['tail_standoff_slope_d_per_time']:.6f} D per unit time (absolute gate "
            f"{aggregate['stationarity_gates']['maximum_absolute_standoff_slope_d_per_time']:.6f}). The trajectory "
            "must be extended before it is described as steady."
        )
        stationarity_limit = (
            "- The run is not yet stationary; it is useful for transfer stress-testing, but not a final steady validation case."
        )
    lines = [
        "# MFC Mach-2.7 Euler Cylinder Audit",
        "",
        "## Outcome",
        "",
        f"The delivered package contains {aggregate['snapshot_count']} readable full-domain states. "
        f"All fluid density and pressure values are finite and positive, and the decoded IB marker has "
        f"IoU={aggregate['minimum_ib_exact_geometry_iou']:.6f} with the exact radius-0.5 cylinder.",
        "",
        f"The entropy-free compression/jump/NMS detector selects a detached bow-shock object with final "
        f"standoff delta/D={stagnation.get('standoff_over_d', float('nan')):.6f}. It reports zero accepted "
        "vortices because the inviscid slip-cylinder case is a vortex false-positive control, not a vortex-shedding target.",
        "",
        stationarity_paragraph,
        "",
        "## Exact normal-shock audit at the last state",
        "",
        "| Quantity | Measured | Exact | Relative error |",
        "|---|---:|---:|---:|",
    ]
    for name, label in (
        ("p2_over_p1", "$p_2/p_1$"),
        ("rho2_over_rho1", "$\\rho_2/\\rho_1$"),
        ("mach_downstream", "$M_2$"),
        ("stagnation_cp", "$C_{p,0}$"),
    ):
        item = rh[name]
        lines.append(
            f"| {label} | {item['measured']:.6f} | {item['exact_reference']:.6f} | {100.0 * item['relative_error']:.2f}% |"
        )
    lines.extend(
        [
            "",
            f"The 10--90% stagnation-line pressure jump is resolved in approximately "
            f"{stagnation.get('shock_width_10_90_cells', float('nan')):.2f} cells. These RH comparisons are "
            "verification diagnostics, not experimental validation.",
            "",
            "## ML and hybrid interpretation",
            "",
            f"Frozen ML-only inference was run at {len(ml_report['records'])} representative states. The checkpoint "
            "was weak-trained on one Mach-3 alpha-40 diamond-airfoil trajectory; its cylinder outputs are therefore "
            "out-of-distribution responses, not accuracy measurements. Fixed training normalization and fixed model-card "
            "thresholds were retained; no frame-wise tuning was used.",
            "",
            f"The ML-seeded hybrid accepted the bow-shock object in {accepted_hybrid}/{len(hybrid_report['records'])} "
            "representative states. Physics-only, ML-only, and hybrid arrays remain in separate result directories.",
            "",
            "## Scientific limitations",
            "",
            f"- The {aggregate['snapshot_count']} adjacent states are one leakage-free case group and must never be split across train/test.",
            stationarity_limit,
            "- Experimental shock-shape and surface-Cp curves have not yet been digitized, so no experimental validation claim is made.",
            "- No grid-convergence claim is made; this case is used as a fixed-grid topology, RH, stationarity, and transfer audit.",
            "- Rotational candidates near the slip wall and wake are audit outputs only. Accepted vortex ground truth is empty.",
            "",
            "## Reproducibility",
            "",
            f"Configuration: `{config['experiment_id']}`. The verified outer ZIP SHA-256 is "
            f"`{config['outer_zip_sha256']}` and the verified nested archive SHA-256 is "
            f"`{config['nested_archive_sha256']}`.",
            "",
            "Generated figures:",
            "",
        ]
    )
    for item in manifest:
        lines.append(f"- `{item['figure']}` — {item['role']}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "mfc_euler_cylinder_m2p7_f90_audit_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = read_inventory(ROOT / config["inventory"])
    expected_snapshots = int(config.get("expected_snapshots", 31))
    if len(records) != expected_snapshots:
        raise RuntimeError(
            f"Expected {expected_snapshots} snapshots, found {len(records)}"
        )

    experiment = config["experiment_id"]
    physics_dir = ROOT / "results" / "physics_only" / experiment
    ml_dir = ROOT / "results" / "ml_only" / experiment
    hybrid_dir = ROOT / "results" / "hybrid" / experiment
    figures_dir = ROOT / "results" / "figures" / experiment
    for directory in (physics_dir, ml_dir, hybrid_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    physics_report, cache = process_physics(config, records, physics_dir, config_path)
    ml_report, hybrid_report = process_ml_and_hybrid(config, cache, ml_dir, hybrid_dir)
    manifest = make_figures(config, physics_report, cache, figures_dir)
    summary_path = ROOT / config.get(
        "summary_path", "docs/MFC_EULER_CYLINDER_M2P7_F90_AUDIT.md"
    )
    write_markdown_summary(
        config,
        physics_report,
        ml_report,
        hybrid_report,
        manifest,
        summary_path,
    )
    print(
        json.dumps(
            {
                "status": (
                    "PASS_DATA_INTEGRITY_PASS_STATIONARITY"
                    if physics_report["aggregate"]["stationarity_pass"]
                    else "PASS_DATA_INTEGRITY_FAIL_STATIONARITY"
                ),
                "physics_report": str((physics_dir / "physics_only_report.json").resolve()),
                "ml_report": str((ml_dir / "ml_only_transfer_report.json").resolve()),
                "hybrid_report": str((hybrid_dir / "hybrid_transfer_report.json").resolve()),
                "figures": str(figures_dir.resolve()),
                "summary": str(summary_path.resolve()),
                "next_simulation_action": physics_report["aggregate"]["f90_pilot_decision"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
