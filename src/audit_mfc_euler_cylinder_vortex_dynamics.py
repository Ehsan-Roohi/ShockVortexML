#!/usr/bin/env python3
"""Raw-field vortex/shedding audit for the Mach-2.7 MFC Euler cylinder.

The learned Stage-6 response is deliberately not an input to this analysis.
Vorticity, swirling strength, Gamma-2 topology, object tracks, probe spectra,
and the limited numerical-repeat evidence are reported separately.  Missing
grid and solver-time-step controls remain explicit missing evidence.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from analyze_mfc_euler_cylinder import (  # noqa: E402
    Frame,
    read_frame,
    read_inventory,
    save_figure,
    write_json,
)
from physics_proposals import derivatives  # noqa: E402


CHARCOAL = "#202020"
VERMILLION = "#D55E00"
GOLD = "#E69F00"
MAGENTA = "#CC79A7"
MID_GRAY = "#777777"
LIGHT_GRAY = "#D0D0D0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def offset_slices(
    shape: tuple[int, int], offset_y: int, offset_x: int
) -> tuple[tuple[slice, slice], tuple[slice, slice]]:
    """Return target/source slices for a non-wrapping integer offset."""

    height, width = shape
    y0 = max(0, -offset_y)
    y1 = min(height, height - offset_y)
    x0 = max(0, -offset_x)
    x1 = min(width, width - offset_x)
    target = (slice(y0, y1), slice(x0, x1))
    source = (
        slice(y0 + offset_y, y1 + offset_y),
        slice(x0 + offset_x, x1 + offset_x),
    )
    return target, source


def gamma2_field(
    u: np.ndarray,
    v: np.ndarray,
    valid: np.ndarray,
    *,
    dx: float,
    dy: float,
    radius_cells: int,
) -> np.ndarray:
    r"""Compute Graftieaux's :math:`\Gamma_2` with a disk neighbourhood.

    The local convection velocity is the valid-neighbour mean.  Geometry is
    excluded from both the local mean and the circulation sum.  A positive
    value has the same sign as positive ``dv/dx-du/dy`` for solid-body
    rotation in the Cartesian convention used by MFC.
    """

    if radius_cells < 1:
        raise ValueError("radius_cells must be positive")
    if not (u.shape == v.shape == valid.shape):
        raise ValueError("u, v, and valid must have identical shapes")
    offsets = [
        (oy, ox)
        for oy in range(-radius_cells, radius_cells + 1)
        for ox in range(-radius_cells, radius_cells + 1)
        if oy * oy + ox * ox <= radius_cells * radius_cells
    ]
    kernel = np.zeros((2 * radius_cells + 1, 2 * radius_cells + 1), dtype=float)
    for oy, ox in offsets:
        kernel[oy + radius_cells, ox + radius_cells] = 1.0
    valid_float = valid.astype(float)
    count = ndimage.convolve(valid_float, kernel, mode="constant", cval=0.0)
    mean_u = np.divide(
        ndimage.convolve(np.where(valid, u, 0.0), kernel, mode="constant", cval=0.0),
        count,
        out=np.zeros_like(u, dtype=float),
        where=count > 0,
    )
    mean_v = np.divide(
        ndimage.convolve(np.where(valid, v, 0.0), kernel, mode="constant", cval=0.0),
        count,
        out=np.zeros_like(v, dtype=float),
        where=count > 0,
    )
    accumulator = np.zeros_like(u, dtype=float)
    contributors = np.zeros_like(u, dtype=np.int16)
    velocity_scale = max(float(np.nanmax(np.hypot(u[valid], v[valid]))), 1.0)
    epsilon = np.finfo(float).eps * velocity_scale * 64.0
    for oy, ox in offsets:
        if oy == 0 and ox == 0:
            continue
        target, source = offset_slices(u.shape, oy, ox)
        allowed = valid[target] & valid[source]
        du = u[source] - mean_u[target]
        dv = v[source] - mean_v[target]
        speed = np.hypot(du, dv)
        radius = math.hypot(ox * dx, oy * dy)
        usable = allowed & (speed > epsilon)
        term = np.zeros_like(speed)
        term[usable] = (
            (ox * dx) * dv[usable] - (oy * dy) * du[usable]
        ) / (radius * speed[usable])
        accumulator[target] += term
        contributors[target] += usable.astype(np.int16)
    minimum_contributors = max(4, int(0.5 * (len(offsets) - 1)))
    result = np.divide(
        accumulator,
        contributors,
        out=np.zeros_like(accumulator),
        where=contributors >= minimum_contributors,
    )
    result[~valid] = 0.0
    return np.clip(result, -1.0, 1.0).astype(np.float32)


def inertia_aspect_ratio(
    rows: np.ndarray,
    columns: np.ndarray,
    weights: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> float:
    if len(rows) < 3:
        return float("inf")
    x = columns.astype(float) * dx
    y = rows.astype(float) * dy
    total = float(np.sum(weights))
    if total <= 0.0:
        weights = np.ones_like(x)
        total = float(len(x))
    x -= float(np.sum(weights * x) / total)
    y -= float(np.sum(weights * y) / total)
    covariance = np.asarray(
        [
            [np.sum(weights * x * x), np.sum(weights * x * y)],
            [np.sum(weights * x * y), np.sum(weights * y * y)],
        ],
        dtype=float,
    ) / total
    eigenvalues = np.linalg.eigvalsh(covariance)
    if eigenvalues[0] <= np.finfo(float).eps:
        return float("inf")
    return float(math.sqrt(eigenvalues[1] / eigenvalues[0]))


def detect_vortex_objects(
    frame: Frame,
    gamma2: np.ndarray,
    *,
    shock_mask: np.ndarray,
    geometry: np.ndarray,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, Any]], dict[str, np.ndarray]]:
    """Extract compact rotational objects from fixed nondimensional tests."""

    dx = float(np.mean(np.diff(frame.x)))
    dy = float(np.mean(np.diff(frame.y)))
    physical = config["physics"]
    diagnostic = config["vortex_diagnostics"]
    exclusion = config["shock_wall_exclusion"]
    u_scale = float(physical["u_inf"]) / float(physical["diameter"])
    flow = derivatives(frame.rho, frame.pressure, frame.u, frame.v, dx=dx, dy=dy)
    wall_distance = ndimage.distance_transform_edt(~geometry, sampling=(dy, dx))
    expanded_shock = ndimage.binary_dilation(
        shock_mask,
        iterations=int(exclusion["shock_dilation_cells"]),
    )
    shock_distance = (
        ndimage.distance_transform_edt(~shock_mask, sampling=(dy, dx))
        if np.any(shock_mask)
        else None
    )
    yy, xx = np.meshgrid(frame.y, frame.x, indexing="ij")
    bounds = [float(value) for value in diagnostic["wake_bounds_D"]]
    wake = (
        (xx >= bounds[0])
        & (xx <= bounds[1])
        & (yy >= bounds[2])
        & (yy <= bounds[3])
    )
    permitted = (
        wake
        & ~geometry
        & ~expanded_shock
        & (wall_distance >= float(exclusion["wall_band_D"]))
    )
    lambda_seed = float(diagnostic["lambda_ci_seed_over_uinf_D"]) * u_scale
    lambda_support = float(diagnostic["lambda_ci_support_over_uinf_D"]) * u_scale
    omega_seed = float(diagnostic["abs_vorticity_seed_over_uinf_D"]) * u_scale
    omega_support = float(diagnostic["abs_vorticity_support_over_uinf_D"]) * u_scale
    q_scale = u_scale * u_scale
    q_seed = float(diagnostic["q_seed_over_uinf_D_squared"]) * q_scale
    q_support = float(diagnostic["q_support_over_uinf_D_squared"]) * q_scale
    seed = (
        permitted
        & (np.abs(gamma2) >= float(diagnostic["gamma2_seed"]))
        & (flow.lambda_ci >= lambda_seed)
        & (np.abs(flow.vorticity) >= omega_seed)
        & (flow.q_deviatoric >= q_seed)
    )
    support = (
        permitted
        & (np.abs(gamma2) >= float(diagnostic["gamma2_support"]))
        & (flow.lambda_ci >= lambda_support)
        & (np.abs(flow.vorticity) >= omega_support)
        & (flow.q_deviatoric >= q_support)
    )
    grown = ndimage.binary_propagation(seed, mask=support)
    grown = ndimage.binary_closing(grown, structure=np.ones((3, 3))) & permitted
    labels, count = ndimage.label(grown, structure=np.ones((3, 3), dtype=np.uint8))
    accepted = np.zeros_like(labels, dtype=np.uint16)
    objects: list[dict[str, Any]] = []
    next_id = 1
    minimum_pixels = int(diagnostic["minimum_pixels"])
    for component_id in range(1, count + 1):
        component = labels == component_id
        rows, columns = np.where(component)
        pixels = int(len(rows))
        if pixels < minimum_pixels:
            continue
        radius = float(math.sqrt(pixels * dx * dy / math.pi))
        if radius < float(diagnostic["minimum_equivalent_radius_D"]):
            continue
        weights = np.maximum(flow.lambda_ci[component], np.finfo(float).eps)
        aspect_ratio = inertia_aspect_ratio(rows, columns, weights, dx=dx, dy=dy)
        if aspect_ratio > float(diagnostic["maximum_aspect_ratio"]):
            continue
        mean_omega = float(np.average(flow.vorticity[component], weights=weights))
        sign = 1 if mean_omega >= 0.0 else -1
        sign_fraction = float(np.mean(np.sign(flow.vorticity[component]) == sign))
        if sign_fraction < float(diagnostic["minimum_vorticity_sign_fraction"]):
            continue
        weight_sum = float(np.sum(weights))
        center_x = float(np.sum(frame.x[columns] * weights) / weight_sum)
        center_y = float(np.sum(frame.y[rows] * weights) / weight_sum)
        accepted[component] = next_id
        objects.append(
            {
                "frame_object_id": next_id,
                "step": int(frame.step),
                "time": float(frame.time),
                "center_x_D": center_x,
                "center_y_D": center_y,
                "equivalent_radius_D": radius,
                "pixels": pixels,
                "vorticity_sign": sign,
                "vorticity_sign_fraction": sign_fraction,
                "circulation_over_ainf_D": float(
                    np.sum(flow.vorticity[component]) * dx * dy
                ),
                "mean_vorticity_D_over_ainf": mean_omega,
                "peak_abs_vorticity_D_over_ainf": float(
                    np.max(np.abs(flow.vorticity[component]))
                ),
                "peak_lambda_ci_D_over_ainf": float(
                    np.max(flow.lambda_ci[component])
                ),
                "mean_abs_gamma2": float(np.mean(np.abs(gamma2[component]))),
                "peak_abs_gamma2": float(np.max(np.abs(gamma2[component]))),
                "aspect_ratio": aspect_ratio,
                "minimum_wall_distance_D": float(np.min(wall_distance[component])),
                "minimum_shock_distance_D": float(
                    np.min(shock_distance[component])
                )
                if shock_distance is not None
                else None,
                "track_id": None,
                "status": "raw_field_topology_candidate",
            }
        )
        next_id += 1
    diagnostics = {
        "vorticity": flow.vorticity.astype(np.float32),
        "lambda_ci": flow.lambda_ci.astype(np.float32),
        "q_deviatoric": flow.q_deviatoric.astype(np.float32),
        "permitted": permitted,
        "seed": seed,
        "support": support,
        "expanded_shock": expanded_shock,
        "wall_distance": wall_distance.astype(np.float32),
    }
    return accepted, objects, diagnostics


def sample_field(frame: Frame, field: np.ndarray, x: float, y: float) -> float:
    column = (x - float(frame.x[0])) / float(frame.x[1] - frame.x[0])
    row = (y - float(frame.y[0])) / float(frame.y[1] - frame.y[0])
    return float(
        ndimage.map_coordinates(
            field, [[row], [column]], order=1, mode="nearest"
        )[0]
    )


def pressure_lift_coefficient(frame: Frame, config: dict[str, Any]) -> float:
    physical = config["physics"]
    radius = float(physical["cylinder_radius"])
    dx = float(np.mean(np.diff(frame.x)))
    sample_radius = radius + int(physical["surface_probe_offset_cells"]) * dx
    theta = np.linspace(-math.pi, math.pi, 721)
    sample_x = sample_radius * np.cos(theta)
    sample_y = sample_radius * np.sin(theta)
    columns = (sample_x - frame.x[0]) / (frame.x[1] - frame.x[0])
    rows = (sample_y - frame.y[0]) / (frame.y[1] - frame.y[0])
    pressure = ndimage.map_coordinates(
        frame.pressure, [rows, columns], order=1, mode="nearest"
    )
    force_y = -float(np.trapezoid(pressure * np.sin(theta) * sample_radius, theta))
    dynamic_pressure = (
        0.5 * float(physical["rho_inf"]) * float(physical["u_inf"]) ** 2
    )
    return force_y / (dynamic_pressure * float(physical["diameter"]))


def frame_signals(
    frame: Frame,
    config: dict[str, Any],
    *,
    objects: list[dict[str, Any]] | None = None,
) -> dict[str, float]:
    spectral = config["spectral"]
    pair_y = float(spectral["pressure_pair_y_D"])
    result = {
        "time": float(frame.time),
        "step": float(frame.step),
        "lift_coefficient": pressure_lift_coefficient(frame, config),
    }
    for x in (float(value) for value in spectral["probe_x_D"]):
        tag = str(x).replace(".", "p")
        result[f"v_x{tag}"] = sample_field(frame, frame.v, x, 0.0)
        result[f"dp_x{tag}"] = sample_field(
            frame, frame.pressure, x, pair_y
        ) - sample_field(frame, frame.pressure, x, -pair_y)
    if objects is not None:
        result["candidate_count"] = float(len(objects))
        result["candidate_positive_circulation"] = float(
            sum(max(0.0, float(item["circulation_over_ainf_D"])) for item in objects)
        )
        result["candidate_negative_circulation"] = float(
            sum(min(0.0, float(item["circulation_over_ainf_D"])) for item in objects)
        )
    return result


def periodogram(
    times: np.ndarray,
    values: np.ndarray,
    *,
    u_inf: float,
    minimum_cycles: float,
    minimum_peak_power_fraction: float,
    reference_scale: float = 1.0,
    minimum_rms_normalized_by_reference_scale: float = 0.0,
    maximum_half_window_rms_ratio: float = float("inf"),
) -> dict[str, Any]:
    if len(times) < 6:
        return {"passed": False, "reason": "fewer_than_six_samples"}
    delta = np.diff(times)
    dt = float(np.median(delta))
    if not np.allclose(delta, dt, rtol=1e-6, atol=1e-10):
        raise ValueError("periodogram requires uniformly sampled data")
    coefficient = np.polyfit(times - times[0], values, 1)
    detrended = values - np.polyval(coefficient, times - times[0])
    rms = float(np.sqrt(np.mean(detrended * detrended)))
    midpoint = len(detrended) // 2
    first_rms = float(np.sqrt(np.mean(detrended[:midpoint] ** 2)))
    second_rms = float(np.sqrt(np.mean(detrended[midpoint:] ** 2)))
    half_rms_ratio = max(first_rms, second_rms) / max(
        min(first_rms, second_rms), np.finfo(float).tiny
    )
    windowed = detrended * np.hanning(len(values))
    frequencies = np.fft.rfftfreq(len(values), dt)
    power = np.abs(np.fft.rfft(windowed)) ** 2
    power[0] = 0.0
    positive_power = float(np.sum(power[1:]))
    if positive_power <= np.finfo(float).tiny:
        normalized = np.zeros_like(power)
        peak_index = 0
    else:
        normalized = power / positive_power
        peak_index = int(np.argmax(power[1:]) + 1)
    frequency = float(frequencies[peak_index]) if peak_index else 0.0
    cycles = frequency * float(times[-1] - times[0])
    peak_fraction = float(normalized[peak_index]) if peak_index else 0.0
    nonzero = power[1:]
    median_power = float(np.median(nonzero)) if len(nonzero) else 0.0
    peak_to_median = (
        float(power[peak_index] / max(median_power, np.finfo(float).tiny))
        if peak_index
        else 0.0
    )
    passed = bool(
        peak_index > 1
        and cycles >= minimum_cycles
        and peak_fraction >= minimum_peak_power_fraction
        and rms / reference_scale >= minimum_rms_normalized_by_reference_scale
        and half_rms_ratio <= maximum_half_window_rms_ratio
    )
    return {
        "passed": passed,
        "samples": int(len(values)),
        "sample_delta_time_D_over_ainf": dt,
        "duration_D_over_ainf": float(times[-1] - times[0]),
        "frequency_resolution_ainf_over_D": float(frequencies[1]),
        "nyquist_frequency_ainf_over_D": float(frequencies[-1]),
        "peak_bin": peak_index,
        "peak_frequency_ainf_over_D": frequency,
        "peak_strouhal_fD_over_uinf": frequency / u_inf,
        "cycles_in_window": cycles,
        "peak_power_fraction": peak_fraction,
        "peak_to_median_power": peak_to_median,
        "detrended_rms": rms,
        "rms_normalized_by_reference_scale": rms / reference_scale,
        "first_half_rms": first_rms,
        "second_half_rms": second_rms,
        "half_window_rms_ratio": half_rms_ratio,
        "linear_drift_per_time": float(coefficient[0]),
        "frequencies_ainf_over_D": frequencies.tolist(),
        "strouhal": (frequencies / u_inf).tolist(),
        "normalized_power": normalized.tolist(),
    }


def track_objects(
    frame_objects: list[list[dict[str, Any]]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    tracking = config["tracking"]
    maximum_gap = int(tracking["maximum_gap_frames"])
    maximum_displacement = float(tracking["maximum_displacement_D_per_snapshot"])
    maximum_upstream = float(tracking["maximum_upstream_displacement_D"])
    maximum_crossstream = float(
        tracking["maximum_crossstream_displacement_D_per_snapshot"]
    )
    maximum_radius_ratio = float(tracking["maximum_equivalent_radius_ratio"])
    tracks: dict[int, list[dict[str, Any]]] = {}
    next_track_id = 1
    for frame_index, objects in enumerate(frame_objects):
        active_ids = [
            track_id
            for track_id, points in tracks.items()
            if frame_index - int(points[-1]["frame_index"]) <= maximum_gap + 1
        ]
        cost = np.full((len(active_ids), len(objects)), np.inf, dtype=float)
        for row, track_id in enumerate(active_ids):
            points = tracks[track_id]
            last = points[-1]
            gap = frame_index - int(last["frame_index"])
            predicted_x = float(last["center_x_D"])
            predicted_y = float(last["center_y_D"])
            if len(points) >= 2:
                previous = points[-2]
                dt_old = float(last["time"] - previous["time"])
                dt_new = float(objects[0]["time"] - last["time"]) if objects else 0.0
                if dt_old > 0.0:
                    predicted_x += (
                        float(last["center_x_D"] - previous["center_x_D"])
                        / dt_old
                        * dt_new
                    )
                    predicted_y += (
                        float(last["center_y_D"] - previous["center_y_D"])
                        / dt_old
                        * dt_new
                    )
            for column, item in enumerate(objects):
                if int(item["vorticity_sign"]) != int(last["vorticity_sign"]):
                    continue
                delta_x = float(item["center_x_D"] - last["center_x_D"])
                delta_y = float(item["center_y_D"] - last["center_y_D"])
                if delta_x < -maximum_upstream * gap:
                    continue
                if abs(delta_y) > maximum_crossstream * gap:
                    continue
                radius_ratio = max(
                    float(item["equivalent_radius_D"]),
                    float(last["equivalent_radius_D"]),
                ) / max(
                    min(
                        float(item["equivalent_radius_D"]),
                        float(last["equivalent_radius_D"]),
                    ),
                    1e-9,
                )
                if radius_ratio > maximum_radius_ratio:
                    continue
                distance = math.hypot(
                    float(item["center_x_D"]) - predicted_x,
                    float(item["center_y_D"]) - predicted_y,
                )
                allowed_distance = maximum_displacement * gap
                if distance > allowed_distance:
                    continue
                size_penalty = abs(
                    math.log(
                        max(float(item["equivalent_radius_D"]), 1e-9)
                        / max(float(last["equivalent_radius_D"]), 1e-9)
                    )
                )
                cost[row, column] = distance / allowed_distance + 0.2 * size_penalty
        assigned_objects: set[int] = set()
        if cost.size:
            finite = np.isfinite(cost)
            working = np.where(finite, cost, 1e6)
            rows, columns = linear_sum_assignment(working)
            for row, column in zip(rows, columns):
                if not np.isfinite(cost[row, column]) or cost[row, column] > 1.5:
                    continue
                track_id = active_ids[int(row)]
                item = objects[int(column)]
                item["track_id"] = track_id
                tracks[track_id].append({**item, "frame_index": frame_index})
                assigned_objects.add(int(column))
        for object_index, item in enumerate(objects):
            if object_index in assigned_objects:
                continue
            item["track_id"] = next_track_id
            tracks[next_track_id] = [{**item, "frame_index": frame_index}]
            next_track_id += 1

    summaries: list[dict[str, Any]] = []
    for track_id, points in tracks.items():
        times = np.asarray([item["time"] for item in points], dtype=float)
        x = np.asarray([item["center_x_D"] for item in points], dtype=float)
        y = np.asarray([item["center_y_D"] for item in points], dtype=float)
        speed_x = float(np.polyfit(times, x, 1)[0]) if len(points) >= 2 else 0.0
        step_dx = np.diff(x)
        downstream_fraction = float(np.mean(step_dx >= 0.0)) if len(step_dx) else 0.0
        time_span = float(times[-1] - times[0])
        downstream_displacement = float(x[-1] - x[0])
        persistent = bool(
            len(points) >= int(tracking["minimum_persistent_frames"])
            and time_span >= float(tracking["minimum_persistent_time_D_over_ainf"])
            and downstream_displacement
            >= float(tracking["minimum_downstream_displacement_D"])
            and downstream_fraction
            >= float(tracking["minimum_downstream_step_fraction"])
            and 0.0 < speed_x
            <= float(config["physics"]["u_inf"])
            * float(tracking["maximum_convection_speed_over_uinf"])
        )
        summaries.append(
            {
                "track_id": track_id,
                "vorticity_sign": int(points[0]["vorticity_sign"]),
                "frames": int(len(points)),
                "start_time": float(times[0]),
                "end_time": float(times[-1]),
                "time_span_D_over_ainf": time_span,
                "start_x_D": float(x[0]),
                "end_x_D": float(x[-1]),
                "downstream_displacement_D": downstream_displacement,
                "downstream_step_fraction": downstream_fraction,
                "convection_speed_ainf": speed_x,
                "convection_speed_over_uinf": speed_x
                / float(config["physics"]["u_inf"]),
                "mean_y_D": float(np.mean(y)),
                "median_abs_gamma2": float(
                    np.median([item["mean_abs_gamma2"] for item in points])
                ),
                "median_equivalent_radius_D": float(
                    np.median([item["equivalent_radius_D"] for item in points])
                ),
                "persistent_convecting": persistent,
                "points": [
                    {
                        "step": int(item["step"]),
                        "time": float(item["time"]),
                        "x_D": float(item["center_x_D"]),
                        "y_D": float(item["center_y_D"]),
                        "frame_object_id": int(item["frame_object_id"]),
                    }
                    for item in points
                ],
            }
        )
    summaries.sort(key=lambda item: (-int(item["persistent_convecting"]), -int(item["frames"])))
    return summaries


def pair_mirrored_tracks(
    tracks: list[dict[str, Any]], config: dict[str, Any]
) -> list[dict[str, Any]]:
    """Pair simultaneous opposite-sign tracks related by y-reflection."""

    persistent = [item for item in tracks if item["persistent_convecting"]]
    positive = [item for item in persistent if int(item["vorticity_sign"]) > 0]
    negative = [item for item in persistent if int(item["vorticity_sign"]) < 0]
    cost = np.full((len(positive), len(negative)), np.inf, dtype=float)
    residuals: dict[tuple[int, int], tuple[int, float, float]] = {}
    minimum_common = int(config["tracking"]["mirror_pair_minimum_common_frames"])
    maximum_residual = float(
        config["tracking"]["mirror_pair_maximum_median_residual_D"]
    )
    for row, first in enumerate(positive):
        first_points = {int(item["step"]): item for item in first["points"]}
        for column, second in enumerate(negative):
            second_points = {int(item["step"]): item for item in second["points"]}
            common = sorted(set(first_points) & set(second_points))
            if len(common) < minimum_common:
                continue
            distances = np.asarray(
                [
                    math.hypot(
                        float(first_points[step]["x_D"] - second_points[step]["x_D"]),
                        float(first_points[step]["y_D"] + second_points[step]["y_D"]),
                    )
                    for step in common
                ]
            )
            median = float(np.median(distances))
            maximum = float(np.max(distances))
            if median <= maximum_residual:
                cost[row, column] = median + 0.05 / len(common)
                residuals[(row, column)] = (len(common), median, maximum)
    for item in tracks:
        item["mirror_pair_id"] = None
        item["mirror_paired"] = False
        item["mirror_common_frames"] = 0
        item["mirror_median_residual_D"] = None
        item["mirror_maximum_residual_D"] = None
    if not cost.size:
        return tracks
    rows, columns = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
    pair_id = 1
    for row, column in zip(rows, columns):
        if not np.isfinite(cost[row, column]):
            continue
        common, median, maximum = residuals[(int(row), int(column))]
        for item in (positive[int(row)], negative[int(column)]):
            item["mirror_pair_id"] = pair_id
            item["mirror_paired"] = True
            item["mirror_common_frames"] = common
            item["mirror_median_residual_D"] = median
            item["mirror_maximum_residual_D"] = maximum
        pair_id += 1
    return tracks


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def relative_l2(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    return float(
        np.linalg.norm((first - second)[mask])
        / max(np.linalg.norm(second[mask]), np.finfo(float).tiny)
    )


def repeatability_audit(
    config: dict[str, Any],
    primary_signals: list[dict[str, float]],
) -> dict[str, Any]:
    repeat = config["repeat_run"]
    records = read_inventory(ROOT / repeat["inventory"])
    input_dir = ROOT / repeat["input_dir"] / repeat["silo_leaf_subdir"]
    repeat_signals: list[dict[str, float]] = []
    for record in records:
        step = int(record["step"])
        time = float(record["time"])
        frame = read_frame(input_dir / f"{step}.silo", step, time)
        repeat_signals.append(frame_signals(frame, config))
    signal_names = [
        name
        for name in primary_signals[0]
        if name == "lift_coefficient" or name.startswith("v_") or name.startswith("dp_")
    ]
    primary_times = np.asarray([item["time"] for item in primary_signals])
    repeat_times = np.asarray([item["time"] for item in repeat_signals])
    signal_comparison: dict[str, Any] = {}
    for name in signal_names:
        primary_values = np.asarray([item[name] for item in primary_signals])
        repeat_values = np.asarray([item[name] for item in repeat_signals])
        interpolated = np.interp(repeat_times, primary_times, primary_values)
        difference = repeat_values - interpolated
        if name.startswith("v_"):
            normalization = float(config["physics"]["u_inf"])
        elif name.startswith("dp_"):
            normalization = float(config["physics"]["p_inf"])
        else:
            normalization = 1.0
        signal_comparison[name] = {
            "absolute_rmse": float(np.sqrt(np.mean(difference * difference))),
            "rmse_normalized_by_reference_scale": float(
                np.sqrt(np.mean(difference * difference)) / normalization
            ),
            "maximum_absolute_difference": float(np.max(np.abs(difference))),
        }

    field_comparison: list[dict[str, Any]] = []
    primary_records = read_inventory(ROOT / config["inventory"])
    primary_dir = ROOT / config["input_dir"] / config["silo_leaf_subdir"]
    for target_time in (float(value) for value in config["sensitivity"]["matched_field_times"]):
        primary_record = min(primary_records, key=lambda item: abs(float(item["time"]) - target_time))
        repeat_record = min(records, key=lambda item: abs(float(item["time"]) - target_time))
        mismatch = abs(float(primary_record["time"]) - float(repeat_record["time"]))
        if mismatch > float(config["sensitivity"]["maximum_nearest_time_mismatch"]):
            continue
        primary_frame = read_frame(
            primary_dir / f"{int(primary_record['step'])}.silo",
            int(primary_record["step"]),
            float(primary_record["time"]),
        )
        repeat_frame = read_frame(
            input_dir / f"{int(repeat_record['step'])}.silo",
            int(repeat_record["step"]),
            float(repeat_record["time"]),
        )
        yy, xx = np.meshgrid(primary_frame.y, primary_frame.x, indexing="ij")
        wake = (
            (xx >= 0.58)
            & (xx <= 3.5)
            & (np.abs(yy) <= 1.5)
            & (xx * xx + yy * yy > 0.58 * 0.58)
        )
        field_comparison.append(
            {
                "target_time": target_time,
                "primary_time": float(primary_frame.time),
                "repeat_time": float(repeat_frame.time),
                "absolute_time_mismatch": mismatch,
                "relative_l2_wake_pressure": relative_l2(
                    repeat_frame.pressure, primary_frame.pressure, wake
                ),
                "relative_l2_wake_u": relative_l2(repeat_frame.u, primary_frame.u, wake),
                "relative_l2_wake_v": relative_l2(repeat_frame.v, primary_frame.v, wake),
            }
        )
    primary_case = ROOT / config["input_dir"] / "case.py"
    repeat_case = ROOT / repeat["case_file"]
    return {
        "role": repeat["role"],
        "repeat_snapshot_count": len(records),
        "repeat_time_interval": [float(records[0]["time"]), float(records[-1]["time"])],
        "case_file_sha256": {
            "primary": sha256(primary_case),
            "repeat": sha256(repeat_case),
            "identical": sha256(primary_case) == sha256(repeat_case),
        },
        "signal_comparison_after_primary_time_interpolation": signal_comparison,
        "matched_field_comparison": field_comparison,
        "limitations": [
            "The two runs use the same f90 grid and the same solver time step.",
            "Nonzero matched-field differences include the recorded time mismatch.",
            "This tests independent repeatability only; it is not grid convergence or solver-time-step sensitivity.",
        ],
        "repeat_signals": repeat_signals,
    }


def spectral_audit(
    signals: list[dict[str, float]], config: dict[str, Any]
) -> dict[str, Any]:
    spectral = config["spectral"]
    selected = [
        item for item in signals if item["time"] >= float(spectral["analysis_start_time"])
    ]
    times = np.asarray([item["time"] for item in selected], dtype=float)
    names = [
        name
        for name in selected[0]
        if name == "lift_coefficient" or name.startswith("v_") or name.startswith("dp_")
    ]
    results: dict[str, Any] = {}
    for name in names:
        values = np.asarray([item[name] for item in selected], dtype=float)
        reference_scale = (
            float(config["physics"]["u_inf"])
            if name.startswith("v_")
            else float(config["physics"]["p_inf"])
            if name.startswith("dp_")
            else 1.0
        )
        results[name] = periodogram(
            times,
            values,
            u_inf=float(config["physics"]["u_inf"]),
            minimum_cycles=float(spectral["minimum_cycles"]),
            minimum_peak_power_fraction=float(spectral["minimum_peak_power_fraction"]),
            reference_scale=reference_scale,
            minimum_rms_normalized_by_reference_scale=float(
                spectral["minimum_rms_normalized_by_reference_scale"]
            ),
            maximum_half_window_rms_ratio=float(
                spectral["maximum_half_window_rms_ratio"]
            ),
        )
    passed = [item for item in results.values() if item.get("passed")]
    consensus = False
    consensus_frequency = None
    if len(passed) >= int(spectral["minimum_consensus_signals"]):
        frequencies = np.asarray(
            [item["peak_frequency_ainf_over_D"] for item in passed], dtype=float
        )
        resolution = float(passed[0]["frequency_resolution_ainf_over_D"])
        median_frequency = float(np.median(frequencies))
        close = np.abs(frequencies - median_frequency) <= (
            int(spectral["maximum_consensus_bins"]) * resolution + 1e-12
        )
        consensus = bool(
            np.count_nonzero(close) >= int(spectral["minimum_consensus_signals"])
        )
        if consensus:
            consensus_frequency = median_frequency

    decimated_results: dict[str, Any] = {}
    decimated = selected[::2]
    decimated_times = np.asarray([item["time"] for item in decimated], dtype=float)
    for name in names:
        reference_scale = (
            float(config["physics"]["u_inf"])
            if name.startswith("v_")
            else float(config["physics"]["p_inf"])
            if name.startswith("dp_")
            else 1.0
        )
        decimated_results[name] = periodogram(
            decimated_times,
            np.asarray([item[name] for item in decimated], dtype=float),
            u_inf=float(config["physics"]["u_inf"]),
            minimum_cycles=float(spectral["minimum_cycles"]),
            minimum_peak_power_fraction=float(spectral["minimum_peak_power_fraction"]),
            reference_scale=reference_scale,
            minimum_rms_normalized_by_reference_scale=float(
                spectral["minimum_rms_normalized_by_reference_scale"]
            ),
            maximum_half_window_rms_ratio=float(
                spectral["maximum_half_window_rms_ratio"]
            ),
        )
    return {
        "analysis_time_interval": [float(times[0]), float(times[-1])],
        "signals": results,
        "consensus_peak_pass": consensus,
        "consensus_frequency_ainf_over_D": consensus_frequency,
        "consensus_strouhal_fD_over_uinf": (
            None
            if consensus_frequency is None
            else consensus_frequency / float(config["physics"]["u_inf"])
        ),
        "saved_snapshot_cadence_sensitivity": decimated_results,
        "cadence_limitation": "Decimation tests observation cadence only and does not test the MFC solver time step.",
    }


def draw_body(axis: plt.Axes, radius: float) -> None:
    axis.add_patch(
        Circle(
            (0.0, 0.0),
            radius,
            facecolor="white",
            edgecolor=CHARCOAL,
            linewidth=0.9,
            zorder=8,
        )
    )


def make_figures(
    config: dict[str, Any],
    cached: dict[int, dict[str, Any]],
    tracks: list[dict[str, Any]],
    signals: list[dict[str, float]],
    spectral: dict[str, Any],
    repeatability: dict[str, Any],
    figure_dir: Path,
) -> list[dict[str, str]]:
    dpi = int(config["figures"]["dpi"])
    radius = float(config["physics"]["cylinder_radius"])
    manifests: list[dict[str, str]] = []

    steps = [int(value) for value in config["figures"]["representative_steps"]]
    figure, axes = plt.subplots(len(steps), 4, figsize=(14.0, 8.0), constrained_layout=True)
    for row, step in enumerate(steps):
        item = cached[step]
        frame: Frame = item["frame"]
        extent = [frame.x[0], frame.x[-1], frame.y[0], frame.y[-1]]
        omega = item["vorticity"]
        lambda_ci = item["lambda_ci"]
        gamma2 = item["gamma2"]
        labels = item["object_labels"]
        shock = item["shock"]
        panels = [
            (omega, SymLogNorm(linthresh=0.5, vmin=-60, vmax=60), "PuOr_r", r"$\omega_zD/a_\infty$"),
            (lambda_ci, None, "magma", r"$\lambda_{ci}D/a_\infty$"),
            (gamma2, TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1), "PuOr_r", r"$\Gamma_2$"),
        ]
        for column, (field, norm, cmap, title) in enumerate(panels):
            image = axes[row, column].imshow(
                field,
                origin="lower",
                extent=extent,
                cmap=cmap,
                norm=norm,
                vmin=None if norm is not None else 0.0,
                vmax=None if norm is not None else 12.0,
                interpolation="nearest",
                rasterized=True,
            )
            figure.colorbar(image, ax=axes[row, column], fraction=0.046, pad=0.02)
            axes[row, column].set_title(title)
        axes[row, 3].imshow(
            np.log1p(np.maximum(frame.schlieren, 0.0)),
            origin="lower",
            extent=extent,
            cmap="Greys",
            interpolation="nearest",
            rasterized=True,
        )
        if np.any(labels):
            axes[row, 3].contour(
                frame.x, frame.y, labels > 0, levels=[0.5], colors=[VERMILLION], linewidths=0.8
            )
        axes[row, 3].set_title("retained topology candidates")
        for axis in axes[row]:
            if np.any(shock):
                axis.contour(
                    frame.x,
                    frame.y,
                    shock,
                    levels=[0.5],
                    colors=[MID_GRAY],
                    linewidths=0.65,
                    linestyles="dashed",
                )
            draw_body(axis, radius)
            axis.set_xlim(-1.35, 3.5)
            axis.set_ylim(-1.5, 1.5)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel(r"$x/D$")
            axis.set_ylabel(r"$y/D$")
        axes[row, 0].text(
            0.02,
            0.96,
            rf"$t={frame.time:.3f}$",
            transform=axes[row, 0].transAxes,
            va="top",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8},
        )
    figure.suptitle("Euler-cylinder raw rotational diagnostics (wall and bow-shock layers excluded)")
    base = figure_dir / "figure_01_raw_rotational_diagnostics"
    save_figure(figure, base, dpi)
    manifests.append({"figure": base.name, "role": "raw diagnostics and candidate masks"})

    final = cached[steps[-1]]["frame"]
    figure, axis = plt.subplots(figsize=(9.0, 4.7), constrained_layout=True)
    axis.imshow(
        np.log1p(np.maximum(final.schlieren, 0.0)),
        origin="lower",
        extent=[final.x[0], final.x[-1], final.y[0], final.y[-1]],
        cmap="Greys",
        interpolation="nearest",
        rasterized=True,
    )
    plotted_tracks = sorted(
        [track for track in tracks if track["persistent_convecting"]],
        key=lambda track: (-int(track["frames"]), int(track["track_id"])),
    )[:14]
    for track in plotted_tracks:
        points = track["points"]
        color = VERMILLION if int(track["vorticity_sign"]) > 0 else MAGENTA
        x = [point["x_D"] for point in points]
        y = [point["y_D"] for point in points]
        paired = bool(track.get("mirror_paired"))
        axis.plot(
            x,
            y,
            color=color,
            linewidth=1.0 if paired else 1.8,
            linestyle="--" if paired else "-",
            alpha=0.72 if paired else 0.95,
            marker="o",
            markersize=2.3,
        )
    draw_body(axis, radius)
    axis.set_xlim(-1.35, 3.5)
    axis.set_ylim(-1.5, 1.5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(r"$x/D$")
    axis.set_ylabel(r"$y/D$")
    axis.set_title("Longest persistent raw-field tracks; dashed tracks have a mirror partner")
    axis.legend(
        handles=[
            Line2D([0], [0], color=VERMILLION, marker="o", label=r"$\omega_z>0$"),
            Line2D([0], [0], color=MAGENTA, marker="o", label=r"$\omega_z<0$"),
            Line2D([0], [0], color=MID_GRAY, linestyle="--", label="mirror-paired"),
            Line2D([0], [0], color=MID_GRAY, linestyle="-", linewidth=1.8, label="unpaired"),
        ],
        loc="upper right",
    )
    base = figure_dir / "figure_02_raw_vortex_tracks"
    save_figure(figure, base, dpi)
    manifests.append({"figure": base.name, "role": "persistent object tracks"})

    times = np.asarray([item["time"] for item in signals])
    figure, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), constrained_layout=True)
    axes[0, 0].plot(times, [item["lift_coefficient"] for item in signals], color=CHARCOAL, label=r"$C_L^p$")
    for color, name in zip([VERMILLION, GOLD, MAGENTA], ["dp_x0p75", "dp_x1p0", "dp_x1p5"]):
        axes[0, 0].plot(times, [item[name] for item in signals], color=color, linewidth=1.0, label=name.replace("dp_", r"$\Delta p$ "))
    axes[0, 0].axvline(float(config["spectral"]["analysis_start_time"]), color=MID_GRAY, linestyle="--")
    axes[0, 0].set_title("Odd-symmetry pressure/lift signals")
    axes[0, 0].set_xlabel(r"$t a_\infty/D$")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(alpha=0.2)
    for color, name in zip([CHARCOAL, VERMILLION, GOLD, MAGENTA], ["v_x0p75", "v_x1p0", "v_x1p5", "v_x2p0"]):
        axes[0, 1].plot(times, [item[name] for item in signals], color=color, linewidth=1.0, label=name.replace("v_", r"$v$ "))
    axes[0, 1].axvline(float(config["spectral"]["analysis_start_time"]), color=MID_GRAY, linestyle="--")
    axes[0, 1].set_title("Wake centerline transverse velocity")
    axes[0, 1].set_xlabel(r"$t a_\infty/D$")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(alpha=0.2)

    for color, name in zip([CHARCOAL, VERMILLION, GOLD, MAGENTA], ["lift_coefficient", "v_x0p75", "v_x1p0", "dp_x1p0"]):
        item = spectral["signals"][name]
        axes[1, 0].plot(item["strouhal"][1:], item["normalized_power"][1:], color=color, linewidth=1.2, label=name)
    axes[1, 0].set_xlim(0.0, 0.7)
    axes[1, 0].set_xlabel(r"$St=fD/U_\infty$")
    axes[1, 0].set_ylabel("normalized windowed power")
    axes[1, 0].set_title("Tail-window spectra")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(alpha=0.2)

    candidate_count = [item.get("candidate_count", 0.0) for item in signals]
    positive = [item.get("candidate_positive_circulation", 0.0) for item in signals]
    negative = [item.get("candidate_negative_circulation", 0.0) for item in signals]
    axes[1, 1].plot(times, candidate_count, color=CHARCOAL, label="object count")
    twin = axes[1, 1].twinx()
    twin.plot(times, positive, color=VERMILLION, label="positive circulation")
    twin.plot(times, negative, color=MAGENTA, label="negative circulation")
    axes[1, 1].set_xlabel(r"$t a_\infty/D$")
    axes[1, 1].set_ylabel("candidate objects")
    twin.set_ylabel(r"candidate circulation / $a_\infty D$")
    axes[1, 1].set_title("Raw-field candidate population")
    axes[1, 1].grid(alpha=0.2)
    lines1, labels1 = axes[1, 1].get_legend_handles_labels()
    lines2, labels2 = twin.get_legend_handles_labels()
    axes[1, 1].legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    base = figure_dir / "figure_03_probe_signals_and_spectra"
    save_figure(figure, base, dpi)
    manifests.append({"figure": base.name, "role": "probe histories and Strouhal spectra"})

    field_rows = repeatability["matched_field_comparison"]
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), constrained_layout=True)
    compare_times = [item["target_time"] for item in field_rows]
    for color, key, label in [
        (CHARCOAL, "relative_l2_wake_pressure", "pressure"),
        (VERMILLION, "relative_l2_wake_u", "u"),
        (MAGENTA, "relative_l2_wake_v", "v"),
    ]:
        axes[0].plot(compare_times, [item[key] for item in field_rows], marker="o", color=color, label=label)
    axes[0].set_yscale("symlog", linthresh=1e-12)
    axes[0].set_xlabel(r"matched $t a_\infty/D$")
    axes[0].set_ylabel("relative wake-field difference")
    axes[0].set_title("Independent repeat, same f90 grid and solver time step")
    axes[0].legend()
    axes[0].grid(alpha=0.2)
    names = ["lift_coefficient", "v_x0p75", "v_x1p0", "dp_x1p0"]
    full_st = [spectral["signals"][name].get("peak_strouhal_fD_over_uinf", 0.0) for name in names]
    half_st = [spectral["saved_snapshot_cadence_sensitivity"][name].get("peak_strouhal_fD_over_uinf", 0.0) for name in names]
    index = np.arange(len(names))
    axes[1].plot(index, full_st, marker="o", color=CHARCOAL, label="all saved frames")
    axes[1].plot(index, half_st, marker="s", color=VERMILLION, label="every second frame")
    axes[1].set_xticks(index, names, rotation=25, ha="right")
    axes[1].set_ylabel("dominant spectral-bin St")
    axes[1].set_title("Saved-cadence sensitivity (not solver-dt sensitivity)")
    axes[1].legend()
    axes[1].grid(alpha=0.2)
    base = figure_dir / "figure_04_available_numerical_sensitivity"
    save_figure(figure, base, dpi)
    manifests.append({"figure": base.name, "role": "same-grid repeat and snapshot-cadence sensitivity"})
    return manifests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "mfc_euler_cylinder_m2p7_f90_vortex_dynamics_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    records = read_inventory(ROOT / config["inventory"])
    input_dir = ROOT / config["input_dir"] / config["silo_leaf_subdir"]
    physics_dir = ROOT / config["physics_result_dir"]
    output_dir = ROOT / "results" / "physics_only" / config["experiment_id"]
    figure_dir = ROOT / "results" / "figures" / config["experiment_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    representative_steps = set(int(value) for value in config["figures"]["representative_steps"])
    frame_reports: list[dict[str, Any]] = []
    frame_objects: list[list[dict[str, Any]]] = []
    signals: list[dict[str, float]] = []
    cached: dict[int, dict[str, Any]] = {}
    mfc_omega_checks: list[float] = []
    for frame_index, record in enumerate(records):
        step = int(record["step"])
        time = float(record["time"])
        frame = read_frame(input_dir / f"{step}.silo", step, time)
        yy, xx = np.meshgrid(frame.y, frame.x, indexing="ij")
        radius = float(config["physics"]["cylinder_radius"])
        geometry = xx * xx + yy * yy <= radius * radius
        physics = np.load(physics_dir / f"step_{step:05d}_physics_only.npz")
        shock = physics["shock_envelope"].astype(bool)
        dx = float(np.mean(np.diff(frame.x)))
        dy = float(np.mean(np.diff(frame.y)))
        flow = derivatives(frame.rho, frame.pressure, frame.u, frame.v, dx=dx, dy=dy)
        gamma2 = gamma2_field(
            frame.u,
            frame.v,
            ~geometry,
            dx=dx,
            dy=dy,
            radius_cells=int(config["vortex_diagnostics"]["gamma2_radius_cells"]),
        )
        labels, objects, diagnostic = detect_vortex_objects(
            frame,
            gamma2,
            shock_mask=shock,
            geometry=geometry,
            config=config,
        )
        for item in objects:
            item["frame_index"] = frame_index
        frame_objects.append(objects)
        signal_record = frame_signals(frame, config, objects=objects)
        permitted = diagnostic["permitted"]
        mirror_permitted = permitted & np.flipud(permitted)
        v_antisymmetric = 0.5 * (frame.v - np.flipud(frame.v))
        v_symmetry_breaking = 0.5 * (frame.v + np.flipud(frame.v))
        antisymmetric_norm = float(np.linalg.norm(v_antisymmetric[mirror_permitted]))
        symmetry_breaking_norm = float(np.linalg.norm(v_symmetry_breaking[mirror_permitted]))
        symmetry_breaking_ratio = (
            symmetry_breaking_norm / antisymmetric_norm
            if antisymmetric_norm > 1e-12
            else 0.0 if symmetry_breaking_norm <= 1e-12 else float("inf")
        )
        signal_record["mirror_symmetry_breaking_v_ratio"] = symmetry_breaking_ratio
        signals.append(signal_record)
        omega_denominator = float(np.linalg.norm(flow.vorticity[permitted]))
        omega_relative_l2 = (
            float(np.linalg.norm((frame.omega - flow.vorticity)[permitted]) / omega_denominator)
            if omega_denominator > 1e-10
            else None
        )
        if omega_relative_l2 is not None:
            mfc_omega_checks.append(omega_relative_l2)
        wall_violation = (
            (diagnostic["wall_distance"] < float(config["shock_wall_exclusion"]["wall_band_D"]))
            & ~geometry
        )
        frame_reports.append(
            {
                "frame_index": frame_index,
                "step": step,
                "time": time,
                "candidate_objects": len(objects),
                "positive_objects": sum(int(item["vorticity_sign"]) > 0 for item in objects),
                "negative_objects": sum(int(item["vorticity_sign"]) < 0 for item in objects),
                "candidate_pixels": int(np.count_nonzero(labels)),
                "candidate_geometry_overlap_pixels": int(np.count_nonzero((labels > 0) & geometry)),
                "candidate_shock_exclusion_overlap_pixels": int(
                    np.count_nonzero((labels > 0) & diagnostic["expanded_shock"])
                ),
                "candidate_wall_exclusion_overlap_pixels": int(
                    np.count_nonzero((labels > 0) & wall_violation)
                ),
                "maximum_abs_gamma2": float(np.max(np.abs(gamma2[permitted]))),
                "maximum_lambda_ci_D_over_ainf": float(np.max(diagnostic["lambda_ci"][permitted])),
                "mfc_omega3_vs_recomputed_relative_l2_in_permitted_wake": omega_relative_l2,
                "mirror_symmetry_breaking_v_ratio": symmetry_breaking_ratio,
            }
        )
        np.savez_compressed(
            output_dir / f"step_{step:05d}_objects.npz",
            step=np.asarray(step, dtype=np.int64),
            time=np.asarray(time, dtype=np.float64),
            object_labels=labels,
        )
        if step in representative_steps:
            np.savez_compressed(
                output_dir / f"step_{step:05d}_diagnostics.npz",
                step=np.asarray(step, dtype=np.int64),
                time=np.asarray(time, dtype=np.float64),
                vorticity=diagnostic["vorticity"],
                lambda_ci=diagnostic["lambda_ci"],
                q_deviatoric=diagnostic["q_deviatoric"],
                gamma2=gamma2,
                object_labels=labels,
                permitted=diagnostic["permitted"].astype(np.uint8),
                shock_exclusion=diagnostic["expanded_shock"].astype(np.uint8),
                geometry=geometry.astype(np.uint8),
            )
            cached[step] = {
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
                "vorticity": diagnostic["vorticity"],
                "lambda_ci": diagnostic["lambda_ci"],
                "gamma2": gamma2,
                "object_labels": labels,
                "shock": shock,
            }
        print(
            f"raw-vortex step={step:05d} t={time:.3f} "
            f"objects={len(objects)} pixels={np.count_nonzero(labels)}",
            flush=True,
        )

    tracks = pair_mirrored_tracks(track_objects(frame_objects, config), config)
    by_track = {
        (int(point["step"]), int(point["frame_object_id"])): int(track["track_id"])
        for track in tracks
        for point in track["points"]
    }
    object_rows: list[dict[str, Any]] = []
    for objects in frame_objects:
        for item in objects:
            item["track_id"] = by_track[(int(item["step"]), int(item["frame_object_id"]))]
            object_rows.append(item)
    persistent = [item for item in tracks if item["persistent_convecting"]]
    persistent_positive = [item for item in persistent if item["vorticity_sign"] > 0]
    persistent_negative = [item for item in persistent if item["vorticity_sign"] < 0]
    mirror_paired = [item for item in persistent if item["mirror_paired"]]
    unpaired = [item for item in persistent if not item["mirror_paired"]]
    unpaired_positive = [item for item in unpaired if item["vorticity_sign"] > 0]
    unpaired_negative = [item for item in unpaired if item["vorticity_sign"] < 0]
    ordered_birth_signs = [
        int(item["vorticity_sign"])
        for item in sorted(unpaired, key=lambda value: (value["start_time"], value["track_id"]))
    ]
    alternations = sum(
        first != second for first, second in zip(ordered_birth_signs[:-1], ordered_birth_signs[1:])
    )
    alternation_fraction = alternations / max(len(ordered_birth_signs) - 1, 1)
    mirror_paired_fraction = len(mirror_paired) / max(len(persistent), 1)
    topology_gate = bool(
        len(unpaired_positive) >= 2
        and len(unpaired_negative) >= 2
        and alternation_fraction >= 0.5
        and mirror_paired_fraction
        <= float(
            config["tracking"]["maximum_mirror_paired_fraction_for_alternating_shedding"]
        )
    )

    spectra = spectral_audit(signals, config)
    repeatability = repeatability_audit(config, signals)
    figure_manifest = make_figures(
        config, cached, tracks, signals, spectra, repeatability, figure_dir
    )
    grid_available = config["sensitivity"]["grid_check"] != "UNAVAILABLE_NO_SECOND_EULER_GRID"
    dt_available = config["sensitivity"]["solver_time_step_check"] != "UNAVAILABLE_NO_SECOND_EULER_TIME_STEP"
    validated = bool(
        topology_gate
        and spectra["consensus_peak_pass"]
        and grid_available
        and dt_available
    )
    conclusion = (
        "VALIDATED_EULER_SHOCK_GENERATED_VORTEX_SHEDDING"
        if validated
        else "NOT_VALIDATED_WITH_AVAILABLE_DATA"
    )
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "result_role": "physics_only_raw_field_audit; independent of neural vortex output; not human ground truth",
        "case_group_id": config["case_group_id"],
        "frames": frame_reports,
        "objects": object_rows,
        "tracks": tracks,
        "spectral": spectra,
        "repeatability": {key: value for key, value in repeatability.items() if key != "repeat_signals"},
        "decision": {
            "persistent_tracks": len(persistent),
            "persistent_positive_tracks": len(persistent_positive),
            "persistent_negative_tracks": len(persistent_negative),
            "mirror_paired_persistent_tracks": len(mirror_paired),
            "mirror_paired_persistent_fraction": mirror_paired_fraction,
            "unpaired_persistent_positive_tracks": len(unpaired_positive),
            "unpaired_persistent_negative_tracks": len(unpaired_negative),
            "unpaired_track_birth_sign_alternation_fraction": alternation_fraction,
            "alternating_shedding_topology_gate_pass": topology_gate,
            "rotational_structure_tracks_supported": len(persistent) > 0,
            "final_v_mirror_symmetry_breaking_ratio": float(
                frame_reports[-1]["mirror_symmetry_breaking_v_ratio"]
            ),
            "spectral_consensus_gate_pass": bool(spectra["consensus_peak_pass"]),
            "independent_same_grid_repeat_available": True,
            "independent_euler_grid_check_available": grid_available,
            "independent_solver_time_step_check_available": dt_available,
            "periodic_shedding_validated": validated,
            "conclusion": conclusion,
            "allowed_wording": (
                "Euler shock-generated vortex shedding"
                if validated
                else "raw-field rotational candidates / evolving inviscid wake; periodic Euler shedding not established"
            ),
        },
        "diagnostic_consistency": {
            "mfc_omega3_vs_recomputed_relative_l2_permitted_wake_minimum": float(np.min(mfc_omega_checks)),
            "mfc_omega3_vs_recomputed_relative_l2_permitted_wake_median": float(np.median(mfc_omega_checks)),
            "mfc_omega3_vs_recomputed_relative_l2_permitted_wake_maximum": float(np.max(mfc_omega_checks)),
        },
        "structural_qa": {
            "all_candidate_masks_exclude_geometry": all(
                item["candidate_geometry_overlap_pixels"] == 0 for item in frame_reports
            ),
            "all_candidate_masks_exclude_dilated_bow_shock": all(
                item["candidate_shock_exclusion_overlap_pixels"] == 0 for item in frame_reports
            ),
            "all_candidate_masks_exclude_wall_band": all(
                item["candidate_wall_exclusion_overlap_pixels"] == 0 for item in frame_reports
            ),
        },
        "missing_required_evidence": [
            "one otherwise identical Euler-cylinder run on a second grid",
            "one otherwise identical f90 Euler-cylinder run with a different solver time step",
            "a longer statistically stationary tail if fewer than several consistent cycles are resolved",
        ],
        "provenance": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "source_manifest": str((ROOT / "data/processed/mfc_euler_cylinder_m2p7_f90_t8_31/data_manifest.json").resolve()),
            "neural_outputs_used": False,
        },
    }
    write_json(output_dir / "vortex_dynamics_report.json", report)

    object_fields = [
        "frame_index", "step", "time", "frame_object_id", "track_id", "center_x_D", "center_y_D",
        "equivalent_radius_D", "pixels", "vorticity_sign", "vorticity_sign_fraction",
        "circulation_over_ainf_D", "mean_vorticity_D_over_ainf", "peak_abs_vorticity_D_over_ainf",
        "peak_lambda_ci_D_over_ainf", "mean_abs_gamma2", "peak_abs_gamma2", "aspect_ratio",
        "minimum_wall_distance_D", "minimum_shock_distance_D", "status",
    ]
    write_csv(output_dir / "vortex_objects.csv", object_rows, object_fields)
    track_rows = [{key: value for key, value in item.items() if key != "points"} for item in tracks]
    write_csv(
        output_dir / "vortex_tracks.csv",
        track_rows,
        [
            "track_id", "vorticity_sign", "frames", "start_time", "end_time", "time_span_D_over_ainf",
            "start_x_D", "end_x_D", "downstream_displacement_D", "downstream_step_fraction",
            "convection_speed_ainf", "convection_speed_over_uinf", "mean_y_D", "median_abs_gamma2",
            "median_equivalent_radius_D", "persistent_convecting", "mirror_pair_id", "mirror_paired",
            "mirror_common_frames", "mirror_median_residual_D", "mirror_maximum_residual_D",
        ],
    )
    write_csv(output_dir / "probe_signals.csv", signals, list(signals[0].keys()))
    spectrum_rows: list[dict[str, Any]] = []
    for name, item in spectra["signals"].items():
        for frequency, st, power in zip(
            item["frequencies_ainf_over_D"], item["strouhal"], item["normalized_power"]
        ):
            spectrum_rows.append(
                {
                    "signal": name,
                    "frequency_ainf_over_D": frequency,
                    "strouhal_fD_over_uinf": st,
                    "normalized_power": power,
                }
            )
    write_csv(
        output_dir / "spectra.csv",
        spectrum_rows,
        ["signal", "frequency_ainf_over_D", "strouhal_fD_over_uinf", "normalized_power"],
    )
    write_csv(figure_dir / "figure_manifest.csv", figure_manifest, ["figure", "role"])
    print(json.dumps(report["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
