#!/usr/bin/env python3
"""Three-run raw-field sensitivity audit for the Mach-2.7 Euler cylinder.

This program compares the production baseline with an otherwise identical
half-CFL run and a two-times-finer spatial grid.  The fine-grid finite-volume
fields are conservatively restricted to the common 90-cell-per-D observation
grid before any shock or vortex diagnostic is evaluated.  Neural outputs are
not inputs to this audit and detected rotational objects are candidates, not
human vortex ground truth.
"""

from __future__ import annotations

import argparse
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
    select_main_bow_component,
    write_json,
)
from audit_mfc_euler_cylinder_vortex_dynamics import (  # noqa: E402
    detect_vortex_objects,
    frame_signals,
    gamma2_field,
    pair_mirrored_tracks,
    spectral_audit,
    track_objects,
    write_csv,
)
from physics_proposals import derivatives, shock_proposal  # noqa: E402


CHARCOAL = "#202020"
VERMILLION = "#D55E00"
GOLD = "#E69F00"
MAGENTA = "#CC79A7"
MID_GRAY = "#777777"
LIGHT_GRAY = "#D0D0D0"
LINE_COLORS = [CHARCOAL, VERMILLION, GOLD, MAGENTA]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def block_average(field: np.ndarray, factor: int) -> np.ndarray:
    """Conservatively restrict a cell-centred finite-volume raster."""

    if factor == 1:
        return np.asarray(field)
    ny, nx = field.shape
    if ny % factor or nx % factor:
        raise ValueError(f"shape {field.shape} is not divisible by factor {factor}")
    return field.reshape(ny // factor, factor, nx // factor, factor).mean(axis=(1, 3))


def restrict_frame(frame: Frame, factor: int) -> Frame:
    if factor == 1:
        return frame
    if len(frame.x) % factor or len(frame.y) % factor:
        raise ValueError("coordinate counts are not divisible by restriction factor")
    x = frame.x.reshape(-1, factor).mean(axis=1)
    y = frame.y.reshape(-1, factor).mean(axis=1)
    return Frame(
        step=frame.step,
        time=frame.time,
        x=x,
        y=y,
        rho=block_average(frame.rho, factor),
        u=block_average(frame.u, factor),
        v=block_average(frame.v, factor),
        pressure=block_average(frame.pressure, factor),
        omega=block_average(frame.omega, factor),
        ib=block_average(frame.ib, factor),
        schlieren=block_average(frame.schlieren, factor),
    )


def crop_frame(frame: Frame, bounds: list[float]) -> tuple[Frame, tuple[slice, slice]]:
    xmin, xmax, ymin, ymax = bounds
    columns = np.flatnonzero((frame.x >= xmin) & (frame.x <= xmax))
    rows = np.flatnonzero((frame.y >= ymin) & (frame.y <= ymax))
    if not len(columns) or not len(rows):
        raise ValueError("vortex crop does not intersect the observation grid")
    index = (slice(int(rows[0]), int(rows[-1]) + 1), slice(int(columns[0]), int(columns[-1]) + 1))
    return Frame(
        step=frame.step,
        time=frame.time,
        x=frame.x[index[1]],
        y=frame.y[index[0]],
        rho=frame.rho[index],
        u=frame.u[index],
        v=frame.v[index],
        pressure=frame.pressure[index],
        omega=frame.omega[index],
        ib=frame.ib[index],
        schlieren=frame.schlieren[index],
    ), index


def exact_geometry(frame: Frame, radius: float) -> np.ndarray:
    yy, xx = np.meshgrid(frame.y, frame.x, indexing="ij")
    return xx * xx + yy * yy <= radius * radius


def entropy_free_bow_shock(
    frame: Frame, config: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    """Return the anchored compression/jump/NMS bow-front envelope."""

    dx = float(np.mean(np.diff(frame.x)))
    dy = float(np.mean(np.diff(frame.y)))
    radius = float(config["physics"]["cylinder_radius"])
    detector = config["shock_detector"]
    geometry = exact_geometry(frame, radius)
    flow = derivatives(frame.rho, frame.pressure, frame.u, frame.v, dx=dx, dy=dy)
    proposal = shock_proposal(
        frame.rho,
        frame.pressure,
        flow,
        geometry,
        dx=dx,
        dy=dy,
        jump_offset_pixels=float(detector["jump_offset_pixels_on_common_grid"]),
        leading_edge_anchor_distance=float(detector["leading_edge_anchor_distance_D"]),
    )
    main, components = select_main_bow_component(
        proposal.centerline,
        frame.x,
        frame.y,
        guard_radius=float(detector["guard_radius_D"]),
        dy=dy,
    )
    envelope = ndimage.binary_dilation(
        main, iterations=int(detector["envelope_dilation_cells_on_common_grid"])
    ) & ~geometry
    yy, xx = np.meshgrid(frame.y, frame.x, indexing="ij")
    axis = main & (np.abs(yy) <= max(3.0 * dy, 0.04)) & (xx < -radius)
    if np.any(axis):
        axis_x = xx[axis]
        shock_nose_x = float(np.max(axis_x))
        standoff = float(-radius - shock_nose_x)
    else:
        shock_nose_x = None
        standoff = None
    return envelope, {
        "centerline_pixels": int(np.count_nonzero(main)),
        "envelope_pixels": int(np.count_nonzero(envelope)),
        "shock_nose_x_D": shock_nose_x,
        "standoff_D": standoff,
        "components": components,
        "entropy_used": False,
    }


def case_paths(case: dict[str, Any]) -> tuple[Path, Path]:
    inventory = ROOT / case["inventory"]
    silo = ROOT / case["input_dir"] / case["silo_leaf_subdir"]
    return inventory, silo


def read_common_frame(case: dict[str, Any], record: dict[str, Any]) -> Frame:
    _, silo = case_paths(case)
    step = int(record["step"])
    frame = read_frame(silo / f"{step}.silo", step, float(record["time"]))
    return restrict_frame(frame, int(case["restriction_factor_to_common_grid"]))


def grid_signature(frame: Frame) -> str:
    digest = hashlib.sha256()
    # A restricted f180 cell centre and its analytically identical native-f90
    # centre can differ by one floating-point ulp.  Canonicalise coordinates
    # before hashing while retaining far more precision than the grid spacing.
    digest.update(np.round(np.asarray(frame.x, dtype="<f8"), 14).tobytes())
    digest.update(np.round(np.asarray(frame.y, dtype="<f8"), 14).tobytes())
    return digest.hexdigest()


def topology_decision(tracks: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    persistent = [item for item in tracks if item["persistent_convecting"]]
    positive = [item for item in persistent if int(item["vorticity_sign"]) > 0]
    negative = [item for item in persistent if int(item["vorticity_sign"]) < 0]
    paired = [item for item in persistent if item.get("mirror_paired")]
    unpaired = [item for item in persistent if not item.get("mirror_paired")]
    up = [item for item in unpaired if int(item["vorticity_sign"]) > 0]
    un = [item for item in unpaired if int(item["vorticity_sign"]) < 0]
    signs = [
        int(item["vorticity_sign"])
        for item in sorted(unpaired, key=lambda value: (value["start_time"], value["track_id"]))
    ]
    alternations = sum(a != b for a, b in zip(signs[:-1], signs[1:]))
    alternation_fraction = alternations / max(len(signs) - 1, 1)
    paired_fraction = len(paired) / max(len(persistent), 1)
    gate = bool(
        len(up) >= 2
        and len(un) >= 2
        and alternation_fraction >= 0.5
        and paired_fraction
        <= float(config["tracking"]["maximum_mirror_paired_fraction_for_alternating_shedding"])
    )
    return {
        "persistent_tracks": len(persistent),
        "persistent_positive_tracks": len(positive),
        "persistent_negative_tracks": len(negative),
        "mirror_paired_persistent_tracks": len(paired),
        "mirror_paired_persistent_fraction": paired_fraction,
        "unpaired_persistent_positive_tracks": len(up),
        "unpaired_persistent_negative_tracks": len(un),
        "unpaired_track_birth_sign_alternation_fraction": alternation_fraction,
        "alternating_shedding_topology_gate_pass": gate,
        "rotational_structure_tracks_supported": bool(persistent),
    }


def analyze_case(
    case: dict[str, Any],
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    inventory_path, _ = case_paths(case)
    records = read_inventory(inventory_path)
    expected = int(case["expected_snapshots"])
    if len(records) != expected:
        raise ValueError(f"{case['name']}: expected {expected} snapshots, found {len(records)}")
    case_dir = output_root / case["name"]
    case_dir.mkdir(parents=True, exist_ok=True)
    frame_reports: list[dict[str, Any]] = []
    frame_objects: list[list[dict[str, Any]]] = []
    signals: list[dict[str, float]] = []
    object_rows: list[dict[str, Any]] = []
    final_cache: dict[str, Any] | None = None
    reference_signature: str | None = None
    analysis_index = 0
    for record_index, record in enumerate(records):
        frame = read_common_frame(case, record)
        signature = grid_signature(frame)
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            raise ValueError(f"{case['name']}: observation grid changed in time")
        shock, shock_info = entropy_free_bow_shock(frame, config)
        objects: list[dict[str, Any]] = []
        labels = np.zeros((1, 1), dtype=np.uint16)
        diagnostic: dict[str, np.ndarray] | None = None
        gamma2: np.ndarray | None = None
        crop: Frame | None = None
        crop_index: tuple[slice, slice] | None = None
        if frame.time >= float(config["vortex_diagnostics"]["analysis_start_time"]):
            crop, crop_index = crop_frame(
                frame, [float(value) for value in config["vortex_diagnostics"]["crop_bounds_D"]]
            )
            geometry = exact_geometry(crop, float(config["physics"]["cylinder_radius"]))
            dx = float(np.mean(np.diff(crop.x)))
            dy = float(np.mean(np.diff(crop.y)))
            gamma2 = gamma2_field(
                crop.u,
                crop.v,
                ~geometry,
                dx=dx,
                dy=dy,
                radius_cells=int(config["vortex_diagnostics"]["gamma2_radius_cells"]),
            )
            labels, objects, diagnostic = detect_vortex_objects(
                crop,
                gamma2,
                shock_mask=shock[crop_index],
                geometry=geometry,
                config=config,
            )
            for item in objects:
                item["frame_index"] = analysis_index
            frame_objects.append(objects)
            analysis_index += 1
        signal = frame_signals(frame, config, objects=objects)
        signals.append(signal)
        geometry_full = exact_geometry(frame, float(config["physics"]["cylinder_radius"]))
        fluid = ~geometry_full
        finite = bool(
            np.all(np.isfinite(frame.rho[fluid]))
            and np.all(np.isfinite(frame.pressure[fluid]))
            and np.all(np.isfinite(frame.u[fluid]))
            and np.all(np.isfinite(frame.v[fluid]))
        )
        report = {
            "record_index": record_index,
            "step": int(frame.step),
            "time": float(frame.time),
            "candidate_objects": len(objects),
            "positive_objects": sum(int(item["vorticity_sign"]) > 0 for item in objects),
            "negative_objects": sum(int(item["vorticity_sign"]) < 0 for item in objects),
            "candidate_pixels": int(np.count_nonzero(labels)),
            "shock_centerline_pixels": shock_info["centerline_pixels"],
            "shock_envelope_pixels": shock_info["envelope_pixels"],
            "bow_shock_standoff_D": shock_info["standoff_D"],
            "all_fluid_primitive_values_finite": finite,
            "minimum_fluid_density": float(np.min(frame.rho[fluid])),
            "minimum_fluid_pressure": float(np.min(frame.pressure[fluid])),
        }
        if diagnostic is not None and crop is not None:
            report["maximum_lambda_ci_D_over_ainf_in_permitted_wake"] = float(
                np.max(diagnostic["lambda_ci"][diagnostic["permitted"]])
            )
            report["candidate_geometry_overlap_pixels"] = int(
                np.count_nonzero((labels > 0) & exact_geometry(crop, float(config["physics"]["cylinder_radius"])))
            )
            report["candidate_shock_exclusion_overlap_pixels"] = int(
                np.count_nonzero((labels > 0) & diagnostic["expanded_shock"])
            )
        frame_reports.append(report)
        if record_index == len(records) - 1 and crop is not None and diagnostic is not None:
            final_cache = {
                "frame": frame,
                "shock": shock,
                "crop": crop,
                "crop_index": crop_index,
                "labels": labels,
                "gamma2": gamma2,
                "vorticity": diagnostic["vorticity"],
            }
        print(
            f"three-run {case['name']} [{record_index + 1:03d}/{len(records):03d}] "
            f"t={frame.time:.3f} objects={len(objects)} shock={shock_info['centerline_pixels']}",
            flush=True,
        )

    tracks = pair_mirrored_tracks(track_objects(frame_objects, config), config)
    object_to_track = {
        (int(point["step"]), int(point["frame_object_id"])): int(track["track_id"])
        for track in tracks
        for point in track["points"]
    }
    for objects in frame_objects:
        for item in objects:
            item["track_id"] = object_to_track[(int(item["step"]), int(item["frame_object_id"]))]
            object_rows.append(item)
    spectra = spectral_audit(signals, config)
    decision = topology_decision(tracks, config)
    decision["spectral_consensus_gate_pass"] = bool(spectra["consensus_peak_pass"])
    case_report = {
        "case": case,
        "snapshot_count": len(records),
        "time_interval_D_over_ainf": [float(records[0]["time"]), float(records[-1]["time"])],
        "common_observation_grid_shape": list(final_cache["frame"].rho.shape) if final_cache else None,
        "common_observation_grid_sha256": reference_signature,
        "frames": frame_reports,
        "tracks": tracks,
        "spectral": spectra,
        "decision": decision,
        "structural_qa": {
            "all_fluid_values_finite": all(item["all_fluid_primitive_values_finite"] for item in frame_reports),
            "candidate_masks_exclude_geometry": all(item.get("candidate_geometry_overlap_pixels", 0) == 0 for item in frame_reports),
            "candidate_masks_exclude_dilated_bow_shock": all(item.get("candidate_shock_exclusion_overlap_pixels", 0) == 0 for item in frame_reports),
        },
    }
    write_json(case_dir / "case_report.json", case_report)
    write_csv(case_dir / "frame_summary.csv", frame_reports, list(frame_reports[-1].keys()))
    write_csv(case_dir / "probe_signals.csv", signals, list(signals[0].keys()))
    object_fields = [
        "frame_index", "step", "time", "frame_object_id", "track_id", "center_x_D", "center_y_D",
        "equivalent_radius_D", "pixels", "vorticity_sign", "vorticity_sign_fraction",
        "circulation_over_ainf_D", "mean_vorticity_D_over_ainf", "peak_abs_vorticity_D_over_ainf",
        "peak_lambda_ci_D_over_ainf", "mean_abs_gamma2", "peak_abs_gamma2", "aspect_ratio",
        "minimum_wall_distance_D", "minimum_shock_distance_D", "status",
    ]
    write_csv(case_dir / "vortex_objects.csv", object_rows, object_fields)
    write_csv(
        case_dir / "vortex_tracks.csv",
        [{key: value for key, value in item.items() if key != "points"} for item in tracks],
        [
            "track_id", "vorticity_sign", "frames", "start_time", "end_time", "time_span_D_over_ainf",
            "start_x_D", "end_x_D", "downstream_displacement_D", "downstream_step_fraction",
            "convection_speed_ainf", "convection_speed_over_uinf", "mean_y_D", "median_abs_gamma2",
            "median_equivalent_radius_D", "persistent_convecting", "mirror_pair_id", "mirror_paired",
            "mirror_common_frames", "mirror_median_residual_D", "mirror_maximum_residual_D",
        ],
    )
    if final_cache is None:
        raise RuntimeError(f"{case['name']}: no final diagnostic cache")
    return {
        "report": case_report,
        "records": records,
        "signals": signals,
        "tracks": tracks,
        "final": final_cache,
    }


def symmetric_relative_l2(first: np.ndarray, second: np.ndarray, mask: np.ndarray) -> float:
    numerator = float(np.linalg.norm((first - second)[mask]))
    denominator = 0.5 * (
        float(np.linalg.norm(first[mask])) + float(np.linalg.norm(second[mask]))
    )
    return numerator / max(denominator, np.finfo(float).tiny)


def nearest_record(records: list[dict[str, Any]], target: float, tolerance: float = 1e-7) -> dict[str, Any]:
    record = min(records, key=lambda item: abs(float(item["time"]) - target))
    if abs(float(record["time"]) - target) > tolerance:
        raise ValueError(f"no snapshot matches t={target}; nearest is {record['time']}")
    return record


def compare_fields(
    comparison: dict[str, Any],
    cases: dict[str, dict[str, Any]],
    analyses: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    first_name = comparison["first"]
    second_name = comparison["second"]
    first_case = cases[first_name]
    second_case = cases[second_name]
    first_records = analyses[first_name]["records"]
    second_records = analyses[second_name]["records"]
    if comparison["matched_times"] == "all_81_exact_times":
        times = [float(item["time"]) for item in first_records]
    else:
        times = [float(value) for value in comparison["matched_times"]]
    rows: list[dict[str, Any]] = []
    for index, target in enumerate(times):
        first_record = nearest_record(first_records, target)
        second_record = nearest_record(second_records, target)
        first = read_common_frame(first_case, first_record)
        second = read_common_frame(second_case, second_record)
        if not (np.allclose(first.x, second.x) and np.allclose(first.y, second.y)):
            raise ValueError(f"{comparison['name']}: common grids do not match")
        dy, dx = float(np.mean(np.diff(first.y))), float(np.mean(np.diff(first.x)))
        first_flow = derivatives(first.rho, first.pressure, first.u, first.v, dx=dx, dy=dy)
        second_flow = derivatives(second.rho, second.pressure, second.u, second.v, dx=dx, dy=dy)
        yy, xx = np.meshgrid(first.y, first.x, indexing="ij")
        wake = (
            (xx >= 0.58) & (xx <= 3.5) & (np.abs(yy) <= 1.5)
            & (xx * xx + yy * yy > 0.58 * 0.58)
        )
        rows.append(
            {
                "comparison": comparison["name"],
                "target_time": target,
                "first_step": int(first.step),
                "second_step": int(second.step),
                "relative_l2_wake_pressure": symmetric_relative_l2(first.pressure, second.pressure, wake),
                "relative_l2_wake_u": symmetric_relative_l2(first.u, second.u, wake),
                "relative_l2_wake_v": symmetric_relative_l2(first.v, second.v, wake),
                "relative_l2_wake_vorticity": symmetric_relative_l2(first_flow.vorticity, second_flow.vorticity, wake),
            }
        )
        print(f"field-compare {comparison['name']} [{index + 1:03d}/{len(times):03d}] t={target:.3f}", flush=True)
    return rows


def interpolate_track(track: dict[str, Any], times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    points = track["points"]
    source_t = np.asarray([point["time"] for point in points], dtype=float)
    source_x = np.asarray([point["x_D"] for point in points], dtype=float)
    source_y = np.asarray([point["y_D"] for point in points], dtype=float)
    return np.interp(times, source_t, source_x), np.interp(times, source_t, source_y)


def match_tracks(
    first: list[dict[str, Any]], second: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    a = [track for track in first if track["persistent_convecting"]]
    b = [track for track in second if track["persistent_convecting"]]
    cost = np.full((len(a), len(b)), np.inf)
    details: dict[tuple[int, int], dict[str, Any]] = {}
    maximum = float(config["tracking"]["cross_case_track_match_maximum_median_distance_D"])
    for row, one in enumerate(a):
        for column, two in enumerate(b):
            if int(one["vorticity_sign"]) != int(two["vorticity_sign"]):
                continue
            start = max(float(one["start_time"]), float(two["start_time"]))
            end = min(float(one["end_time"]), float(two["end_time"]))
            if end - start < float(config["tracking"]["minimum_persistent_time_D_over_ainf"]):
                continue
            times = np.linspace(start, end, 25)
            ax, ay = interpolate_track(one, times)
            bx, by = interpolate_track(two, times)
            distances = np.hypot(ax - bx, ay - by)
            median = float(np.median(distances))
            if median <= maximum:
                cost[row, column] = median
                details[(row, column)] = {
                    "first_track_id": int(one["track_id"]),
                    "second_track_id": int(two["track_id"]),
                    "vorticity_sign": int(one["vorticity_sign"]),
                    "overlap_time": end - start,
                    "median_trajectory_distance_D": median,
                    "maximum_trajectory_distance_D": float(np.max(distances)),
                }
    matches: list[dict[str, Any]] = []
    if cost.size:
        rows, columns = linear_sum_assignment(np.where(np.isfinite(cost), cost, 1e6))
        for row, column in zip(rows, columns):
            if np.isfinite(cost[row, column]):
                matches.append(details[(int(row), int(column))])
    return {
        "first_persistent_tracks": len(a),
        "second_persistent_tracks": len(b),
        "matched_tracks": len(matches),
        "symmetric_match_fraction": 2 * len(matches) / max(len(a) + len(b), 1),
        "median_matched_trajectory_distance_D": float(np.median([item["median_trajectory_distance_D"] for item in matches])) if matches else None,
        "matches": matches,
    }


def aggregate_field_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    metrics = [
        "relative_l2_wake_pressure", "relative_l2_wake_u",
        "relative_l2_wake_v", "relative_l2_wake_vorticity",
    ]
    for comparison in sorted({row["comparison"] for row in rows}):
        subset = [row for row in rows if row["comparison"] == comparison]
        for metric in metrics:
            values = np.asarray([row[metric] for row in subset], dtype=float)
            output.append(
                {
                    "comparison": comparison,
                    "metric": metric,
                    "samples": len(values),
                    "median": float(np.median(values)),
                    "maximum": float(np.max(values)),
                    "final": float(values[-1]),
                }
            )
    return output


def draw_body(axis: plt.Axes, radius: float) -> None:
    axis.add_patch(Circle((0.0, 0.0), radius, facecolor="white", edgecolor=CHARCOAL, linewidth=0.8, zorder=9))


def make_figures(
    config: dict[str, Any],
    analyses: dict[str, dict[str, Any]],
    field_rows: list[dict[str, Any]],
    figure_dir: Path,
) -> list[dict[str, str]]:
    figure_dir.mkdir(parents=True, exist_ok=True)
    dpi = int(config["figures"]["dpi"])
    radius = float(config["physics"]["cylinder_radius"])
    case_names = [case["name"] for case in config["cases"]]
    labels = ["baseline: f90, CFL 0.20", "time-step control: f90, CFL 0.10", "grid control: f180, CFL 0.20"]
    manifest: list[dict[str, str]] = []

    figure, axes = plt.subplots(3, 3, figsize=(12.2, 9.4), constrained_layout=True)
    for row, (name, label) in enumerate(zip(case_names, labels)):
        cached = analyses[name]["final"]
        frame: Frame = cached["frame"]
        crop: Frame = cached["crop"]
        extent = [frame.x[0], frame.x[-1], frame.y[0], frame.y[-1]]
        axes[row, 0].imshow(np.log1p(np.maximum(frame.schlieren, 0.0)), origin="lower", extent=extent, cmap="Greys", interpolation="nearest", rasterized=True)
        axes[row, 0].contour(frame.x, frame.y, cached["shock"], levels=[0.5], colors=[VERMILLION], linewidths=1.0)
        axes[row, 0].set_title(f"{label}\nentropy-free bow-shock front")
        omega = cached["vorticity"]
        omega_limit = max(float(np.nanpercentile(np.abs(omega), 99.5)), 1.0)
        image = axes[row, 1].imshow(
            omega, origin="lower", extent=[crop.x[0], crop.x[-1], crop.y[0], crop.y[-1]],
            cmap="PuOr_r", norm=SymLogNorm(linthresh=0.5, vmin=-omega_limit, vmax=omega_limit),
            interpolation="nearest", rasterized=True,
        )
        figure.colorbar(image, ax=axes[row, 1], fraction=0.046, pad=0.02)
        axes[row, 1].set_title(r"signed raw-field $\omega_zD/a_\infty$")
        axes[row, 2].imshow(np.log1p(np.maximum(crop.schlieren, 0.0)), origin="lower", extent=[crop.x[0], crop.x[-1], crop.y[0], crop.y[-1]], cmap="Greys", interpolation="nearest", rasterized=True)
        labels_mask = cached["labels"]
        if np.any(labels_mask):
            axes[row, 2].contour(crop.x, crop.y, labels_mask > 0, levels=[0.5], colors=[GOLD], linewidths=1.0)
        axes[row, 2].set_title("topology-supported rotational candidates")
        for axis in axes[row]:
            draw_body(axis, radius)
            axis.set_xlim(-1.35, 3.5)
            axis.set_ylim(-1.5, 1.5)
            axis.set_aspect("equal", adjustable="box")
            axis.set_xlabel(r"$x/D$")
            axis.set_ylabel(r"$y/D$")
    base = figure_dir / "figure_01_three_run_final_fields"
    save_figure(figure, base, dpi)
    manifest.append({"figure": base.name, "role": "final common-grid shock and rotational fields"})

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    for axis, name, label in zip(axes, case_names, labels):
        tracks = analyses[name]["tracks"]
        for track in tracks:
            if not track["persistent_convecting"]:
                continue
            color = VERMILLION if int(track["vorticity_sign"]) > 0 else MAGENTA
            points = track["points"]
            axis.plot([p["x_D"] for p in points], [p["y_D"] for p in points], color=color, linewidth=1.1, alpha=0.8)
        draw_body(axis, radius)
        axis.set_xlim(-0.65, 3.5)
        axis.set_ylim(-1.25, 1.25)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(label)
        axis.set_xlabel(r"$x/D$")
        axis.set_ylabel(r"$y/D$")
        axis.grid(alpha=0.15)
    figure.suptitle("Persistent raw-field candidate tracks: positive (vermillion), negative (magenta)")
    base = figure_dir / "figure_02_three_run_candidate_tracks"
    save_figure(figure, base, dpi)
    manifest.append({"figure": base.name, "role": "persistent candidate trajectories"})

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), constrained_layout=True)
    metrics = [
        ("relative_l2_wake_pressure", "pressure"),
        ("relative_l2_wake_u", r"$u$"),
        ("relative_l2_wake_v", r"$v$"),
        ("relative_l2_wake_vorticity", r"$\omega_z$"),
    ]
    comparison_names = ["solver_time_step", "spatial_grid_at_matched_dt"]
    titles = ["CFL 0.20 vs 0.10 on f90", "f90 vs restricted f180 at matched solver dt"]
    for axis, comparison_name, title in zip(axes, comparison_names, titles):
        subset = [row for row in field_rows if row["comparison"] == comparison_name]
        for color, (key, label) in zip(LINE_COLORS, metrics):
            axis.plot([row["target_time"] for row in subset], [row[key] for row in subset], color=color, linewidth=1.2, label=label)
        axis.set_yscale("log")
        axis.set_xlabel(r"$t a_\infty/D$")
        axis.set_ylabel("symmetric relative wake-field L2")
        axis.set_title(title)
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    base = figure_dir / "figure_03_grid_and_time_step_field_sensitivity"
    save_figure(figure, base, dpi)
    manifest.append({"figure": base.name, "role": "matched-time wake-field sensitivity"})

    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), constrained_layout=True)
    spectrum_names = ["v_x0p75", "v_x1p0", "v_x1p5", "v_x2p0"]
    for axis, name, label in zip(axes, case_names, labels):
        spectral = analyses[name]["report"]["spectral"]
        for color, signal_name in zip(LINE_COLORS, spectrum_names):
            item = spectral["signals"][signal_name]
            axis.plot(item["strouhal"][1:], item["normalized_power"][1:], color=color, linewidth=1.1, label=signal_name.replace("v_", r"$v$ "))
        axis.set_xlim(0.0, 0.7)
        axis.set_xlabel(r"$St=fD/U_\infty$")
        axis.set_ylabel("normalized power")
        axis.set_title(label)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.suptitle("Wake-probe spectra; peaks are audited, not presumed physical")
    base = figure_dir / "figure_04_three_run_wake_spectra"
    save_figure(figure, base, dpi)
    manifest.append({"figure": base.name, "role": "three-run transverse-velocity spectra"})
    write_csv(figure_dir / "figure_manifest.csv", manifest, ["figure", "role"])
    return manifest


def build_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# MFC Euler cylinder: grid and solver-time-step sensitivity",
        "",
        "This is a physics-only audit of raw MFC fields. No neural-network output or human vortex ground truth is used.",
        "The f180 fields are restricted by 2x2 finite-volume cell averaging to the f90 observation grid before diagnostics.",
        "",
        "## Case decisions",
        "",
        "| case | persistent + | persistent - | mirror paired | topology gate | spectral consensus |",
        "|---|---:|---:|---:|:---:|:---:|",
    ]
    for name, item in report["cases"].items():
        d = item["decision"]
        lines.append(
            f"| {name} | {d['persistent_positive_tracks']} | {d['persistent_negative_tracks']} | "
            f"{d['mirror_paired_persistent_tracks']} | {'PASS' if d['alternating_shedding_topology_gate_pass'] else 'FAIL'} | "
            f"{'PASS' if d['spectral_consensus_gate_pass'] else 'FAIL'} |"
        )
    lines += [
        "",
        "## Matched-field sensitivity",
        "",
        "All values are symmetric relative L2 norms in the fixed wake window. The fine field is restricted to the common f90 observation grid first.",
        "",
        "| control | field | samples | median | maximum | final |",
        "|---|---|---:|---:|---:|---:|",
    ]
    short_names = {
        "solver_time_step": "CFL 0.20 vs 0.10 (f90)",
        "spatial_grid_at_matched_dt": "f90 vs f180 (matched dt)",
    }
    field_names = {
        "relative_l2_wake_pressure": "pressure",
        "relative_l2_wake_u": "u",
        "relative_l2_wake_v": "v",
        "relative_l2_wake_vorticity": "vorticity",
    }
    for item in report["matched_field_summary"]:
        lines.append(
            f"| {short_names[item['comparison']]} | {field_names[item['metric']]} | {item['samples']} | "
            f"{item['median']:.4f} | {item['maximum']:.4f} | {item['final']:.4f} |"
        )
    lines += [
        "",
        "## Track repeatability",
        "",
        "| control | persistent tracks (first/second) | matched | symmetric match fraction | median matched distance D |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in report["cross_case_track_matching"].items():
        distance = item["median_matched_trajectory_distance_D"]
        distance_text = "--" if distance is None else f"{distance:.4f}"
        lines.append(
            f"| {short_names[name]} | {item['first_persistent_tracks']}/{item['second_persistent_tracks']} | "
            f"{item['matched_tracks']} | {item['symmetric_match_fraction']:.4f} | {distance_text} |"
        )
    lines += [
        "",
        "## Cross-control result",
        "",
        f"**Conclusion:** `{report['decision']['conclusion']}`",
        "",
        report["decision"]["allowed_wording"],
        "",
        "A periodic Euler shedding claim is allowed only if alternating signed tracks, a common Strouhal peak, and both numerical controls all pass. Rotational candidates may still be shown as an out-of-distribution/numerical-sensitivity audit when this gate fails.",
        "",
        "## Provenance and limitations",
        "",
        "- Shock support uses compression, pressure/density jumps and normal-direction NMS; entropy is not used.",
        "- Wall/solid and dilated bow-shock layers are excluded before vortex-object acceptance.",
        "- Each complete trajectory is a separate leakage-free case group.",
        "- These are Euler/slip-cylinder runs; they are not viscous cylinder ground truth.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "mfc_euler_cylinder_three_run_sensitivity_v1.json",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output_root = ROOT / config["output_dir"]
    figure_dir = ROOT / config["figure_dir"]
    output_root.mkdir(parents=True, exist_ok=True)
    cases = {case["name"]: case for case in config["cases"]}
    analyses: dict[str, dict[str, Any]] = {}
    for case in config["cases"]:
        analyses[case["name"]] = analyze_case(case, config, output_root)
    signatures = {item["report"]["common_observation_grid_sha256"] for item in analyses.values()}
    if len(signatures) != 1:
        raise ValueError("the three cases do not share the same restricted observation grid")

    field_rows: list[dict[str, Any]] = []
    track_matches: dict[str, Any] = {}
    for comparison in config["comparisons"]:
        field_rows.extend(compare_fields(comparison, cases, analyses))
        track_matches[comparison["name"]] = match_tracks(
            analyses[comparison["first"]]["tracks"],
            analyses[comparison["second"]]["tracks"],
            config,
        )
    field_summary = aggregate_field_rows(field_rows)
    write_csv(output_root / "matched_field_comparisons.csv", field_rows, list(field_rows[0].keys()))
    write_csv(output_root / "matched_field_summary.csv", field_summary, list(field_summary[0].keys()))

    case_decisions = {name: item["report"]["decision"] for name, item in analyses.items()}
    topology_all = all(item["alternating_shedding_topology_gate_pass"] for item in case_decisions.values())
    spectral_all = all(item["spectral_consensus_gate_pass"] for item in case_decisions.values())
    strouhal = [
        analyses[name]["report"]["spectral"]["consensus_strouhal_fD_over_uinf"]
        for name in cases
    ]
    common_strouhal_stable = False
    if spectral_all and all(value is not None for value in strouhal):
        frequencies = [float(value) for value in strouhal]
        resolutions = [
            float(analyses[name]["report"]["spectral"]["signals"]["v_x1p0"]["frequency_resolution_ainf_over_D"])
            / float(config["physics"]["u_inf"])
            for name in cases
        ]
        common_strouhal_stable = max(frequencies) - min(frequencies) <= max(resolutions)
    validated = bool(topology_all and spectral_all and common_strouhal_stable)
    conclusion = "VALIDATED_EULER_SHOCK_GENERATED_VORTEX_SHEDDING" if validated else "NOT_VALIDATED_AS_PHYSICAL_EULER_SHEDDING"
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "result_role": "physics_only_raw_field_three_run_sensitivity; no neural outputs; not human ground truth",
        "cases": {
            name: {
                "case_group_id": cases[name]["case_group_id"],
                "snapshot_count": item["report"]["snapshot_count"],
                "final_shock_standoff_D": item["report"]["frames"][-1]["bow_shock_standoff_D"],
                "final_candidate_objects": item["report"]["frames"][-1]["candidate_objects"],
                "final_candidate_pixels": item["report"]["frames"][-1]["candidate_pixels"],
                "decision": item["report"]["decision"],
                "spectral_consensus_strouhal_fD_over_uinf": item["report"]["spectral"]["consensus_strouhal_fD_over_uinf"],
                "structural_qa": item["report"]["structural_qa"],
            }
            for name, item in analyses.items()
        },
        "matched_field_summary": field_summary,
        "cross_case_track_matching": track_matches,
        "decision": {
            "all_three_topology_gates_pass": topology_all,
            "all_three_spectral_consensus_gates_pass": spectral_all,
            "common_strouhal_stable_within_one_frequency_bin": common_strouhal_stable,
            "periodic_euler_shedding_validated": validated,
            "conclusion": conclusion,
            "allowed_wording": (
                "Euler shock-generated vortex shedding supported under grid and solver-time-step controls."
                if validated
                else "Raw-field rotational wake candidates with documented grid/time-step sensitivity; periodic physical Euler shedding is not established."
            ),
        },
        "provenance": {
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "script": str(Path(__file__).resolve()),
            "script_sha256": sha256(Path(__file__).resolve()),
            "outer_package_sha256": config["package"]["outer_zip_sha256"],
            "common_observation_grid_sha256": next(iter(signatures)),
            "fine_grid_restriction": config["common_observation_grid"]["f180_restriction"],
            "entropy_used_for_shock_detection": False,
            "neural_outputs_used": False,
        },
    }
    make_figures(config, analyses, field_rows, figure_dir)
    write_json(output_root / "three_run_sensitivity_report.json", report)
    build_markdown(report, ROOT / "docs" / "MFC_EULER_CYLINDER_THREE_RUN_SENSITIVITY.md")
    print(json.dumps(report["decision"], indent=2), flush=True)


if __name__ == "__main__":
    main()
