"""Physics-based weak-label proposals for shock and vortex annotation.

These functions create reviewable proposals and continuous audit scores.  They
do not create human ground truth and are not the final ML system.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


@dataclass(frozen=True)
class FlowDerivatives:
    grad_log_rho: np.ndarray
    grad_log_pressure: np.ndarray
    compression: np.ndarray
    divergence: np.ndarray
    vorticity: np.ndarray
    q_deviatoric: np.ndarray
    lambda_ci: np.ndarray
    pressure_normal_x: np.ndarray
    pressure_normal_y: np.ndarray


@dataclass(frozen=True)
class ShockProposal:
    centerline: np.ndarray
    band: np.ndarray
    uncertainty: np.ndarray
    score: np.ndarray
    jump_log_pressure: np.ndarray
    jump_log_density: np.ndarray


@dataclass(frozen=True)
class VortexProposal:
    core: np.ndarray
    uncertainty: np.ndarray
    instances: np.ndarray
    score: np.ndarray
    objects: list[dict[str, float | int | bool | str]]


def derivatives(
    rho: np.ndarray,
    pressure: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> FlowDerivatives:
    """Compute entropy-free compression and rotational diagnostics."""

    log_rho = np.log(np.maximum(rho, np.finfo(np.float64).tiny))
    log_pressure = np.log(np.maximum(pressure, np.finfo(np.float64).tiny))
    dlrho_dy, dlrho_dx = np.gradient(log_rho, dy, dx, edge_order=2)
    dlp_dy, dlp_dx = np.gradient(log_pressure, dy, dx, edge_order=2)
    du_dy, du_dx = np.gradient(u, dy, dx, edge_order=2)
    dv_dy, dv_dx = np.gradient(v, dy, dx, edge_order=2)

    grad_p = np.hypot(dlp_dx, dlp_dy)
    normal_x = np.divide(dlp_dx, grad_p, out=np.zeros_like(grad_p), where=grad_p > 0)
    normal_y = np.divide(dlp_dy, grad_p, out=np.zeros_like(grad_p), where=grad_p > 0)
    divergence = du_dx + dv_dy
    vorticity = dv_dx - du_dy
    sxx_dev = du_dx - 0.5 * divergence
    syy_dev = dv_dy - 0.5 * divergence
    sxy = 0.5 * (du_dy + dv_dx)
    omega_xy = 0.5 * (du_dy - dv_dx)
    strain_dev_sq = sxx_dev**2 + syy_dev**2 + 2.0 * sxy**2
    rotation_sq = 2.0 * omega_xy**2
    q_deviatoric = 0.5 * (rotation_sq - strain_dev_sq)
    discriminant = (du_dx - dv_dy) ** 2 + 4.0 * du_dy * dv_dx
    lambda_ci = 0.5 * np.sqrt(np.maximum(-discriminant, 0.0))
    return FlowDerivatives(
        grad_log_rho=np.hypot(dlrho_dx, dlrho_dy),
        grad_log_pressure=grad_p,
        compression=np.maximum(-divergence, 0.0),
        divergence=divergence,
        vorticity=vorticity,
        q_deviatoric=q_deviatoric,
        lambda_ci=lambda_ci,
        pressure_normal_x=normal_x,
        pressure_normal_y=normal_y,
    )


def _normal_samples(
    field: np.ndarray,
    normal_x: np.ndarray,
    normal_y: np.ndarray,
    offset_pixels: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = np.indices(field.shape, dtype=np.float64)
    plus = ndimage.map_coordinates(
        field,
        [rows + offset_pixels * normal_y, cols + offset_pixels * normal_x],
        order=1,
        mode="nearest",
    )
    minus = ndimage.map_coordinates(
        field,
        [rows - offset_pixels * normal_y, cols - offset_pixels * normal_x],
        order=1,
        mode="nearest",
    )
    return plus, minus


def shock_proposal(
    rho: np.ndarray,
    pressure: np.ndarray,
    flow: FlowDerivatives,
    geometry_mask: np.ndarray,
    *,
    dx: float,
    dy: float,
    jump_offset_pixels: float = 2.0,
    leading_edge_anchor_distance: float = 0.18,
) -> ShockProposal:
    """Build a thin entropy-free shock proposal with normal-direction NMS.

    ``leading_edge_anchor_distance`` is expressed in the same nondimensional
    physical coordinates as ``dx`` and ``dy``.  The historical default is
    retained for the diamond-airfoil datasets.  Blunt bodies with a detached
    bow shock may pass a larger, fixed geometry-specific distance; this does
    not alter the compression/jump/NMS evidence or tune a test-frame quantile.
    """

    log_rho = np.log(np.maximum(rho, np.finfo(np.float64).tiny))
    log_pressure = np.log(np.maximum(pressure, np.finfo(np.float64).tiny))
    p_plus, p_minus = _normal_samples(
        log_pressure, flow.pressure_normal_x, flow.pressure_normal_y, jump_offset_pixels
    )
    r_plus, r_minus = _normal_samples(
        log_rho, flow.pressure_normal_x, flow.pressure_normal_y, jump_offset_pixels
    )
    jump_p = np.abs(p_plus - p_minus)
    jump_rho = np.abs(r_plus - r_minus)

    grad_plus, grad_minus = _normal_samples(
        flow.grad_log_pressure, flow.pressure_normal_x, flow.pressure_normal_y, 1.0
    )
    nms = (flow.grad_log_pressure >= grad_plus) & (flow.grad_log_pressure >= grad_minus)

    compression_support = np.clip(flow.compression / 8.0, 0.0, 1.0)
    pressure_support = np.clip(jump_p / 0.20, 0.0, 1.0)
    density_support = np.clip(jump_rho / 0.12, 0.0, 1.0)
    gradient_support = np.clip(flow.grad_log_pressure / 18.0, 0.0, 1.0)
    score = (
        compression_support
        * np.sqrt(pressure_support * density_support)
        * np.sqrt(gradient_support)
    )

    wall_distance = ndimage.distance_transform_edt(~geometry_mask, sampling=(dy, dx))
    geometry_exclusion = geometry_mask | (wall_distance < 0.012)
    base_support = (
        (flow.compression >= 1.5)
        & (jump_p >= 0.055)
        & (jump_rho >= 0.025)
        & (flow.grad_log_pressure >= 2.0)
        & nms
        & ~geometry_exclusion
    )
    # Pressure rings around vortices can contain compression and jumps but are
    # not safe shock labels.  Only promote pixels where compression is not
    # dominated by rotation.  Rotationally dominated pixels remain uncertain,
    # which preserves possible shock-vortex overlap for human review.
    rotation_dominated = (
        (flow.compression < 0.20 * np.abs(flow.vorticity))
        | (flow.compression < 0.30 * flow.lambda_ci)
    )
    hard_support = base_support & ~rotation_dominated
    # Join subpixel NMS gaps along an otherwise coherent ridge, then thin again.
    joined = ndimage.binary_closing(hard_support, structure=np.ones((3, 3)), iterations=1)
    provisional_centerline = joined & nms
    labels, count = ndimage.label(
        provisional_centerline, structure=np.ones((3, 3), dtype=np.uint8)
    )
    centerline = np.zeros_like(provisional_centerline)
    geometry_rows, geometry_cols = np.where(geometry_mask)
    if geometry_cols.size:
        leading_col = int(geometry_cols.min())
        leading_rows = geometry_rows[geometry_cols <= leading_col + 1]
        leading_row = int(np.median(leading_rows))
    else:
        leading_col = leading_row = -10_000
    if count:
        for component_id in range(1, count + 1):
            rows, cols = np.where(labels == component_id)
            pixel_count = int(cols.size)
            extent = float(
                np.hypot((cols.max() - cols.min()) * dx, (rows.max() - rows.min()) * dy)
            )
            touches_window = bool(
                rows.min() <= 2
                or cols.min() <= 2
                or rows.max() >= centerline.shape[0] - 3
                or cols.max() >= centerline.shape[1] - 3
            )
            leading_edge_distance = float(
                np.min(
                    np.hypot(
                        (cols - leading_col) * dx,
                        (rows - leading_row) * dy,
                    )
                )
            )
            anchored_to_leading_edge = (
                leading_edge_distance <= leading_edge_anchor_distance
            )
            # Only long coherent fronts are high-confidence weak labels. Short
            # shocklets and closed pressure rings are retained as uncertain so
            # the network is never taught that they are definitely background.
            if (
                pixel_count >= 100
                and extent >= 1.25
                and (touches_window or anchored_to_leading_edge)
            ):
                centerline[rows, cols] = True
    demoted_short_fronts = provisional_centerline & ~centerline
    band = ndimage.binary_dilation(centerline, iterations=2)
    partial_support = (
        (score >= 0.035)
        & (flow.grad_log_pressure >= 1.0)
        & ~geometry_mask
    )
    uncertainty = partial_support & ~band
    uncertainty |= base_support & rotation_dominated & ~band
    uncertainty |= demoted_short_fronts & ~band
    uncertainty |= (
        (flow.grad_log_pressure >= 5.0)
        & (wall_distance < 0.04)
        & ~geometry_mask
        & ~band
    )
    return ShockProposal(
        centerline=centerline,
        band=band,
        uncertainty=uncertainty,
        score=np.asarray(score, dtype=np.float32),
        jump_log_pressure=np.asarray(jump_p, dtype=np.float32),
        jump_log_density=np.asarray(jump_rho, dtype=np.float32),
    )


def vortex_proposal(
    flow: FlowDerivatives,
    geometry_mask: np.ndarray,
    shock: ShockProposal,
    x: np.ndarray,
    y: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> VortexProposal:
    """Create topology-oriented vortex-core candidates without shock vetoes."""

    rotational_support = np.clip(flow.lambda_ci / 12.0, 0.0, 1.0)
    q_support = np.clip(np.maximum(flow.q_deviatoric, 0.0) / 100.0, 0.0, 1.0)
    omega_support = np.clip(np.abs(flow.vorticity) / 25.0, 0.0, 1.0)
    score = rotational_support * np.sqrt(q_support * omega_support)
    wall_distance = ndimage.distance_transform_edt(~geometry_mask, sampling=(dy, dx))

    seed = (
        (flow.lambda_ci >= 5.0)
        & (flow.q_deviatoric >= 25.0)
        & (np.abs(flow.vorticity) >= 10.0)
        & ~geometry_mask
    )
    labels, count = ndimage.label(seed, structure=np.ones((3, 3), dtype=np.uint8))
    yy, xx = np.meshgrid(y, x, indexing="ij")
    core = np.zeros_like(seed)
    uncertain = np.zeros_like(seed)
    instances = np.zeros(seed.shape, dtype=np.int32)
    objects: list[dict[str, float | int | bool | str]] = []
    for component_id in range(1, count + 1):
        component = labels == component_id
        pixels = int(np.count_nonzero(component))
        radius = float(np.sqrt(pixels * dx * dy / np.pi))
        if pixels < 5 or radius < 0.015:
            continue
        weights = flow.lambda_ci[component]
        weight_sum = float(weights.sum())
        cx = float(np.sum(xx[component] * weights) / weight_sum)
        cy = float(np.sum(yy[component] * weights) / weight_sum)
        center_i = int(np.argmin(np.abs(x - cx)))
        center_j = int(np.argmin(np.abs(y - cy)))
        center_wall_distance = float(wall_distance[center_j, center_i])
        near_wall = center_wall_distance < 0.05
        shock_overlap = bool(np.any(shock.band[component]))
        classification = (
            "near_wall_unresolved_rotation_or_shear"
            if near_wall
            else "off_wall_rotational_candidate"
        )
        if near_wall:
            uncertain |= component
        else:
            core |= component
            instances[component] = component_id
        mean_omega = float(np.mean(flow.vorticity[component]))
        objects.append(
            {
                "component_id": int(component_id),
                "center_x": cx,
                "center_y": cy,
                "equivalent_radius": radius,
                "pixels": pixels,
                "vorticity_sign": int(np.sign(mean_omega)),
                "mean_vorticity": mean_omega,
                "peak_lambda_ci": float(np.max(flow.lambda_ci[component])),
                "peak_q_deviatoric": float(np.max(flow.q_deviatoric[component])),
                "center_wall_distance": center_wall_distance,
                "near_wall": near_wall,
                "shock_overlap": shock_overlap,
                "triage_class": classification,
                "review_status": "unreviewed_pseudo_label",
            }
        )
    objects.sort(key=lambda item: float(item["peak_lambda_ci"]), reverse=True)
    return VortexProposal(
        core=core,
        uncertainty=uncertain,
        instances=instances,
        score=np.asarray(score, dtype=np.float32),
        objects=objects,
    )
