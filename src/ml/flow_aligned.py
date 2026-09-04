"""Geometry-agnostic, physically normalized CFD image channels.

The legacy Stage-5 model standardized laboratory-frame ``u`` and ``v`` with
statistics from one Mach-3, alpha=40-degree airfoil trajectory.  That makes a
perfectly ordinary horizontal inflow an extreme out-of-distribution input.
This module instead expresses velocity in a freestream-aligned basis and uses
only nondimensional, fixed transforms.  No statistic is fitted on a test
case.

The four primitive channels are suitable for an ML-only ablation.  The three
bounded invariant diagnostic channels may be appended for the learned hybrid
model.  They are inputs to a neural decision function, not binary labels or
post-hoc accept/reject gates.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

import numpy as np


PRIMITIVE_CHANNELS = (
    "log_rho_over_rho_inf",
    "log_pressure_over_pressure_inf",
    "u_parallel_over_u_inf_minus_one",
    "u_normal_over_u_inf",
)

HYBRID_CHANNELS = (
    *PRIMITIVE_CHANNELS,
    "bounded_vorticity",
    "bounded_q_deviatoric",
    "bounded_lambda_ci",
)


@dataclass(frozen=True)
class FreestreamReference:
    """Freestream and length scales used by the canonical transform."""

    rho_inf: float
    pressure_inf: float
    u_inf_x: float
    u_inf_y: float
    gamma: float = 1.4
    reference_length: float = 1.0
    source: str = "explicit_metadata"

    @property
    def speed_inf(self) -> float:
        return math.hypot(self.u_inf_x, self.u_inf_y)

    @property
    def sound_speed_inf(self) -> float:
        return math.sqrt(self.gamma * self.pressure_inf / self.rho_inf)

    @property
    def mach_inf(self) -> float:
        return self.speed_inf / self.sound_speed_inf

    @property
    def flow_angle_rad(self) -> float:
        return math.atan2(self.u_inf_y, self.u_inf_x)

    def validate(self) -> None:
        values = (
            self.rho_inf,
            self.pressure_inf,
            self.u_inf_x,
            self.u_inf_y,
            self.gamma,
            self.reference_length,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("freestream reference contains a non-finite value")
        if self.rho_inf <= 0.0 or self.pressure_inf <= 0.0:
            raise ValueError("freestream density and pressure must be positive")
        if self.gamma <= 1.0:
            raise ValueError("gamma must exceed one")
        if self.reference_length <= 0.0:
            raise ValueError("reference length must be positive")
        if self.speed_inf <= np.finfo(float).eps:
            raise ValueError("freestream speed must be nonzero")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            speed_inf=self.speed_inf,
            sound_speed_inf=self.sound_speed_inf,
            mach_inf=self.mach_inf,
            flow_angle_deg=math.degrees(self.flow_angle_rad),
        )
        return result


def infer_reference_length(
    x: np.ndarray,
    y: np.ndarray,
    geometry: np.ndarray,
) -> float:
    """Infer a body scale from its larger principal-axis span.

    A user-supplied physical chord/diameter remains preferable.  The inference
    is deterministic, rotation-aware, and adequate for a single airfoil or
    circular cylinder.
    """

    geometry = np.asarray(geometry, dtype=bool)
    if geometry.shape != (len(y), len(x)) or not np.any(geometry):
        raise ValueError("a nonempty geometry mask matching x/y is required")
    rows, columns = np.where(geometry)
    dx = float(np.median(np.diff(np.asarray(x, dtype=float))))
    dy = float(np.median(np.diff(np.asarray(y, dtype=float))))
    coordinates = np.column_stack(
        (
            np.asarray(x, dtype=float)[columns],
            np.asarray(y, dtype=float)[rows],
        )
    )
    centered = coordinates - np.mean(coordinates, axis=0, keepdims=True)
    covariance = centered.T @ centered / max(len(centered) - 1, 1)
    _values, axes = np.linalg.eigh(covariance)
    projected = centered @ axes
    principal_spans = np.ptp(projected, axis=0)
    # Add one representative cell width because the mask stores cell centers.
    scale = float(np.max(principal_spans) + max(abs(dx), abs(dy)))
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("could not infer a positive reference length")
    return scale


def estimate_freestream_from_border(
    rho: np.ndarray,
    pressure: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    *,
    fluid_mask: np.ndarray | None = None,
    gamma: float = 1.4,
    reference_length: float = 1.0,
    border_fraction: float = 0.08,
) -> FreestreamReference:
    """Robustly estimate freestream values from the outer image border.

    Medians make the estimate resistant to a shock or wake crossing part of
    the boundary.  Public inference should prefer explicit solver metadata
    when it is available and record which route was used.
    """

    arrays = [np.asarray(value, dtype=float) for value in (rho, pressure, u, v)]
    if len({value.shape for value in arrays}) != 1 or arrays[0].ndim != 2:
        raise ValueError("rho, pressure, u, and v must be same-shaped 2-D arrays")
    height, width = arrays[0].shape
    band_y = max(1, int(round(height * border_fraction)))
    band_x = max(1, int(round(width * border_fraction)))
    border = np.zeros((height, width), dtype=bool)
    border[:band_y] = True
    border[-band_y:] = True
    border[:, :band_x] = True
    border[:, -band_x:] = True
    if fluid_mask is not None:
        border &= np.asarray(fluid_mask, dtype=bool)
    valid = border.copy()
    for value in arrays:
        valid &= np.isfinite(value)
    valid &= arrays[0] > 0.0
    valid &= arrays[1] > 0.0
    if np.count_nonzero(valid) < 32:
        raise ValueError("insufficient valid outer-border fluid samples")
    reference = FreestreamReference(
        rho_inf=float(np.median(arrays[0][valid])),
        pressure_inf=float(np.median(arrays[1][valid])),
        u_inf_x=float(np.median(arrays[2][valid])),
        u_inf_y=float(np.median(arrays[3][valid])),
        gamma=float(gamma),
        reference_length=float(reference_length),
        source="robust_outer_border_median",
    )
    reference.validate()
    return reference


def velocity_in_freestream_basis(
    u: np.ndarray,
    v: np.ndarray,
    reference: FreestreamReference,
) -> tuple[np.ndarray, np.ndarray]:
    """Return velocity parallel and normal to the freestream direction."""

    reference.validate()
    cosine = reference.u_inf_x / reference.speed_inf
    sine = reference.u_inf_y / reference.speed_inf
    parallel = cosine * np.asarray(u) + sine * np.asarray(v)
    normal = -sine * np.asarray(u) + cosine * np.asarray(v)
    return parallel, normal


def velocity_diagnostics(
    u: np.ndarray,
    v: np.ndarray,
    *,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute signed vorticity, deviatoric Q, and swirling strength."""

    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    du_dy, du_dx = np.gradient(u, dy, dx)
    dv_dy, dv_dx = np.gradient(v, dy, dx)
    vorticity = dv_dx - du_dy
    divergence = du_dx + dv_dy
    du_dx_dev = du_dx - 0.5 * divergence
    dv_dy_dev = dv_dy - 0.5 * divergence
    strain_xy = 0.5 * (du_dy + dv_dx)
    rotation_xy = 0.5 * (du_dy - dv_dx)
    q_deviatoric = 0.5 * (
        2.0 * rotation_xy * rotation_xy
        - du_dx_dev * du_dx_dev
        - dv_dy_dev * dv_dy_dev
        - 2.0 * strain_xy * strain_xy
    )
    discriminant = (0.5 * (du_dx - dv_dy)) ** 2 + du_dy * dv_dx
    lambda_ci = np.sqrt(np.maximum(-discriminant, 0.0))
    return vorticity, q_deviatoric, lambda_ci


def canonical_channels(
    rho: np.ndarray,
    pressure: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    reference: FreestreamReference,
    *,
    dx: float | None = None,
    dy: float | None = None,
    vorticity: np.ndarray | None = None,
    q_deviatoric: np.ndarray | None = None,
    lambda_ci: np.ndarray | None = None,
    include_diagnostics: bool = True,
) -> np.ndarray:
    """Build fixed, physically normalized channels with no fitted statistics."""

    reference.validate()
    rho = np.asarray(rho, dtype=np.float64)
    pressure = np.asarray(pressure, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    if len({value.shape for value in (rho, pressure, u, v)}) != 1:
        raise ValueError("all primitive fields must have the same shape")
    tiny = np.finfo(np.float64).tiny
    parallel, normal = velocity_in_freestream_basis(u, v, reference)
    primitive = [
        np.clip(np.log(np.maximum(rho, tiny) / reference.rho_inf), -4.0, 4.0) / 2.0,
        np.clip(
            np.log(np.maximum(pressure, tiny) / reference.pressure_inf), -6.0, 6.0
        )
        / 3.0,
        np.clip(parallel / reference.speed_inf - 1.0, -2.0, 2.0),
        np.clip(normal / reference.speed_inf, -2.0, 2.0),
    ]
    if not include_diagnostics:
        return np.ascontiguousarray(np.stack(primitive).astype(np.float32))

    if vorticity is None or q_deviatoric is None or lambda_ci is None:
        if dx is None or dy is None:
            raise ValueError("dx and dy are required when diagnostics are not supplied")
        vorticity, q_deviatoric, lambda_ci = velocity_diagnostics(u, v, dx=dx, dy=dy)
    time_scale = reference.reference_length / reference.speed_inf
    omega_nd = np.asarray(vorticity, dtype=np.float64) * time_scale
    q_nd = np.asarray(q_deviatoric, dtype=np.float64) * time_scale * time_scale
    lambda_nd = np.asarray(lambda_ci, dtype=np.float64) * time_scale
    diagnostics = [
        np.tanh(omega_nd / 8.0),
        np.tanh(q_nd / 24.0),
        np.tanh(lambda_nd / 8.0),
    ]
    result = np.stack([*primitive, *diagnostics]).astype(np.float32)
    if not np.isfinite(result).all():
        raise ValueError("canonical channels contain a non-finite value")
    return np.ascontiguousarray(result)
