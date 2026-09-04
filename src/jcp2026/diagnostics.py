"""One audited 2-D gradient convention for every input family and deployment."""
from __future__ import annotations

import numpy as np
from ml.flow_aligned import FreestreamReference, canonical_channels


def gradient_diagnostics(u: np.ndarray, v: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    u, v, x, y = [np.asarray(a, dtype=np.float64) for a in (u, v, x, y)]
    if x.ndim != 1 or y.ndim != 1 or min(len(x), len(y)) < 3:
        raise ValueError("At least three 1-D coordinates per axis are required")
    if u.shape != v.shape or u.shape != (len(y), len(x)):
        raise ValueError("Velocity/coordinate shape mismatch")
    if not all(np.isfinite(a).all() for a in (u, v, x, y)):
        raise ValueError("Non-finite velocity or coordinates")
    if not (np.all(np.diff(x) > 0) and np.all(np.diff(y) > 0)):
        raise ValueError("Coordinates must be strictly increasing; reorder fields with coordinates first")
    uy, ux = np.gradient(u, y, x, edge_order=1)
    vy, vx = np.gradient(v, y, x, edge_order=1)
    divergence = ux + vy
    # Qdev = .5 (||Omega||_F^2 - ||S - tr(A) I/2||_F^2).
    # Evaluating the 2-D form avoids subtracting large trace terms.
    qdev = -0.25 * (ux - vy) ** 2 - uy * vx
    return {
        "vorticity": vx - uy,
        "q_deviatoric": qdev,
        "q_legacy": -0.5 * (ux * ux + vy * vy + 2.0 * uy * vx),
        "lambda_ci": np.sqrt(np.maximum(qdev, 0.0)),
        "divergence": divergence,
    }


def build_inputs(fields: dict, reference: FreestreamReference) -> tuple[np.ndarray, dict]:
    diagnostics = gradient_diagnostics(fields["u"], fields["v"], fields["x"], fields["y"])
    channels = canonical_channels(
        fields["rho"], fields["pressure"], fields["u"], fields["v"], reference,
        vorticity=diagnostics["vorticity"], q_deviatoric=diagnostics["q_deviatoric"],
        lambda_ci=diagnostics["lambda_ci"],
    )
    return channels, diagnostics
