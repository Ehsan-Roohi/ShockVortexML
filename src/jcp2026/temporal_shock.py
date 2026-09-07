"""Geometry-agnostic temporal hysteresis for learned shock probabilities.

This is a development-stage decoder.  It never uses a case ROI, schlieren
ridge, pressure gradient, or a geometry-specific direction.  High-confidence
neural pixels seed components grown through temporally persistent lower-score
neural support.  Exact solids and a fixed wall-clearance band remain excluded.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def decode_temporal_hysteresis(
    probabilities: list[np.ndarray],
    domains: list[np.ndarray],
    geometries: list[np.ndarray],
    *,
    seed_threshold: float = 0.97,
    support_threshold: float = 0.55,
    temporal_radius: int = 1,
    closing_iterations: int = 2,
    wall_clearance_cells: int = 2,
    minimum_component_pixels: int = 9,
) -> list[np.ndarray]:
    """Decode a sequence while retaining only components with neural seeds."""
    if not (len(probabilities) == len(domains) == len(geometries)) or not probabilities:
        raise ValueError("probabilities, domains and geometries must have equal nonzero length")
    shape = probabilities[0].shape
    if any(a.shape != shape for seq in (probabilities, domains, geometries) for a in seq):
        raise ValueError("all sequence arrays must have one common shape")
    structure = np.ones((3, 3), dtype=bool)
    answers: list[np.ndarray] = []
    for i, probability in enumerate(probabilities):
        lo, hi = max(0, i-temporal_radius), min(len(probabilities), i+temporal_radius+1)
        persistent = np.median(np.stack(probabilities[lo:hi]), axis=0)
        allowed = domains[i].astype(bool) & ~ndi.binary_dilation(
            geometries[i].astype(bool), iterations=wall_clearance_cells)
        seeds = (probability >= seed_threshold) & allowed
        support = (persistent >= support_threshold) & allowed
        if closing_iterations > 0:
            support = ndi.binary_closing(support, structure=structure,
                                         iterations=closing_iterations)
        support &= allowed
        labels, count = ndi.label(support, structure)
        seeded = np.unique(labels[seeds])
        seeded = seeded[seeded != 0]
        result = np.isin(labels, seeded)
        if count:
            sizes = np.bincount(labels.ravel())
            result &= sizes[labels] >= minimum_component_pixels
        answers.append(result & allowed)
    return answers
