"""Case-wise, physically normalized data loader for geometry-robust training.

The loader deliberately keeps the immutable airfoil tensors in place and
loads the materialized canonical cylinder records produced by
``build_geometry_robust_v2_dataset.py``.  Split membership is read from the
dataset index; a trajectory/case group is never divided between splits.

All targets are weak proposals until a named expert freezes the corresponding
annotation pack.  The class therefore exposes ``annotation_status`` in every
sample and never calls these arrays ground truth.
"""

from __future__ import annotations

from collections import OrderedDict, defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from ml.flow_aligned import FreestreamReference, canonical_channels
from ml.stage5_model import Stage5JointNet


FOCUS_NAMES = (
    "shock",
    "vortex_core",
    "hard_negative",
    "wall",
    "uncertain",
    "uniform",
)


def _reference(record: dict[str, Any]) -> FreestreamReference:
    values = record["reference"]
    return FreestreamReference(
        rho_inf=float(values["rho_inf"]),
        pressure_inf=float(values["pressure_inf"]),
        u_inf_x=float(values["u_inf_x"]),
        u_inf_y=float(values["u_inf_y"]),
        gamma=float(values.get("gamma", 1.4)),
        reference_length=float(values["reference_length"]),
        source=str(values.get("source", "dataset_index")),
    )


def _airfoil_frame(record: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(Path(record["source_frame"])) as data:
        fields = data["fields"].astype(np.float32)
        names = [str(value) for value in data["field_names"].tolist()]
        lookup = {name: fields[index] for index, name in enumerate(names)}
        geometry = data["body_mask"].astype(bool)
        fluid = data["fluid_mask"].astype(bool)
        base_valid = data["label_valid_mask"].astype(bool) & fluid
        shock = data["shock_mask"].astype(bool) & fluid
        vortex = data["vortex_instances"].astype(np.uint16) > 0
        vortex_score = np.maximum(
            np.abs(data["vortex_positive_heatmap"].astype(np.float32)),
            np.abs(data["vortex_negative_heatmap"].astype(np.float32)),
        )
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)

    diagnostics = canonical_channels(
        lookup["rho"],
        lookup["pressure"],
        lookup["u"],
        lookup["v"],
        _reference(record),
        vorticity=lookup["omega_z"],
        q_deviatoric=lookup["q_criterion"],
        lambda_ci=lookup["lambda_ci"],
    )
    # Fixed spatial widths are proposal-boundary ignore zones, not per-frame
    # thresholds.  They keep weak ridge/core edges from dominating the loss.
    # Ignore only the exterior localization tolerance.  The seed itself must
    # remain a valid positive even when it is a one-cell ridge/core.
    shock_boundary = ndimage.binary_dilation(shock, iterations=3) & ~shock
    vortex_boundary = ndimage.binary_dilation(vortex, iterations=3) & ~vortex
    weak_vortex_ambiguity = (vortex_score >= 0.15) & ~vortex
    shock_uncertain = (shock_boundary | (~base_valid & fluid)) & ~geometry
    vortex_uncertain = (
        vortex_boundary
        | weak_vortex_ambiguity
        | ((shock & ndimage.binary_dilation(vortex, iterations=5)) & ~vortex)
    ) & fluid & ~geometry

    wall_band = ndimage.binary_dilation(geometry, iterations=10) & fluid
    # Channel 4/5/6 are fixed bounded nondimensional rotation diagnostics.
    coherent_rotation = (
        (np.abs(diagnostics[4]) >= np.tanh(0.5 / 8.0))
        & (diagnostics[5] >= np.tanh(0.02 / 24.0))
        & (diagnostics[6] >= np.tanh(0.2 / 8.0))
    )
    shear_like = (np.abs(diagnostics[4]) >= np.tanh(0.5 / 8.0)) & ~coherent_rotation
    hard_negative = ((wall_band | shear_like) & ~vortex & ~vortex_uncertain) | geometry

    shock &= ~geometry
    vortex &= ~geometry
    shock_valid = base_valid & ~shock_uncertain
    vortex_valid = base_valid & ~vortex_uncertain
    if not bool(record.get("shock_loss_eligible", True)):
        shock_valid[:] = False
    else:
        shock_valid[shock] = True
    # A topology-supported positive remains a positive even when it overlaps a
    # generic wall/shock ignore band; only the exterior ambiguity is ignored.
    vortex_valid[vortex] = True
    # Exact masks are loss supervision only; they are never model inputs.
    shock_valid[geometry] = True
    vortex_valid[geometry] = True
    shock[geometry] = False
    vortex[geometry] = False
    shock_uncertain[geometry] = False
    vortex_uncertain[geometry] = False
    background = ~(shock | vortex | geometry | shock_uncertain | vortex_uncertain)
    background_valid = shock_valid & vortex_valid & ~geometry
    every_pixel = np.ones_like(geometry, dtype=bool)
    targets = np.stack(
        (shock, vortex, background, geometry, shock_uncertain, vortex_uncertain)
    ).astype(np.float32)
    valid = np.stack(
        (shock_valid, vortex_valid, background_valid, every_pixel, every_pixel, every_pixel)
    ).astype(np.float32)
    return {
        "inputs": diagnostics,
        "targets": targets,
        "valid": valid,
        "geometry": geometry,
        "wall_band": wall_band,
        "hard_negative": hard_negative,
        "x": x,
        "y": y,
    }


def apply_temporal_vortex_override(
    *,
    vortex: np.ndarray,
    vortex_uncertain: np.ndarray,
    hard_negative: np.ndarray,
    geometry: np.ndarray,
    temporal_positive: np.ndarray,
    recall_candidate: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge a machine-reviewed temporal override without inventing negatives.

    Persistent learned candidates expand the weak positive set.  Rejected
    two-level candidates become ignore pixels, not background labels.  This is
    self-training supervision and must never be described as human ground truth.
    """

    if temporal_positive.shape != vortex.shape or recall_candidate.shape != vortex.shape:
        raise ValueError("Temporal vortex override shape does not match the source frame")
    fluid = ~geometry
    positive = (vortex | temporal_positive.astype(bool)) & fluid
    uncertain = (vortex_uncertain | recall_candidate.astype(bool)) & fluid & ~positive
    negatives = hard_negative.astype(bool) & ~positive & ~uncertain
    return positive, uncertain, negatives


def _cylinder_frame(
    record: dict[str, Any], override_path: Path | None = None
) -> dict[str, np.ndarray]:
    with np.load(Path(record["source_frame"])) as data:
        inputs = data["input_fields"].astype(np.float32)
        geometry = data["geometry_mask"].astype(bool)
        shock = data["shock_mask"].astype(bool) & ~geometry
        vortex = data["vortex_mask"].astype(bool) & ~geometry
        shock_uncertain = data["shock_uncertain"].astype(bool) & ~geometry
        vortex_uncertain = data["vortex_uncertain"].astype(bool) & ~geometry
        hard_negative = data["hard_negative_mask"].astype(bool)
        wall_band = data["wall_band"].astype(bool)
        x = data["x"].astype(np.float32)
        y = data["y"].astype(np.float32)

    if override_path is not None:
        with np.load(override_path, allow_pickle=False) as override:
            vortex, vortex_uncertain, hard_negative = apply_temporal_vortex_override(
                vortex=vortex,
                vortex_uncertain=vortex_uncertain,
                hard_negative=hard_negative,
                geometry=geometry,
                temporal_positive=override["temporal_positive"].astype(bool),
                recall_candidate=override["recall_candidate"].astype(bool),
            )

    fluid = ~geometry
    shock_valid = fluid & ~shock_uncertain
    vortex_valid = fluid & ~vortex_uncertain
    if not bool(record.get("shock_loss_eligible", True)):
        shock_valid[:] = False
    shock_valid[geometry] = True
    vortex_valid[geometry] = True
    background = ~(shock | vortex | geometry | shock_uncertain | vortex_uncertain)
    background_valid = shock_valid & vortex_valid & fluid
    every_pixel = np.ones_like(geometry, dtype=bool)
    targets = np.stack(
        (shock, vortex, background, geometry, shock_uncertain, vortex_uncertain)
    ).astype(np.float32)
    valid = np.stack(
        (shock_valid, vortex_valid, background_valid, every_pixel, every_pixel, every_pixel)
    ).astype(np.float32)
    return {
        "inputs": np.ascontiguousarray(inputs),
        "targets": np.ascontiguousarray(targets),
        "valid": np.ascontiguousarray(valid),
        "geometry": geometry,
        "wall_band": wall_band,
        "hard_negative": hard_negative,
        "x": x,
        "y": y,
    }


class GeometryRobustFrames:
    """Load one case-wise split and sample case-balanced focused patches."""

    def __init__(
        self,
        index_path: Path,
        *,
        split: str,
        cache_size: int = 8,
        override_index_path: Path | None = None,
    ) -> None:
        self.index_path = index_path.resolve()
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.frames = [record for record in self.index["records"] if record["split"] == split]
        if not self.frames:
            raise ValueError(f"No records found for split {split!r}")
        self.split = split
        self.input_channels = tuple(self.index["input_channels"])
        self.cache_size = int(cache_size)
        self.override_index_path = (
            override_index_path.resolve() if override_index_path is not None else None
        )
        self.override_index: dict[str, Any] | None = None
        self.overrides: dict[str, dict[str, Any]] = {}
        if self.override_index_path is not None:
            self.override_index = json.loads(
                self.override_index_path.read_text(encoding="utf-8")
            )
            self.overrides = {
                str(item["dataset_id"]): item
                for item in self.override_index["records"]
            }
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()
        grouped: dict[str, list[int]] = defaultdict(list)
        for index, record in enumerate(self.frames):
            grouped[str(record["leakage_group_id"])].append(index)
        self.group_names = tuple(sorted(grouped))
        self.grouped_indices = {name: tuple(grouped[name]) for name in self.group_names}
        self.group_family = {
            name: str(self.frames[indices[0]]["family"])
            for name, indices in self.grouped_indices.items()
        }
        family_groups: dict[str, list[str]] = defaultdict(list)
        for name in self.group_names:
            family_groups[self.group_family[name]].append(name)
        self.family_names = tuple(sorted(family_groups))
        self.family_groups = {
            family: tuple(groups) for family, groups in family_groups.items()
        }

    def __len__(self) -> int:
        return len(self.frames)

    def load(self, frame_index: int) -> dict[str, Any]:
        if frame_index in self._cache:
            loaded = self._cache.pop(frame_index)
            self._cache[frame_index] = loaded
            return loaded
        record = self.frames[frame_index]
        if record["source_kind"] == "mfc_cv_full_npz":
            loaded = _airfoil_frame(record)
        elif record["source_kind"] == "canonical_cylinder_npz":
            override = self.overrides.get(str(record["dataset_id"]))
            loaded = _cylinder_frame(
                record,
                Path(override["override_file"]) if override is not None else None,
            )
        else:
            raise ValueError(f"Unsupported source_kind: {record['source_kind']}")
        if loaded["inputs"].shape[0] != len(self.input_channels):
            raise RuntimeError("Canonical input-channel contract is inconsistent")
        if loaded["targets"].shape[0] != len(Stage5JointNet.output_names):
            raise RuntimeError("Target/head contract is inconsistent")
        override = self.overrides.get(str(record["dataset_id"]))
        loaded.update(
            record=record,
            frame_index=frame_index,
            annotation_status=(
                str(override["annotation_status"])
                if override is not None else record["annotation_status"]
            ),
            label_override=override,
        )
        self._cache[frame_index] = loaded
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return loaded

    def sample_patch(
        self,
        rng: np.random.Generator,
        *,
        patch_size: int,
        focus_probabilities: tuple[float, ...],
        case_balanced: bool = True,
        family_balanced: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        probabilities = np.asarray(focus_probabilities, dtype=np.float64)
        if len(probabilities) != len(FOCUS_NAMES) or not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("focus probabilities must match FOCUS_NAMES and sum to one")
        focus = FOCUS_NAMES[int(rng.choice(len(FOCUS_NAMES), p=probabilities))]

        def choose_frame() -> int:
            if family_balanced:
                family = self.family_names[int(rng.integers(len(self.family_names)))]
                family_groups = self.family_groups[family]
                group = family_groups[int(rng.integers(len(family_groups)))]
                choices = self.grouped_indices[group]
                return int(choices[int(rng.integers(len(choices)))])
            if case_balanced:
                group = self.group_names[int(rng.integers(len(self.group_names)))]
                choices = self.grouped_indices[group]
                return int(choices[int(rng.integers(len(choices)))])
            return int(rng.integers(len(self.frames)))

        frame_index = choose_frame()
        loaded = self.load(frame_index)
        height, width = loaded["targets"].shape[-2:]
        if patch_size > min(height, width):
            raise ValueError("patch size exceeds frame dimensions")
        focus_mask = np.zeros((height, width), dtype=bool)
        for _ in range(16):
            if focus == "shock":
                focus_mask = (loaded["targets"][0] > 0.5) & (loaded["valid"][0] > 0.5)
            elif focus == "vortex_core":
                focus_mask = (loaded["targets"][1] > 0.5) & (loaded["valid"][1] > 0.5)
            elif focus == "hard_negative":
                focus_mask = loaded["hard_negative"]
            elif focus == "wall":
                focus_mask = loaded["wall_band"]
            elif focus == "uncertain":
                focus_mask = (loaded["targets"][4] > 0.5) | (loaded["targets"][5] > 0.5)
            else:
                break
            if np.any(focus_mask):
                break
            frame_index = choose_frame()
            loaded = self.load(frame_index)
            height, width = loaded["targets"].shape[-2:]

        locations = np.argwhere(focus_mask)
        if locations.size:
            center_y, center_x = locations[int(rng.integers(len(locations)))]
            jitter = max(1, patch_size // 5)
            center_y += int(rng.integers(-jitter, jitter + 1))
            center_x += int(rng.integers(-jitter, jitter + 1))
            top = int(np.clip(center_y - patch_size // 2, 0, height - patch_size))
            left = int(np.clip(center_x - patch_size // 2, 0, width - patch_size))
        else:
            top = int(rng.integers(0, height - patch_size + 1))
            left = int(rng.integers(0, width - patch_size + 1))
        rows = slice(top, top + patch_size)
        columns = slice(left, left + patch_size)
        metadata = {
            "frame_index": frame_index,
            "dataset_id": loaded["record"]["dataset_id"],
            "family": loaded["record"]["family"],
            "case": loaded["record"]["case"],
            "focus": focus,
            "top": top,
            "left": left,
        }
        return (
            np.ascontiguousarray(loaded["inputs"][:, rows, columns]),
            np.ascontiguousarray(loaded["targets"][:, rows, columns]),
            np.ascontiguousarray(loaded["valid"][:, rows, columns]),
            metadata,
        )
