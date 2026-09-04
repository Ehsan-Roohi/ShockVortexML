"""Weak-supervision dataset for the Stage-5 joint dense network."""

from __future__ import annotations

from collections import OrderedDict
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

from ml.stage5_model import Stage5JointNet


FOCUS_NAMES = (
    "shock",
    "vortex_core",
    "shock_uncertainty",
    "vortex_uncertainty",
    "wall_hard_negative",
    "uniform",
)
FOCUS_TO_HEAD = {0: 0, 1: 1, 2: 4, 3: 5}


class Stage5WeakFrames:
    """Load primitive inputs and six independent weak-label targets.

    All frames from a trajectory are kept in one group.  ``training_only``
    excludes initialization and other explicitly ineligible frames but never
    creates a within-trajectory validation split.
    """

    def __init__(
        self,
        index_path: Path,
        *,
        input_channels: list[str] | tuple[str, ...],
        input_clip: float | None = None,
        cache_size: int = 6,
        training_only: bool = True,
    ) -> None:
        self.index_path = index_path.resolve()
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        all_frames = self.index["frames"]
        self.frames = (
            [
                frame
                for frame in all_frames
                if bool(frame.get("training_eligible_by_default", True))
            ]
            if training_only
            else list(all_frames)
        )
        self.input_channels = list(input_channels)
        if "geometry_mask" in self.input_channels:
            raise ValueError(
                "Stage-5 geometry must remain an auxiliary output, not an input shortcut"
            )
        unsupported = set(self.input_channels) - set(self.index["input_channels"])
        if unsupported:
            raise ValueError(
                "Stage-5 ML-only inputs must be primitive fields; unsupported: "
                + ", ".join(sorted(unsupported))
            )
        stats = self.index["normalization_all_frames_for_inspection_only"]
        primitive_names = self.index["input_channels"]
        if isinstance(stats, dict) and "mean" in stats:
            lookup = {
                name: {
                    "mean": stats["mean"][channel],
                    "std": stats["std"][channel],
                }
                for channel, name in enumerate(primitive_names)
            }
        else:
            lookup = stats
        self.mean = np.asarray(
            [lookup[name]["mean"] for name in self.input_channels], dtype=np.float32
        )
        self.std = np.maximum(
            np.asarray(
                [lookup[name]["std"] for name in self.input_channels], dtype=np.float32
            ),
            1e-6,
        )
        self.input_clip = input_clip
        self.cache_size = cache_size
        self._cache: OrderedDict[int, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.frames)

    def load(self, frame_index: int) -> dict[str, np.ndarray]:
        if frame_index in self._cache:
            loaded = self._cache.pop(frame_index)
            self._cache[frame_index] = loaded
            return loaded

        frame = self.frames[frame_index]
        with np.load(Path(frame["frame_file"])) as data:
            primitive = data["input_fields"].astype(np.float32)
            geometry = data["geometry_mask"].astype(bool)
            shock = data["shock_mask"].astype(bool)
            shock_uncertain = data["shock_uncertain"].astype(bool)
            base_valid = data["valid_loss_mask"].astype(bool)
        with np.load(Path(frame["temporal_vortex_label_file"])) as temporal:
            vortex = temporal["vortex_mask"].astype(bool)
            vortex_uncertain = temporal["vortex_uncertain"].astype(bool)
            vortex_valid = temporal["vortex_valid_loss_mask"].astype(bool)

        channel_lookup = {
            name: primitive[channel]
            for channel, name in enumerate(self.index["input_channels"])
        }
        inputs = np.stack([channel_lookup[name] for name in self.input_channels])
        inputs = (inputs - self.mean[:, None, None]) / self.std[:, None, None]
        if self.input_clip is not None:
            inputs = np.clip(inputs, -self.input_clip, self.input_clip)

        shock_valid = base_valid & ~shock_uncertain
        if not bool(frame.get("shock_loss_eligible", True)):
            shock_valid[:] = False
        vortex_valid &= ~vortex_uncertain & ~geometry

        # Solid cells are certain negatives for the two fluid-object heads even
        # when the frame's weak shock supervision is otherwise disabled.
        shock[geometry] = False
        vortex[geometry] = False
        shock_uncertain[geometry] = False
        vortex_uncertain[geometry] = False
        shock_valid[geometry] = True
        vortex_valid[geometry] = True
        background = ~(
            shock | vortex | geometry | shock_uncertain | vortex_uncertain
        )
        background_valid = shock_valid & vortex_valid & ~geometry
        every_pixel = np.ones_like(geometry, dtype=bool)
        targets = np.stack(
            (
                shock,
                vortex,
                background,
                geometry,
                shock_uncertain,
                vortex_uncertain,
            )
        ).astype(np.float32)
        valid = np.stack(
            (
                shock_valid,
                vortex_valid,
                background_valid,
                every_pixel,
                every_pixel,
                every_pixel,
            )
        ).astype(np.float32)
        if targets.shape[0] != len(Stage5JointNet.output_names):
            raise RuntimeError("Stage-5 target/head contract is inconsistent")
        wall_band = ndimage.binary_dilation(geometry, iterations=12) & ~geometry
        loaded = {
            "inputs": np.ascontiguousarray(inputs.astype(np.float32)),
            "targets": np.ascontiguousarray(targets),
            "valid": np.ascontiguousarray(valid),
            "geometry": geometry,
            "wall_band": wall_band,
            "step": np.asarray(int(frame["step"]), dtype=np.int64),
            "time": np.asarray(float(frame["time"]), dtype=np.float64),
        }
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
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        probabilities = np.asarray(focus_probabilities, dtype=np.float64)
        if len(probabilities) != len(FOCUS_NAMES):
            raise ValueError(
                f"Expected {len(FOCUS_NAMES)} focus probabilities, got "
                f"{len(probabilities)}"
            )
        if not np.isclose(probabilities.sum(), 1.0):
            raise ValueError("focus probabilities must sum to one")
        focus_index = int(rng.choice(len(FOCUS_NAMES), p=probabilities))
        frame_index = int(rng.integers(0, len(self.frames)))
        loaded = self.load(frame_index)
        height, width = loaded["targets"].shape[-2:]
        if patch_size > min(height, width):
            raise ValueError("patch size exceeds frame dimensions")

        positives = np.empty((0, 2), dtype=np.int64)
        target_head = FOCUS_TO_HEAD.get(focus_index)
        if target_head is not None:
            for _ in range(12):
                positives = np.argwhere(
                    (loaded["targets"][target_head] > 0.5)
                    & (loaded["valid"][target_head] > 0.5)
                )
                if positives.size:
                    break
                frame_index = int(rng.integers(0, len(self.frames)))
                loaded = self.load(frame_index)
        elif FOCUS_NAMES[focus_index] == "wall_hard_negative":
            positives = np.argwhere(loaded["wall_band"])

        if positives.size:
            center_y, center_x = positives[int(rng.integers(0, len(positives)))]
            jitter = patch_size // 5
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
            "step": int(loaded["step"]),
            "focus": FOCUS_NAMES[focus_index],
            "top": top,
            "left": left,
        }
        return (
            loaded["inputs"][:, rows, columns],
            loaded["targets"][:, rows, columns],
            loaded["valid"][:, rows, columns],
            metadata,
        )
