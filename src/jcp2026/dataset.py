from __future__ import annotations

from collections import OrderedDict, defaultdict
from pathlib import Path
import numpy as np

from jcp2026.common import ROOT, CONTRACT, read_json, validate_groups, sha256
from ml.geometry_robust_dataset import GeometryRobustFrames


class Frames(GeometryRobustFrames):
    """Reuse the case-balanced sampler, but read versioned harmonized arrays."""
    def __init__(self, index_path: Path, split: str, cache_size: int = 8):
        self.index_path = index_path
        self.index = read_json(index_path)
        if self.index["provenance"]["diagnostic_contract"] != CONTRACT:
            raise ValueError("Unrecognized diagnostic contract")
        validate_groups(self.index["records"])
        self.frames = [r for r in self.index["records"] if r["split"] == split]
        if not self.frames:
            raise ValueError("Empty requested split")
        self.input_channels = tuple(self.index["input_channels"])
        self.cache_size, self._cache = cache_size, OrderedDict()
        self._verified = set()
        self.split = split
        groups = defaultdict(list)
        for i, record in enumerate(self.frames):
            groups[record["leakage_group_id"]].append(i)
        self.grouped_indices = dict(groups)
        self.group_names = tuple(sorted(groups))
        self.group_family = {g: self.frames[ix[0]]["family"] for g, ix in groups.items()}
        family_groups = defaultdict(list)
        for group in self.group_names:
            family_groups[self.group_family[group]].append(group)
        self.family_groups = dict(family_groups)
        self.family_names = tuple(sorted(family_groups))

    def load(self, frame_index: int) -> dict:
        if frame_index in self._cache:
            value = self._cache.pop(frame_index)
            self._cache[frame_index] = value
            return value
        record = self.frames[frame_index]
        if frame_index not in self._verified:
            if sha256(ROOT / record["cache_file"]) != record["cache_sha256"]:
                raise ValueError("Input cache changed after freeze")
            self._verified.add(frame_index)
        with np.load(ROOT / record["cache_file"], allow_pickle=False) as z:
            value = {name: z[name] for name in ("inputs", "targets", "valid", "geometry", "wall_band", "hard_negative", "observation_mask", "x", "y")}
        value.update(record=record, frame_index=frame_index)
        self._cache[frame_index] = value
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return value
