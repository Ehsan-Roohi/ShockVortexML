"""An identical, hashed training patch schedule across models and seeds."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from jcp2026.common import ROOT, read_json, write_json, sha256
from jcp2026.dataset import Frames
from ml.geometry_robust_dataset import apply_temporal_vortex_override


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, default=ROOT / "configs/jcp_v4_campaign.json")
    args = p.parse_args()
    cfg = read_json(args.config)
    t = cfg["training"]
    root = ROOT / cfg["output"]
    out = root / "patches"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists():
        raise FileExistsError("Frozen patch bank exists; refusing overwrite")
    ds = Frames(root / "dataset_index.json", "train", cache_size=16)
    rng = np.random.default_rng(t["patch_seed"])
    n = t["epochs"] * t["steps_per_epoch"] * t["batch_size"]
    size = t["patch_size"]
    arrays = {}
    for name, channels, dtype in [("inputs",7,"float32"),("targets",6,"uint8"),("valid",6,"uint8"),("temporal_targets",6,"uint8"),("temporal_valid",6,"uint8")]:
        arrays[name] = np.lib.format.open_memmap(out / (name+".npy"), mode="w+", dtype=dtype, shape=(n,channels,size,size))
    override_doc = read_json(ROOT / cfg["temporal_override_index"])
    overrides = {r["dataset_id"]: r for r in override_doc["records"]}
    metadata, temporal_cache = [], {}
    for i in range(n):
        inputs, targets, valid, meta = ds.sample_patch(rng, patch_size=size, focus_probabilities=tuple(t["focus_probabilities"]), family_balanced=True)
        arrays["inputs"][i], arrays["targets"][i], arrays["valid"][i] = inputs, targets, valid
        yt, vt = targets.copy(), valid.copy()
        if meta["dataset_id"] in overrides:
            if meta["dataset_id"] not in temporal_cache:
                frame = ds.load(meta["frame_index"])
                if frame["record"]["split"] != "train":
                    raise ValueError("Temporal targets may only be used on train")
                with np.load(overrides[meta["dataset_id"]]["override_file"], allow_pickle=False) as z:
                    positive, uncertain, _ = apply_temporal_vortex_override(vortex=frame["targets"][1].astype(bool), vortex_uncertain=frame["targets"][5].astype(bool), hard_negative=frame["hard_negative"], geometry=frame["geometry"], temporal_positive=z["temporal_positive"], recall_candidate=z["recall_candidate"])
                temporal_cache[meta["dataset_id"]] = positive, uncertain
            positive, uncertain = temporal_cache[meta["dataset_id"]]
            rows = slice(meta["top"],meta["top"]+size)
            cols = slice(meta["left"],meta["left"]+size)
            positive, uncertain = positive[rows,cols], uncertain[rows,cols]
            yt[1], yt[5] = positive, uncertain
            vt[1] = ((valid[1]>0) | positive) & ~uncertain
            yt[2] = ~(yt[0].astype(bool)|positive|yt[3].astype(bool)|yt[4].astype(bool)|uncertain)
            vt[2] = (vt[0]>0)&(vt[1]>0)&~yt[3].astype(bool)
        arrays["temporal_targets"][i], arrays["temporal_valid"][i] = yt, vt
        metadata.append(meta)
        if (i+1)%100==0:
            print(f"patches {i+1}/{n}",flush=True)
    for value in arrays.values():
        value.flush()
    hashes = {name: sha256(out/(name+".npy")) for name in arrays}
    write_json(out / "manifest.json", {"config_sha256":sha256(args.config), "dataset_index_sha256":sha256(root/"dataset_index.json"), "patch_count":n,"patch_seed":t["patch_seed"],"hashes":hashes,"temporal_override_index_sha256":sha256(ROOT/cfg["temporal_override_index"]),"temporal_target_policy":cfg["temporal_policy"],"patches":metadata})


if __name__ == "__main__":
    main()
