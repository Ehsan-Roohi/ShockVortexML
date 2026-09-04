"""Rebuild inputs without altering original fields, legacy labels or outputs."""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from jcp2026.common import ROOT, CONTRACT, read_json, write_json, sha256, validate_groups
from jcp2026.diagnostics import build_inputs
from ml.geometry_robust_dataset import GeometryRobustFrames, _reference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/jcp_v4_campaign.json")
    args = parser.parse_args()
    config = read_json(args.config)
    destination = ROOT / config["output"]
    source_index = ROOT / config["source_index"]
    source = read_json(source_index)
    validate_groups(source["records"])
    provenance = {"config_sha256": sha256(args.config), "source_index_sha256": sha256(source_index),
                  "diagnostic_contract": CONTRACT, "diagnostic_source_sha256": sha256(Path(__file__).with_name("diagnostics.py"))}
    index_path = destination / "dataset_index.json"
    if index_path.exists():
        old = read_json(index_path)
        if old["provenance"] != provenance:
            raise ValueError("Existing version has different provenance; create a new version instead")
        print("Verified existing preparation; use a new output version for changes", flush=True)
        return
    from audit_mfc_euler_cylinder_three_run_sensitivity import read_common_frame, case_paths
    three = read_json(ROOT / "configs/mfc_euler_cylinder_three_run_sensitivity_v1.json")
    cases = {case["name"]: case for case in three["cases"]}
    datasets = {s: GeometryRobustFrames(source_index, split=s, cache_size=1) for s in ("train", "validation", "test")}
    indices = {s: {r["dataset_id"]: i for i, r in enumerate(ds.frames)} for s, ds in datasets.items()}
    rows, audit = [], []
    for number, original in enumerate(source["records"]):
        record = dict(original)
        loaded = datasets[record["split"]].load(indices[record["split"]][record["dataset_id"]])
        if sha256(Path(record["source_frame"])) != record["source_sha256"].lower():
            raise ValueError(f"Legacy source hash changed: {record['dataset_id']}")
        if record["family"] == "airfoil":
            with np.load(record["source_frame"], allow_pickle=False) as z:
                lookup = dict(zip(z["field_names"].tolist(), z["fields"].astype(np.float64)))
                fields = {k: lookup[k] for k in ("rho", "pressure", "u", "v")}
                fields.update(x=z["x"].astype(np.float64), y=z["y"].astype(np.float64))
                observation = z["fluid_mask"].astype(bool) & ~loaded["geometry"]
            raw_path = Path(record["source_frame"])
        else:
            case = cases[record["case"]]
            frame = read_common_frame(case, record)
            fields = {k: np.asarray(getattr(frame, k), dtype=np.float64) for k in ("rho", "pressure", "u", "v", "x", "y")}
            raw_path = case_paths(case)[1] / f"{record['step']}.silo"
            observation = ~loaded["geometry"]
        reference = _reference(record)
        channels, diagnostic = build_inputs(fields, reference)
        fluid = observation
        if not all(np.isfinite(fields[k][fluid]).all() for k in ("rho", "pressure", "u", "v")):
            raise ValueError("Non-finite fluid primitive")
        if np.any(fields["rho"][fluid] <= 0) or np.any(fields["pressure"][fluid] <= 0):
            raise ValueError("Non-positive fluid density/pressure")
        error = diagnostic["q_legacy"] - (diagnostic["q_deviatoric"] - diagnostic["divergence"]**2 / 4)
        relative = float(np.max(np.abs(error)) / max(1.0, np.max(np.abs(diagnostic["q_legacy"]))))
        if relative > 1e-10:
            raise ValueError("Q identity failed")
        delta = np.abs(channels - loaded["inputs"])
        gy, gx = np.gradient(np.log(np.maximum(fields["rho"], 1e-12)), fields["y"], fields["x"])
        py, px = np.gradient(np.log(np.maximum(fields["pressure"], 1e-12)), fields["y"], fields["x"])
        evidence = np.stack([np.hypot(gx, gy), np.hypot(px, py), np.maximum(-diagnostic["divergence"], 0), diagnostic["vorticity"], diagnostic["q_deviatoric"], diagnostic["lambda_ci"]]).astype(np.float32)
        target = destination / "frames" / f"f{number:04d}.npz"
        target.parent.mkdir(parents=True, exist_ok=True)
        # Unobserved raster padding is not physical fluid or a supervised negative.
        valid = loaded["valid"].astype(np.uint8)
        valid[:, ~(observation | loaded["geometry"])] = 0
        np.savez_compressed(target, inputs=channels, targets=loaded["targets"].astype(np.uint8), valid=valid, geometry=loaded["geometry"], observation_mask=observation, wall_band=loaded["wall_band"], hard_negative=loaded["hard_negative"], diagnostics=evidence, **{k: v.astype(np.float32) for k, v in fields.items()})
        record.update(cache_file=str(target.relative_to(ROOT)), cache_sha256=sha256(target), primitive_source=str(raw_path.relative_to(ROOT)), primitive_source_sha256=sha256(raw_path), diagnostic_contract=CONTRACT, evaluation_role="previously_inspected_development" if record["split"] == "test" else record["split"])
        rows.append(record)
        audit.append({"dataset_id": record["dataset_id"], "case": record["case"], "unobserved_padding_pixels": int(np.count_nonzero(~(observation | loaded["geometry"]))), "q_identity_relative_residual": relative, "mean_channel_change_fluid": delta[:, fluid].mean(axis=1).tolist(), "max_channel_change_fluid": delta[:, fluid].max(axis=1).tolist()})
        print(f"{number+1}/{len(source['records'])} {record['dataset_id']} recomputed", flush=True)
    index = {"schema_version": "jcp_v4", "created_utc": datetime.now(timezone.utc).isoformat(), "provenance": provenance, "input_channels": source["input_channels"], "records": rows, "annotation_status": "unreviewed_weak_labels_unchanged", "accuracy_claim": False, "untouched_final_test_available": False}
    write_json(index_path, index)
    write_json(destination / "harmonization_audit.json", {"status": "PASS", "raw_modified": False, "legacy_labels_modified": False, "record_count": len(rows), "contract": CONTRACT, "wall_stencil_note": "Stored-grid derivatives include wall-adjacent stencils; coherent near-wall structures require independent review, not a blanket wall veto.", "records": audit})
    groups = defaultdict(list)
    for r in rows:
        groups[r["split"]].append(r["leakage_group_id"])
    write_json(destination / "split_freeze.json", {"index_sha256": sha256(index_path), "groups": {s: sorted(set(v)) for s, v in groups.items()}, "final_test": config["final_test"], "independent_reference": "pending_named_expert"})


if __name__ == "__main__":
    main()
