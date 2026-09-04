#!/usr/bin/env python3
"""Freeze validation thresholds and audit v2 on held-out case groups.

Every reported score is agreement with unreviewed weak proposals.  The script
does not call these values accuracy and does not alter thresholds after reading
the test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch

from analyze_mfc_euler_cylinder import predict_tiled
from evaluate_stage5_joint import object_match_counts
from ml.geometry_robust_dataset import GeometryRobustFrames
from ml.stage5_model import Stage5JointNet


HEAD_NAMES = Stage5JointNet.output_names
OBJECT_HEADS = {"shock": 0, "vortex_core": 1}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_components(mask: np.ndarray, minimum_pixels: int) -> np.ndarray:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = np.bincount(labels.ravel())
    keep = sizes >= minimum_pixels
    keep[0] = False
    return keep[labels]


def counts(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> np.ndarray:
    prediction = prediction.astype(bool) & valid
    target = target.astype(bool) & valid
    return np.asarray(
        [
            np.count_nonzero(prediction & target),
            np.count_nonzero(prediction & ~target),
            np.count_nonzero(~prediction & target & valid),
        ],
        dtype=np.int64,
    )


def metrics(values: np.ndarray) -> dict[str, float | int]:
    true_positive, false_positive, false_negative = [int(value) for value in values]
    return {
        "true_positive_pixels": true_positive,
        "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative,
        "weak_dice": 2.0 * true_positive / max(2 * true_positive + false_positive + false_negative, 1),
        "weak_precision": true_positive / max(true_positive + false_positive, 1),
        "weak_recall": true_positive / max(true_positive + false_negative, 1),
    }


def object_metrics(values: np.ndarray) -> dict[str, float | int | str]:
    matched, false_positive, false_negative = [int(value) for value in values]
    return {
        "matched_objects": matched,
        "false_positive_objects": false_positive,
        "false_negative_objects": false_negative,
        "weak_object_f1": 2.0 * matched / max(2 * matched + false_positive + false_negative, 1),
        "weak_object_precision": matched / max(matched + false_positive, 1),
        "weak_object_recall": matched / max(matched + false_negative, 1),
        "scope": "component agreement with unreviewed weak proposals; not object accuracy",
    }


def describe_vortex_components(
    mask: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    bounded_vorticity: np.ndarray,
    *,
    minimum_pixels: int,
) -> list[dict[str, float | int]]:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    objects: list[dict[str, float | int]] = []
    for label in range(1, count + 1):
        if sizes[label] < minimum_pixels:
            continue
        rows, columns = np.where(labels == label)
        mean_rotation = float(np.mean(bounded_vorticity[rows, columns]))
        objects.append({
            "local_id": len(objects) + 1,
            "area_pixels": int(len(rows)),
            "center_x": float(np.mean(x[columns])),
            "center_y": float(np.mean(y[rows])),
            "rotation_sign": int(np.sign(mean_rotation)),
            "mean_bounded_vorticity": mean_rotation,
        })
    return objects


def track_vortex_objects(
    frame_reports: list[dict[str, Any]],
    *,
    maximum_centroid_jump: float = 0.45,
) -> dict[str, Any]:
    """Associate cylinder candidates in time; tracks remain review proposals."""

    tracks: list[dict[str, Any]] = []
    next_id = 1
    previous: list[tuple[int, dict[str, Any]]] = []
    for frame in sorted(
        (item for item in frame_reports if item["family"] == "cylinder"),
        key=lambda item: (item["case"], item["time"]),
    ):
        if previous and frame["case"] != previous[0][1]["case"]:
            previous = []
        current = frame["vortex_objects"]
        assignments: dict[int, int] = {}
        if previous and current:
            cost = np.full((len(previous), len(current)), 1e6, dtype=np.float64)
            for old_index, (_track_id, old) in enumerate(previous):
                for new_index, new in enumerate(current):
                    if old["rotation_sign"] and new["rotation_sign"] and old["rotation_sign"] != new["rotation_sign"]:
                        continue
                    distance = float(np.hypot(new["center_x"] - old["center_x"], new["center_y"] - old["center_y"]))
                    if distance <= maximum_centroid_jump:
                        cost[old_index, new_index] = distance
            rows, columns = linear_sum_assignment(cost)
            for old_index, new_index in zip(rows, columns):
                if cost[old_index, new_index] <= maximum_centroid_jump:
                    assignments[int(new_index)] = int(previous[int(old_index)][0])
        new_previous: list[tuple[int, dict[str, Any]]] = []
        for object_index, item in enumerate(current):
            track_id = assignments.get(object_index)
            if track_id is None:
                track_id = next_id
                next_id += 1
                tracks.append({"track_id": track_id, "case": frame["case"], "observations": []})
            track = next(value for value in tracks if value["track_id"] == track_id)
            observation = {
                "dataset_id": frame["dataset_id"],
                "time": frame["time"],
                **item,
            }
            track["observations"].append(observation)
            item["track_id"] = track_id
            new_previous.append((track_id, {**item, "case": frame["case"]}))
        previous = new_previous
    for track in tracks:
        observations = track["observations"]
        track["frames"] = len(observations)
        track["persistent_three_frames"] = len(observations) >= 3
        if len(observations) >= 2:
            elapsed = float(observations[-1]["time"] - observations[0]["time"])
            track["mean_streamwise_speed"] = (
                float(observations[-1]["center_x"] - observations[0]["center_x"]) / elapsed
                if elapsed > 0 else None
            )
        else:
            track["mean_streamwise_speed"] = None
    return {
        "method": "nearest-centroid Hungarian association with 0.45 reference-length gate and rotation-sign consistency",
        "claim_boundary": "candidate identity audit only; not validated physical vortex tracks",
        "tracks": tracks,
        "track_count": len(tracks),
        "persistent_track_count": sum(bool(track["persistent_three_frames"]) for track in tracks),
    }


def predict(model: Stage5JointNet, loaded: dict[str, Any], evaluation: dict[str, Any]) -> np.ndarray:
    return predict_tiled(
        model,
        torch,
        loaded["inputs"],
        tile_size=int(evaluation["tile_size"]),
        overlap=int(evaluation["tile_overlap"]),
        batch_size=4,
    )


def output_path(output_dir: Path, record: dict[str, Any]) -> Path:
    # Keep paths below the classic Windows MAX_PATH limit while retaining
    # uniqueness through split/family/case directories plus the solver step.
    return (
        output_dir
        / record["split"]
        / record["family"]
        / record["case"]
        / f"s{int(record['step']):08d}.npz"
    )


def save_prediction(path: Path, loaded: dict[str, Any], probability: np.ndarray) -> None:
    record = loaded["record"]
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        dataset_id=np.asarray(record["dataset_id"]),
        family=np.asarray(record["family"]),
        case=np.asarray(record["case"]),
        split=np.asarray(record["split"]),
        step=np.asarray(record["step"], dtype=np.int64),
        time=np.asarray(record["time"], dtype=np.float64),
        probabilities=probability.astype(np.float16),
        x=loaded["x"],
        y=loaded["y"],
    )


def validation_thresholds(
    model: Stage5JointNet,
    dataset: GeometryRobustFrames,
    output_dir: Path,
    evaluation: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    candidates = [float(value) for value in evaluation["thresholds"]]
    minimum_pixels = int(evaluation["minimum_component_pixels"])
    aggregate = {
        name: {threshold: np.zeros(3, dtype=np.int64) for threshold in candidates}
        for name in OBJECT_HEADS
    }
    by_family = {
        family: {
            name: {threshold: np.zeros(3, dtype=np.int64) for threshold in candidates}
            for name in OBJECT_HEADS
        }
        for family in sorted({record["family"] for record in dataset.frames})
    }
    for frame_index in range(len(dataset)):
        loaded = dataset.load(frame_index)
        probability = predict(model, loaded, evaluation)
        save_prediction(output_path(output_dir, loaded["record"]), loaded, probability)
        for name, head_index in OBJECT_HEADS.items():
            target = loaded["targets"][head_index] >= 0.5
            valid = loaded["valid"][head_index] >= 0.5
            for threshold in candidates:
                prediction = clean_components(
                    (probability[head_index] >= threshold) & ~loaded["geometry"], minimum_pixels
                )
                value = counts(prediction, target, valid)
                aggregate[name][threshold] += value
                by_family[loaded["record"]["family"]][name][threshold] += value
        print(f"validation {frame_index + 1:02d}/{len(dataset):02d} {loaded['record']['dataset_id']}", flush=True)

    selected: dict[str, float] = {}
    report: dict[str, Any] = {
        "selection_objective": (
            "maximize macro mean weak Dice across geometry families using one global threshold; "
            "tie-break by worst-family Dice, macro precision, then larger threshold"
        ),
        "candidate_curves": {},
        "by_family": {},
    }
    for name in OBJECT_HEADS:
        curve = []
        for threshold in candidates:
            family_metrics = {
                family: metrics(values[name][threshold])
                for family, values in by_family.items()
            }
            macro_dice = float(np.mean([value["weak_dice"] for value in family_metrics.values()]))
            macro_precision = float(
                np.mean([value["weak_precision"] for value in family_metrics.values()])
            )
            item = {
                "threshold": threshold,
                **metrics(aggregate[name][threshold]),
                "macro_family_weak_dice": macro_dice,
                "worst_family_weak_dice": float(
                    min(value["weak_dice"] for value in family_metrics.values())
                ),
                "macro_family_weak_precision": macro_precision,
                "family_metrics": family_metrics,
            }
            curve.append(item)
        winner = max(
            curve,
            key=lambda item: (
                item["macro_family_weak_dice"],
                item["worst_family_weak_dice"],
                item["macro_family_weak_precision"],
                item["threshold"],
            ),
        )
        selected[name] = float(winner["threshold"])
        report["candidate_curves"][name] = curve
    for family, family_values in by_family.items():
        report["by_family"][family] = {
            name: metrics(family_values[name][selected[name]]) for name in OBJECT_HEADS
        }
    report["selected_thresholds"] = selected
    return selected, report


def legacy_v1_path(record: dict[str, Any]) -> Path:
    root = Path("results/ml_only/mfc_euler_cylinder_three_run_frozen_ml_hybrid_v1").resolve()
    return root / record["case"] / f"step_{int(record['step']):05d}_ml_only.npz"


def evaluate_test(
    model: Stage5JointNet,
    dataset: GeometryRobustFrames,
    output_dir: Path,
    evaluation: dict[str, Any],
    thresholds: dict[str, float],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    minimum_pixels = int(evaluation["minimum_component_pixels"])
    total = {name: np.zeros(3, dtype=np.int64) for name in OBJECT_HEADS}
    by_family: dict[str, dict[str, np.ndarray]] = {}
    by_case: dict[str, dict[str, np.ndarray]] = {}
    vortex_objects_total = np.zeros(3, dtype=np.int64)
    vortex_objects_by_family: dict[str, np.ndarray] = {}
    legacy = {name: np.zeros(3, dtype=np.int64) for name in OBJECT_HEADS}
    legacy_frames = 0
    saturation_sum = np.zeros(len(dataset.input_channels), dtype=np.int64)
    saturation_denominator = 0
    frame_reports: list[dict[str, Any]] = []
    for frame_index in range(len(dataset)):
        loaded = dataset.load(frame_index)
        record = loaded["record"]
        probability = predict(model, loaded, evaluation)
        masks: dict[str, np.ndarray] = {}
        for name, head_index in OBJECT_HEADS.items():
            masks[name] = clean_components(
                (probability[head_index] >= thresholds[name]) & ~loaded["geometry"], minimum_pixels
            )
        background = ~(masks["shock"] | masks["vortex_core"] | loaded["geometry"])
        destination = output_path(output_dir, record)
        save_prediction(destination, loaded, probability)
        with np.load(destination) as saved:
            payload = {key: saved[key] for key in saved.files}
        np.savez_compressed(
            destination,
            **payload,
            shock=masks["shock"].astype(np.uint8),
            vortex_core=masks["vortex_core"].astype(np.uint8),
            background_other=background.astype(np.uint8),
            geometry=loaded["geometry"].astype(np.uint8),
        )
        family = record["family"]
        case = record["case"]
        by_family.setdefault(family, {name: np.zeros(3, dtype=np.int64) for name in OBJECT_HEADS})
        by_case.setdefault(case, {name: np.zeros(3, dtype=np.int64) for name in OBJECT_HEADS})
        vortex_objects_by_family.setdefault(family, np.zeros(3, dtype=np.int64))
        frame_item: dict[str, Any] = {
            "dataset_id": record["dataset_id"],
            "family": family,
            "case": case,
            "step": int(record["step"]),
            "time": float(record["time"]),
            "prediction_file": str(destination),
            "weak_proposal_agreement": {},
            "predicted_pixels": {name: int(np.count_nonzero(mask)) for name, mask in masks.items()},
            "shock_vortex_overlap_pixels": int(np.count_nonzero(masks["shock"] & masks["vortex_core"])),
            "vortex_objects": describe_vortex_components(
                masks["vortex_core"], loaded["x"], loaded["y"], loaded["inputs"][4],
                minimum_pixels=minimum_pixels,
            ),
        }
        for name, head_index in OBJECT_HEADS.items():
            value = counts(
                masks[name], loaded["targets"][head_index] >= 0.5, loaded["valid"][head_index] >= 0.5
            )
            total[name] += value
            by_family[family][name] += value
            by_case[case][name] += value
            frame_item["weak_proposal_agreement"][name] = metrics(value)
        vortex_object_value = np.asarray(
            object_match_counts(
                masks["vortex_core"],
                (loaded["targets"][1] >= 0.5) & (loaded["valid"][1] >= 0.5),
                minimum_pixels=minimum_pixels,
                minimum_iou=0.1,
            ),
            dtype=np.int64,
        )
        vortex_objects_total += vortex_object_value
        vortex_objects_by_family[family] += vortex_object_value
        frame_item["vortex_object_weak_proposal_agreement"] = object_metrics(vortex_object_value)
        saturation_sum += np.count_nonzero(np.abs(loaded["inputs"]) >= 0.99, axis=(1, 2))
        saturation_denominator += loaded["inputs"].shape[1] * loaded["inputs"].shape[2]

        old_path = legacy_v1_path(record)
        if family == "cylinder" and old_path.exists():
            with np.load(old_path) as old:
                for name, head_index in OBJECT_HEADS.items():
                    old_mask = old[name].astype(bool) & ~loaded["geometry"]
                    legacy[name] += counts(
                        old_mask,
                        loaded["targets"][head_index] >= 0.5,
                        loaded["valid"][head_index] >= 0.5,
                    )
            legacy_frames += 1
        frame_reports.append(frame_item)
        print(f"test {frame_index + 1:02d}/{len(dataset):02d} {record['dataset_id']}", flush=True)

    tracking = track_vortex_objects(frame_reports)
    result: dict[str, Any] = {
        "overall": {name: metrics(total[name]) for name in OBJECT_HEADS},
        "by_family": {
            family: {name: metrics(values[name]) for name in OBJECT_HEADS}
            for family, values in by_family.items()
        },
        "by_case": {
            case: {name: metrics(values[name]) for name in OBJECT_HEADS}
            for case, values in by_case.items()
        },
        "vortex_object_weak_proposal_agreement": {
            "overall": object_metrics(vortex_objects_total),
            "by_family": {
                family: object_metrics(values) for family, values in vortex_objects_by_family.items()
            },
        },
        "cylinder_vortex_candidate_tracking": tracking,
        "canonical_input_fraction_abs_ge_0p99": {
            name: float(saturation_sum[index] / max(saturation_denominator, 1))
            for index, name in enumerate(dataset.input_channels)
        },
        "legacy_v1_cylinder_comparison": {
            "matched_frames": legacy_frames,
            "weak_proposal_agreement": {
                name: metrics(legacy[name]) for name in OBJECT_HEADS
            },
        },
    }
    return result, frame_reports


def load_saved_probability(output_dir: Path, record: dict[str, Any]) -> np.ndarray:
    with np.load(output_path(output_dir, record)) as data:
        return data["probabilities"].astype(np.float32)


def plot_cylinder_audit(
    dataset: GeometryRobustFrames,
    output_dir: Path,
    figure_dir: Path,
    thresholds: dict[str, float],
    requested_times: list[float],
) -> Path:
    records = [record for record in dataset.frames if record["family"] == "cylinder"]
    chosen: list[int] = []
    for requested in requested_times:
        candidates = [index for index, record in enumerate(dataset.frames) if record["family"] == "cylinder"]
        chosen.append(min(candidates, key=lambda index: abs(float(dataset.frames[index]["time"]) - requested)))
    chosen = list(dict.fromkeys(chosen))
    figure, axes = plt.subplots(
        len(chosen), 4, figsize=(15.5, 3.2 * len(chosen)), squeeze=False,
        layout="constrained",
    )
    for row, frame_index in enumerate(chosen):
        loaded = dataset.load(frame_index)
        record = loaded["record"]
        probability = load_saved_probability(output_dir, record)
        extent = [float(loaded["x"][0]), float(loaded["x"][-1]), float(loaded["y"][0]), float(loaded["y"][-1])]
        pressure_view = loaded["inputs"][1]
        axes[row, 0].imshow(pressure_view, origin="lower", extent=extent, cmap="Greys", aspect="equal")
        axes[row, 0].contour(loaded["targets"][0], levels=[0.5], colors=["#d55e00"], linewidths=0.8, origin="lower", extent=extent)
        if np.any(loaded["targets"][1] > 0.5):
            axes[row, 0].contour(loaded["targets"][1], levels=[0.5], colors=["#cc79a7"], linewidths=0.9, origin="lower", extent=extent)
        axes[row, 0].set_title(f"weak seeds; t={record['time']:.2f}")
        first = axes[row, 1].imshow(probability[0], origin="lower", extent=extent, cmap="inferno", vmin=0, vmax=1, aspect="equal")
        axes[row, 1].set_title("ML P(shock)")
        second = axes[row, 2].imshow(probability[1], origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1, aspect="equal")
        axes[row, 2].set_title("ML P(vortex core)")
        shock = clean_components((probability[0] >= thresholds["shock"]) & ~loaded["geometry"], 9)
        vortex = clean_components((probability[1] >= thresholds["vortex_core"]) & ~loaded["geometry"], 9)
        axes[row, 3].imshow(pressure_view, origin="lower", extent=extent, cmap="Greys", aspect="equal")
        if np.any(shock):
            axes[row, 3].contour(shock, levels=[0.5], colors=["#d55e00"], linewidths=1.0, origin="lower", extent=extent)
        if np.any(vortex):
            axes[row, 3].contour(vortex, levels=[0.5], colors=["#cc79a7"], linewidths=1.0, origin="lower", extent=extent)
        axes[row, 3].set_title("frozen masks")
        for axis in axes[row]:
            axis.contour(loaded["geometry"], levels=[0.5], colors="black", linewidths=0.8, origin="lower", extent=extent)
            axis.set_xlim(-1.25, 2.5)
            axis.set_ylim(-1.25, 1.25)
            axis.set_xlabel("x/D")
            axis.set_ylabel("y/D")
    figure.colorbar(
        first, ax=list(axes[:, 1]), shrink=0.72, pad=0.02, label="P(shock)"
    )
    figure.colorbar(
        second, ax=list(axes[:, 2]), shrink=0.72, pad=0.02, label="P(vortex core)"
    )
    figure.suptitle(
        "Fine-grid cylinder development audit — shock=orange, vortex=magenta"
    )
    figure_dir.mkdir(parents=True, exist_ok=True)
    path = figure_dir / "heldout_cylinder_v2_audit.png"
    figure.savefig(path, dpi=250)
    plt.close(figure)
    return path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/geometry_robust_v2.json"))
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    evaluation = config["evaluation"]
    index_path = Path(config["output"]["dataset"]).resolve() / "dataset_index.json"
    checkpoint_path = Path(config["output"]["model"]).resolve()
    output_dir = Path(config["output"]["evaluation"]).resolve()
    figure_dir = Path(config["output"]["figures"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    override_value = config.get("label_overrides")
    override_index_path = Path(override_value).resolve() if override_value else None

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = Stage5JointNet(
        input_channels=len(checkpoint["input_channels"]), base_channels=int(checkpoint["base_channels"])
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.set_num_threads(int(config["training"].get("torch_num_threads", 8)))

    validation = GeometryRobustFrames(
        index_path,
        split=evaluation["threshold_selection_split"],
        cache_size=3,
        override_index_path=override_index_path,
    )
    test = GeometryRobustFrames(
        index_path,
        split=evaluation["final_report_split"],
        cache_size=3,
        override_index_path=override_index_path,
    )
    thresholds, validation_report = validation_thresholds(model, validation, output_dir, evaluation)
    threshold_path = output_dir / "frozen_thresholds.json"
    threshold_path.write_text(json.dumps({
        "selected_only_on_split": validation.split,
        "test_split_read_during_selection": False,
        "thresholds": thresholds,
        "minimum_component_pixels": int(evaluation["minimum_component_pixels"]),
        "checkpoint_sha256": sha256(checkpoint_path),
        "validation": validation_report,
    }, indent=2), encoding="utf-8")
    print(json.dumps({"frozen_thresholds": thresholds}, indent=2), flush=True)

    test_report, frame_reports = evaluate_test(model, test, output_dir, evaluation, thresholds)
    figure = plot_cylinder_audit(
        test, output_dir, figure_dir, thresholds, [float(value) for value in evaluation["figure_times"]]
    )
    report = {
        "schema_version": "geometry_robust_v2_weak_evaluation",
        "status": "PASS",
        "claim_boundary": evaluation["metrics_wording"],
        "annotation_status": test.index["annotation_status"],
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "dataset_index": str(index_path),
        "dataset_index_sha256": sha256(index_path),
        "label_override_index": str(override_index_path) if override_index_path else None,
        "label_override_index_sha256": (
            sha256(override_index_path) if override_index_path else None
        ),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "threshold_selection": {
            "split": validation.split,
            "records": len(validation),
            "threshold_file": str(threshold_path),
            "thresholds": thresholds,
        },
        "final_test": {"split": test.split, "records": len(test), **test_report},
        "figures": [str(figure)],
        "frames": frame_reports,
    }
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "report": str(report_path),
        "figure": str(figure),
        "test_weak_proposal_agreement": test_report["by_family"],
        "legacy_v1_cylinder": test_report["legacy_v1_cylinder_comparison"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
