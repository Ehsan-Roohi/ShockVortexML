#!/usr/bin/env python3
"""Calibrate and visually audit the Stage-5 joint dense weak-label pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch

from ml.stage5_dataset import Stage5WeakFrames
from ml.stage5_model import Stage5JointNet


HEAD_NAMES = Stage5JointNet.output_names
PHYSICAL_HEAD_INDICES = (0, 1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or overlap < 0 or overlap >= tile_size:
        raise ValueError("tile_size must be positive and 0 <= overlap < tile_size")
    if length <= tile_size:
        return [0]
    step = tile_size - overlap
    starts = list(range(0, length - tile_size + 1, step))
    if starts[-1] != length - tile_size:
        starts.append(length - tile_size)
    return starts


def predict_tiled(
    model: Stage5JointNet,
    inputs: np.ndarray,
    *,
    tile_size: int,
    overlap: int,
    batch_size: int = 1,
) -> np.ndarray:
    """Blend fixed-size sigmoid predictions without per-frame normalization."""

    channels, height, width = inputs.shape
    padded_height = max(height, tile_size)
    padded_width = max(width, tile_size)
    padded = np.pad(
        inputs,
        ((0, 0), (0, padded_height - height), (0, padded_width - width)),
        mode="edge",
    )
    starts_y = tile_starts(padded_height, tile_size, overlap)
    starts_x = tile_starts(padded_width, tile_size, overlap)
    window_1d = np.hanning(tile_size).astype(np.float32)
    window_1d = np.maximum(window_1d, 0.05)
    window = window_1d[:, None] * window_1d[None, :]
    probability_sum = np.zeros(
        (len(HEAD_NAMES), padded_height, padded_width), dtype=np.float32
    )
    weight_sum = np.zeros((padded_height, padded_width), dtype=np.float32)
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    positions = [(top, left) for top in starts_y for left in starts_x]
    with torch.inference_mode():
        for first in range(0, len(positions), batch_size):
            batch_positions = positions[first : first + batch_size]
            tiles = np.stack(
                [
                    padded[:, top : top + tile_size, left : left + tile_size]
                    for top, left in batch_positions
                ],
                axis=0,
            )
            tensor = torch.from_numpy(np.ascontiguousarray(tiles))
            probabilities = torch.sigmoid(model(tensor)).cpu().numpy()
            for probability, (top, left) in zip(probabilities, batch_positions):
                probability_sum[
                    :, top : top + tile_size, left : left + tile_size
                ] += probability * window[None, ...]
                weight_sum[top : top + tile_size, left : left + tile_size] += window
    if np.any(weight_sum <= 0):
        raise RuntimeError("Tiled inference left uncovered pixels")
    return (
        probability_sum[:, :height, :width]
        / weight_sum[None, :height, :width]
    ).astype(np.float32)


def counts_for_threshold(
    probability: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    threshold: float,
) -> np.ndarray:
    predicted = probability >= threshold
    truth = target >= 0.5
    allowed = valid >= 0.5
    return np.asarray(
        (
            np.count_nonzero(predicted & truth & allowed),
            np.count_nonzero(predicted & ~truth & allowed),
            np.count_nonzero(~predicted & truth & allowed),
        ),
        dtype=np.int64,
    )


def metric_record(counts: np.ndarray) -> dict[str, float | int]:
    true_positive, false_positive, false_negative = (int(value) for value in counts)
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "weak_dice": 2.0 * true_positive
        / max(2.0 * true_positive + false_positive + false_negative, 1),
        "weak_iou": true_positive
        / max(true_positive + false_positive + false_negative, 1),
        "weak_precision": true_positive / max(true_positive + false_positive, 1),
        "weak_recall": true_positive / max(true_positive + false_negative, 1),
    }


def component_count(mask: np.ndarray, minimum_pixels: int = 3) -> int:
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return 0
    sizes = np.bincount(labels.ravel())[1:]
    return int(np.count_nonzero(sizes >= minimum_pixels))


def object_match_counts(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    minimum_pixels: int = 3,
    minimum_iou: float = 0.1,
) -> tuple[int, int, int]:
    """One-to-one connected-component matching for vortex weak-label audit."""

    structure = np.ones((3, 3), dtype=np.uint8)
    prediction_labels, prediction_count = ndimage.label(prediction, structure=structure)
    target_labels, target_count = ndimage.label(target, structure=structure)
    prediction_sizes = np.bincount(prediction_labels.ravel(), minlength=prediction_count + 1)
    target_sizes = np.bincount(target_labels.ravel(), minlength=target_count + 1)
    prediction_ids = np.flatnonzero(prediction_sizes >= minimum_pixels)
    prediction_ids = prediction_ids[prediction_ids != 0]
    target_ids = np.flatnonzero(target_sizes >= minimum_pixels)
    target_ids = target_ids[target_ids != 0]
    if not len(prediction_ids) or not len(target_ids):
        return 0, int(len(prediction_ids)), int(len(target_ids))

    prediction_lookup = {int(label): index for index, label in enumerate(prediction_ids)}
    target_lookup = {int(label): index for index, label in enumerate(target_ids)}
    iou = np.zeros((len(target_ids), len(prediction_ids)), dtype=np.float64)
    overlap = (target_labels > 0) & (prediction_labels > 0)
    if np.any(overlap):
        pairs, intersections = np.unique(
            np.stack((target_labels[overlap], prediction_labels[overlap]), axis=1),
            axis=0,
            return_counts=True,
        )
        for (target_id, prediction_id), intersection in zip(pairs, intersections):
            if int(target_id) not in target_lookup or int(prediction_id) not in prediction_lookup:
                continue
            union = (
                target_sizes[int(target_id)]
                + prediction_sizes[int(prediction_id)]
                - int(intersection)
            )
            iou[target_lookup[int(target_id)], prediction_lookup[int(prediction_id)]] = (
                float(intersection) / max(float(union), 1.0)
            )
    target_assignment, prediction_assignment = linear_sum_assignment(-iou)
    matches = int(
        np.count_nonzero(iou[target_assignment, prediction_assignment] >= minimum_iou)
    )
    return matches, int(len(prediction_ids) - matches), int(len(target_ids) - matches)


def symmetric_mask_distance(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    spacing: tuple[float, float],
) -> dict[str, float | None]:
    if not np.any(prediction) or not np.any(target):
        return {"symmetric_mean_distance": None, "symmetric_p95_distance": None}
    to_target = ndimage.distance_transform_edt(~target, sampling=spacing)[prediction]
    to_prediction = ndimage.distance_transform_edt(~prediction, sampling=spacing)[target]
    distances = np.concatenate((to_target, to_prediction))
    return {
        "symmetric_mean_distance": float(np.mean(distances)),
        "symmetric_p95_distance": float(np.percentile(distances, 95.0)),
    }


def overlay(shock: np.ndarray, vortex: np.ndarray, geometry: np.ndarray) -> np.ndarray:
    rgba = np.zeros((*shock.shape, 4), dtype=np.float32)
    rgba[shock] = (1.0, 0.05, 0.0, 0.72)
    rgba[vortex] = (0.0, 0.82, 1.0, 0.76)
    rgba[shock & vortex] = (0.66, 0.0, 0.95, 0.9)
    rgba[geometry] = (0.03, 0.03, 0.03, 1.0)
    return rgba


def load_baseline_comparison(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for baseline in config.get("baselines", []):
        path = (root / baseline["report"]).resolve()
        if not path.exists():
            rows.append({"name": baseline["name"], "status": "missing", "report": str(path)})
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        if baseline["kind"] == "public_yolo":
            metric = report["mfc_aggregate_weak_shock_agreement"]
            rows.append(
                {
                    "name": baseline["name"],
                    "kind": baseline["kind"],
                    "shock_weak_dice": metric["dice"],
                    "vortex_weak_dice": None,
                    "evaluation_coverage": "four non-initialization selected MFC frames",
                    "direct_full_trajectory_comparison": False,
                    "report": str(path),
                }
            )
        else:
            metric = report["aggregate_weak_label_agreement"]
            rows.append(
                {
                    "name": baseline["name"],
                    "kind": baseline["kind"],
                    "shock_weak_dice": metric["shock"]["weak_dice"],
                    "vortex_weak_dice": metric["vortex_core"]["weak_dice"],
                    "evaluation_coverage": "60 weak-training frames",
                    "direct_full_trajectory_comparison": True,
                    "report": str(path),
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_index", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    root = Path(__file__).resolve().parents[1]
    config_path = args.config.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    figures_dir = output_dir / "figures"
    predictions_dir = output_dir / "selected_predictions"
    figures_dir.mkdir(parents=True, exist_ok=True)
    predictions_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if checkpoint["model_class"] != "ml.stage5_model.Stage5JointNet":
        raise ValueError("Checkpoint is not a Stage5JointNet")
    model = Stage5JointNet(
        input_channels=len(checkpoint["input_channels"]),
        base_channels=int(checkpoint["base_channels"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    torch.set_num_threads(int(checkpoint["config"].get("torch_num_threads", 8)))
    training_dataset = Stage5WeakFrames(
        args.dataset_index,
        input_channels=checkpoint["input_channels"],
        input_clip=checkpoint["config"].get("input_clip"),
        cache_size=2,
        training_only=True,
    )
    audit_dataset = Stage5WeakFrames(
        args.dataset_index,
        input_channels=checkpoint["input_channels"],
        input_clip=checkpoint["config"].get("input_clip"),
        cache_size=2,
        training_only=False,
    )
    evaluation = config["evaluation"]
    thresholds = np.asarray(evaluation["thresholds"], dtype=np.float64)
    tile_size = int(evaluation["tile_size"])
    overlap_size = int(evaluation["tile_overlap"])

    # Fit one global threshold per output on the intact weak-pretraining group.
    threshold_counts = np.zeros(
        (len(HEAD_NAMES), len(thresholds), 3), dtype=np.int64
    )
    raw_inside_geometry = np.zeros((2, len(thresholds)), dtype=np.int64)
    near_wall = np.zeros((2, len(thresholds)), dtype=np.int64)
    for frame_index, frame in enumerate(training_dataset.frames):
        loaded = training_dataset.load(frame_index)
        raw_probability = predict_tiled(
            model,
            loaded["inputs"],
            tile_size=tile_size,
            overlap=overlap_size,
        )
        geometry = loaded["geometry"]
        wall = ndimage.binary_dilation(geometry, iterations=5) & ~geometry
        probability = raw_probability.copy()
        probability[: len(PHYSICAL_HEAD_INDICES), geometry] = 0.0
        for head_index in range(len(HEAD_NAMES)):
            for threshold_index, threshold in enumerate(thresholds):
                threshold_counts[head_index, threshold_index] += counts_for_threshold(
                    probability[head_index],
                    loaded["targets"][head_index],
                    loaded["valid"][head_index],
                    float(threshold),
                )
                if head_index < 2:
                    raw_inside_geometry[head_index, threshold_index] += np.count_nonzero(
                        (raw_probability[head_index] >= threshold) & geometry
                    )
                    near_wall[head_index, threshold_index] += np.count_nonzero(
                        (probability[head_index] >= threshold) & wall
                    )
        print(f"calibration {frame_index + 1:02d}/{len(training_dataset)} step {int(frame['step']):05d}", flush=True)

    calibration: dict[str, Any] = {}
    selected_thresholds = np.zeros(len(HEAD_NAMES), dtype=np.float32)
    for head_index, head_name in enumerate(HEAD_NAMES):
        curve = []
        for threshold_index, threshold in enumerate(thresholds):
            item = metric_record(threshold_counts[head_index, threshold_index])
            item["threshold"] = float(threshold)
            if head_index < 2:
                item["raw_inside_geometry_pixels"] = int(
                    raw_inside_geometry[head_index, threshold_index]
                )
                item["near_wall_band_pixels"] = int(
                    near_wall[head_index, threshold_index]
                )
            curve.append(item)
        best = max(curve, key=lambda item: (item["weak_dice"], item["weak_precision"]))
        selected_thresholds[head_index] = float(best["threshold"])
        calibration[head_name] = {
            "selection_rule": "maximum aggregate weak-label Dice; precision tie-break",
            "best_fixed_threshold": float(best["threshold"]),
            "best_fit_metrics": best,
            "curve": curve,
        }

    calibration_figure, axes = plt.subplots(2, 3, figsize=(16.0, 9.0), sharex=True)
    for head_index, axis in enumerate(axes.ravel()):
        curve = calibration[HEAD_NAMES[head_index]]["curve"]
        axis.plot(thresholds, [item["weak_dice"] for item in curve], label="Dice")
        axis.plot(thresholds, [item["weak_precision"] for item in curve], label="precision")
        axis.plot(thresholds, [item["weak_recall"] for item in curve], label="recall")
        axis.axvline(selected_thresholds[head_index], color="black", linestyle="--")
        axis.set_title(
            f"{HEAD_NAMES[head_index]} @ {selected_thresholds[head_index]:.3f}"
        )
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    calibration_figure.suptitle(
        "Stage-5 fixed-threshold weak-fit calibration — no held-out validation"
    )
    calibration_figure.tight_layout(rect=[0, 0, 1, 0.96])
    calibration_figure_path = figures_dir / "stage5_fixed_threshold_calibration.png"
    calibration_figure.savefig(calibration_figure_path, dpi=180)
    plt.close(calibration_figure)

    grid = np.load(Path(audit_dataset.index["grid"]["grid_file"]))
    x = grid["x"]
    y = grid["y"]
    dx = abs(float(x[1] - x[0]))
    dy = abs(float(y[1] - y[0]))
    figure_steps = {int(step) for step in evaluation["figure_steps"]}
    save_steps = figure_steps | {0}
    figure_roles = {int(step): role for step, role in evaluation["figure_roles"].items()}
    selected_data: dict[int, dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    aggregate_counts = np.zeros((len(HEAD_NAMES), 3), dtype=np.int64)
    vortex_object_counts = np.zeros(3, dtype=np.int64)
    shock_distance_means: list[float] = []
    shock_distance_p95: list[float] = []
    for frame_index, frame in enumerate(audit_dataset.frames):
        loaded = audit_dataset.load(frame_index)
        raw_probability = predict_tiled(
            model,
            loaded["inputs"],
            tile_size=tile_size,
            overlap=overlap_size,
        )
        geometry = loaded["geometry"]
        probability = raw_probability.copy()
        probability[: len(PHYSICAL_HEAD_INDICES), geometry] = 0.0
        binary = probability >= selected_thresholds[:, None, None]
        heads: dict[str, Any] = {}
        for head_index, head_name in enumerate(HEAD_NAMES):
            counts = counts_for_threshold(
                probability[head_index],
                loaded["targets"][head_index],
                loaded["valid"][head_index],
                float(selected_thresholds[head_index]),
            )
            aggregate_counts[head_index] += counts
            heads[head_name] = metric_record(counts)
            heads[head_name]["threshold"] = float(selected_thresholds[head_index])
            reference_positive_pixels = int(
                np.count_nonzero(
                    (loaded["targets"][head_index] >= 0.5)
                    & (loaded["valid"][head_index] >= 0.5)
                )
            )
            heads[head_name]["reference_positive_pixels"] = reference_positive_pixels
            if reference_positive_pixels == 0:
                heads[head_name]["positive_reference_score_eligible"] = False
                for metric_name in (
                    "weak_dice",
                    "weak_iou",
                    "weak_precision",
                    "weak_recall",
                ):
                    heads[head_name][metric_name] = None
            else:
                heads[head_name]["positive_reference_score_eligible"] = True
        heads["shock"]["raw_inside_geometry_pixels"] = int(
            np.count_nonzero(
                (raw_probability[0] >= selected_thresholds[0]) & geometry
            )
        )
        heads["vortex_core"]["raw_inside_geometry_pixels"] = int(
            np.count_nonzero(
                (raw_probability[1] >= selected_thresholds[1]) & geometry
            )
        )
        heads["shock"]["inside_geometry_after_domain_mask_pixels"] = int(
            np.count_nonzero(binary[0] & geometry)
        )
        heads["vortex_core"]["inside_geometry_after_domain_mask_pixels"] = int(
            np.count_nonzero(binary[1] & geometry)
        )
        shock_distance = symmetric_mask_distance(
            binary[0], loaded["targets"][0] >= 0.5, spacing=(dy, dx)
        )
        if (
            bool(frame.get("shock_loss_eligible", True))
            and shock_distance["symmetric_mean_distance"] is not None
        ):
            shock_distance_means.append(float(shock_distance["symmetric_mean_distance"]))
            shock_distance_p95.append(float(shock_distance["symmetric_p95_distance"]))
        object_counts = np.asarray(
            object_match_counts(binary[1], loaded["targets"][1] >= 0.5),
            dtype=np.int64,
        )
        vortex_object_counts += object_counts
        step = int(loaded["step"])
        record = {
            "step": step,
            "time": float(loaded["time"]),
            "training_eligible": bool(frame.get("training_eligible_by_default", True)),
            "shock_loss_eligible": bool(frame.get("shock_loss_eligible", True)),
            "role": figure_roles.get(step),
            "heads": heads,
            "shock_topology": {
                "predicted_components_min_3_pixels": component_count(binary[0]),
                "weak_components_min_3_pixels": component_count(
                    loaded["targets"][0] >= 0.5
                ),
                **shock_distance,
            },
            "vortex_object_agreement": {
                "matched": int(object_counts[0]),
                "false_positive_objects": int(object_counts[1]),
                "false_negative_objects": int(object_counts[2]),
                "matching": "8-connected components >=3 pixels; Hungarian IoU >=0.1",
            },
        }
        records.append(record)
        if step in save_steps:
            output_path = predictions_dir / f"step_{step:05d}_stage5_joint.npz"
            np.savez_compressed(
                output_path,
                step=np.asarray(step, dtype=np.int64),
                time=np.asarray(float(loaded["time"]), dtype=np.float64),
                probabilities=probability.astype(np.float16),
                raw_probabilities=raw_probability.astype(np.float16),
                binary_masks=binary.astype(np.uint8),
                weak_targets=loaded["targets"].astype(np.uint8),
                valid_masks=loaded["valid"].astype(np.uint8),
                geometry_mask=geometry.astype(np.uint8),
                thresholds=selected_thresholds,
                output_heads=np.asarray(HEAD_NAMES, dtype="U32"),
            )
            with np.load(Path(frame["frame_file"])) as raw:
                schlieren = raw["diagnostics"][0].astype(np.float32)
            selected_data[step] = {
                "probability": probability,
                "binary": binary,
                "targets": loaded["targets"].astype(bool),
                "geometry": geometry,
                "schlieren": schlieren,
                "prediction_file": str(output_path),
            }
            record["prediction_file"] = str(output_path)
        print(f"audit {frame_index + 1:02d}/{len(audit_dataset)} step {step:05d}", flush=True)

    if figure_steps - set(selected_data):
        raise RuntimeError(f"Missing selected audit steps: {sorted(figure_steps - set(selected_data))}")
    rows = [selected_data[step] for step in evaluation["figure_steps"]]
    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
    crop = (-0.55, 3.55, -0.8, 2.9)
    montage, axes = plt.subplots(
        len(rows), 6, figsize=(23.0, 4.1 * len(rows)), sharex=True, sharey=True
    )
    for row_index, (step, data) in enumerate(zip(evaluation["figure_steps"], rows)):
        geometry = data["geometry"]
        schlieren = data["schlieren"]
        binary = data["binary"]
        targets = data["targets"]
        probability = data["probability"]
        axes[row_index, 0].imshow(
            schlieren,
            origin="lower",
            extent=extent,
            cmap="gray_r",
            vmin=0,
            vmax=float(evaluation.get("schlieren_vmax", 20.0)),
            interpolation="nearest",
        )
        axes[row_index, 0].set_title(
            f"{figure_roles[int(step)]}\nstep={int(step)}, t={int(step)/5400:.2f}"
        )
        for column, shock, vortex, title in (
            (1, targets[0], targets[1], "unreviewed weak labels"),
            (2, binary[0], binary[1], "joint learned prediction"),
        ):
            axes[row_index, column].imshow(
                schlieren,
                origin="lower",
                extent=extent,
                cmap="gray_r",
                vmin=0,
                vmax=float(evaluation.get("schlieren_vmax", 20.0)),
                interpolation="nearest",
            )
            axes[row_index, column].imshow(
                overlay(shock, vortex, geometry),
                origin="lower",
                extent=extent,
                interpolation="nearest",
            )
            axes[row_index, column].set_title(title + "\nred shock, cyan vortex")
        for column, head_index, title, cmap in (
            (3, 0, "P(shock)", "magma"),
            (4, 1, "P(vortex core)", "viridis"),
        ):
            image = axes[row_index, column].imshow(
                probability[head_index],
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=0,
                vmax=1,
                interpolation="nearest",
            )
            axes[row_index, column].contour(
                x, y, geometry, levels=[0.5], colors="white", linewidths=0.55
            )
            axes[row_index, column].set_title(title)
            montage.colorbar(image, ax=axes[row_index, column], fraction=0.046, pad=0.04)
        uncertainty = np.maximum(probability[4], probability[5])
        image = axes[row_index, 5].imshow(
            uncertainty,
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axes[row_index, 5].contour(
            x, y, geometry, levels=[0.5], colors="cyan", linewidths=0.55
        )
        axes[row_index, 5].set_title("max learned review-uncertainty")
        montage.colorbar(image, ax=axes[row_index, 5], fraction=0.046, pad=0.04)
        for axis in axes[row_index]:
            axis.set_xlim(crop[0], crop[1])
            axis.set_ylim(crop[2], crop[3])
            axis.set_aspect("equal")
            axis.set_xlabel("x/c")
            axis.set_ylabel("y/c")
    montage.suptitle(
        "Stage-5 joint dense ML-only pilot — weak-fit visual audit, not validation",
        fontsize=15,
    )
    montage.tight_layout(rect=[0, 0, 1, 0.975])
    montage_path = figures_dir / "stage5_joint_dense_montage.png"
    montage.savefig(montage_path, dpi=180)
    plt.close(montage)

    negative = selected_data[0]
    negative_figure, negative_axes = plt.subplots(1, 4, figsize=(16.0, 4.2))
    negative_axes[0].imshow(
        negative["schlieren"],
        origin="lower",
        extent=extent,
        cmap="gray_r",
        vmin=0,
        vmax=float(evaluation.get("schlieren_vmax", 20.0)),
        interpolation="nearest",
    )
    negative_axes[0].set_title("step 0: uniform-flow/geometry negative")
    for axis, head_index, title, cmap in (
        (negative_axes[1], 0, "P(shock)", "magma"),
        (negative_axes[2], 1, "P(vortex core)", "viridis"),
        (negative_axes[3], 3, "P(learned geometry)", "cividis"),
    ):
        image = axis.imshow(
            negative["probability"][head_index],
            origin="lower",
            extent=extent,
            cmap=cmap,
            vmin=0,
            vmax=1,
            interpolation="nearest",
        )
        axis.contour(x, y, negative["geometry"], levels=[0.5], colors="white", linewidths=0.7)
        axis.set_title(title)
        negative_figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    for axis in negative_axes:
        axis.set_xlim(crop[0], crop[1])
        axis.set_ylim(crop[2], crop[3])
        axis.set_aspect("equal")
        axis.set_xlabel("x/c")
        axis.set_ylabel("y/c")
    negative_figure.suptitle(
        "Untouched initialization negative audit — not used for training or threshold fitting"
    )
    negative_figure.tight_layout(rect=[0, 0, 1, 0.94])
    negative_figure_path = figures_dir / "stage5_step0_negative_audit.png"
    negative_figure.savefig(negative_figure_path, dpi=180)
    plt.close(negative_figure)

    aggregate = {
        head_name: metric_record(aggregate_counts[head_index])
        for head_index, head_name in enumerate(HEAD_NAMES)
    }
    object_tp, object_fp, object_fn = (int(value) for value in vortex_object_counts)
    object_summary = {
        "matched": object_tp,
        "false_positive_objects": object_fp,
        "false_negative_objects": object_fn,
        "weak_object_f1": 2.0 * object_tp / max(2 * object_tp + object_fp + object_fn, 1),
        "weak_object_precision": object_tp / max(object_tp + object_fp, 1),
        "weak_object_recall": object_tp / max(object_tp + object_fn, 1),
        "scope": "agreement with proposal components; partly circular and not object accuracy",
    }
    shock_topology_summary = {
        "frame_mean_symmetric_mean_distance_x_over_c": (
            float(np.mean(shock_distance_means)) if shock_distance_means else None
        ),
        "frame_mean_symmetric_p95_distance_x_over_c": (
            float(np.mean(shock_distance_p95)) if shock_distance_p95 else None
        ),
        "scope": "distance to unreviewed weak shock masks; not ground-truth front error",
    }
    comparison = load_baseline_comparison(config, root)
    comparison.append(
        {
            "name": "Stage-5 joint dense v1",
            "kind": "joint_dense_ml_only",
            "shock_weak_dice": calibration["shock"]["best_fit_metrics"]["weak_dice"],
            "vortex_weak_dice": calibration["vortex_core"]["best_fit_metrics"]["weak_dice"],
            "evaluation_coverage": "60 weak-training frames (step 0 reported separately as a negative audit)",
            "direct_full_trajectory_comparison": True,
        }
    )
    report = {
        "schema_version": "2.0",
        "experiment_id": checkpoint["experiment_id"],
        "scientific_scope": (
            "fit agreement with unreviewed weak labels from one intact trajectory; "
            "not accuracy, human ground truth, uncertainty calibration, or independent validation"
        ),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "dataset_index": str(args.dataset_index.resolve()),
        "dataset_index_sha256": sha256(args.dataset_index.resolve()),
        "case_group_id": audit_dataset.index["split_policy"]["case_group_id"],
        "split_policy": audit_dataset.index["split_policy"],
        "training_frames_used_for_threshold_fit": len(training_dataset),
        "audit_frames": len(audit_dataset),
        "threshold_policy": (
            "one global fixed threshold per head fitted on the weak-pretraining group; "
            "no per-frame quantile or threshold tuning"
        ),
        "selected_thresholds": {
            head_name: float(selected_thresholds[index])
            for index, head_name in enumerate(HEAD_NAMES)
        },
        "calibration": calibration,
        "aggregate_weak_label_agreement": aggregate,
        "vortex_object_agreement": object_summary,
        "shock_topology_agreement": shock_topology_summary,
        "geometry_domain_audit": {
            "shock_inside_exact_geometry_after_mask_all_frames": sum(
                int(record["heads"]["shock"]["inside_geometry_after_domain_mask_pixels"])
                for record in records
            ),
            "vortex_inside_exact_geometry_after_mask_all_frames": sum(
                int(record["heads"]["vortex_core"]["inside_geometry_after_domain_mask_pixels"])
                for record in records
            ),
            "raw_shock_inside_geometry_all_frames": sum(
                int(record["heads"]["shock"]["raw_inside_geometry_pixels"])
                for record in records
            ),
            "raw_vortex_inside_geometry_all_frames": sum(
                int(record["heads"]["vortex_core"]["raw_inside_geometry_pixels"])
                for record in records
            ),
        },
        "baseline_comparison": comparison,
        "figures": {
            "montage": str(montage_path),
            "fixed_threshold_calibration": str(calibration_figure_path),
            "untouched_step0_negative_audit": str(negative_figure_path),
        },
        "selected_predictions": [
            {
                "step": int(step),
                "role": figure_roles[int(step)],
                "file": selected_data[int(step)]["prediction_file"],
            }
            for step in evaluation["figure_steps"]
        ],
        "frames": records,
        "elapsed_seconds": time.perf_counter() - started,
    }
    report_path = output_dir / "stage5_joint_dense_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "report": str(report_path),
                "montage": str(montage_path),
                "selected_thresholds": report["selected_thresholds"],
                "aggregate_weak_label_agreement": aggregate,
                "vortex_object_agreement": object_summary,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
