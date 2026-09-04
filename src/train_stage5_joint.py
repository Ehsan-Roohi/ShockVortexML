#!/usr/bin/env python3
"""Train the Stage-5 joint dense model on one weak-pretraining case group."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as functional

from ml.stage5_dataset import FOCUS_NAMES, Stage5WeakFrames
from ml.stage5_model import Stage5JointNet


HEAD_NAMES = Stage5JointNet.output_names


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def masked_bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    positive_weight_cap: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid_count = valid.sum()
    if float(valid_count.detach()) < 1.0:
        zero = logits.sum() * 0.0
        return zero, {"bce": 0.0, "dice_loss": 0.0, "positive_weight": 1.0}
    positive = (target * valid).sum()
    negative = valid_count - positive
    if float(positive.detach()) > 0.0:
        positive_weight = torch.clamp(
            negative / positive, min=1.0, max=positive_weight_cap
        )
    else:
        positive_weight = torch.ones((), dtype=logits.dtype, device=logits.device)
    bce_map = functional.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=positive_weight
    )
    bce = (bce_map * valid).sum() / valid_count.clamp_min(1.0)
    probability = torch.sigmoid(logits)
    intersection = (probability * target * valid).sum()
    denominator = (probability * valid).sum() + (target * valid).sum()
    dice_loss = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
    return 0.8 * bce + 0.2 * dice_loss, {
        "bce": float(bce.detach()),
        "dice_loss": float(dice_loss.detach()),
        "positive_weight": float(positive_weight.detach()),
    }


def binary_counts(
    logits: torch.Tensor, target: torch.Tensor, valid: torch.Tensor
) -> tuple[int, int, int]:
    predicted = torch.sigmoid(logits) >= 0.5
    truth = target >= 0.5
    allowed = valid >= 0.5
    return (
        int((predicted & truth & allowed).sum().item()),
        int((predicted & ~truth & allowed).sum().item()),
        int((~predicted & truth & allowed).sum().item()),
    )


def metrics_from_counts(counts: np.ndarray) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for head_index, head_name in enumerate(HEAD_NAMES):
        true_positive, false_positive, false_negative = counts[head_index]
        report[head_name] = {
            "weak_dice": (2.0 * true_positive)
            / max(2.0 * true_positive + false_positive + false_negative, 1.0),
            "weak_precision": true_positive / max(true_positive + false_positive, 1.0),
            "weak_recall": true_positive / max(true_positive + false_negative, 1.0),
        }
    return report


def flatten_gradients(gradients: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def project_conflicting_gradients(
    shock_gradients: list[torch.Tensor],
    vortex_gradients: list[torch.Tensor],
    epsilon: float = 1e-12,
) -> tuple[list[torch.Tensor], list[torch.Tensor], float, bool]:
    """Apply symmetric PCGrad projection to two shared-encoder gradients."""

    shock_flat = flatten_gradients(shock_gradients)
    vortex_flat = flatten_gradients(vortex_gradients)
    dot = torch.dot(shock_flat, vortex_flat)
    shock_norm = torch.dot(shock_flat, shock_flat)
    vortex_norm = torch.dot(vortex_flat, vortex_flat)
    cosine = float(
        (dot / torch.sqrt(shock_norm.clamp_min(epsilon) * vortex_norm.clamp_min(epsilon)))
        .detach()
        .cpu()
    )
    if float(dot.detach()) >= 0.0:
        return shock_gradients, vortex_gradients, cosine, False
    dot_value = dot.detach()
    shock_scale = dot_value / vortex_norm.detach().clamp_min(epsilon)
    vortex_scale = dot_value / shock_norm.detach().clamp_min(epsilon)
    projected_shock = [
        shock - shock_scale * vortex
        for shock, vortex in zip(shock_gradients, vortex_gradients)
    ]
    projected_vortex = [
        vortex - vortex_scale * shock
        for shock, vortex in zip(shock_gradients, vortex_gradients)
    ]
    return projected_shock, projected_vortex, cosine, True


def gradients_or_zeros(
    loss: torch.Tensor, parameters: list[nn.Parameter], *, retain_graph: bool
) -> list[torch.Tensor]:
    gradients = torch.autograd.grad(
        loss, parameters, retain_graph=retain_graph, allow_unused=True
    )
    return [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, gradients)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_index", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--steps-per-epoch", type=int, default=None)
    args = parser.parse_args()

    started = time.perf_counter()
    index_path = args.dataset_index.resolve()
    config_path = args.config.resolve()
    output_dir = args.output_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if tuple(config["output_heads"]) != HEAD_NAMES:
        raise ValueError("Configured output heads do not match Stage5JointNet")
    if "geometry_mask" in config["input_channels"]:
        raise ValueError("Exact geometry cannot be used as a Stage-5 input shortcut")
    epochs = int(args.epochs or config["epochs"])
    steps_per_epoch = int(args.steps_per_epoch or config["steps_per_epoch"])
    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(int(config.get("torch_num_threads", 8)))
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(seed)

    dataset = Stage5WeakFrames(
        index_path,
        input_channels=config["input_channels"],
        input_clip=config.get("input_clip"),
        training_only=True,
    )
    split_policy = dataset.index["split_policy"]
    if split_policy.get("assignment") != "weak_pretrain_only":
        raise ValueError("Stage-5 pilot expects the intact weak_pretrain_only group")
    model = Stage5JointNet(
        input_channels=len(config["input_channels"]),
        base_channels=int(config["base_channels"]),
    )
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(epochs * steps_per_epoch, 1),
        eta_min=float(config.get("minimum_learning_rate", 1e-5)),
    )
    head_weights = torch.as_tensor(config["head_loss_weights"], dtype=torch.float32)
    positive_caps = config["positive_weight_cap"]
    batch_size = int(config["batch_size"])
    patch_size = int(config["patch_size"])
    focus_probabilities = tuple(float(value) for value in config["focus_probabilities"])
    pcgrad_enabled = bool(config.get("pcgrad_shared_encoder", True))
    pcgrad_interval = int(config.get("pcgrad_interval", 1))
    if pcgrad_interval < 1:
        raise ValueError("pcgrad_interval must be at least one")
    trainable_parameters = sum(parameter.numel() for parameter in model.parameters())
    print(
        json.dumps(
            {
                "device": str(device),
                "torch": torch.__version__,
                "model": "Stage5JointNet",
                "trainable_parameters": trainable_parameters,
                "eligible_frames": len(dataset),
                "case_group_id": split_policy["case_group_id"],
                "epochs": epochs,
                "steps_per_epoch": steps_per_epoch,
                "pcgrad_shared_encoder": pcgrad_enabled,
                "pcgrad_interval": pcgrad_interval,
            },
            indent=2,
        ),
        flush=True,
    )

    history: list[dict[str, object]] = []
    shared_parameters = model.shared_encoder_parameters()
    for epoch in range(epochs):
        epoch_started = time.perf_counter()
        model.train()
        total_sum = 0.0
        head_loss_sum = np.zeros(len(HEAD_NAMES), dtype=np.float64)
        counts = np.zeros((len(HEAD_NAMES), 3), dtype=np.int64)
        focus_counts = {name: 0 for name in FOCUS_NAMES}
        gradient_cosines: list[float] = []
        conflict_steps = 0
        pcgrad_steps = 0
        for step_index in range(steps_per_epoch):
            patches = [
                dataset.sample_patch(
                    rng,
                    patch_size=patch_size,
                    focus_probabilities=focus_probabilities,
                )
                for _batch in range(batch_size)
            ]
            inputs = torch.from_numpy(np.stack([patch[0] for patch in patches])).to(device)
            targets = torch.from_numpy(np.stack([patch[1] for patch in patches])).to(device)
            valid = torch.from_numpy(np.stack([patch[2] for patch in patches])).to(device)
            for patch in patches:
                focus_counts[patch[3]["focus"]] += 1

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            head_losses: list[torch.Tensor] = []
            for head_index, head_name in enumerate(HEAD_NAMES):
                cap = (
                    float(positive_caps[head_index])
                    if isinstance(positive_caps, list)
                    else float(positive_caps)
                )
                head_loss, _details = masked_bce_dice_loss(
                    logits[:, head_index],
                    targets[:, head_index],
                    valid[:, head_index],
                    cap,
                )
                head_losses.append(head_loss)
                head_loss_sum[head_index] += float(head_loss.detach())
                counts[head_index] += np.asarray(
                    binary_counts(
                        logits[:, head_index], targets[:, head_index], valid[:, head_index]
                    ),
                    dtype=np.int64,
                )
            shock_group = head_weights[0] * head_losses[0] + head_weights[4] * head_losses[4]
            vortex_group = head_weights[1] * head_losses[1] + head_weights[5] * head_losses[5]
            auxiliary_group = (
                head_weights[2] * head_losses[2] + head_weights[3] * head_losses[3]
            )
            total_loss = shock_group + vortex_group + auxiliary_group
            if not torch.isfinite(total_loss):
                raise FloatingPointError(f"Non-finite loss at epoch {epoch + 1}")

            apply_pcgrad = pcgrad_enabled and step_index % pcgrad_interval == 0
            if apply_pcgrad:
                pcgrad_steps += 1
                shock_gradients = gradients_or_zeros(
                    shock_group, shared_parameters, retain_graph=True
                )
                vortex_gradients = gradients_or_zeros(
                    vortex_group, shared_parameters, retain_graph=True
                )
                auxiliary_gradients = gradients_or_zeros(
                    auxiliary_group, shared_parameters, retain_graph=True
                )
                (
                    shock_gradients,
                    vortex_gradients,
                    cosine,
                    had_conflict,
                ) = project_conflicting_gradients(shock_gradients, vortex_gradients)
                gradient_cosines.append(cosine)
                conflict_steps += int(had_conflict)
                total_loss.backward()
                for parameter, shock_gradient, vortex_gradient, auxiliary_gradient in zip(
                    shared_parameters,
                    shock_gradients,
                    vortex_gradients,
                    auxiliary_gradients,
                ):
                    parameter.grad = (
                        shock_gradient + vortex_gradient + auxiliary_gradient
                    ).detach()
            else:
                total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            scheduler.step()
            total_sum += float(total_loss.detach())

        record: dict[str, object] = {
            "epoch": epoch + 1,
            "mean_total_loss": total_sum / steps_per_epoch,
            "mean_head_losses": {
                head_name: head_loss_sum[index] / steps_per_epoch
                for index, head_name in enumerate(HEAD_NAMES)
            },
            "patch_weak_label_agreement_at_0.5": metrics_from_counts(counts),
            "focus_counts": focus_counts,
            "pcgrad": {
                "enabled": pcgrad_enabled,
                "interval": pcgrad_interval,
                "applied_steps": pcgrad_steps,
                "mean_shared_encoder_task_gradient_cosine": (
                    float(np.mean(gradient_cosines)) if gradient_cosines else None
                ),
                "conflict_steps": conflict_steps,
                "conflict_fraction": (
                    conflict_steps / pcgrad_steps if pcgrad_steps else None
                ),
            },
            "learning_rate_end": optimizer.param_groups[0]["lr"],
            "wall_seconds": time.perf_counter() - epoch_started,
        }
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

    checkpoint = {
        "schema_version": "2.0",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "scope": config["scope"],
        "model_class": "ml.stage5_model.Stage5JointNet",
        "model_state_dict": model.state_dict(),
        "input_channels": config["input_channels"],
        "output_heads": list(HEAD_NAMES),
        "base_channels": int(config["base_channels"]),
        "normalization_mean": dataset.mean.tolist(),
        "normalization_std": dataset.std.tolist(),
        "dataset_index": str(index_path),
        "dataset_index_sha256": sha256(index_path),
        "case_group_id": split_policy["case_group_id"],
        "split_assignment": split_policy["assignment"],
        "config": config,
        "history": history,
    }
    torch.save(checkpoint, checkpoint_path)
    checkpoint_hash = sha256(checkpoint_path)
    report = {
        "schema_version": "2.0",
        "experiment_id": config["experiment_id"],
        "scientific_scope": (
            "joint dense weak-supervision training on one intact trajectory; "
            "not accuracy, human ground truth, calibration, or independent validation"
        ),
        "model": {
            "class": checkpoint["model_class"],
            "trainable_parameters": trainable_parameters,
            "input_channels": config["input_channels"],
            "output_heads": list(HEAD_NAMES),
            "independent_shock_vortex_logits": True,
            "shared_encoder_pcgrad": pcgrad_enabled,
        },
        "dataset_index": str(index_path),
        "dataset_index_sha256": checkpoint["dataset_index_sha256"],
        "case_group_id": split_policy["case_group_id"],
        "split_assignment": split_policy["assignment"],
        "eligible_training_frames": len(dataset),
        "config_file": str(config_path),
        "config_sha256": sha256(config_path),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "torch_version": torch.__version__,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    report_path = output_dir / "training_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    epochs_axis = [int(item["epoch"]) for item in history]
    axes[0].plot(
        epochs_axis,
        [float(item["mean_total_loss"]) for item in history],
        marker="o",
        label="total",
    )
    for head_name in HEAD_NAMES:
        axes[0].plot(
            epochs_axis,
            [float(item["mean_head_losses"][head_name]) for item in history],
            marker=".",
            label=head_name,
        )
    axes[0].set_ylabel("masked weak-label loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].plot(
        epochs_axis,
        [
            float(item["pcgrad"]["mean_shared_encoder_task_gradient_cosine"])
            for item in history
        ],
        marker="o",
        label="shock-vortex gradient cosine",
    )
    axes[1].plot(
        epochs_axis,
        [float(item["pcgrad"]["conflict_fraction"]) for item in history],
        marker="s",
        label="conflict fraction",
    )
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("shared-encoder interference audit")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    figure.suptitle("Stage-5 joint dense pilot — one weak-pretraining group")
    figure.tight_layout(rect=[0, 0, 1, 0.97])
    curve_path = output_dir / "training_and_gradient_audit.png"
    figure.savefig(curve_path, dpi=180)
    plt.close(figure)
    print(
        json.dumps(
            {
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256": checkpoint_hash,
                "report": str(report_path),
                "figure": str(curve_path),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
