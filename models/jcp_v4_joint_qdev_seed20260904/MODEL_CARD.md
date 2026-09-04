# v4 joint Qdev model: development checkpoint

- Architecture: shared-encoder, separate-decoder six-logit CNN; 268,496 parameters.
- Input: four physically normalized primitive channels and three recomputed,
  bounded differential channels. Exact geometry is not a network input.
- Training: from scratch, seed 20260904, 600 optimizer steps, batch four,
  128 x 128 patches, weak labels, train-only case-group sampling.
- Calibration: validation-family macro weak Dice only. Frozen thresholds are
  0.97 for shock and 0.85 for vortex core; these are not human-calibrated.
- Checkpoint SHA-256:
  `ea7b6bfdae4b0fc7c520d6b7679337ad3ff173cc9a8ceb4f57133029dd07850a`.
- Current scope: the first model in a five-variant, three-initialization campaign.

Training, validation and previously inspected development-control groups are
kept distinct. The development controls are not genuinely untouched cases.
Any provided metrics are agreement with weak proposals, not independent
accuracy. Expert review is outstanding. The public examples are training-case
snapshots and must not be used to claim held-out performance.

Shock/core overlap is permitted. Near-wall shear, shocks, solver transitions,
derivative noise and unfamiliar flow regimes can still cause false positives
and missed cores. Sigmoid outputs and ambiguity heads are not independently
calibrated uncertainty. This checkpoint is not endorsed as superior to the
other variants before the complete comparison is finished.

The portable CLI makes ML-only predictions. Hybrid and physics-only products
are separate research branches; the CLI does not require their masks. A
successful program exit means the pipeline executed, not that its detections
are scientifically correct.
