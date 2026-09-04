# Reproducibility and release contract

## Status

This v4 lineage targets JCP, but it is a development campaign, not an accepted
method or completed independent benchmark. The author approved the independent
public repository `Ehsan-Roohi/ShockVortexML` with the MIT license and inclusion
of suitable data. This bootstrap contains one completed development model and
two numerical-QC-passed primitive-field samples, not the full training dataset.
See the root README, `PUBLIC_ASSET_MANIFEST.json`, `RELEASE_QA.json` and the
sample/model cards. Preserve the author-derived manuscript.

## Training and comparisons on the full project data

On a portable Python 3.12 installation, run `python -m pip install -e '.[research]'`
from this checkout. The author's machine-specific runtime helper is deliberately
not distributed. The original raw-data exports,
legacy source manifest, weak proposals and temporal override archives are
required for the full preparation stage; they are NOT in a source-only ZIP.

```text
python -m unittest discover -s tests -p "test_jcp*.py" -v
python -m jcp2026.prepare
python -m jcp2026.patch_bank
python -m jcp2026.campaign
```

The campaign freezes source hashes, trains each of five variants for each of
three initialization seeds, then calibrates only on the validation groups.
Evaluation uses validation and previously inspected development-control cases.
It also runs clean, adverse, late-wake and deterministic noisy Mach-3/alpha-40
regressions. No test-driven checkpoint selection or threshold changes occur.
Training budgets match optimizer updates and sampled patches, not parameter
count or runtime. Initial-seed dispersion is not a case-generalization CI.

## Portable inference without reference masks

An inference input is an NPZ archive with 2-D arrays `rho`, `pressure`, `u`,
`v`; strictly increasing one-dimensional vectors `x`, `y`; and preferably
boolean `geometry` and `observation_mask`. Unobserved raster padding is not
fluid. Fields must use consistent nondimensional units and positive observed
density and pressure. The coordinates correspond to rows `y` and columns `x`.

A separate JSON supplies `rho_inf`, `pressure_inf`, `u_inf_x`, `u_inf_y`,
`gamma` and `reference_length`. For example, a Mach-2.7 horizontal inflow
with rho_inf=1 and pressure_inf=1/1.4 has u_inf_x=2.7, u_inf_y=0,
gamma=1.4 and reference_length=1. Do not use these values for unrelated cases.

```text
python -m jcp2026.infer --input fields.npz --reference freestream.json \
  --checkpoint checkpoint.pt --thresholds thresholds.json --output prediction.npz
```

The checkpoint and frozen-threshold file must have matching hashes. Output
contains separate `p_shock`, `p_vortex_core`, binary masks and frame-local
vortex component IDs. Shock/core overlap is legal. `background_other` is the
complement of accepted shock/core masks on observed fluid, while
`p_background_head` is a separate sigmoid head; the independent heads are not
a mutually exclusive softmax. Scores are not claimed independently calibrated
probabilities. Frame-local IDs are not temporal tracks or physical truth.

The new CLI does not need physics labels or a case-specific detector to make
ML-only predictions. It does not promise correctness on every geometry,
resolution, Mach number, solver or rarefaction regime.

## Expert review and final test

Follow `docs/JCP_EXPERT_REVIEW.md`. After actual review, the expert fills a
COPY of `configs/jcp_expert_attestation_TEMPLATE.json`; the supplied template
deliberately has false/empty fields and cannot unlock independent evaluation.

```text
python tools/jcp_freeze_expert_review.py --attestation signed_review.json \
  --output results/jcp_v4/expert_reference.json
python tools/evaluate_jcp_expert_reference.py \
  --reference results/jcp_v4/expert_reference.json \
  --run results/jcp_v4/joint_qdev/20260904
```

That comparison excludes train rows and uses already frozen outputs. Human
review of previously inspected cases is still a development reference. A new
whole case must be registered before opening its fields and evaluated once
after model/config/threshold freeze to support an untouched-test claim.

## Distribution safeguards

The candidate builder excludes emails, manuscripts, raw CFD archives and
third-party installed packages. It includes local scientific-source dependency
closure, pinned requirements, configuration, tests and only completed model
artifacts. Its manifest lists exactly which runs are present and which are
missing, plus SHA-256 and size for every file. An incomplete campaign cannot
be silently described as a complete three-seed benchmark.

Author-approved source, model and the two samples are included in this MIT
bootstrap; installed dependencies retain their own licenses. No third-party
papers or pretrained weights are redistributed. Full training-data distribution
is separate from the lightweight inference artifact. A larger validated release
still requires the completed campaign and expert reference. Git line-ending
conversion is disabled in `.gitattributes` to preserve recorded byte hashes.
